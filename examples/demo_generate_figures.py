import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from tqdm import trange

from xai.attention_maps import extract_last_layer_class_attention, attention_vector_to_heatmap
from xai.integrated_gradients import integrated_gradients, attribution_to_heatmap
from xai.attention_rollout import attention_rollout, rollout_cls_to_patches
from xai.attention_similarity import mean_cosine_similarity
from utils.plotting import save_line_over_time, save_heatmap

import os
os.makedirs('outputs', exist_ok=True)

# ---- Tiny Toy Models (for demo only) ----
class TinySelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B,T,D = x.shape
        qkv = self.to_qkv(x).reshape(B,T,3,self.heads,D//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # [B,H,T,d]
        attn = (q @ k.transpose(-2,-1)) * self.scale  # [B,H,T,T]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B,T,D)
        out = self.out(out)
        return out, attn

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim),
        )

    def forward(self, x):
        h, attn = self.attn(self.ln1(x))
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, attn

class TinyTransformer(nn.Module):
    def __init__(self, dim=64, depth=3, heads=4, num_classes=10, seq_len=51):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1,seq_len,dim))
        self.blocks = nn.ModuleList([TinyTransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, return_attn=False):
        # x: [B, W, C] -> we pretend tokens already embedded; project to dim
        B,W,C = x.shape
        # make tokens: CLS + 50 patches (toy)
        if W != 50:
            x = x[:, :50, :]  # trim/pad for demo
        tok = torch.cat([self.cls.expand(B,-1,-1), torch.randn(B,50,self.cls.shape[-1], device=x.device)], dim=1)
        tok = tok + self.pos[:, :tok.shape[1], :]
        attn_list = []
        for blk in self.blocks:
            tok, attn = blk(tok)
            attn_list.append(attn)
        logits = self.head(tok[:,0,:])
        if return_attn:
            return logits, attn_list
        return logits

def synthetic_loader(B=16, W=200, C=6, batches=8):
    for _ in range(batches):
        x = torch.randn(B, W, C)
        y = torch.randint(0, 10, (B,))
        yield x, y

def target_fn_from_pred(logits):
    # choose predicted logit per sample (for IG)
    preds = logits.argmax(dim=-1)
    return logits.gather(1, preds.view(-1,1)).squeeze(1)

def main():
    device = torch.device('cpu')
    teacher = TinyTransformer(dim=64, depth=3, heads=4, num_classes=10, seq_len=51).to(device)
    student = TinyTransformer(dim=64, depth=2, heads=4, num_classes=10, seq_len=51).to(device)

    # ---- Attention Visualization (class-token) ----
    x, y = next(iter(synthetic_loader(B=8)))
    x = x.to(device)
    s_logits, s_attn = student(x, return_attn=True)
    cls_vec = extract_last_layer_class_attention(s_attn).mean(dim=0)  # [T-1]
    from utils.plotting import save_line_over_time
    save_line_over_time(cls_vec.detach().cpu().numpy(), xlabel="Temporal patch index", ylabel="Attention weight",
                        title="Student class-token attention over time (UTD-like)", path="outputs/attention_utd.png")
    save_line_over_time(cls_vec.detach().cpu().numpy(), xlabel="Temporal patch index", ylabel="Attention weight",
                        title="Student class-token attention over time (MM-Fit-like)", path="outputs/attention_mmfit.png")

    # ---- Integrated Gradients ----
    x, y = next(iter(synthetic_loader(B=1)))
    x = x.to(device).requires_grad_(True)
    ig = integrated_gradients(student, x, target_fn_from_pred, baseline="zero", steps=32)[0]  # [W,C]
    ig_hm = attribution_to_heatmap(ig)  # [C,W]
    save_heatmap(ig_hm, xlabel="Temporal patches (T0–T49)", ylabel="IMU Channels",
                 title="IG heatmap (UTD-like)", path="outputs/ig_utd_num.png")
    save_heatmap(ig_hm, xlabel="Temporal patches", ylabel="IMU Channels",
                 title="IG heatmap (MM-Fit-like)", path="outputs/ig_mmfit_notitle.png")

    # ---- Attention rollout ----
    _, s_attn = student(x, return_attn=True)
    roll = attention_rollout(s_attn)  # [B,T,T]
    vec = rollout_cls_to_patches(roll)[0]  # [T-1]
    hm = vec.detach().cpu().numpy()[None,:]
    save_heatmap(hm, xlabel="Temporal patches", ylabel="Rollout (CLS→patch)", title="Attention rollout (UTD-like)",
                 path="outputs/rollout_utd_notitle.png")
    save_heatmap(hm, xlabel="Temporal patches", ylabel="Rollout (CLS→patch)", title="Attention rollout (MM-Fit-like)",
                 path="outputs/rollout_mmfit_notitle.png")

    # ---- Attention similarity (student vs teacher) ----
    sims = []
    for x, y in synthetic_loader(B=8, batches=12):
        x = x.to(device)
        _, s_attn = student(x, return_attn=True)
        _, t_attn = teacher(x, return_attn=True)
        s_roll = rollout_cls_to_patches(attention_rollout(s_attn))  # [B,T-1]
        t_roll = rollout_cls_to_patches(attention_rollout(t_attn))  # [B,T-1]
        mean_sim = (s_roll / (s_roll.norm(dim=-1, keepdim=True)+1e-8) * (t_roll / (t_roll.norm(dim=-1, keepdim=True)+1e-8))).sum(dim=-1)
        sims.append(mean_sim.detach().cpu().numpy())
    sims = np.concatenate(sims, axis=0)
    print(f"Mean cosine similarity (toy): {sims.mean():.3f} ± {sims.std():.3f}")

    # save a similarity heatmap for a batch (cosine per patch is 1D; for paper-style figure, we plot a 2D row)
    # here we just reuse the rollout vec as proxy visualization
    save_heatmap(hm, xlabel="Temporal patches", ylabel="Cosine-sim row", title="Attention similarity (UTD-like)",
                 path="outputs/similarity_utd_notitle.png")
    save_heatmap(hm, xlabel="Temporal patches", ylabel="Cosine-sim row", title="Attention similarity (MM-Fit-like)",
                 path="outputs/similarity_mmfit_notitle.png")

if __name__ == "__main__":
    main()
