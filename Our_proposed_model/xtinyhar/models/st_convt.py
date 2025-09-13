
import torch
import torch.nn as nn
from torch import Tensor
from .blocks import TransformerEncoderLayer, PositionalEmbeddingLearned

class SpatialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_t=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_t), padding=(0, kernel_t//2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_t), padding=(0, kernel_t//2))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x  # (B, Cout, J, W)

class TemporalBlock(nn.Module):
    def __init__(self, dim=128, depth=2, heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=drop)
        for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.norm(x)
        return x, attn_maps

class STConvTTeacher(nn.Module):
    def __init__(self, num_classes=27, dim=128, depth=2, heads=4, mlp_ratio=2.0, drop=0.1,
                 J=20, W=100, sk_in_channels=3):
        super().__init__()
        self.spatial = SpatialBlock(in_channels=sk_in_channels, out_channels=32, kernel_t=9)
        patch_len = 20
        self.N = W // patch_len
        self.skel_proj = nn.Linear(32 * J * patch_len, dim)
        self.cls_s = nn.Parameter(torch.zeros(1,1,dim))
        self.cls_i = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = PositionalEmbeddingLearned(seq_len=self.N + 1, dim=dim)

        self.temp_s = TemporalBlock(dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio, drop=drop)
        self.temp_i = TemporalBlock(dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio, drop=drop)
        self.fuse = TemporalBlock(dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio, drop=drop)

        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_s, std=0.02)
        nn.init.trunc_normal_(self.cls_i, std=0.02)

    def _tokenize_skeleton(self, x_skel: Tensor) -> Tensor:
        B, C, J, W = x_skel.shape
        h = self.spatial(x_skel)  # (B, 32, J, W)
        patch_len = W // self.N
        tokens = []
        for p in range(self.N):
            seg = h[:, :, :, p*patch_len:(p+1)*patch_len]  # (B, 32, J, patch_len)
            tokens.append(seg.flatten(1))  # (B, 32*J*patch_len)
        tok = torch.stack(tokens, dim=1)  # (B, N, 32*J*patch_len)
        tok = self.skel_proj(tok)         # (B, N, D)
        return tok

    def _add_cls_pos(self, tok: Tensor, cls_tok: Tensor) -> Tensor:
        B = tok.size(0)
        cls = cls_tok.expand(B, -1, -1)
        x = torch.cat([cls, tok], dim=1)
        x = self.pos(x)
        return x

    def forward(self, x_iner_tok: Tensor, x_skel: Tensor):
        s_tok = self._tokenize_skeleton(x_skel)
        s_seq = self._add_cls_pos(s_tok, self.cls_s)
        s_h, s_attn = self.temp_s(s_seq)

        i_seq = self._add_cls_pos(x_iner_tok, self.cls_i)
        i_h, i_attn = self.temp_i(i_seq)

        comb = s_h + i_h
        comb, f_attn = self.fuse(comb)

        logits = self.head(comb[:,0])
        return logits, {"s_attn": s_attn, "i_attn": i_attn, "f_attn": f_attn}
