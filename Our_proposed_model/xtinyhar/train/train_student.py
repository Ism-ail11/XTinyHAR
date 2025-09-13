
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.inertial_transformer import InertialTransformer
from ..kd.losses import DistillationLoss

def train_student_kd(train_loader: DataLoader, val_loader: DataLoader,
                     teacher, num_classes=27, device="cuda",
                     in_channels=6, window_size=100, patch_size=20,
                     dim=128, depth=2, heads=4,
                     lr=1e-4, epochs=20, alpha=0.7, T=3.0):
    student = InertialTransformer(in_channels=in_channels, window_size=window_size, patch_size=patch_size,
                                  dim=dim, depth=depth, heads=heads, num_classes=num_classes).to(device)
    kd = DistillationLoss(alpha=alpha, T=T)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    best_val = -1.0
    best_state = None

    for ep in range(epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"Student KD Train [{ep+1}/{epochs}]")
        run_loss = 0.0
        for batch in pbar:
            x_iner = batch["iner"].to(device)
            y = batch["label"].to(device)

            with torch.no_grad():
                B, W, C = x_iner.shape
                P = patch_size
                N = W // P
                x_pool = x_iner[:, :N*P, :].reshape(B, N, P*C)
                proj = getattr(train_student_kd, "_teacher_proj", None)
                if proj is None:
                    proj = nn.Linear(P*C, dim).to(device)
                    proj.weight.requires_grad_(False)
                    proj.bias.requires_grad_(False)
                    train_student_kd._teacher_proj = proj
                iner_tok = proj(x_pool)
                t_logits, _ = teacher(iner_tok, batch["skel"].to(device))

            opt.zero_grad(set_to_none=True)
            s_logits, _ = student(x_iner)
            loss, parts = kd(s_logits, t_logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            pbar.set_postfix(loss=run_loss / (pbar.n+1), ce=float(parts["ce"]), kl=float(parts["kl"]))

        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = student(batch["iner"].to(device))
                pred = logits.argmax(1).cpu()
                correct += (pred == batch["label"]).sum().item()
                total += pred.numel()
        val_acc = correct / max(1,total)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
    return best_state, best_val
