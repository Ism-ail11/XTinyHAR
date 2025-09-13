
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from xtinyhar.train.train_teacher import train_teacher_loop
from xtinyhar.train.train_student import train_student_kd
from xtinyhar.models.st_convt import STConvTTeacher

class DummyHAR(Dataset):
    def __init__(self, n=256, W=100, C=6, J=20, num_classes=27, split="train"):
        rng = np.random.RandomState(42 if split=="train" else 7)
        self.iner = rng.randn(n, W, C).astype(np.float32)
        self.skel = rng.randn(n, 3, J, W).astype(np.float32)
        self.y = rng.randint(0, num_classes, size=(n,)).astype(np.int64)
        self.W, self.C, self.J = W, C, J

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {
            "iner": torch.from_numpy(self.iner[idx]),
            "skel": torch.from_numpy(self.skel[idx]),
            "label": torch.tensor(self.y[idx], dtype=torch.long)
        }

def collate_teacher(batch, dim=128, patch_size=20):
    W = batch[0]["iner"].shape[0]
    C = batch[0]["iner"].shape[1]
    B = len(batch)
    X_iner = torch.stack([b["iner"] for b in batch], 0)
    X_skel = torch.stack([b["skel"] for b in batch], 0)
    y = torch.stack([b["label"] for b in batch], 0)

    P = patch_size
    N = W // P
    X_pool = X_iner[:, :N*P, :].reshape(B, N, P*C)
    proj = getattr(collate_teacher, "_proj", None)
    if proj is None:
        proj = nn.Linear(P*C, dim)
        with torch.no_grad():
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        collate_teacher._proj = proj
    with torch.no_grad():
        iner_tok = proj(X_pool)

    return {"iner_tok": iner_tok, "skel": X_skel, "label": y}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_kd.yaml")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    W = cfg["dataset"]["window_size"]
    C = cfg["dataset"]["in_channels"]
    num_classes = cfg["dataset"]["num_classes"]

    train_ds = DummyHAR(n=256, W=W, C=C, num_classes=num_classes, split="train")
    val_ds   = DummyHAR(n=128, W=W, C=C, num_classes=num_classes, split="val")
    train_loader_teacher = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                                      collate_fn=lambda b: collate_teacher(b, dim=cfg["teacher"]["dim"],
                                                                           patch_size=cfg["student"]["patch_size"]))
    val_loader_teacher = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                                    collate_fn=lambda b: collate_teacher(b, dim=cfg["teacher"]["dim"],
                                                                         patch_size=cfg["student"]["patch_size"]))

    teacher_state, best_val = train_teacher_loop(train_loader_teacher, val_loader_teacher,
                                                 num_classes=num_classes, device=args.device,
                                                 dim=cfg["teacher"]["dim"], depth=cfg["teacher"]["depth"],
                                                 heads=cfg["teacher"]["heads"],
                                                 lr=cfg["optim"]["lr"], epochs=cfg["optim"]["epochs_teacher"])
    print(f"[Teacher] Best val acc: {best_val:.4f}")

    teacher = STConvTTeacher(num_classes=num_classes, dim=cfg["teacher"]["dim"],
                             depth=cfg["teacher"]["depth"], heads=cfg["teacher"]["heads"]).to(args.device)
    teacher.load_state_dict(teacher_state)

    train_loader_student = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader_student   = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    best_student_state, best_student_val = train_student_kd(train_loader_student, val_loader_student,
                                                            teacher, num_classes=num_classes, device=args.device,
                                                            in_channels=C, window_size=W,
                                                            patch_size=cfg["student"]["patch_size"],
                                                            dim=cfg["student"]["dim"], depth=cfg["student"]["depth"],
                                                            heads=cfg["student"]["heads"],
                                                            lr=cfg["optim"]["lr"], epochs=cfg["optim"]["epochs_student"],
                                                            alpha=cfg["kd"]["alpha"], T=cfg["kd"]["T"])
    print(f"[Student] Best val acc: {best_student_val:.4f}")

if __name__ == "__main__":
    main()
