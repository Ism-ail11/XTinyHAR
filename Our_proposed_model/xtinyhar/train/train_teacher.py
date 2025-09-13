
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.st_convt import STConvTTeacher

def train_teacher_loop(train_loader: DataLoader, val_loader: DataLoader, num_classes=27, device="cuda",
                       dim=128, depth=2, heads=4, lr=1e-4, epochs=10):
    model = STConvTTeacher(num_classes=num_classes, dim=dim, depth=depth, heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Teacher Train [{ep+1}/{epochs}]")
        running = 0.0
        for batch in pbar:
            x_iner_tok = batch["iner_tok"].to(device)
            x_skel = batch["skel"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x_iner_tok, x_skel)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
            pbar.set_postfix(loss=running / (pbar.n+1))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = model(batch["iner_tok"].to(device), batch["skel"].to(device))
                pred = logits.argmax(1).cpu()
                correct += (pred == batch["label"]).sum().item()
                total += pred.numel()
        val_acc = correct / max(1,total)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return best_state, best_val
