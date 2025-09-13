
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, T=3.0):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, target):
        ce = self.ce(student_logits, target)
        T = self.T
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        kl = F.kl_div(s, t, reduction="batchmean") * (T * T)
        return (1 - self.alpha) * ce + self.alpha * kl, {"ce": ce.detach(), "kl": kl.detach()}
