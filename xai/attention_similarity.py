import torch
import numpy as np
from typing import Tuple

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute cosine similarity between vectors a and b along last dim.
    Args:
        a, b: [..., D]
    Returns:
        sim: [...]
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)

def mean_cosine_similarity(student_vecs: torch.Tensor, teacher_vecs: torch.Tensor) -> Tuple[float, float]:
    """Return mean and std of cosine similarity across batch.
    Args:
        student_vecs: [B, T-1]
        teacher_vecs: [B, T-1]
    """
    sims = cosine_similarity(student_vecs, teacher_vecs)  # [B]
    sims_np = sims.detach().cpu().numpy()
    return float(sims_np.mean()), float(sims_np.std())
