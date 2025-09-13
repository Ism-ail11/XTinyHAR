import torch
import numpy as np
from typing import List

def attention_rollout(attn_list: List[torch.Tensor], residual: float = 0.5) -> torch.Tensor:
    """Residual-aware rollout: A_hat = residual * I + (1-residual) * A; then cumulative matmul across layers.
    Args:
        attn_list: list of [B,H,T,T]
        residual: default 0.5 per literature (often (I + A)/2)
    Returns:
        rollout: [B, T, T] cumulative attention
    """
    B, H, T, _ = attn_list[0].shape
    I = torch.eye(T, device=attn_list[0].device).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    joint = None
    for A in attn_list:
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        A_hat = residual * I + (1 - residual) * A  # [B,H,T,T]
        A_hat = A_hat.mean(dim=1)  # average heads -> [B,T,T]
        joint = A_hat if joint is None else torch.matmul(joint, A_hat)
    return joint  # [B,T,T]

def rollout_cls_to_patches(roll: torch.Tensor) -> torch.Tensor:
    """Extract CLS->tokens (excluding self) from rollout matrix [B,T,T].
    Returns [B, T-1] vector.
    """
    return roll[:, 0, 1:]
