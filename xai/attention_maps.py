import torch
import numpy as np
from typing import List, Tuple, Optional

def extract_last_layer_class_attention(attn_list: List[torch.Tensor]) -> torch.Tensor:
    """Extract class-token -> patch attention from the last layer.
    Args:
        attn_list: list of attention tensors [L] of shape [B, H, T, T].
    Returns:
        class_attn: [B, T-1] averaged over heads (excluding CLS->CLS).
    """
    last = attn_list[-1]  # [B,H,T,T]
    # average over heads
    last_mean = last.mean(dim=1)  # [B,T,T]
    # class token is index 0; we want attention weights from CLS to others
    cls_to_tokens = last_mean[:, 0, 1:]  # [B, T-1]
    return cls_to_tokens

def attention_vector_to_heatmap(vec: torch.Tensor) -> np.ndarray:
    """Convert a 1D attention vector [T-1] to a 2D heatmap (1 x T-1) for plotting convenience."""
    if vec.dim() != 1:
        raise ValueError("Expected a 1D vector for heatmap conversion.")
    v = vec.detach().cpu().numpy()
    v = v / (v.max() + 1e-8)
    return v[None, :]  # shape (1, T-1)
