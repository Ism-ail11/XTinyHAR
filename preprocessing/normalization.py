import numpy as np
from typing import Tuple, Dict

def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    x: [T, C] IMU
    Returns normalized x and stats dict.
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + eps
    x_norm = (x - mean) / std
    return x_norm.astype(np.float32), {"mean": mean.squeeze(0), "std": std.squeeze(0)}

def recenter_and_scale_skeleton(skel: np.ndarray, ref_joint: int = 0) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    skel: [T, J, D] (D=2 or 3)
    Recenter to pelvis (ref_joint) and scale by RMS joint distance.
    """
    skel = skel.astype(np.float32).copy()
    center = skel[:, ref_joint:ref_joint+1, :]  # [T,1,D]
    skel -= center
    rms = np.sqrt((skel**2).sum(axis=-1)).mean() + 1e-8
    skel /= rms
    return skel, {"ref_joint": ref_joint, "scale_rms": float(rms)}
