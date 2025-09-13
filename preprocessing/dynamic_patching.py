import numpy as np

def motion_variance(imu_window: np.ndarray) -> float:
    """
    imu_window: [W, C] (z-scored)
    Returns average per-channel std over the window (scalar).
    """
    return float(imu_window.std(axis=0).mean())

def choose_patch_size(vw: float, p_min: int, p_max: int, var_low: float, var_high: float) -> int:
    if vw <= var_low:
        return p_max
    if vw >= var_high:
        return p_min
    # linear interpolation (vw in (var_low, var_high))
    ratio = (vw - var_low) / max(var_high - var_low, 1e-8)
    p = int(round(p_max - ratio * (p_max - p_min)))
    return max(p_min, min(p_max, p))

def to_patches(x: np.ndarray, P: int) -> np.ndarray:
    """
    x: [W, C] → patches [N, P*C], dropping residual if W % P != 0.
    """
    W, C = x.shape
    N = W // P
    x = x[:N*P]
    x = x.reshape(N, P*C)
    return x.astype(np.float32)

def to_patches_skeleton(s: np.ndarray, P: int) -> np.ndarray:
    """
    s: [W, J, D] → patches [N, P*J*D]
    """
    W, J, D = s.shape
    N = W // P
    s = s[:N*P]
    s = s.reshape(N, P*J*D)
    return s.astype(np.float32)
