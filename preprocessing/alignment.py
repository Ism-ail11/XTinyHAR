import numpy as np

def resample_to_grid(t_src: np.ndarray, x_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of x_src onto t_grid (per-channel).
    x_src: [T, ...]  (IMU: [T,C], Skeleton: [T,J,D])
    """
    if x_src.ndim == 2:
        # [T, C]
        out = np.empty((len(t_grid), x_src.shape[1]), dtype=np.float32)
        for c in range(x_src.shape[1]):
            out[:, c] = np.interp(t_grid, t_src, x_src[:, c])
        return out
    elif x_src.ndim == 3:
        # [T, J, D]
        T, J, D = x_src.shape
        out = np.empty((len(t_grid), J, D), dtype=np.float32)
        for j in range(J):
            for d in range(D):
                out[:, j, d] = np.interp(t_grid, t_src, x_src[:, j, d])
        return out
    else:
        raise ValueError("Unsupported x_src ndim")

def make_time_grid(t_start: float, t_end: float, target_hz: float) -> np.ndarray:
    n = int(np.floor((t_end - t_start) * target_hz)) + 1
    return np.linspace(t_start, t_start + (n-1)/target_hz, n, dtype=np.float64)
