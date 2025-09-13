import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    """
    x: [T, C]
    """
    if cutoff is None or cutoff >= fs / 2.0:
        return x
    b, a = butter(order, cutoff / (fs / 2.0), btype='low', analog=False)
    y = np.empty_like(x)
    for c in range(x.shape[1]):
        y[:, c] = filtfilt(b, a, x[:, c], method="gust")
    return y.astype(np.float32)

def add_gaussian_noise(x: np.ndarray, rel_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    x: [T, C] (assumed z-scored). rel_std is relative to per-channel std (=1 after z-score).
    """
    noise = rng.normal(0.0, rel_std, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)

def moving_average_skeleton(s: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return s
    out = np.copy(s)
    for d in range(s.shape[-1]):
        for j in range(s.shape[1]):
            v = s[:, j, d]
            w = np.convolve(v, np.ones(k)/k, mode='same')
            out[:, j, d] = w
    return out.astype(np.float32)
