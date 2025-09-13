from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    # Target synchronized sampling rate (Hz)
    target_hz: float = 50.0

    # Sliding window (seconds) and overlap fraction
    window_seconds: float = 3.0
    overlap: float = 0.5

    # Dynamic patching bounds and thresholds (Sec. 3.2)
    p_min: int = 10
    p_max: int = 30
    var_low: float = 0.05
    var_high: float = 0.25

    # Filtering / augmentation (optional)
    accel_lowpass_hz: float = 20.0
    apply_lowpass: bool = True
    add_gaussian_noise: bool = False
    noise_std: float = 0.02  # relative to per-channel std

    # Skeleton smoothing (moving average over frames)
    skel_ma_window: int = 3

    # Seed for any stochastic ops
    seed: int = 42
