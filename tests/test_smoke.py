import numpy as np
from preprocessing import Preprocessor, PreprocessConfig

def test_smoke():
    # tiny synthetic
    T = 2000
    t = np.linspace(0, 40, T)
    imu = np.stack([np.sin(2*np.pi*0.5*t), np.cos(2*np.pi*0.25*t)], axis=1).astype(np.float32)
    y = (imu[:,0] > 0).astype(np.int64)

    pp = Preprocessor(PreprocessConfig(window_seconds=2.0, overlap=0.5, target_hz=50.0))
    out = pp.fit_transform(t, imu, y, augment=False)

    assert len(out["imu_patches"]) > 0
    assert "meta" in out
