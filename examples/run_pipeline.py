import numpy as np
from preprocessing import Preprocessor, PreprocessConfig

def synth_data(T=10_000, imu_C=6, J=20, D=3, imu_hz=52.0, skel_hz=30.0):
    t_imu = np.cumsum(np.full(T, 1.0/imu_hz))
    imu = np.zeros((T, imu_C), dtype=np.float32)
    # simple synthetic patterns
    imu[:, 0] = np.sin(2*np.pi*1.0*t_imu)   # accel-x
    imu[:, 1] = np.cos(2*np.pi*0.7*t_imu)
    imu[:, 2] = np.sin(2*np.pi*0.4*t_imu + 0.5)
    imu[:, 3:] = 0.5*np.random.randn(T, imu_C-3)

    # labels per-timestep (two classes switching)
    y = (np.sin(2*np.pi*0.02*t_imu) > 0).astype(np.int64)

    # skeleton (optional)
    t_skel = np.cumsum(np.full(int(T*skel_hz/imu_hz), 1.0/skel_hz))
    skel = np.random.randn(len(t_skel), J, D).astype(np.float32) * 0.05
    skel[:, 0, :] = 0.0  # pelvis at origin (before normalization)

    return t_imu, imu, y, t_skel, skel

def main():
    t_imu, imu, y, t_skel, skel = synth_data()

    cfg = PreprocessConfig(
        target_hz=50.0,
        window_seconds=3.0,
        overlap=0.5,
        p_min=10, p_max=30,
        var_low=0.05, var_high=0.25,
        apply_lowpass=True,
        add_gaussian_noise=False
    )

    pp = Preprocessor(cfg)
    result = pp.fit_transform(
        t_imu=t_imu, imu=imu, labels=y,
        t_skel=t_skel, skel=skel,
        augment=False,
        out_dir="./preprocessed"
    )

    print(f"Windows: {len(result['imu_patches'])}")
    print(f"First window P: {result['meta'][0]['chosen_P']}, var Vw: {result['meta'][0]['variance_Vw']:.4f}")
    if 'skel_patches' in result:
        print("Skeleton patches included.")

if __name__ == "__main__":
    main()
