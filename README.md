# Data Pre-Processing for XTinyHAR

Implements the pre-processing pipeline described in the paper:

1) Sliding-window segmentation  
2) Signal normalization (IMU z-score; skeleton recenter + scale)  
3) Modality alignment & resampling to a common rate (e.g., 50 Hz)  
4) Dynamic patch construction via intra-window motion variance \( \mathcal{V}_w \) → adaptive patch size \(P\)  
5) Noise-robust augmentation & filtering (optional)  
6) Transformer-ready embeddings (patch tokens + metadata)

## Install
```bash
pip install -r requirements.txt
```

## Quick start (demo with synthetic data)
```bash
python -m examples.run_pipeline
```

This writes `.npz` windows + a `meta.jsonl` into `./preprocessed/`.

## Use in your project
```python
from preprocessing import Preprocessor, PreprocessConfig

pp = Preprocessor(PreprocessConfig(
    target_hz=50.0,
    window_seconds=3.0, overlap=0.5,
    p_min=10, p_max=30, var_low=0.05, var_high=0.25
))

result = pp.fit_transform(
    t_imu, imu, labels,      # imu: [T, C], timestamps [T]
    t_skel=t_skel, skel=skel,# optional: [T, J, D]
    augment=False,
    out_dir="./preprocessed"
)
```

**Outputs**
- `imu_patches`: list of `[N, P*C]` arrays (one per window)  
- `skel_patches` (optional): list of `[N, P*J*D]` arrays  
- `y`: per-window labels  
- `meta`: list of dicts with window stats (`chosen_P`, `variance_Vw`, times, z-score stats, etc.)
