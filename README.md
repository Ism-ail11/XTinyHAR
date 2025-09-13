<<<<<<< HEAD
# XAI for XTinyHAR

This package provides explainability utilities used in the paper section *Explainable AI*:
- Attention visualization (class-token attention over temporal patches)
- Integrated Gradients (IG) for time-series IMU input
- Attention rollout across transformer layers
- Attention similarity (student vs teacher), with mean cosine similarity

It is **model-agnostic** but expects your model to optionally return attention maps.

## Installation
=======
<<<<<<< HEAD
# Data Pre-Processing for XTinyHAR

Implements the pre-processing pipeline described in the paper:

1) Sliding-window segmentation  
2) Signal normalization (IMU z-score; skeleton recenter + scale)  
3) Modality alignment & resampling to a common rate (e.g., 50 Hz)  
4) Dynamic patch construction via intra-window motion variance \( \mathcal{V}_w \) → adaptive patch size \(P\)  
5) Noise-robust augmentation & filtering (optional)  
6) Transformer-ready embeddings (patch tokens + metadata)

## Install
>>>>>>> 13bc9c0595d906eb00cb77a1cc3b8986b3701f0e
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
## Expected Model Interface

Your PyTorch `forward` can follow either of these:
1) `logits = model(x)` (no attention)
2) `logits, attn_list = model(x, return_attn=True)` where `attn_list` is a list of attention tensors,
   each with shape `[B, H, T, T]` (T = number of tokens; first token is the class token).

If you already have a model that returns attention differently, write a small adapter that formats
`attn_list` in this standard shape.

## Quick Demo (Synthetic)
```bash
python -m examples.demo_generate_figures
```

This will:
- Create synthetic IMU sequences (B=16, W=200, C=6),
- Run a tiny Transformer (toy) for both a "teacher" and "student",
- Save figures in `outputs/` with names compatible with the paper:
  - attention_utd.png, attention_mmfit.png
  - ig_utd_num.png, ig_mmfit_notitle.png
  - rollout_utd_notitle.png, rollout_mmfit_notitle.png
  - similarity_utd_notitle.png, similarity_mmfit_notitle.png
- Print mean cosine similarity over the synthetic test set (as an example).

## Using with Your Real Checkpoints
Replace the toy models in `examples/demo_generate_figures.py` with your loaded teacher/student:
```python
teacher = torch.load("teacher.ckpt", map_location="cpu")
student = torch.load("student.ckpt", map_location="cpu")
```
Provide a real DataLoader that yields `x` tensors shaped `[B, W, C]` (IMU).

## Notes
- The IG baseline is zero by default; change via `baseline="zero"|"mean"|np.ndarray`.
- Attention rollout uses residual-aware rule: `A_hat = 0.5 * (I + A)` and cumulative product.
- Cosine similarity is reported as the mean over samples.
=======
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
=======
# XTinyHAR
>>>>>>> fdc40a47a6b71f93bd795eebb041bd330379fdb2
>>>>>>> 13bc9c0595d906eb00cb77a1cc3b8986b3701f0e
