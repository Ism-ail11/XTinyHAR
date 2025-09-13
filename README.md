# XAI for XTinyHAR

This package provides explainability utilities used in the paper section *Explainable AI*:
- Attention visualization (class-token attention over temporal patches)
- Integrated Gradients (IG) for time-series IMU input
- Attention rollout across transformer layers
- Attention similarity (student vs teacher), with mean cosine similarity

It is **model-agnostic** but expects your model to optionally return attention maps.

## Installation
```bash
pip install -r requirements.txt
```

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
