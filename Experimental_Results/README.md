# Experimental_Results

Utilities to reproduce figures and LaTeX tables used in the Results section for XTinyHAR.

## Quick start
```bash
pip install -r requirements.txt
python examples/run_results_demo.py
```

Artifacts will be written to:
- `images/` — all figures (PNG)
- `tables/` — all LaTeX tables (.tex)

To plug in your real results:
- Edit arrays in `examples/run_results_demo.py` (marked with "REPLACE WITH REAL DATA").
- Or import `src/metrics.py`, `src/plots.py`, `src/tables.py` into your own training scripts.
