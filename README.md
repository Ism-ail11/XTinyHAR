# A Tiny Inertial Transformer for Human Activity Recognition via Multimodal Knowledge Distillation and Explainable AI

XTinyHAR is a compact, edge-deployable Human Activity Recognition (HAR) framework. A multimodal **teacher** (skeleton + IMU) transfers knowledge to a lightweight **Inertial Transformer** student that uses **only IMU** at inference. The pipeline includes dynamic patching, positional embeddings, explainability (IG, attention rollout, attention similarity), and export to ONNX/TFLite for Raspberry Pi / Jetson Nano.

## Cite Our work if you use our model: 

Lamaakal, I., Yahyati, C., Maleh, Y. et al. A tiny inertial transformer for human activity recognition via multimodal knowledge distillation and explainable AI. Sci Rep 15, 42335 (2025). https://doi.org/10.1038/s41598-025-26297-2

## Model Architecture

<!-- Optional: control size with HTML if it’s huge -->
<p align="center">
  <img src="Model_Arch (1).png" alt="XTinyHAR Architecture" width="720">
</p>


---

## 🔧 Repository Structure

```
XTinyHAR/
├─ datasets/                        # dataset loaders + examples (UTD-MHAD, MM-Fit)
│  ├─ xtinyhar_data/
│  │  ├─ datasets/
│  │  │  ├─ base.py
│  │  │  ├─ utd_mhad.py
│  │  │  └─ mm_fit.py
│  └─ examples/
│     └─ inspect_first_batch.py
│
├─ Data_preprocessing/              # preprocessing library + CLI tools
│  ├─ xtinyhar_preproc/
│  │  ├─ sliding_window.py
│  │  ├─ normalization.py
│  │  ├─ align_resample.py
│  │  ├─ dynamic_patching.py
│  │  └─ augment_filter.py
│  └─ cli/
│     └─ build_tokens.py
│
├─ Our_proposed_model/              # models (student IT + teacher ST-ConvT) + training
│  ├─ xtinyhar_models/
│  │  ├─ student_it.py
│  │  ├─ teacher_stconvt.py
│  │  ├─ kd_loss.py
│  │  ├─ train_teacher.py
│  │  ├─ train_student.py
│  │  ├─ export_onnx.py
│  │  └─ export_tflite.py
│  └─ configs/
│     ├─ utd_mhad.yaml
│     └─ mm_fit.yaml
│
├─ XAI/                             # explainability (IG, rollout, attention similarity)
│  ├─ xtinyhar_xai/
│  │  ├─ integrated_gradients.py
│  │  ├─ attention_rollout.py
│  │  └─ attention_similarity.py
│  └─ examples/
│     └─ run_xai_demo.py
│
├─ Experimental_Results/            # figure/table generators for the paper
│  ├─ examples/
│  │  └─ run_results_demo.py
│  ├─ images/  (generated)
│  └─ tables/  (generated)
│
├─ images/                          # paper figures (confusion matrices, curves, etc.)
├─ requirements.txt
└─ README.md
```

> If your local names differ, keep the same relative roles. The commands below assume this layout.

---

## 🧱 Installation

### Option A — Conda (recommended)
```bash
conda create -n xtinyhar python=3.9 -y
conda activate xtinyhar
pip install -r requirements.txt
```

### Option B — venv (Python 3.9+)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> PyTorch: choose the wheel matching your CUDA/CPU from https://pytorch.org if needed.

---

## 📥 Datasets & Expected Layout

This project uses **UTD-MHAD** and **MM-Fit** (IMU + skeleton). Place them anywhere and point to the root in configs (see below).

**UTD-MHAD (example layout)**
```
<UTD_MHAD_ROOT>/
├─ imu/                 # CSV or MAT per trial (50 Hz)
└─ skeleton/            # Kinect 3D joints per frame
```

**MM-Fit (example layout)**
```
<MM_FIT_ROOT>/
├─ imu/                 # per subject/session sensor CSVs
└─ skeleton/            # OpenPose keypoints per frame
```

> The exact loaders in `datasets/xtinyhar_data/datasets/*.py` explain the expected filenames; adapt if your local naming differs.

---

## ⚙️ Configuration

Edit or copy the YAMLs in `Our_proposed_model/configs/`:

```yaml
# Our_proposed_model/configs/utd_mhad.yaml
data:
  root: "/absolute/path/to/UTD_MHAD_ROOT"
  imu_sr: 50
  window: 100       # W
  stride: 50
  imu_channels: 6   # acc+gyro (x,y,z)
preproc:
  resample_hz: 50
  norm: "zscore"
  dynamic_patch:
    p_min: 10
    p_max: 30
    tau_low: 0.08
    tau_high: 0.20
model:
  student:
    L: 2
    D: 128
    heads: 4
    mlp_ratio: 2.0
    p_default: 20
  teacher:
    D: 256
    L: 4
train:
  batch_size: 64
  epochs: 20
  lr: 1e-4
  weight_decay: 1e-5
  seed: 42
distill:
  T: 3
  alpha: 0.7
eval:
  folds: 5
  averaging: "micro"   # or "macro"
```

---

## ▶️ Preprocessing (tokens for Transformer)

Build tokenized sequences (sliding windows, normalization, alignment, dynamic patching):

```bash
# macOS/Linux
python Data_preprocessing/cli/build_tokens.py \
  --config Our_proposed_model/configs/utd_mhad.yaml \
  --dataset utd-mhad \
  --out data_cache/utd_tokens.npz

# Windows PowerShell
python .\Data_preprocessing\cli\build_tokens.py `
  --config .\Our_proposed_model\configs\utd_mhad.yaml `
  --dataset utd-mhad `
  --out .\data_cache\utd_tokens.npz
```

Repeat for MM-Fit by swapping `--config` and `--dataset`.

---

## 🧠 Training

### 1) Train the Teacher (multimodal)
```bash
python Our_proposed_model/xtinyhar_models/train_teacher.py \
  --config Our_proposed_model/configs/utd_mhad.yaml \
  --tokens data_cache/utd_tokens.npz \
  --save runs/teacher_utd.pt
```

### 2) Train the Student (IMU-only) with KD
```bash
python Our_proposed_model/xtinyhar_models/train_student.py \
  --config Our_proposed_model/configs/utd_mhad.yaml \
  --tokens data_cache/utd_tokens.npz \
  --teacher runs/teacher_utd.pt \
  --save runs/student_it_utd.pt
```

Key KD params are in the YAML (`distill.T=3`, `distill.alpha=0.7`) and implemented in `kd_loss.py`.

---

## 📊 Evaluation & Reproduction

### Cross-validation, metrics, confusion matrices
- Metrics computed: Accuracy, Precision/Recall/F1 (per-class + macro/micro), Cohen’s κ, FLOPs, latency, model size.
- Our paper’s figures/tables can be regenerated:

```bash
python Experimental_Results/examples/run_results_demo.py
```

This script outputs:
- PNGs in `Experimental_Results/images/` (training curves, confusion matrices, bar plots)
- LaTeX tables in `Experimental_Results/tables/` to paste directly into your paper

> Replace the synthetic arrays inside the demo with your actual logs if you want exact reproduction.

---

## 🔍 Explainability (XAI)

We provide three tools: Integrated Gradients (IG), Attention Rollout, and Attention Similarity.

```bash
python XAI/examples/run_xai_demo.py \
  --model Our_proposed_model/runs/student_it_utd.pt \
  --config Our_proposed_model/configs/utd_mhad.yaml \
  --tokens data_cache/utd_tokens.npz \
  --out xai_outputs/
```

This will generate:
- `ig_*` heatmaps (per-channel, per-time)
- rollout matrices across layers
- teacher↔student attention cosine similarity (mean reported in paper)

See:
- `XAI/xtinyhar_xai/integrated_gradients.py`
- `XAI/xtinyhar_xai/attention_rollout.py`
- `XAI/xtinyhar_xai/attention_similarity.py`

---

## 📦 Edge Deployment (ONNX / TFLite)

### Export to ONNX
```bash
python Our_proposed_model/xtinyhar_models/export_onnx.py \
  --model runs/student_it_utd.pt \
  --config Our_proposed_model/configs/utd_mhad.yaml \
  --out exports/xtinyhar_student.onnx
```

### Export to TFLite (via ONNX→TF)
```bash
python Our_proposed_model/xtinyhar_models/export_tflite.py \
  --onnx exports/xtinyhar_student.onnx \
  --out exports/xtinyhar_student.tflite
```

**Raspberry Pi 4B / Jetson Nano runtime**
- Copy the `.tflite` or `.onnx` to the device.
- Use `tflite_runtime` on Pi; TensorRT or ONNX Runtime on Nano.
- Batch-1 latency on our devices: **CPU 3.1 ms**, **GPU 1.2 ms** (see paper; numbers depend on clocks/OS).

---

## 🔬 Key Results (paper highlights)

- **UTD-MHAD**: Test **98.71%** Acc / **98.71%** F1, κ=0.985  
- **MM-Fit**: Test **98.55%** Acc / **98.55%** F1, κ=0.983  
- **Student size**: **2.45 MB**; **FLOPs**: **11.3 M**; **Latency**: **3.1 ms CPU**, **1.2 ms GPU**  
- **Ablations**: KD & positional embeddings provide the largest gains; dynamic patching beats fixed by up to **+1.1%**.  
- **XAI**: Teacher↔Student attention similarity (cosine) ~ **0.84–0.86**.

Full figure/table generators live in `Experimental_Results/`.

---

## 🧪 Reproducibility

- We fix seeds for NumPy/PyTorch/CUDA (see `train_*.py`).
- 5-fold cross-validation, subject-independent splits (see dataset loaders).
- Macro vs. micro averaging supported via config (`eval.averaging`).

---

## 🔐 Privacy & Ethics

- **IMU-only inference** preserves privacy vs. video-based HAR.
- Supports on-device processing; no raw movement data leaves the device if you keep inference local.
- Consider federated KD or DP-SGD if you plan cross-site training.

---

## 🛠 Troubleshooting

- **CRLF warnings on Windows**: safe to ignore; run `git config core.autocrlf true` if desired.
- **“refusing to merge unrelated histories”**: pull with `--allow-unrelated-histories` or force-push only if you intend to overwrite the remote:
  ```bash
  git pull origin main --allow-unrelated-histories
  # resolve conflicts, commit, then:
  git push origin main
  ```
- **CUDA/torch mismatch**: reinstall torch wheel matching your CUDA version.

---

## 📄 License

This project is released under the **MIT License** (see `LICENSE`).

---


## 🙏 Acknowledgments

We thank the maintainers of UTD-MHAD and MM-Fit datasets, and the open-source contributors of PyTorch, ONNX Runtime, and TensorFlow Lite.

---

## 💬 Contact

- Maintainer: Ismail Lamaakal — ismail.lamaakal@ieee.org  
- Issues/bugs: please open a GitHub issue with logs, config, and environment details.

---
