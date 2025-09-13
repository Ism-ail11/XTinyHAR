XTinyHAR: Lightweight, Explainable HAR via Multimodal→Unimodal Distillation

XTinyHAR is a compact, edge-deployable Human Activity Recognition (HAR) framework. A multimodal teacher (skeleton + IMU) transfers knowledge to a lightweight Inertial Transformer student that uses only IMU at inference. The pipeline includes dynamic patching, positional embeddings, explainability (IG, attention rollout, attention similarity), and export to ONNX/TFLite for Raspberry Pi / Jetson Nano.

<p align="center"> <img src="images/Model_Arch.png" alt="XTinyHAR Overview" width="640"/> </p>

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
