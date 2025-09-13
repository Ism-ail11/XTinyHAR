import os
import numpy as np
from src.metrics import confusion_matrix
from src.plots import plot_training_curves, plot_confusion_matrix, plot_bar
from src.tables import (
    table_metrics_summary, table_per_class_f1_mmfit, table_ablation_extended,
    table_averaging_ablation, table_kd_params, table_positional_encoding,
    table_ablation_patching, table_ablation_arch, table_robustness
)

IMAGES_DIR = "images"
TABLES_DIR = "tables"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------- REPLACE WITH REAL DATA ----------
# Synthetic training curves to mimic your plots (acc_32.png, acc_64.png)
epochs = 20
train_acc_utd = np.clip(np.linspace(0.75, 0.995, epochs), 0, 1)
val_acc_utd   = np.clip(np.linspace(0.72, 0.987, epochs), 0, 1)
train_loss_utd = np.linspace(0.9, 0.02, epochs)
val_loss_utd   = np.linspace(1.0, 0.03, epochs)

train_acc_mm  = np.clip(np.linspace(0.73, 0.992, epochs), 0, 1)
val_acc_mm    = np.clip(np.linspace(0.70, 0.985, epochs), 0, 1)
train_loss_mm = np.linspace(1.1, 0.03, epochs)
val_loss_mm   = np.linspace(1.2, 0.04, epochs)

# Synthetic confusion matrices (replace with your real CM)
n_classes_utd = 27
cm_utd = np.eye(n_classes_utd, dtype=int) * 30
cm_utd[3, 4] += 1; cm_utd[4, 3] += 1   # small off-diagonal
classes_utd = [f"C{i+1}" for i in range(n_classes_utd)]

n_classes_mm = 12
cm_mm = np.eye(n_classes_mm, dtype=int) * 50
cm_mm[1, 2] += 2; cm_mm[2, 1] += 1
classes_mm = [f"A{i+1}" for i in range(n_classes_mm)]
# -------------------------------------------

# CURVES
plot_training_curves(train_acc_utd, val_acc_utd, train_loss_utd, val_loss_utd,
                     out_path=os.path.join(IMAGES_DIR, "acc_32.png"),
                     title="UTD-MHAD: Accuracy/Loss")
plot_training_curves(train_acc_mm, val_acc_mm, train_loss_mm, val_loss_mm,
                     out_path=os.path.join(IMAGES_DIR, "acc_64.png"),
                     title="MM-Fit: Accuracy/Loss")

# CONFUSION MATRICES
plot_confusion_matrix(cm_utd, classes_utd,
                      out_path=os.path.join(IMAGES_DIR, "Matrice_UTD.png"),
                      title="Confusion Matrix (UTD-MHAD)")
plot_confusion_matrix(cm_mm, classes_mm,
                      out_path=os.path.join(IMAGES_DIR, "Matrice_MM.png"),
                      title="Confusion Matrix (MM-Fit)")

# SIMPLE BAR COMPARISON (your Plot_Bar.png)
datasets = ["UTD-MHAD", "MM-Fit"]
acc_values = [98.71, 98.55]
plot_bar(datasets, acc_values,
         out_path=os.path.join(IMAGES_DIR, "Plot_Bar.png"),
         ylabel="Accuracy (%)",
         title="Dataset Accuracy Comparison")

# TABLES
table_metrics_summary(os.path.join(TABLES_DIR, "metrics_summary.tex"))
table_per_class_f1_mmfit(os.path.join(TABLES_DIR, "per_class_f1_mmfit.tex"))
table_ablation_extended(os.path.join(TABLES_DIR, "ablation_study_extended.tex"))
table_averaging_ablation(os.path.join(TABLES_DIR, "averaging_ablation.tex"))
table_kd_params(os.path.join(TABLES_DIR, "ablation_kd_params.tex"))
table_positional_encoding(os.path.join(TABLES_DIR, "positional_encoding.tex"))
table_ablation_patching(os.path.join(TABLES_DIR, "ablation_patching.tex"))
table_ablation_arch(os.path.join(TABLES_DIR, "ablation_arch.tex"))
table_robustness(os.path.join(TABLES_DIR, "robustness_ablation.tex"))

print("Generated figures in ./images and LaTeX tables in ./tables")
