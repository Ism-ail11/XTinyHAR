import os

def _write_tex(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def table_metrics_summary(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive performance metrics of the proposed XTinyHAR model evaluated on UTD-MHAD and MM-Fit datasets.}
\label{tab:metrics_summary}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{UTD-MHAD} & \textbf{MM-Fit} \\
\midrule
Train Accuracy (\%)        & 99.01 & 98.79 \\
Validation Accuracy (\%)   & 98.64 & 98.42 \\
Test Accuracy (\%)         & 98.71 & 98.55 \\
Precision (\%)             & 98.72 & 98.56 \\
Recall (\%)                & 98.71 & 98.55 \\
F1-Score (\%)              & 98.71 & 98.55 \\
Cohen's Kappa              & 0.985 & 0.983 \\
Model Size (MB)            & 2.45  & 2.45 \\
Memory Footprint (MB)      & 7.2   & 7.1 \\
Inference Time CPU (ms)    & 3.1   & 3.0 \\
Inference Time GPU (ms)    & 1.2   & 1.1 \\
Floating-point Operations Per Second (FLOPs) (M) & 11.3  & 11.3 \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_per_class_f1_mmfit(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Per-class F1-scores (\%) of XTinyHAR on the MM-Fit dataset.}
\label{tab:per_class_f1_mmfit}
\begin{tabular}{lc}
\toprule
\textbf{Activity Class} & \textbf{F1-Score (\%)} \\
\midrule
Push-up       & 98.41 \\
Sit-to-Stand  & 98.23 \\
Squat         & 98.55 \\
Jumping Jack  & 98.72 \\
Lunge         & 98.39 \\
Walk          & 98.88 \\
Stand Still   & 98.34 \\
Arm Circle    & 98.61 \\
Torso Twist   & 98.50 \\
Knee Raise    & 98.43 \\
\midrule
\textbf{Micro Average} & \textbf{98.55} \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_ablation_extended(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Extended Ablation Study on UTD-MHAD dataset showing the impact of removing or isolating key components from the XTinyHAR model across multiple evaluation metrics.}
\label{tab:ablation_study_extended}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Variant} & \textbf{Test Accuracy (\%)} & \textbf{F1-Score (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{Cohen's Kappa} & \textbf{FLOPs (M)} \\
\midrule
\textbf{Full Model (KD + PE + AR)}          & \textbf{98.71} & \textbf{98.71} & \textbf{98.72} & \textbf{98.71} & \textbf{0.985} & 11.3 \\
\midrule
Without Knowledge Distillation (KD)              & 96.84 & 96.80 & 96.87 & 96.84 & 0.962 & 11.3 \\
Without Positional Embedding                & 97.42 & 97.38 & 97.40 & 97.42 & 0.973 & 11.3 \\
Without Attention Rollout                   & 98.11 & 98.08 & 98.12 & 98.11 & 0.978 & 11.3 \\
\midrule
Only Knowledge Distillation (KD only) & 96.41 & 96.40 & 96.45 & 96.41 & 0.960 & 11.3 \\
Only Positional Embedding (PE only)         & 95.78 & 95.73 & 95.70 & 95.78 & 0.954 & 11.3 \\
Only Attention Rollout (AR only)            & 95.01 & 94.94 & 94.99 & 95.01 & 0.944 & 11.3 \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
    _write_tex(path, tex)

def table_averaging_ablation(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of macro- and micro-averaging for XTinyHAR under 5-fold cross-validation.}
\label{tab:averaging_ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Averaging Method} & \textbf{Accuracy (\%)} & \textbf{F1-Score (\%)} & \textbf{Precision (\%)} \\
\midrule
\multirow{2}{*}{UTD-MHAD} 
 & Macro-Averaging & 98.41 & 98.33 & 98.29 \\
 & Micro-Averaging & \textbf{98.71} & \textbf{98.65} & \textbf{98.61} \\
\midrule
\multirow{2}{*}{MM-Fit} 
 & Macro-Averaging & 98.21 & 98.17 & 98.10 \\
 & Micro-Averaging & \textbf{98.55} & \textbf{98.50} & \textbf{98.46} \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_kd_params(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study for temperature $T$ and distillation coefficient $\alpha$ on the UTD-MHAD dataset. Results are reported as classification accuracy (\%).}
\label{tab:ablation_kd_params}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\diagbox{$\alpha$}{$T$} & 1 & 2 & 3 & 4 & 5 \\
\hline
0.3 & 95.71 & 96.23 & 96.47 & 96.11 & 95.89 \\
0.5 & 96.12 & 96.55 & 97.02 & 96.88 & 96.34 \\
0.7 & 96.89 & 97.38 & \textbf{98.71} & 97.41 & 96.92 \\
0.9 & 96.44 & 96.90 & 97.32 & 96.74 & 96.28 \\
\hline
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_positional_encoding(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison between learnable and sinusoidal positional encodings on UTD-MHAD and MM-Fit datasets.}
\label{tab:positional_encoding}
\begin{tabular}{lcc}
\toprule
\textbf{Positional Encoding Type} & \textbf{UTD-MHAD Accuracy (\%)} & \textbf{MM-Fit Accuracy (\%)} \\
\midrule
Sinusoidal Encoding (Fixed) & 97.89 & 97.62 \\
Learnable Positional Embedding (Ours) & \textbf{98.71} & \textbf{98.55} \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_ablation_patching(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study of patching strategy on UTD-MHAD and MM-Fit datasets.}
\label{tab:ablation_patching}
\begin{tabular}{lcc}
\toprule
\textbf{Patching Method} & \textbf{UTD-MHAD Accuracy (\%)} & \textbf{MM-Fit Accuracy (\%)} \\
\midrule
Fixed Patch ($P=10$) & 97.15 & 97.20 \\
Fixed Patch ($P=20$) & 97.59 & 97.45 \\
Fixed Patch ($P=30$) & 96.94 & 96.88 \\
\textbf{Dynamic Patch (Ours)} & \textbf{98.71} & \textbf{98.40} \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_ablation_arch(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation on architecture hyperparameters using UTD-MHAD.}
\label{tab:ablation_arch}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Accuracy (\%)} & \textbf{Model Size (MB)} & \textbf{FLOPs (M)} \\
\midrule
$L = 1$ (D=128, P=20) & 96.13 & 1.43 & 6.4 \\
$L = 2$ (default)     & \textbf{98.71} & 2.45 & 11.3 \\
$L = 3$               & 98.40 & 3.36 & 16.5 \\
\midrule
$D = 64$ (L=2, P=20)  & 96.89 & 1.61 & 7.8 \\
$D = 128$ (default)   & \textbf{98.71} & 2.45 & 11.3 \\
$D = 256$             & 98.83 & 4.28 & 19.1 \\
\midrule
$P = 10$ (L=2, D=128) & 96.97 & 2.45 & 11.2 \\
$P = 20$ (default)    & \textbf{98.71} & 2.45 & 11.3 \\
$P = 40$              & 98.55 & 2.45 & 11.9 \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)

def table_robustness(path):
    tex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study showing XTinyHAR performance under sensor noise and missing data on UTD-MHAD.}
\label{tab:robustness_ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Perturbation Type} & \textbf{Setting} & \textbf{Accuracy (\%)} & \textbf{F1-Score (\%)} & \textbf{Kappa} \\
\midrule
None (Clean) & -- & 98.71 & 98.71 & 0.985 \\
Gaussian Noise & $\sigma = 0.05$ & 97.46 & 97.41 & 0.973 \\
Gaussian Noise & $\sigma = 0.1$ & 96.02 & 95.96 & 0.960 \\
Gaussian Noise & $\sigma = 0.2$ & 94.12 & 94.00 & 0.938 \\
Missing Segments & 10\% masked & 95.85 & 95.80 & 0.955 \\
\bottomrule
\end{tabular}
\end{table}
"""
    _write_tex(path, tex)
