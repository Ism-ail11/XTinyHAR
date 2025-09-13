import os
import numpy as np
import matplotlib.pyplot as plt

def save_line_over_time(vec, xlabel, ylabel, title, path):
    v = np.asarray(vec).reshape(-1)
    plt.figure(figsize=(8,3))
    plt.plot(v, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_heatmap(mat, xlabel, ylabel, title, path):
    m = np.asarray(mat)
    plt.figure(figsize=(8,3))
    plt.imshow(m, aspect='auto', origin='lower')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_bar(values, labels, ylabel, title, path):
    vals = np.asarray(values)
    idx = np.arange(len(vals))
    plt.figure(figsize=(8,3))
    plt.bar(idx, vals)
    plt.xticks(idx, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
