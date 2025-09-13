import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(train_acc, val_acc, train_loss, val_loss, out_path, title="Training/Validation Curves"):
    epochs = np.arange(1, len(train_acc) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(epochs, train_acc, marker="o", label="Train Acc")
    ax1.plot(epochs, val_acc, marker="s", label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss, marker="^", label="Train Loss")
    ax2.plot(epochs, val_loss, marker="v", label="Val Loss")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_bar(categories, values, out_path, ylabel="Value", title="Bar Plot"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
