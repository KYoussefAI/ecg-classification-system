import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from .config import CLASSES


def training_plots(directory, history, counts):
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r["epoch"] for r in history]
    for key in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    for key in ("validation_macro_auroc", "validation_macro_auprc"):
        axes[1].plot(epochs, [r[key] for r in history], label=key)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, values) in enumerate(counts.items()):
        ax.bar(np.arange(5) + i * 0.25, values, 0.25, label=name)
    ax.set_xticks(np.arange(5) + 0.125, CLASSES)
    ax.set_ylabel("Positive records")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "class_distribution.png", dpi=160)
    plt.close(fig)


def evaluation_plots(directory, targets, scores, report):
    directory.mkdir(exist_ok=True)
    curves = {}
    for kind in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(7, 5))
        curves[kind] = {}
        for i, cls in enumerate(CLASSES):
            if len(np.unique(targets[:, i])) != 2:
                continue
            if kind == "roc":
                x, y, _ = roc_curve(targets[:, i], scores[:, i])
            else:
                y, x, _ = precision_recall_curve(targets[:, i], scores[:, i])
            ax.plot(x, y, label=cls)
            # Bounded representation for browser plotting, not metric calculation.
            indices = np.unique(
                np.linspace(0, len(x) - 1, min(200, len(x))).astype(int)
            )
            curves[kind][cls] = [[float(x[j]), float(y[j])] for j in indices]
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="False positive rate" if kind == "roc" else "Recall",
            ylabel="Sensitivity" if kind == "roc" else "Precision",
            title="Held-out PTB-XL Fold 10",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / (kind + "_curves.png"), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, cls in zip(axes, CLASSES):
        matrix = np.array(report["per_class"][cls]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(
            title=cls, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1]
        )
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    return curves
