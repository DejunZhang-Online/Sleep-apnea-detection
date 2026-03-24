import csv
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_history_row(path: str, row: Dict[str, float]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history: Dict[str, list], output_path: str):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["train_acc"], label="train_acc")
    axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def classification_report_dict(y_true, y_score):
    y_pred = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, tn, fp, fn = cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    f1 = f1_score(y_true, y_pred, average="binary")
    fpr, tpr, _ = roc_curve((y_true == 1).astype(int), y_score[:, 1])
    po = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / ((tp + tn + fp + fn) ** 2 + 1e-12)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return {
        "accuracy": acc,
        "sensitivity": sn,
        "specificity": sp,
        "f1": f1,
        "auc": auc(fpr, tpr),
        "kappa": kappa,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
    }


def save_confusion_matrix(cm, labels, output_path):
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(cm, cmap="Reds")
    axis.figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, str(cm[i, j]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve(fpr, tpr, auc_value, output_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

