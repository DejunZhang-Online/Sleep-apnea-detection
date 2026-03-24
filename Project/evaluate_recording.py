import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate subject-level sleep apnea performance from minute-level predictions.")
    parser.add_argument("--predictions", required=True, help="CSV file with columns: subject, y_true, y_score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metadata",
        default=os.path.join("metadata", "additional-information.txt"),
        help="Optional Apnea-ECG metadata table containing reference AHI values.",
    )
    parser.add_argument(
        "--ahi-threshold",
        type=float,
        default=5.0,
        help="AHI threshold for subject-level sleep apnea detection.",
    )
    return parser.parse_args()


def load_metadata_table(path: str) -> Dict[str, Dict[str, float]]:
    if not path or not os.path.exists(path):
        return {}

    rows = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("Additional information") or line.startswith("Record") or line.startswith("minutes"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            subject = parts[0]
            try:
                rows[subject] = {
                    "length_minutes": float(parts[1]),
                    "ahi": float(parts[7]),
                }
            except ValueError:
                continue
    return rows


def build_subject_table(predictions: pd.DataFrame, metadata_rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    required_cols = {"subject", "y_true", "y_score"}
    missing = required_cols.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for subject, frame in predictions.groupby("subject", sort=True):
        frame = frame.reset_index(drop=True)
        minutes = len(frame)
        hours = minutes / 60.0
        true_apnea_minutes = int(frame["y_true"].sum())
        pred_apnea_minutes = float((frame["y_score"] >= 0.5).sum())

        pred_ahi = 60.0 * pred_apnea_minutes / max(minutes, 1)
        true_ahi_from_labels = 60.0 * true_apnea_minutes / max(minutes, 1)

        metadata = metadata_rows.get(subject)
        if metadata is not None:
            true_ahi = float(metadata["ahi"])
            record_length_minutes = float(metadata["length_minutes"])
        else:
            true_ahi = true_ahi_from_labels
            record_length_minutes = float(minutes)

        mean_score = float(frame["y_score"].mean())
        rows.append(
            {
                "subject": subject,
                "minutes_pred": minutes,
                "hours_pred": hours,
                "record_length_minutes": record_length_minutes,
                "true_apnea_minutes": true_apnea_minutes,
                "pred_apnea_minutes": int(pred_apnea_minutes),
                "true_ahi": true_ahi,
                "pred_ahi": pred_ahi,
                "true_ahi_from_labels": true_ahi_from_labels,
                "mean_score": mean_score,
            }
        )

    return pd.DataFrame(rows)


def compute_record_metrics(subject_df: pd.DataFrame, ahi_threshold: float) -> Dict[str, float]:
    y_true = (subject_df["true_ahi"].to_numpy() >= ahi_threshold).astype(np.int64)
    y_score = subject_df["pred_ahi"].to_numpy()
    y_pred = (y_score >= ahi_threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, tn, fp, fn = cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    if len(subject_df) > 1:
        corr = float(np.corrcoef(subject_df["true_ahi"], subject_df["pred_ahi"])[0, 1])
    else:
        corr = float("nan")

    return {
        "accuracy": float(acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc_value),
        "corr": corr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "cm": cm,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
    }


def save_ahi_scatter(subject_df: pd.DataFrame, output_path: str):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(subject_df["true_ahi"], subject_df["pred_ahi"], alpha=0.8)
    if len(subject_df) > 0:
        min_val = min(subject_df["true_ahi"].min(), subject_df["pred_ahi"].min())
        max_val = max(subject_df["true_ahi"].max(), subject_df["pred_ahi"].max())
        axis.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axis.set_xlabel("Reference AHI")
    axis.set_ylabel("Predicted AHI")
    axis.set_title("AHI Regression")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    predictions = pd.read_csv(args.predictions)
    metadata_rows = load_metadata_table(args.metadata)
    subject_df = build_subject_table(predictions, metadata_rows)
    subject_df["true_disorder"] = (subject_df["true_ahi"] >= args.ahi_threshold).astype(int)
    subject_df["pred_disorder"] = (subject_df["pred_ahi"] >= args.ahi_threshold).astype(int)

    metrics = compute_record_metrics(subject_df, args.ahi_threshold)

    subject_df.to_csv(os.path.join(args.output_dir, "recording_predictions.csv"), index=False)
    save_confusion_matrix(
        metrics["cm"],
        labels=["SA", "Normal"],
        output_path=os.path.join(args.output_dir, "recording_confusion_matrix.png"),
    )
    save_roc_curve(
        metrics["fpr"],
        metrics["tpr"],
        metrics["auc"],
        os.path.join(args.output_dir, "recording_roc_curve.png"),
    )
    save_ahi_scatter(subject_df, os.path.join(args.output_dir, "ahi_scatter.png"))

    summary = {
        "ahi_threshold": float(args.ahi_threshold),
        "record_count": int(len(subject_df)),
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "auc": metrics["auc"],
        "corr": metrics["corr"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
