import argparse
import json
import os
from collections import Counter
from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from model.focal_loss import FocalLoss
from model.main_model import Model, SegmentOnlyModel, TransformerModel
from utils import (
    append_history_row,
    classification_report_dict,
    ensure_dir,
    plot_history,
    save_confusion_matrix,
    save_roc_curve,
    set_seed,
)


class ApneaECGDataset(Dataset):
    def __init__(
            self,
            data_path,
            mode="train",
            seq_len=60,
            val_subject_count=5,
            split_mode="subject_mixed",
            split_seed=42,
            val_ratio=0.2,
    ):
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.val_subject_count = val_subject_count
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.val_ratio = val_ratio

        self.data, self.label, self.trans_idx, self.subjects = self._load()
        print(f"{mode}: windows={len(self.label)}, seq_len={self.seq_len}")
        print(f"{mode}: label distribution={Counter(self.label.reshape(-1).tolist())}")
        print(f"{mode}: transition distribution={Counter(self.trans_idx.reshape(-1).tolist())}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),
            torch.from_numpy(self.label[idx]).long(),
            torch.from_numpy(self.trans_idx[idx]).long(),
        )

    @staticmethod
    def trans_stage_label(label):
        transition = np.zeros(len(label), dtype=np.int64)
        for index in range(len(label)):
            if index > 0 and label[index] != label[index - 1]:
                transition[index] = 1
                continue
            if index < len(label) - 1 and label[index] != label[index + 1]:
                transition[index] = 1
        return transition

    def _select_files(self):
        data_files = sorted(glob(os.path.join(self.data_path, "*.npz")))
        if len(data_files) <= self.val_subject_count and self.mode in {"train", "val"} and self.split_mode != "sample":
            raise ValueError("Not enough files to split train/val set.")

        if self.mode in {"train", "val"}:
            if self.split_mode == "subject_fixed":
                train_files = data_files[:-self.val_subject_count]
                val_files = data_files[-self.val_subject_count:]
            elif self.split_mode == "subject_mixed":
                rng = np.random.default_rng(self.split_seed)
                shuffled = list(data_files)
                rng.shuffle(shuffled)
                val_files = sorted(shuffled[:self.val_subject_count])
                train_files = sorted(shuffled[self.val_subject_count:])
            elif self.split_mode == "sample":
                train_files = data_files
                val_files = data_files
            else:
                raise ValueError("split_mode must be 'subject_fixed', 'subject_mixed', or 'sample'")

        if self.mode == "train":
            return train_files
        if self.mode == "val":
            return val_files
        if self.mode == "test":
            return data_files
        raise ValueError("mode must be 'train', 'val', or 'test'")

    def _load(self):
        all_data, all_label, all_trans, all_subjects = [], [], [], []
        for path in self._select_files():
            npz = np.load(path)
            data = npz["x"].astype(np.float32)
            label = npz["y"].astype(np.int64)
            trans_idx = self.trans_stage_label(label)
            subject = os.path.splitext(os.path.basename(path))[0]
            data, label, trans_idx, subjects = self._reshape_to_windows(data, label, trans_idx, subject)
            all_data.append(data)
            all_label.append(label)
            all_trans.append(trans_idx)
            all_subjects.append(subjects)

        data = np.concatenate(all_data, axis=0)
        label = np.concatenate(all_label, axis=0)
        trans_idx = np.concatenate(all_trans, axis=0)
        subjects = np.concatenate(all_subjects, axis=0)
        return self._split_loaded_samples(data, label, trans_idx, subjects)

    def _split_loaded_samples(self, data, label, trans_idx, subjects):
        if self.split_mode != "sample" or self.mode not in {"train", "val"}:
            return data, label, trans_idx, subjects

        total_windows = len(label)
        if total_windows < 2:
            raise ValueError("Need at least two windows for sample-level train/val split.")

        val_count = max(1, int(round(total_windows * self.val_ratio)))
        val_count = min(val_count, total_windows - 1)

        rng = np.random.default_rng(self.split_seed)
        indices = np.arange(total_windows)
        rng.shuffle(indices)
        val_indices = np.sort(indices[:val_count])
        train_indices = np.sort(indices[val_count:])

        selected_indices = train_indices if self.mode == "train" else val_indices
        return data[selected_indices], label[selected_indices], trans_idx[selected_indices], subjects[selected_indices]

    def _reshape_to_windows(self, data, label, trans_idx, subject):
        if data.shape[0] != label.shape[0]:
            raise ValueError("Data and label length do not match.")

        remainder = data.shape[0] % self.seq_len
        if remainder != 0:
            padding = self.seq_len - remainder
            data = np.concatenate([data, data[-padding:]], axis=0)
            label = np.concatenate([label, label[-padding:]], axis=0)
            trans_idx = np.concatenate([trans_idx, trans_idx[-padding:]], axis=0)

        window_count = data.shape[0] // self.seq_len
        data = data.reshape(window_count, self.seq_len, data.shape[1], data.shape[2])
        label = label.reshape(window_count, self.seq_len)
        trans_idx = trans_idx.reshape(window_count, self.seq_len)
        subjects = np.full(window_count, subject, dtype=object)
        return data, label, trans_idx, subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the sparse-attention apnea ECG sequence model.")
    parser.add_argument("--train-dir", default=os.path.join("dataset", "Apnea_ECG", "train"))
    parser.add_argument("--test-dir", default=os.path.join("dataset", "Apnea_ECG", "test"))
    parser.add_argument("--output-dir", default=os.path.join("result", "apnea_ecg"))
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.2, help="Transition auxiliary loss weight.")
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subject-count", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation ratio used only when split-mode=sample.")
    parser.add_argument(
        "--split-mode",
        choices=["subject_fixed", "subject_mixed", "sample"],
        default="subject_mixed",
        help="subject_fixed: last N subjects for validation; subject_mixed: shuffle subjects before splitting; sample: load all subjects first, then split windows.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(model, loader, optimizer, scheduler, criterion_main, criterion_trans, alpha, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    all_main_logits = []
    all_main_labels = []
    all_trans_logits = []
    all_trans_labels = []

    for batch_x, batch_y, batch_trans in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_trans = batch_trans.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits, logits_trans = model(batch_x)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = batch_y.reshape(-1)
            logits_trans_flat = logits_trans.reshape(-1, logits_trans.size(-1))
            trans_flat = batch_trans.reshape(-1)

            loss_main = criterion_main(logits_flat, labels_flat)
            loss_trans = criterion_trans(logits_trans_flat, trans_flat)
            loss = loss_main + alpha * loss_trans
            # loss = loss_main

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        predictions = torch.argmax(logits_flat, dim=1)
        total_loss += loss.item() * labels_flat.size(0)
        total_correct += (predictions == labels_flat).sum().item()
        total_points += labels_flat.size(0)
        all_main_logits.append(logits.detach().cpu())
        all_main_labels.append(batch_y.detach().cpu())
        all_trans_logits.append(logits_trans.detach().cpu())
        all_trans_labels.append(batch_trans.detach().cpu())

    return {
        "loss": total_loss / max(total_points, 1),
        "acc": total_correct / max(total_points, 1),
        "logits": torch.cat(all_main_logits, dim=0).numpy(),
        "labels": torch.cat(all_main_labels, dim=0).numpy(),
        "trans_logits": torch.cat(all_trans_logits, dim=0).numpy(),
        "trans_labels": torch.cat(all_trans_labels, dim=0).numpy(),
    }


def flatten_predictions(logits, labels):
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    probs_flat = probs.reshape(-1, probs.shape[-1])
    labels_flat = labels.reshape(-1)
    return probs, probs_flat, labels_flat


def evaluate_and_save(model, loader, subjects, criterion_main, criterion_trans, alpha, device, output_dir, prefix):
    epoch_result = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        criterion_main=criterion_main,
        criterion_trans=criterion_trans,
        alpha=alpha,
        device=device,
        training=False,
    )
    probs, probs_flat, labels_flat = flatten_predictions(epoch_result["logits"], epoch_result["labels"])
    metrics = classification_report_dict(labels_flat, probs_flat)
    trans_probs, trans_probs_flat, trans_labels_flat = flatten_predictions(
        epoch_result["trans_logits"], epoch_result["trans_labels"]
    )
    trans_metrics = classification_report_dict(trans_labels_flat, trans_probs_flat)

    confusion = confusion_matrix(labels_flat, metrics["y_pred"], labels=[1, 0])
    save_confusion_matrix(confusion, ["Apnea", "Normal"], os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    save_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"], os.path.join(output_dir, f"{prefix}_roc_curve.png"))

    summary = {
        "loss": float(epoch_result["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "kappa": float(metrics["kappa"]),
        "transition_accuracy": float(trans_metrics["accuracy"]),
        "transition_f1": float(trans_metrics["f1"]),
        "window_count": int(epoch_result["labels"].shape[0]),
        "window_length": int(epoch_result["labels"].shape[1]),
    }
    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_dataset = ApneaECGDataset(
        args.train_dir,
        mode="train",
        seq_len=args.seq_len,
        val_subject_count=args.val_subject_count,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
    )
    val_dataset = ApneaECGDataset(
        args.train_dir,
        mode="val",
        seq_len=args.seq_len,
        val_subject_count=args.val_subject_count,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
    )
    test_dataset = ApneaECGDataset(args.test_dir, mode="test", seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TransformerModel(seq_length=args.seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * len(train_loader), 1),
        eta_min=1e-6,
    )
    criterion_main = nn.CrossEntropyLoss()
    criterion_trans = FocalLoss(gamma=2, alpha=0.6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    history_path = os.path.join(args.output_dir, "history.csv")
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))

    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, optimizer, scheduler, criterion_main, criterion_trans, args.alpha, device, True
        )
        val_result = run_epoch(
            model, val_loader, None, None, criterion_main, criterion_trans, args.alpha, device, False
        )

        _, train_probs_flat, train_labels_flat = flatten_predictions(train_result["logits"], train_result["labels"])
        _, val_probs_flat, val_labels_flat = flatten_predictions(val_result["logits"], val_result["labels"])
        train_metrics = classification_report_dict(train_labels_flat, train_probs_flat)
        val_metrics = classification_report_dict(val_labels_flat, val_probs_flat)

        history["train_loss"].append(train_result["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_result["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        append_history_row(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_result["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        writer.add_scalars("loss", {"train": train_result["loss"], "val": val_result["loss"]}, epoch)
        writer.add_scalars("accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
        writer.add_scalars("f1", {"train": train_metrics["f1"], "val": val_metrics["f1"]}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_result['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_result['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pt"))
    plot_history(history, os.path.join(args.output_dir, "history.png"))

    if best_state is not None:
        model.load_state_dict(best_state)

    val_summary = evaluate_and_save(
        model,
        val_loader,
        val_dataset.subjects,
        criterion_main,
        criterion_trans,
        args.alpha,
        device,
        args.output_dir,
        "val",
    )
    test_summary = evaluate_and_save(
        model,
        test_loader,
        test_dataset.subjects,
        criterion_main,
        criterion_trans,
        args.alpha,
        device,
        args.output_dir,
        "test",
    )

    final_summary = {"best_val_f1": float(best_val_f1), "val": val_summary, "test": test_summary}
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
