#!/usr/bin/env python3
"""Evaluation script: compute medical metrics and save confusion matrix."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

sys_path = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(sys_path))

from src.config import load_config
from src.data import get_dataloaders
from src.models import build_model
from src.metrics import compute_metrics, get_confusion_matrix


def _class_names(loader):
    if hasattr(loader.dataset, "classes"):
        return getattr(loader.dataset, "classes", ["0", "1"])
    return ["normal", "cancer"]


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())
        all_scores.append(probs[:, 1].cpu().numpy())  # probability of positive class
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_scores),
    )


def plot_confusion_matrix(cm, class_names, save_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt", help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", load_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    loaders = get_dataloaders(config)
    if args.split not in loaders:
        raise ValueError(f"No {args.split} split. Use train, val, or test.")
    loader = loaders[args.split]
    y_pred, y_true, y_score = predict_all(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, y_score)
    print(f"Split: {args.split}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    if metrics["auc_roc"] is not None:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")

    class_names = _class_names(loader)
    cm, _ = get_confusion_matrix(y_true, y_pred, class_names=class_names)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
