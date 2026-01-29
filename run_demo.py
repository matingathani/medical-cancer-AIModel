#!/usr/bin/env python3
"""
One-command demo: create data (if needed) → train → evaluate → save sample predictions.
Use this to showcase the pipeline to a professor or stakeholder in ~2–5 minutes.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import subprocess
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import load_config
from src.data import get_dataloaders, get_transforms
from src.models import build_model
from src.metrics import compute_metrics, get_confusion_matrix


def ensure_data_and_config():
    """Create dummy data and config if missing."""
    root = Path("data")
    if not (root / "train" / "cancer").exists() or not (root / "train" / "normal").exists():
        print("No data found. Creating dummy data for demo...")
        subprocess.run([sys.executable, "scripts/create_dummy_data.py"], check=True)
    config_path = Path("config.yaml")
    if not config_path.exists():
        import shutil
        shutil.copy("config.example.yaml", "config.yaml")
        print("Created config.yaml from config.example.yaml")
    return load_config()


def train_demo(config, demo_epochs=5):
    """Train with fewer epochs for a quick demo."""
    # Override for demo: fewer epochs
    config = dict(config)
    config.setdefault("train", {})
    config["train"] = dict(config["train"])
    config["train"]["epochs"] = demo_epochs
    config["train"]["output_dir"] = "outputs_demo"
    config["data"] = dict(config.get("data", {}))
    config["data"]["num_workers"] = 0  # avoid DataLoader shared-memory issues in terminals/sandboxes
    Path(config["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    import torch.nn as nn
    from torch.optim import AdamW
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["train"].get("seed", 42))
    loaders = get_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"].get("lr", 1e-3),
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )
    epochs = config["train"]["epochs"]
    output_dir = Path(config["train"]["output_dir"])
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loaders["val"]:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, output_dir / "best.pt")
        print(f"  Epoch {epoch}  val_acc={val_acc:.4f}")
    print(f"Demo training done. Best val accuracy: {best_val_acc:.4f}")
    return output_dir / "best.pt", config


@torch.no_grad()
def evaluate_and_show_metrics(ckpt_path, config):
    """Evaluate on test split and print medical metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", config)
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    loaders = get_dataloaders(config)
    if "test" not in loaders:
        loader = loaders["val"]
        split_name = "val"
    else:
        loader = loaders["test"]
        split_name = "test"

    all_preds, all_labels, all_scores = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.numpy())
        all_scores.append(probs[:, 1].cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    metrics = compute_metrics(y_true, y_pred, y_score)
    print("\n--- Demo evaluation (medical metrics) ---")
    print(f"  Split: {split_name}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}  (among true positives, how many we catch)")
    print(f"  Specificity: {metrics['specificity']:.4f}  (among true negatives, how many we label correctly)")
    if metrics["auc_roc"] is not None:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print("----------------------------------------\n")
    return model, config, loader, metrics


def save_sample_predictions(model, config, loader, device, out_dir: Path, num_samples=6):
    """Run model on a few images and save visualizations (image + predicted class + confidence)."""
    from src.data import FolderDataset
    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    transform = get_transforms(image_size, is_train=False)
    class_names = getattr(loader.dataset, "classes", ["normal", "cancer"])

    # Get paths from dataset (FolderDataset has .samples as (path, label))
    samples = getattr(loader.dataset, "samples", [])
    if not samples:
        print("No sample paths in dataset; skipping prediction visuals.")
        return
    indices = np.linspace(0, len(samples) - 1, min(num_samples, len(samples)), dtype=int)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        path, true_label = samples[idx]
        img_pil = Image.open(path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(logits.argmax(dim=1).item())
        conf = float(probs[pred_label])
        pred_name = class_names[pred_label]
        true_name = class_names[true_label]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img_pil)
        ax.axis("off")
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"True: {true_name}  →  Predicted: {pred_name} ({conf:.0%})", fontsize=12, color=color)
        fig.text(0.5, 0.02, "Demo data: synthetic placeholders (not real medical images)", ha="center", fontsize=9, color="gray")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(out_dir / f"demo_pred_{i+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(f"Sample predictions saved to {out_dir}/demo_pred_*.png")


def main():
    print("=== Medical Cancer CNN — Demo for showcase ===\n")
    config = ensure_data_and_config()
    ckpt_path, config = train_demo(config, demo_epochs=5)
    model, config, loader, _ = evaluate_and_show_metrics(ckpt_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.get("train", {}).get("output_dir", "outputs_demo"))
    save_sample_predictions(model, config, loader, device, out_dir, num_samples=6)
    print("Done. Show the professor:")
    print(f"  1. Checkpoint: {ckpt_path}")
    print(f"  2. Sample prediction images: {out_dir}/demo_pred_*.png")
    print(f"  3. Run on a new image: python demo_predict.py --checkpoint {ckpt_path} --image <path>")


if __name__ == "__main__":
    main()
