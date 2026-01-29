#!/usr/bin/env python3
"""
Run the trained model on a single image or a folder of images.
Use this to show the professor "drop an image → get cancer/normal + confidence".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.config import load_config
from src.data import get_transforms
from src.models import build_model


def load_model_and_config(checkpoint_path: Path):
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", load_config())
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def predict_image(model, image_path: Path, config: dict, device):
    """Run prediction on one image; return class index, label, confidence, probs."""
    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    transform = get_transforms(image_size, is_train=False)
    class_names = ["normal", "cancer"]

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(logits.argmax(dim=1).item())
    conf = float(probs[pred_idx])
    label = class_names[pred_idx]
    return pred_idx, label, conf, probs, img


def save_prediction_figure(image_path: Path, img, label: str, conf: float, probs, out_path: Path):
    """Save a figure: image + predicted class and confidence."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Predicted: {label} ({conf:.0%})\n(normal: {probs[0]:.0%}, cancer: {probs[1]:.0%})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict cancer vs normal on image(s)")
    parser.add_argument("--checkpoint", type=str, default="outputs_demo/best.pt", help="Path to checkpoint")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, default="outputs_demo/predictions", help="Folder to save prediction figures")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures, only print")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run: python run_demo.py  (then use outputs_demo/best.pt)")
        sys.exit(1)

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_and_config(ckpt_path)
    model = model.to(device)

    class_names = ["normal", "cancer"]
    if args.image:
        paths = [Path(args.image)]
        if not paths[0].exists():
            print(f"Image not found: {paths[0]}")
            sys.exit(1)
    else:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"Folder not found: {folder}")
            sys.exit(1)
        paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        paths.sort()

    out_dir = Path(args.output) if not args.no_save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPredictions (checkpoint: {ckpt_path.name})\n")
    print(f"{'Image':<40} {'Predicted':<10} {'Confidence':<10} {'P(normal)':<10} {'P(cancer)':<10}")
    print("-" * 80)

    for path in paths:
        pred_idx, label, conf, probs, img = predict_image(model, path, config, device)
        p_norm, p_cancer = probs[0], probs[1]
        name = path.name if len(path.name) <= 38 else path.name[:35] + "..."
        print(f"{name:<40} {label:<10} {conf:.2%}       {p_norm:.2%}       {p_cancer:.2%}")
        if out_dir:
            out_path = out_dir / f"pred_{path.stem}.png"
            save_prediction_figure(path, img, label, conf, probs, out_path)

    if out_dir and paths:
        print(f"\nFigures saved to: {out_dir}")
    print()


if __name__ == "__main__":
    main()
