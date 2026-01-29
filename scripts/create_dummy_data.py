#!/usr/bin/env python3
"""Create minimal dummy images for pipeline testing (not for real training).
Generates grayscale-style synthetic patches that look like placeholder medical
images (tissue-like texture) so demo outputs don't show random rainbow noise."""

from pathlib import Path
import numpy as np
from PIL import Image

# Target size (will be resized by the pipeline to 224x224)
SIZE = 96


def _synthetic_patch(rng: np.random.Generator, class_idx: int) -> np.ndarray:
    """Create a grayscale-ish patch with simple structure (blob + texture)."""
    # Base: dark background (like tissue/slide)
    base = rng.integers(80, 140, (SIZE, SIZE), dtype=np.uint8)
    # Add a "region" (lighter blob) - slightly different for class
    y, x = np.ogrid[:SIZE, :SIZE]
    cy, cx = rng.integers(20, SIZE - 20, 2)
    radius = rng.integers(15, 35)
    mask = ((y - cy) ** 2 + (x - cx) ** 2) < (radius ** 2)
    blob_val = 180 + rng.integers(0, 40) if class_idx == 1 else 120 + rng.integers(0, 50)
    base[mask] = np.clip(base[mask] + blob_val - 120, 0, 255).astype(np.uint8)
    # Light texture (fine grain)
    base = np.clip(base + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8)
    # Return as RGB (same in all channels) so it looks like grayscale medical placeholder
    return np.stack([base, base, base], axis=-1)


def main():
    root = Path(__file__).resolve().parent.parent / "data"
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for cls in ("cancer", "normal"):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            class_idx = 1 if cls == "cancer" else 0
            n = 5 if split == "train" else 2
            for i in range(n):
                arr = _synthetic_patch(rng, class_idx)
                Image.fromarray(arr).save(d / f"img_{i}.jpg")
    print(f"Created dummy images under {root} (synthetic placeholders, not real medical images)")

if __name__ == "__main__":
    main()
