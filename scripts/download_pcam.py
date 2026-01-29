#!/usr/bin/env python3
"""
Download PatchCamelyon (PCam) and export to project folder layout.
Real histopathology images: lymph node metastasis (cancer) vs normal.

Usage:
  python scripts/download_pcam.py
  python scripts/download_pcam.py --max-train 5000 --max-val 500 --max-test 500
  python scripts/download_pcam.py --full   # export full dataset (~7GB download, ~262k train images)

Requires: pip install h5py gdown (see requirements.txt)
"""

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# PCAM from torchvision (downloads from Google Drive, stores as H5)
from torchvision.datasets import PCAM
from tqdm import tqdm

# Where to download raw PCAM (H5 files)
PCAM_CACHE = ROOT / "data_pcam"
# Where to write images for this project (data/train/cancer, etc.)
DATA_ROOT = ROOT / "data"
CLASSES = ("normal", "cancer")  # PCAM label 0 = normal, 1 = metastatic (cancer)


def export_split(split: str, max_per_class: int | None, pcam_cache: Path) -> None:
    """Download PCAM split if needed and save images to data/<split>/cancer|normal/."""
    out_dir = DATA_ROOT / split
    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    print(f"Loading {split} split (download if needed)...")
    ds = PCAM(root=str(pcam_cache), split=split, download=True)
    n_total = len(ds)
    counts = {0: 0, 1: 0}
    target_max = max_per_class if max_per_class else n_total
    saved = 0

    with tqdm(total=min(n_total, target_max * 2), desc=f"Export {split}", unit="img") as pbar:
        for idx in range(n_total):
            if counts[0] >= target_max and counts[1] >= target_max:
                break
            img, label = ds[idx]
            if counts[label] >= target_max:
                pbar.update(1)
                continue
            class_name = CLASSES[label]
            path = out_dir / class_name / f"pcam_{split}_{idx:06d}.png"
            img.save(path)
            counts[label] += 1
            saved += 1
            pbar.update(1)

    print(f"  Saved {saved} images to {out_dir} (normal={counts[0]}, cancer={counts[1]})")


def main():
    parser = argparse.ArgumentParser(description="Download PatchCamelyon and export to data/train|val|test")
    parser.add_argument("--max-train", type=int, default=2000, help="Max images per class for train (default 2000)")
    parser.add_argument("--max-val", type=int, default=400, help="Max images per class for val (default 400)")
    parser.add_argument("--max-test", type=int, default=400, help="Max images per class for test (default 400)")
    parser.add_argument("--full", action="store_true", help="Export full dataset (no limit)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output root (default: data/)")
    args = parser.parse_args()

    if args.out_dir:
        global DATA_ROOT
        DATA_ROOT = Path(args.out_dir).resolve()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PCAM_CACHE.mkdir(parents=True, exist_ok=True)

    if args.full:
        max_train = max_val = max_test = None
        print("Exporting full PCam dataset (this will take a while)...")
    else:
        max_train, max_val, max_test = args.max_train, args.max_val, args.max_test
        print(f"Exporting subset: train≤{max_train}/class, val≤{max_val}/class, test≤{max_test}/class")

    for split, max_per in (("train", max_train), ("val", max_val), ("test", max_test)):
        export_split(split, max_per, PCAM_CACHE)

    print(f"\nDone. Real histopathology images are in: {DATA_ROOT}")
    print("Next: python train.py   (then python run_demo.py uses this data if present)")


if __name__ == "__main__":
    main()
