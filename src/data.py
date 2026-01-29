"""Dataset loaders for medical images: folder-based or CSV."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transforms(image_size: tuple[int, int], is_train: bool):
    base = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            *base,
        ])
    return transforms.Compose(base)


class FolderDataset(Dataset):
    """Images in folders: root/train/cancer/, root/train/normal/."""

    def __init__(self, root: Path, split: str, transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []
        for class_idx, class_name in enumerate(sorted(self.root.iterdir())):
            if not self.root.joinpath(class_name).is_dir():
                continue
            for p in self.root.joinpath(class_name).iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(p), class_idx))
        self.classes = sorted(d.name for d in self.root.iterdir() if self.root.joinpath(d).is_dir())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(config: dict):
    """Build train/val/test DataLoaders from config."""
    data_cfg = config.get("data", {})
    root = Path(data_cfg.get("root", "data")).expanduser().resolve()
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_tf = get_transforms(image_size, is_train=True)
    eval_tf = get_transforms(image_size, is_train=False)

    # pin_memory only on CUDA (not supported on MPS/CPU); avoids Mac warning
    pin_memory = torch.cuda.is_available()
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        ds = FolderDataset(root, split, transform=train_tf if split == "train" else eval_tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train folder under {root}. Create data/train/cancer/ and data/train/normal/.")
    if "val" not in loaders:
        raise FileNotFoundError(f"No val folder under {root}. Create data/val/cancer/ and data/val/normal/.")
    return loaders
