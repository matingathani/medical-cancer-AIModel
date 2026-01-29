"""
Smoke tests: verify the pipeline runs without errors.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_imports():
    """Core modules can be imported."""
    from src import config, data, metrics, models
    assert config.load_config is not None
    assert data.get_dataloaders is not None
    assert metrics.compute_metrics is not None
    assert models.build_model is not None


def test_config_loads():
    """Config loads from example if config.yaml missing."""
    from src.config import load_config
    cfg = load_config()
    assert "data" in cfg or "model" in cfg or "train" in cfg


def test_model_builds():
    """Model builds from config."""
    from src.config import load_config
    from src.models import build_model
    cfg = load_config()
    model = build_model(cfg)
    assert model is not None
    # ResNet18 or SimpleCNN
    assert hasattr(model, "forward")


def test_dummy_data_and_one_epoch(tmp_path):
    """Create dummy data and run 1 training epoch (smoke test)."""
    import torch
    import subprocess
    from src.config import load_config
    from src.data import get_dataloaders
    from src.models import build_model

    # Create dummy data under tmp_path
    data_root = tmp_path / "data"
    for split in ("train", "val"):
        for cls in ("cancer", "normal"):
            (data_root / split / cls).mkdir(parents=True)
    # Add 3 tiny images per folder (create_dummy_data would need root override)
    import numpy as np
    from PIL import Image
    for split in ("train", "val"):
        for cls in ("cancer", "normal"):
            for i in range(3):
                arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(data_root / split / cls / f"img_{i}.jpg")

    cfg = load_config()
    cfg = dict(cfg)
    cfg["data"] = cfg.get("data", {})
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["num_workers"] = 0
    cfg["train"] = cfg.get("train", {})
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["epochs"] = 1
    cfg["train"]["output_dir"] = str(tmp_path / "out")

    loaders = get_dataloaders(cfg)
    assert "train" in loaders and "val" in loaders
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for X, y in loaders["train"]:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        break  # one batch only
    assert True  # no exception
