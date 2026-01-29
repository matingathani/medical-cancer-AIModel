#!/usr/bin/env python3
"""Training script: train CNN on medical images with config-driven pipeline."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.data import get_dataloaders
from src.models import build_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    config = load_config()
    train_cfg = config.get("train", {})
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 20)
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_every = train_cfg.get("save_every", 5)
    best_val_acc = -1.0
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, output_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, output_dir / f"epoch_{epoch}.pt")
        if patience and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
