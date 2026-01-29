"""CNN models for binary/multiclass cancer detection."""

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Lightweight CNN for small datasets."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "resnet18")
    num_classes = model_cfg.get("num_classes", 2)
    pretrained = model_cfg.get("pretrained", True)

    if name == "cnn_simple":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")
