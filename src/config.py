"""Load and validate config from YAML."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "config.example.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}
