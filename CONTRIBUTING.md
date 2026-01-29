# Contributing to Medical Cancer CNN

Thank you for considering contributing. This project aims to be a useful, reproducible open-source pipeline for cancer detection from medical images.

---

## How to contribute

- **Report bugs:** Open an issue with steps to reproduce, Python/environment info, and (if possible) a minimal example.
- **Suggest features:** Open an issue describing the use case (e.g. new dataset, new metric, Grad-CAM).
- **Code:** See below for setup and pull request process.

---

## Development setup

```bash
git clone https://github.com/matingathani/medical-cancer-cnn.git
cd medical-cancer-cnn
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .   # optional: install in editable mode if you add pyproject.toml
```

Run the smoke test to confirm the pipeline works:

```bash
python -m pytest tests/ -v
```

---

## Code style

- **Python:** Follow PEP 8. Use type hints where helpful. Keep functions focused.
- **Docstrings:** Use for public functions/classes (one-line or short description).
- **Config:** Add new options in `config.example.yaml` with a short comment.

---

## Pull request process

1. **Fork** the repo and create a branch from `main` (e.g. `feature/grad-cam` or `fix/dataloader`).
2. **Make your changes** and run the smoke test: `python -m pytest tests/ -v`.
3. **Update docs** if you add options or scripts (README, `docs/`).
4. **Open a PR** with a clear title and description. Link any related issue.
5. Maintainers will review; you may be asked to adjust code or docs.

---

## Areas that need help

- **Grad-CAM / interpretability:** Implement in `visualize.py` (see [docs/ROADMAP.md](docs/ROADMAP.md)).
- **More datasets:** Scripts or instructions for ISIC, chest X-ray, etc. (see [docs/DATASETS.md](docs/DATASETS.md)).
- **Tests:** Expand `tests/` (e.g. unit tests for metrics, config loading).
- **Documentation:** Tutorials, blog posts, or short videos showing real-world use.

---

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.
