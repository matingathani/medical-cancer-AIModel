# Medical Cancer CNN

<p align="center">
  <strong>Open-source CNN pipeline for cancer detection from medical images</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-green.svg" alt="Python 3.9+" /></a>
</p>

---

A **reproducible, config-driven pipeline** for training and evaluating CNNs on medical imaging (histopathology, skin lesions, chest X-rays). Includes **medical-grade metrics** (sensitivity, specificity, AUC-ROC), one-command **PatchCamelyon** download, and a **demo** for research, education, and pitching to stakeholders.

**Use it to:** train models on your own or public datasets, run a quick demo for a professor or collaborator, extend with new architectures or interpretability (e.g. Grad-CAM), and contribute back to an open-source medical imaging toolkit.

---

## Impact & who this is for

| Audience | Use case |
|----------|----------|
| **Researchers** | Reproducible baseline for cancer vs normal (or multi-class) from medical images; add your dataset or model and compare. |
| **Students / educators** | Teach medical ML with real metrics (sensitivity, specificity) and a working codebase; run the demo in minutes. |
| **Clinicians / collaborators** | See a working pipeline and metrics; discuss validation and deployment (see [Ethics & limitations](docs/ETHICS_AND_LIMITATIONS.md)). |
| **Open-source contributors** | Extend with new datasets, Grad-CAM, cross-validation, or deployment; see [CONTRIBUTING.md](CONTRIBUTING.md). |

This project is **not** a medical device and is not for direct clinical diagnosis; see [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md).

---

## Features

- **Dataset-agnostic** — Folder-based (`train/cancer/`, `train/normal/`) or add CSV support; works with any binary/multi-class medical image dataset.
- **Real data in one command** — `python scripts/download_pcam.py` downloads **PatchCamelyon** (histopathology) and exports to `data/` (subset or full).
- **CNNs** — Simple CNN and ResNet18 (transfer learning); easy to add more in `src/models.py`.
- **Medical metrics** — Accuracy, **sensitivity**, **specificity**, **AUC-ROC**, confusion matrix (see [Metrics](#metrics-medical-context)).
- **Reproducibility** — Config-driven, fixed seeds, requirements pinned; optional [Makefile](Makefile) for common commands.
- **Demo** — `run_demo.py` trains and evaluates in minutes; `demo_predict.py` runs inference on single images or folders.
- **CI** — GitHub Actions run smoke tests on push/PR (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/matingathani/medical-cancer-cnn.git
cd medical-cancer-cnn
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get data

**Option A — One-command real data (PatchCamelyon, histopathology):**

```bash
python scripts/download_pcam.py
```

Exports a subset to `data/train/`, `data/val/`, `data/test/` (cancer vs normal). Use `--max-train 2000 --max-val 400 --max-test 400` for a larger subset, or `--full` for the entire dataset. Requires `h5py` and `gdown` (in `requirements.txt`).

**Option B — Your own folder layout:**

```text
data/
  train/
    cancer/   # positive class
    normal/   # negative class
  val/
    cancer/
    normal/
  test/
    cancer/
    normal/
```

See [docs/DATASETS.md](docs/DATASETS.md) for more public datasets (ISIC, chest X-ray, etc.).

**Option C — Quick test (no real data):**

```bash
python scripts/create_dummy_data.py
```

### 3. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`: set `data.root` to your `data/` path, adjust `model.name`, `train.epochs`, etc.

### 4. Train

```bash
python train.py
```

Checkpoints and logs go to `outputs/` (or path in config).

### 5. Evaluate

```bash
python evaluate.py --checkpoint outputs/best.pt
```

Prints accuracy, sensitivity, specificity, AUC-ROC, and saves a confusion matrix.

---

## Demo (pitch / stakeholder)

To show a **working model** in a few minutes (creates dummy data if no `data/`):

```bash
python run_demo.py
```

Then run on a new image:

```bash
python demo_predict.py --checkpoint outputs_demo/best.pt --image path/to/image.jpg
```

See [docs/DEMO_PITCH.md](docs/DEMO_PITCH.md) for a one-pager to use when presenting.

---

## Project structure

```text
medical-cancer-cnn/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CITATION.cff
├── SECURITY.md
├── Makefile
├── requirements.txt
├── config.example.yaml
├── train.py
├── evaluate.py
├── run_demo.py
├── demo_predict.py
├── visualize.py          # Optional Grad-CAM (stub)
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── models.py
├── scripts/
│   ├── create_dummy_data.py
│   └── download_pcam.py
├── tests/
│   └── test_smoke.py
├── docs/
│   ├── DATASETS.md
│   ├── DEMO_PITCH.md
│   ├── ETHICS_AND_LIMITATIONS.md
│   └── ROADMAP.md
├── data/                 # Your data (gitignored)
├── outputs/              # Checkpoints (gitignored)
└── .github/workflows/
    └── ci.yml
```

---

## Metrics (medical context)

- **Sensitivity (recall)** — Among actual positives, how many did we catch? (minimize missed cancers.)
- **Specificity** — Among actual negatives, how many did we label correctly? (minimize false alarms.)
- **AUC-ROC** — Overall discriminative ability across thresholds.
- **Confusion matrix** — Saved as image in `outputs/`.

---

## Making it public on GitHub

1. **Create a new repo** on GitHub (e.g. `medical-cancer-cnn`), then:

   ```bash
git remote add origin https://github.com/matingathani/medical-cancer-cnn.git
  git branch -M main
  git push -u origin main
```

2. **Replace placeholders:** In [CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md), and this README, replace `matingathani` with your GitHub username.

3. **Optional:** Add a **Description** and **Topics** on the repo (e.g. `medical-imaging`, `cancer-detection`, `pytorch`, `histopathology`). Enable **Issues** and **Discussions** if you want community input.

4. **Optional:** Pin the repo to your profile and add a short “About” for impact.

---

## Contributing & citation

- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, suggest features, or send pull requests.
- **Citation:** If you use this software in research, cite it using the metadata in [CITATION.cff](CITATION.cff), or:

  ```bibtex
  @software{medical_cancer_cnn,
    title = {Medical Cancer CNN},
    author = {Medical Cancer CNN Contributors},
    url = {https://github.com/YOUR_USERNAME/medical-cancer-cnn},
    license = {MIT},
    year = {2025}
  }
  ```
  (Replace `matingathani` with your GitHub username.)

- **Ethics & limitations:** [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md)  
- **Roadmap (Grad-CAM, cross-validation, etc.):** [docs/ROADMAP.md](docs/ROADMAP.md)

---

## License

MIT. See [LICENSE](LICENSE).
