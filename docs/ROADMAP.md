# Roadmap: from portfolio-ready to “perfect”

Use this checklist to level up the project for a stronger resume or publication-style report.

## Already in place

- [x] Config-driven pipeline (data, model, train)
- [x] Folder-based dataset loader + preprocessing
- [x] Simple CNN + ResNet18 (transfer learning)
- [x] Medical metrics: accuracy, sensitivity, specificity, AUC-ROC
- [x] Confusion matrix plot
- [x] Reproducibility: seed, requirements, config

## MVP (1–2 days) — done if you ran train + evaluate once

- [ ] Download or create a small dataset and run `train.py`
- [ ] Run `evaluate.py` and inspect metrics + confusion matrix
- [ ] Add 1–2 sentences to README: “Tested on [dataset name]”

## Portfolio-ready (3–5 days)

- [ ] Use a **public medical dataset** (e.g. PatchCamelyon, ISIC subset, or chest X-ray) and document it in README and `docs/DATASETS.md`
- [ ] Report **sensitivity and specificity** in README (and optionally in a short `results.md`)
- [ ] Add **AUC-ROC** to the reported metrics
- [ ] Ensure **reproducibility**: fixed seed in config, same Python/package versions in `requirements.txt`
- [ ] Optional: **Learning curves** (plot train/val loss and accuracy vs epoch)

## “Perfect” (1–2 weeks)

- [ ] **Cross-validation:** 5-fold (or 3-fold) over the training set; report mean ± std for sensitivity, specificity, AUC
- [ ] **Interpretability:** Add Grad-CAM (or similar) in `visualize.py` and save example heatmaps for a few test images
- [ ] **Ablation or comparison:** e.g. ResNet18 vs SimpleCNN, or with/without augmentation; summarize in a short report
- [ ] **Short write-up:** 1–2 page “report” (PDF or Markdown) with: dataset, method, metrics, and one or two result figures (confusion matrix, ROC curve, Grad-CAM)
- [ ] **CI (optional):** GitHub Action that runs a quick sanity check (e.g. 1 epoch on a tiny subset) to ensure code runs

## Optional extras

- [ ] Support **CSV dataset** (columns: path, label) in `src/data.py`
- [ ] **ROC curve** plot in `evaluate.py`
- [ ] **Multi-class** (e.g. multiple cancer types) by increasing `num_classes` and ensuring dataset has that many folders
- [ ] **Mixed precision** (AMP) for faster training on GPU
