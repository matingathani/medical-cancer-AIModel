# Demo pitch: CNN for cancer detection from medical images

**One-pager for presenting to a cancer researcher / professor**

---

## The idea

Use **convolutional neural networks (CNNs)** to classify medical images into **cancer vs normal** (binary). The same pipeline can be adapted to histopathology (e.g. lymph node metastasis), skin lesions, or chest X-rays by swapping the dataset and keeping the training/evaluation code.

**Why it matters:** Reproducible, config-driven pipeline with **medical-grade metrics** (sensitivity, specificity, AUC-ROC) so the model’s behaviour is interpretable and comparable to clinical expectations.

---

## What the demo shows

**Important:** The one-command demo (`run_demo.py`) uses **synthetic placeholder images** (grayscale-style patches), not real medical images. Prediction figures are labeled accordingly. For a credible pitch with the professor, use a small real dataset (see [DATASETS.md](DATASETS.md)) so they see realistic inputs and metrics.

1. **End-to-end pipeline**
   - Data in folders: `train/cancer/`, `train/normal/`, same for `val/` and `test/`.
   - Training with validation and early stopping.
   - Evaluation: accuracy, **sensitivity**, **specificity**, **AUC-ROC**, confusion matrix.

2. **Working model**
   - A small CNN or ResNet18 (transfer learning) trained on your data (or dummy data for a quick demo).
   - Single-image or folder prediction: “drop an image → cancer / normal + confidence”.

3. **Interpretability**
   - Sample prediction images: input image + predicted class + confidence.
   - (Optional future: Grad-CAM to show *where* the model looks.)

---

## How to run the demo (for the professor)

**Option A — One-command full demo (no data needed)**  
Creates dummy data, trains for a few epochs, evaluates, and saves sample predictions:

```bash
cd medical-cancer-AIModel
python -m venv .venv && source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run_demo.py
```

Output: trained checkpoint in `outputs_demo/best.pt`, metrics in the terminal, and images in `outputs_demo/demo_pred_*.png`.

**Option B — Predict on a new image (after running the demo or training)**  
Use the trained model on a single image or a folder:

```bash
python demo_predict.py --checkpoint outputs_demo/best.pt --image path/to/image.jpg
# or
python demo_predict.py --checkpoint outputs_demo/best.pt --folder path/to/images/ --output outputs_demo/predictions
```

Prints predicted class (cancer / normal) and confidence; optionally saves a figure per image.

**Option C — Real data**  
Put your own images in `data/train/cancer/`, `data/train/normal/`, and same for `val/` and `test/`. Copy `config.example.yaml` to `config.yaml`, set `data.root: data`, then:

```bash
python train.py
python evaluate.py --checkpoint outputs/best.pt
python demo_predict.py --checkpoint outputs/best.pt --image path/to/test.jpg
```

---

## Metrics (why they matter clinically)

| Metric        | Meaning |
|---------------|--------|
| **Sensitivity** | Among true cancers, how many did we catch? (minimize missed cancers) |
| **Specificity** | Among true normals, how many did we label correctly? (minimize false alarms) |
| **AUC-ROC**   | Overall discriminative ability across decision thresholds |
| **Confusion matrix** | TP, TN, FP, FN at a glance |

---

## Caveats and next steps (for “real life”)

- **Demo data:** The one-command demo uses random dummy images, so metrics are not meaningful. For a real pitch, use a small subset of a public dataset (e.g. PatchCamelyon, ISIC) so the professor sees realistic numbers.
- **Data quality and ethics:** Only use data you are allowed to use (public or with approval). Do not commit patient data or identifiers.
- **Clinical use:** This is a research/educational pipeline. Deployment in clinical settings would require validation on held-out data, regulatory considerations, and integration with clinical workflows.
- **Improvements:** Cross-validation, more architectures, Grad-CAM or other interpretability, and hyperparameter tuning — see [ROADMAP.md](ROADMAP.md).

---

## Summary

- **Pitch:** “A reproducible CNN pipeline for cancer vs normal classification from medical images, with medical metrics and a working demo you can run in minutes.”
- **Show:** Run `python run_demo.py`, then show `outputs_demo/demo_pred_*.png` and run `demo_predict.py` on one image so they see the model in action.
- **Next:** Use real (allowed) data, tune and validate, then discuss collaboration or extension (e.g. other modalities, interpretability, deployment).
