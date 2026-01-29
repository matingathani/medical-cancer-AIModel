# Public datasets for medical cancer image analysis

Use one of these (or your own data in the same folder layout) to train the model.

## Quick links (where to get real data)

| Dataset | What it is | Direct link |
|--------|------------|-------------|
| **PatchCamelyon (PCam)** | Histopathology, lymph node metastasis (96×96 patches) | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/patch_camelyon) · [GitHub](https://github.com/basveeling/pcam) |
| **ISIC** | Skin lesions (melanoma vs benign) | [ISIC Archive](https://www.isic-archive.com/) — create account, download ISIC 2019/2020 |
| **RSNA Pneumonia** | Chest X-rays (pneumonia vs normal) | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
| **NIH ChestX-ray14** | Chest X-rays (multiple findings) | [NIH](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) — request access |

---

## Folder layout (required)

Place images so that:

```text
data/
  train/
    cancer/    ← positive class (e.g. malignant, abnormal)
    normal/    ← negative class (e.g. benign, normal)
  val/
    cancer/
    normal/
  test/
    cancer/
    normal/
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`. Names of subfolders become class labels (e.g. `cancer`, `normal`).

---

## 1. PatchCamelyon (PCam) — histopathology, lymph node metastasis

- **Task:** Binary classification (metastatic vs normal patch in lymph node histology).
- **Size:** ~320k train images (96×96), small and easy to download.
- **Link:** [TensorFlow Datasets - PatchCamelyon](https://www.tensorflow.org/datasets/catalog/patch_camelyon) or [GitHub - basveeling/pcam](https://github.com/basveeling/pcam).
- **One-command download (this repo):** Run `python scripts/download_pcam.py` to download PCam via torchvision and export a subset into `data/train/`, `data/val/`, `data/test/` with `cancer/` and `normal/` folders. Requires `pip install h5py gdown`. Use `--max-train 2000 --max-val 400 --max-test 400` (default) or `--full` for the entire dataset.
- **How to use (manual):** Download and write images into `data/train/cancer/`, `data/train/normal/`, etc. Resize to 224×224 in config if using ResNet.

---

## 2. ISIC — skin lesion (melanoma vs benign)

- **Task:** Binary (or multi-class) skin lesion classification.
- **Link:** [ISIC Archive](https://www.isic-archive.com/) — create account, download a subset (e.g. ISIC 2019 or 2020).
- **How to use:** After download, place images in `data/train/melanoma/` and `data/train/benign/` (or `cancer` / `normal` and set same in config). You can map “malignant” → `cancer`, “benign” → `normal`.

---

## 3. Chest X-ray (lung / pneumonia)

- **Task:** Binary or multi-label (e.g. pneumonia, normal).
- **Options:**
  - [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Kaggle).
  - [NIH ChestX-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) (need to request).
- **How to use:** Export images and labels so that positive class (e.g. “pneumonia” or “opacity”) goes into `data/train/cancer/`, negative into `data/train/normal/`. Split into train/val/test yourself.

---

## 4. Your own data

- **CSV option (future):** Config can support `train_csv`, `val_csv`, `test_csv` with columns `path`, `label`. Not implemented in the initial release; you can add a CSV dataset in `src/data.py` and keep the same folder layout for now.
- **Ethics & consent:** Only use data you are allowed to use (public, or with institutional approval). Do not commit patient data or identifiers.

---

## Quick test without real data

To test the pipeline without downloading a full dataset:

1. Create minimal folders: `data/train/cancer/`, `data/train/normal/`, `data/val/cancer/`, `data/val/normal/`, `data/test/cancer/`, `data/test/normal/`.
2. Put a few JPEG/PNG images (e.g. 10–20 per class) in each. You can use any small images (e.g. from ImageNet samples or random placeholders).
3. Run `python train.py` with a small number of epochs (e.g. 2) to verify the pipeline.
