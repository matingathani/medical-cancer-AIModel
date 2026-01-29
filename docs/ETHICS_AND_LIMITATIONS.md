# Ethics and limitations (real-world use)

This document is for researchers, clinicians, and anyone considering using or extending this project.

---

## Intended use

- **Research and education:** Reproducible pipeline for training and evaluating CNNs on medical imaging (e.g. histopathology, skin lesions, chest X-rays).
- **Prototyping:** Quick experiments with different models, datasets, and metrics.
- **Teaching:** Demonstrating medical-grade metrics (sensitivity, specificity, AUC-ROC) and interpretability.

---

## Not intended for

- **Direct clinical use:** This software is **not** a medical device. Do not use it to make clinical decisions (diagnosis, treatment) without proper validation, regulatory approval, and integration into clinical workflows.
- **Replacing pathologists or clinicians:** Models are decision-support tools at best; final responsibility lies with qualified healthcare providers.
- **Unvalidated data:** Do not train or evaluate on data you are not allowed to use (e.g. without consent, IRB approval, or dataset terms).

---

## Data and privacy

- **Use only data you are permitted to use:** Public datasets (e.g. PatchCamelyon, ISIC) or data with appropriate consent/approval.
- **Do not commit patient data or identifiers:** Keep `data/` and any patient-related paths out of version control (they are gitignored).
- **De-identification:** If you use real patient data in research, follow your institution’s de-identification and ethics guidelines.

---

## Limitations

- **Generalization:** Models trained on one dataset (e.g. one scanner, one hospital) may not generalize to others.
- **Bias:** Datasets can reflect demographic or acquisition biases; evaluate on diverse, representative data when possible.
- **Interpretability:** Current pipeline does not include Grad-CAM or other explanations by default; add them for transparency.
- **Reproducibility:** Same code and seeds help; different hardware or library versions can still change results slightly.

---

## How to use responsibly

1. **Document your data:** In papers or reports, state dataset source, size, and any limitations.
2. **Report metrics properly:** Always report sensitivity, specificity, and AUC-ROC (and confidence intervals if possible), not only accuracy.
3. **Share responsibly:** If you publish trained weights, ensure the training data may be shared or clearly describe data and limitations.
4. **Cite the project:** Use the citation in the README or [CITATION.cff](CITATION.cff) when building on this code.

---

For questions about ethics or limitations, open an issue on GitHub or contact the maintainers.
