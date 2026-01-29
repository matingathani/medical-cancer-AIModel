"""Medical metrics: sensitivity, specificity, AUC-ROC, confusion matrix."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1):
    """Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=int)
        for i in range(2):
            for j in range(2):
                cm[i, j] = np.sum((y_true == i) & (y_pred == j))
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None):
    """Accuracy, sensitivity, specificity, AUC-ROC (if y_score provided)."""
    acc = accuracy_score(y_true, y_pred)
    sens, spec = sensitivity_specificity(y_true, y_pred)
    out = {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }
    if y_score is not None and len(np.unique(y_true)) >= 2:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc_roc"] = None
    return out


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None):
    """Confusion matrix and optional class names for plotting."""
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    return cm, class_names
