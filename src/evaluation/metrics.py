"""Supervised classification metrics (TECHNICAL_SPEC.md §9)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute the spec metric set.

    Returns:
        accuracy, macro_f1, weighted_f1,
        per_class_precision, per_class_recall, per_class_f1 (lists in class order),
        confusion_matrix (np.ndarray, shape KxK),
        roc_auc_ovr (one-vs-rest, macro-averaged),
        per_class_auc (list, NaN for classes absent from y_true),
        predictions, probabilities.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    K = len(class_names)
    labels = list(range(K))

    out: dict = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    out["per_class_precision"] = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ).tolist()
    out["per_class_recall"] = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ).tolist()
    out["per_class_f1"] = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ).tolist()
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)

    # ROC AUC OvR. Skip classes that do not appear in y_true (sklearn would
    # raise on a degenerate single-class column).
    y_true_bin = label_binarize(y_true, classes=labels)
    if K == 2:
        # label_binarize returns shape (N, 1) for binary — expand.
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    present = [c for c in labels if y_true_bin[:, c].sum() > 0]
    if len(present) >= 2:
        out["roc_auc_ovr"] = float(
            roc_auc_score(
                y_true_bin[:, present],
                y_proba[:, present],
                average="macro",
                multi_class="ovr",
            )
        )
        per_class_auc = [float("nan")] * K
        for c in present:
            per_class_auc[c] = float(
                roc_auc_score(y_true_bin[:, c], y_proba[:, c])
            )
        out["per_class_auc"] = per_class_auc
    else:
        out["roc_auc_ovr"] = float("nan")
        out["per_class_auc"] = [float("nan")] * K

    out["predictions"] = y_pred
    out["probabilities"] = y_proba
    return out
