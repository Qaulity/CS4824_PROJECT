"""Data utilities."""
from __future__ import annotations

import json

import numpy as np

from src.config import DATA_PROCESSED


def load_processed(cancer: str) -> dict:
    """Load preprocessed splits + metadata for a cancer.

    Returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
    class_names, gene_names, scaler_mean, scaler_scale, train_class_counts.
    """
    npz = np.load(DATA_PROCESSED / f"{cancer}.npz")
    meta = json.loads((DATA_PROCESSED / f"{cancer}_meta.json").read_text())
    return {
        "X_train": npz["X_train"],
        "y_train": npz["y_train"],
        "X_val": npz["X_val"],
        "y_val": npz["y_val"],
        "X_test": npz["X_test"],
        "y_test": npz["y_test"],
        **meta,
    }
