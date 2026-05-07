"""Plotting helpers for the evaluation pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive: avoids Tcl/Tk thread errors with xgboost n_jobs=-1
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
    *,
    normalize: bool = True,
) -> None:
    """Render a confusion matrix and save as both PDF and PNG (300 DPI).

    `save_path` should be the file stem (no suffix); the function appends .pdf/.png.
    """
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(class_names), 4 + 0.3 * len(class_names)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else cm_norm.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    thresh = (cm_norm.max() + cm_norm.min()) / 2 if normalize else cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_norm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, save_path)
    plt.close(fig)
