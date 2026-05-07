"""EDA artifacts for Day 2: class-distribution bar charts + t-SNE of train data.

Reads data/processed/{cancer}.npz + {cancer}_meta.json and writes
results/figures/eda/{cancer}_class_distribution.{pdf,png} and
results/figures/eda/{cancer}_tsne.{pdf,png} at 300 DPI.

Run: `python scripts/eda.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CANCERS,
    DATA_PROCESSED,
    RESULTS_FIGURES,
    SEED,
    set_seed,
)

OUT = RESULTS_FIGURES / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, stem: Path) -> None:
    """Save the same figure as both PDF (report) and PNG (slides) at 300 DPI."""
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def class_distribution(cancer: str, meta: dict) -> None:
    counts = meta["train_class_counts"]
    val_counts = meta["val_class_counts"]
    test_counts = meta["test_class_counts"]
    classes = meta["class_names"]

    train_vals = [counts[c] for c in classes]
    val_vals = [val_counts[c] for c in classes]
    test_vals = [test_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    x = np.arange(len(classes))
    ax.bar(x - width, train_vals, width, label="train")
    ax.bar(x, val_vals, width, label="val")
    ax.bar(x + width, test_vals, width, label="test")
    ax.set_xticks(x, classes)
    ax.set_ylabel("samples")
    ax.set_title(f"{cancer.upper()} class distribution per split")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, OUT / f"{cancer}_class_distribution")
    plt.close(fig)


def tsne_plot(cancer: str, npz, meta: dict) -> None:
    X = npz["X_train"]
    y = npz["y_train"]
    classes = meta["class_names"]

    perplexity = max(5, min(30, (len(X) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        m = y == i
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, color=cmap(i), label=cls)
    ax.set_title(
        f"{cancer.upper()} t-SNE of z-scored top-{X.shape[1]} genes (train, "
        f"perplexity={perplexity})"
    )
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(loc="best", fontsize=9)
    _save(fig, OUT / f"{cancer}_tsne")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    for cancer in CANCERS:
        meta = json.loads((DATA_PROCESSED / f"{cancer}_meta.json").read_text())
        npz = np.load(DATA_PROCESSED / f"{cancer}.npz")
        print(f"{cancer.upper()}: building class distribution + t-SNE…")
        class_distribution(cancer, meta)
        tsne_plot(cancer, npz, meta)
    print(f"\nWrote figures under {OUT}")


if __name__ == "__main__":
    main()
