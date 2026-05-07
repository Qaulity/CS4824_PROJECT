"""Build the spec's Day-9 grid figures.

  1. results/figures/cm_grid.{pdf,png}   — 3 cancers × N methods of confusion matrices
                                            (one row per cancer)
  2. results/figures/roc_grid.{pdf,png}  — 3 cancers × 1 OvR ROC overlays
                                            (one panel per cancer, one line per method)

Reloads each saved checkpoint at seed=42 and re-evaluates on the test set so we
have the full confusion matrix and per-class probabilities, neither of which is
stored in `all_results.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.mlp import MLP, checkpoint_path_for
from src.config import CANCERS, RESULTS_FIGURES, get_device
from src.data import load_processed

# Methods plotted across cancers; same set per row so the grid is interpretable.
METHODS = ["none", "smote_tuned", "borderline_tuned", "cvae", "wgan_gp", "mixup"]
PRETTY = {
    "none": "no aug", "smote_tuned": "SMOTE (tuned)",
    "borderline_tuned": "Borderline (tuned)", "cvae": "cVAE",
    "wgan_gp": "WGAN-GP", "mixup": "Mixup",
}
SEED = 42


def load_predictions(cancer: str, method: str, data: dict):
    """Return (y_true, y_pred, y_proba) for the saved seed-42 model, or None if missing."""
    path = checkpoint_path_for(cancer, method, SEED)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(input_dim=data["X_train"].shape[1], n_classes=len(data["class_names"]))
    model.load_state_dict(state)
    device = get_device()
    model.to(device).eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(np.asarray(data["X_test"], dtype=np.float32)).to(device)
        )
        proba = F.softmax(logits, dim=1).cpu().numpy()
    return data["y_test"], proba.argmax(axis=1), proba


def cm_grid() -> None:
    fig, axes = plt.subplots(
        len(CANCERS), len(METHODS),
        figsize=(2.4 * len(METHODS), 2.7 * len(CANCERS)),
        squeeze=False,
    )
    for i, cancer in enumerate(CANCERS):
        data = load_processed(cancer)
        K = len(data["class_names"])
        for j, method in enumerate(METHODS):
            ax = axes[i][j]
            res = load_predictions(cancer, method, data)
            if res is None:
                ax.text(0.5, 0.5, "no checkpoint", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(PRETTY[method], fontsize=10)
                if j == 0:
                    ax.set_ylabel(cancer.upper(), fontsize=11)
                continue
            y_true, y_pred, _ = res
            cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(K), data["class_names"], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(K), data["class_names"], fontsize=7)
            for r in range(K):
                for c in range(K):
                    v = cm_norm[r, c]
                    ax.text(
                        c, r, f"{v:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else "black",
                    )
            if i == 0:
                ax.set_title(PRETTY[method], fontsize=10)
            if j == 0:
                ax.set_ylabel(cancer.upper(), fontsize=11)
    fig.suptitle("Confusion matrices (row-normalized) — seed 42, test set", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = RESULTS_FIGURES / "cm_grid"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png}}")


def roc_grid() -> None:
    """Macro-averaged OvR ROC per (cancer, method) — one panel per cancer."""
    fig, axes = plt.subplots(1, len(CANCERS), figsize=(5 * len(CANCERS), 4.5), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, cancer in enumerate(CANCERS):
        ax = axes[0][i]
        data = load_processed(cancer)
        K = len(data["class_names"])
        for j, method in enumerate(METHODS):
            res = load_predictions(cancer, method, data)
            if res is None:
                continue
            y_true, _, y_proba = res
            y_true_bin = label_binarize(y_true, classes=list(range(K)))
            if K == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            # Average across classes (macro)
            fprs, tprs, aucs = [], [], []
            for c in range(K):
                if y_true_bin[:, c].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_proba[:, c])
                roc_auc = auc(fpr, tpr)
                fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc)
            # Macro: average TPR over a common FPR grid
            grid = np.linspace(0, 1, 200)
            tpr_interp = np.zeros_like(grid)
            for fpr, tpr in zip(fprs, tprs):
                tpr_interp += np.interp(grid, fpr, tpr)
            tpr_interp /= max(1, len(fprs))
            macro_auc = float(np.mean(aucs)) if aucs else float("nan")
            ax.plot(grid, tpr_interp, color=cmap(j), linewidth=1.6,
                    label=f"{PRETTY[method]} (AUC={macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{cancer.upper()} — macro-averaged OvR ROC (seed 42)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    stem = RESULTS_FIGURES / "roc_grid"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png}}")


def main() -> None:
    cm_grid()
    roc_grid()


if __name__ == "__main__":
    main()
