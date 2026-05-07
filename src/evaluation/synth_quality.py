"""Synthetic-data quality metrics + t-SNE overlay (TECHNICAL_SPEC.md §9).

Three metrics:
  - mmd_rbf(X, Y)            : Maximum Mean Discrepancy with RBF kernel (lower = closer)
  - correlation_score(X, Y)  : Pearson correlation between flattened upper triangles
                                of feature-feature corr matrices (Lacan eq. 6, higher=better)
  - plot_tsne_overlay(...)   : t-SNE on combined real+synthetic, colored by class,
                                marker shape distinguishing real vs synthetic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Lower = closer distributions. Uses sklearn's `rbf_kernel`. If `gamma` is None,
    fall back to sklearn's default heuristic 1 / (n_features * variance).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * X.var())
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def correlation_score(X_true: np.ndarray, X_gen: np.ndarray) -> float:
    """Pearson correlation between flattened upper triangles of the gene-gene
    correlation matrices of real vs generated data (Lacan eq. 6).

    Higher (closer to 1) = better preservation of feature correlation structure.
    """
    M_t = np.corrcoef(np.asarray(X_true).T)
    M_g = np.corrcoef(np.asarray(X_gen).T)
    iu = np.triu_indices_from(M_t, k=1)
    real = M_t[iu]
    gen = M_g[iu]
    # Mask any NaN entries (e.g. constant gene columns)
    mask = np.isfinite(real) & np.isfinite(gen)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(real[mask], gen[mask])[0, 1])


def plot_tsne_overlay(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    class_names: list[str],
    save_path: Path,
    *,
    title: str = "",
    perplexity: int | None = None,
    seed: int = 42,
) -> None:
    """t-SNE on the concatenation of real+synthetic points; markers distinguish
    real (filled circle) from synthetic (open triangle), colors mark class.

    Saves both PDF and PNG at 300 DPI. `save_path` is the file stem.
    """
    X = np.vstack([X_real, X_synth])
    y = np.concatenate([y_real, y_synth])
    is_synth = np.concatenate([
        np.zeros(len(X_real), dtype=bool),
        np.ones(len(X_synth), dtype=bool),
    ])

    if perplexity is None:
        perplexity = max(5, min(30, (len(X) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(class_names):
        mask_real = (y == i) & ~is_synth
        mask_synth = (y == i) & is_synth
        ax.scatter(
            Z[mask_real, 0], Z[mask_real, 1], s=18, alpha=0.7, color=cmap(i),
            marker="o", label=f"{cls} (real)",
        )
        ax.scatter(
            Z[mask_synth, 0], Z[mask_synth, 1], s=20, alpha=0.55, color=cmap(i),
            marker="^", facecolors="none", linewidth=1.0, label=f"{cls} (synth)",
        )
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title(title or "t-SNE overlay (real vs synthetic)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
