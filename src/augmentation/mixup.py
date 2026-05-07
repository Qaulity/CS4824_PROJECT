"""Mixup augmentation in input space (TECHNICAL_SPEC.md §8).

For each synthetic point: sample two random training points and λ ~ Beta(α, α),
emit `λ·x_i + (1−λ)·x_j` with hard label = argmax(λ, 1−λ) (per-spec default).

α = 0.2 (Zhang et al. 2018).
"""
from __future__ import annotations

import numpy as np


def mixup_augment(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    *,
    alpha: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_synth, y_synth) of length n_synthetic with hard labels."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = len(X)
    if n < 2 or n_synthetic <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    idx_i = rng.integers(0, n, size=n_synthetic)
    idx_j = rng.integers(0, n, size=n_synthetic)
    # avoid pairing with self
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx_i == idx_j

    lam = rng.beta(alpha, alpha, size=n_synthetic).astype(np.float32)
    X_synth = (lam[:, None] * X[idx_i] + (1.0 - lam)[:, None] * X[idx_j]).astype(np.float32)
    y_synth = np.where(lam >= 0.5, y[idx_i], y[idx_j]).astype(np.int64)
    return X_synth, y_synth
