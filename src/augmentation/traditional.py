"""Traditional oversampling methods, unified interface (TECHNICAL_SPEC.md §5).

`augment(X, y, method, ...)` returns `(X_aug, y_aug)` with synthetic samples
appended to the original. Supported methods:

    "none"        — pass-through, returns the inputs unchanged
    "random"      — RandomOverSampler
    "smote"       — SMOTE
    "adasyn"      — ADASYN
    "borderline"  — BorderlineSMOTE
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    RandomOverSampler,
)

METHODS = ("none", "random", "smote", "adasyn", "borderline")


def _safe_k_neighbors(y: np.ndarray, k_requested: int) -> int:
    """SMOTE/ADASYN/Borderline need k_neighbors < min_class_count.

    Some classes (e.g. PRAD FOXA1, 6 train samples) can't support k=5 if the
    method needs at least k+1 neighbors of the same class. Cap k at
    min_class_count - 1 (and at least 1).
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    nonzero = counts[counts > 0]
    min_count = int(nonzero.min()) if len(nonzero) else 1
    return max(1, min(k_requested, min_count - 1))


def augment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    *,
    sampling_strategy: str | dict = "auto",
    k_neighbors: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if method == "none":
        return X_train, y_train

    if method == "random":
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy, random_state=seed
        )
        return sampler.fit_resample(X_train, y_train)

    k = _safe_k_neighbors(y_train, k_neighbors)
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=sampling_strategy, n_neighbors=k, random_state=seed
        )
    elif method == "borderline":
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable")

    return sampler.fit_resample(X_train, y_train)
