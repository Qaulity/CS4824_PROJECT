"""Preprocessing utilities: filter, gene selection, z-score, stratified split.

All four functions follow the contract in TECHNICAL_SPEC.md §3. Gene selection
and z-score are fit on TRAIN ONLY, then applied to val and test — never call
them on the full dataset.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def filter_samples(
    X: pd.DataFrame, y: pd.Series, min_class_size: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with missing labels and any class with fewer than `min_class_size`
    samples (PRAD safeguard per spec)."""
    y = y.dropna()
    counts = y.value_counts()
    keep_classes = counts[counts >= min_class_size].index
    y = y[y.isin(keep_classes)]
    X = X.loc[X.index.intersection(y.index)]
    y = y.loc[X.index]
    return X, y


def select_top_variable_genes(
    X_train: pd.DataFrame, n: int = 500
) -> list[str]:
    """Return the `n` gene names with highest variance across train samples."""
    variances = X_train.var(axis=0, ddof=1)
    return variances.sort_values(ascending=False).head(n).index.tolist()


class ZScored(NamedTuple):
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    scaler: StandardScaler


def zscore_normalize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> ZScored:
    """Fit StandardScaler on train, transform train/val/test, return all + scaler."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return ZScored(X_train_s, X_val_s, X_test_s, scaler)


class Splits(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Splits:
    """Two-step stratified split: train vs (val+test), then val vs test."""
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0; got {train+val+test}")

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=val + test, stratify=y, random_state=seed
    )
    rel_test = test / (val + test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=rel_test, stratify=y_rest, random_state=seed
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def encode_labels(
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Map string labels to integer indices in 0..K-1 by alphabetical class order."""
    class_names = sorted(set(y_train) | set(y_val) | set(y_test))
    mapping = {name: i for i, name in enumerate(class_names)}
    return (
        y_train.map(mapping).to_numpy(dtype=np.int64),
        y_val.map(mapping).to_numpy(dtype=np.int64),
        y_test.map(mapping).to_numpy(dtype=np.int64),
        class_names,
    )
