"""RandomForest + XGBoost wrappers with the same evaluate interface as the MLP.

Per TECHNICAL_SPEC.md §4: `RandomForestClassifier(n_estimators=500, random_state=SEED)`
and `XGBClassifier(n_estimators=500, max_depth=6, random_state=SEED)`. Used for the
proposal's "we tested multiple classifiers" requirement on no-aug + SMOTE conditions.
"""
from __future__ import annotations

import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import supervised_metrics


def _fit_and_eval(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> tuple[dict, float]:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return supervised_metrics(y_test, y_pred, y_proba, class_names), train_time


def fit_eval_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    *,
    seed: int = 42,
) -> tuple[dict, float]:
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    return _fit_and_eval(rf, X_train, y_train, X_test, y_test, class_names)


def fit_eval_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    *,
    seed: int = 42,
) -> tuple[dict, float]:
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    return _fit_and_eval(clf, X_train, y_train, X_test, y_test, class_names)
