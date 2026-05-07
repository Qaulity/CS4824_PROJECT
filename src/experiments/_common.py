"""Shared helpers for the experiment scripts.

The master results CSV at `results/tables/all_results.csv` follows the schema in
TECHNICAL_SPEC.md §10:

    cancer, method, n_synth, seed,
    accuracy, macro_f1, weighted_f1, roc_auc_ovr,
    class_0_f1, class_1_f1, ..., class_K_f1,
    mmd, correlation_score,
    training_time_seconds, timestamp

`mmd` and `correlation_score` are NaN for non-generative methods.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from src.config import RESULTS_TABLES

MASTER_CSV = RESULTS_TABLES / "all_results.csv"


def metrics_to_row(
    *,
    cancer: str,
    method: str,
    n_synth: int,
    seed: int,
    metrics: dict,
    class_names: list[str],
    training_time_seconds: float,
    mmd: float | None = None,
    correlation_score: float | None = None,
) -> dict:
    """Flatten an `evaluate_mlp` metrics dict into the master-CSV row schema."""
    row = {
        "cancer": cancer,
        "method": method,
        "n_synth": n_synth,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
    }
    # Per-spec column scheme is positional: class_0_f1, class_1_f1, ...
    # Class identity is recoverable from data/processed/{cancer}_meta.json.
    for i, f1 in enumerate(metrics["per_class_f1"]):
        row[f"class_{i}_f1"] = f1
    row["mmd"] = math.nan if mmd is None else mmd
    row["correlation_score"] = (
        math.nan if correlation_score is None else correlation_score
    )
    row["training_time_seconds"] = training_time_seconds
    row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
    return row


def append_rows(rows: list[dict], path: Path = MASTER_CSV) -> None:
    """Append rows to the master CSV, creating it (with header) on first write."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
