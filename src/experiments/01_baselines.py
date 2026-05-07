"""Baseline (no augmentation) MLP results for all 3 cancers, 3 seeds each.

Per PROJECT_PLAN.md Day 3: train no-aug MLP on every cancer, log to the master
results CSV, save confusion matrices.

Run: `python -m src.experiments.01_baselines`
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run with `python src/experiments/01_baselines.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classifier.mlp import MLP, checkpoint_path_for, evaluate_mlp, train_mlp
from src.config import (
    CANCERS,
    RESULTS_FIGURES,
    SEED,
    set_seed,
)
from src.data import load_processed
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)
METHOD = "none"
FIG_DIR = RESULTS_FIGURES / "baselines"


def run_one(cancer: str, seed: int) -> dict:
    set_seed(seed)
    data = load_processed(cancer)

    model = MLP(input_dim=data["X_train"].shape[1], n_classes=len(data["class_names"]))
    history = train_mlp(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        checkpoint_path=checkpoint_path_for(cancer, METHOD, seed),
    )
    metrics = evaluate_mlp(model, data["X_test"], data["y_test"], data["class_names"])

    # Confusion matrix: only need one figure per (cancer, seed=42) for the
    # baseline section of the report. Save all three for completeness.
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data["class_names"],
        title=f"{cancer.upper()} baseline (no aug, seed {seed})",
        save_path=FIG_DIR / f"{cancer}_baseline_seed{seed}",
    )

    print(
        f"  acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"weighted_f1={metrics['weighted_f1']:.3f}  "
        f"roc_auc_ovr={metrics['roc_auc_ovr']:.3f}  "
        f"epochs={history['best_epoch']}  "
        f"train_time={history['train_time_seconds']:.1f}s"
    )

    return metrics_to_row(
        cancer=cancer,
        method=METHOD,
        n_synth=0,
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=history["train_time_seconds"],
    )


def main() -> None:
    rows = []
    for cancer in CANCERS:
        print(f"\n=== {cancer.upper()} (no aug) ===")
        for seed in SEEDS:
            print(f" seed {seed}:")
            rows.append(run_one(cancer, seed))
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
