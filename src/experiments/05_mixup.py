"""Mixup experiment — generate n_synth = original train size, retrain MLP.

Per Day 7 (Damien): "Run Mixup on all 3 cancers, log results". 1:1 ratio matches
cVAE / WGAN-GP for fair comparison.

Method name in master CSV: "mixup".

Run: `python -m src.experiments.05_mixup`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.augmentation.mixup import mixup_augment
from src.classifier.mlp import MLP, checkpoint_path_for, evaluate_mlp, train_mlp
from src.config import CANCERS, RESULTS_FIGURES, set_seed
from src.data import load_processed
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)
METHOD = "mixup"
FIG_DIR = RESULTS_FIGURES / "mixup"


def run_one(cancer: str, seed: int) -> dict:
    set_seed(seed)
    data = load_processed(cancer)

    n_synth = len(data["X_train"])  # 1:1 ratio
    X_synth, y_synth = mixup_augment(
        data["X_train"], data["y_train"], n_synth, seed=seed
    )

    X_aug = np.concatenate([data["X_train"], X_synth]).astype(np.float32)
    y_aug = np.concatenate([data["y_train"], y_synth]).astype(np.int64)

    model = MLP(input_dim=X_aug.shape[1], n_classes=len(data["class_names"]))
    history = train_mlp(
        model,
        X_aug, y_aug,
        data["X_val"], data["y_val"],
        checkpoint_path=checkpoint_path_for(cancer, METHOD, seed),
    )
    metrics = evaluate_mlp(model, data["X_test"], data["y_test"], data["class_names"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data["class_names"],
        title=f"{cancer.upper()} {METHOD} (seed {seed})",
        save_path=FIG_DIR / f"{cancer}_{METHOD}_seed{seed}",
    )

    print(
        f"  +{n_synth:>4d} synth | "
        f"acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"roc_auc={metrics['roc_auc_ovr']:.3f}  "
        f"epochs={history['best_epoch']}  "
        f"train_time={history['train_time_seconds']:.1f}s"
    )

    return metrics_to_row(
        cancer=cancer,
        method=METHOD,
        n_synth=n_synth,
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=history["train_time_seconds"],
    )


def main() -> None:
    rows = []
    for cancer in CANCERS:
        print(f"\n=== {cancer.upper()} | mixup ===")
        for seed in SEEDS:
            print(f" seed {seed}:", end=" ")
            rows.append(run_one(cancer, seed))
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
