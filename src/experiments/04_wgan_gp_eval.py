"""Generate WGAN-GP samples → retrain MLP on real+synth → evaluate (Day 6 Damien).

Method name in master CSV: "wgan_gp".

Run: `python -m src.experiments.04_wgan_gp_eval`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.augmentation.wgan_gp import generate, load_generator, wgan_checkpoint_path
from src.classifier.mlp import MLP, checkpoint_path_for, evaluate_mlp, train_mlp
from src.config import CANCERS, RESULTS_FIGURES, get_device, set_seed
from src.data import load_processed
from src.evaluation.synth_quality import (
    correlation_score,
    mmd_rbf,
    plot_tsne_overlay,
)
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)
METHOD = "wgan_gp"
FIG_DIR = RESULTS_FIGURES / "wgan_gp"


def run_one(cancer: str, seed: int, *, do_quality: bool, do_tsne: bool) -> dict:
    set_seed(seed)
    data = load_processed(cancer)

    G = load_generator(wgan_checkpoint_path(cancer)).to(get_device()).eval()
    n_per_class = {
        i: int(data["train_class_counts"][name])
        for i, name in enumerate(data["class_names"])
    }
    X_synth, y_synth = generate(G, n_per_class)

    X_aug = np.concatenate([data["X_train"], X_synth]).astype(np.float32)
    y_aug = np.concatenate([data["y_train"], y_synth]).astype(np.int64)
    n_added = len(X_synth)

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

    mmd = corr = None
    if do_quality:
        mmd = mmd_rbf(data["X_train"], X_synth)
        corr = correlation_score(data["X_train"], X_synth)

    if do_tsne:
        plot_tsne_overlay(
            data["X_train"], X_synth, data["y_train"], y_synth, data["class_names"],
            save_path=FIG_DIR / f"{cancer}_tsne_overlay",
            title=f"{cancer.upper()} t-SNE overlay (real train vs WGAN-GP synthetic)",
            seed=seed,
        )

    print(
        f"  +{n_added:>4d} synth | "
        f"acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"roc_auc={metrics['roc_auc_ovr']:.3f}  "
        + (f"mmd={mmd:.4f}  corr={corr:.4f}" if mmd is not None else "")
    )

    return metrics_to_row(
        cancer=cancer,
        method=METHOD,
        n_synth=n_added,
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=history["train_time_seconds"],
        mmd=mmd,
        correlation_score=corr,
    )


def main() -> None:
    rows: list[dict] = []
    for cancer in CANCERS:
        print(f"\n=== {cancer.upper()} | wgan_gp ===")
        for i, seed in enumerate(SEEDS):
            print(f" seed {seed}:", end=" ")
            rows.append(
                run_one(cancer, seed, do_quality=(i == 0), do_tsne=(i == 0))
            )
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
