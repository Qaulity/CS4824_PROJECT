"""Generate cVAE samples → retrain MLP on real+synth → evaluate (Day 5 Damien).

For each cancer:
  1. Load trained cVAE checkpoint.
  2. Generate 1:1-per-class synthetic samples (matching `train_class_counts`).
  3. Concatenate real train + synthetic, train fresh MLP, eval test, append rows.
  4. Compute synth-quality metrics (MMD, correlation_score) on a per-class basis,
     stored in the row.
  5. t-SNE overlay (real train vs cVAE synth) for visual sanity check.

Method name in master CSV: "cvae".

Run: `python -m src.experiments.03_cvae_eval`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.augmentation.cvae import cvae_checkpoint_path, load_cvae
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
METHOD = "cvae"
FIG_DIR = RESULTS_FIGURES / "cvae"


def generate_synth(model, train_class_counts: dict[str, int], class_names: list[str], seed: int):
    """1:1 per class — n_synth[c] == train count for class c."""
    set_seed(seed)
    n_per_class = {i: int(train_class_counts[name]) for i, name in enumerate(class_names)}
    return model.generate(n_per_class)


def run_one(cancer: str, seed: int, *, do_tsne: bool, do_quality: bool) -> dict:
    set_seed(seed)
    data = load_processed(cancer)

    cvae = load_cvae(cvae_checkpoint_path(cancer)).to(get_device()).eval()
    X_synth, y_synth = generate_synth(cvae, data["train_class_counts"], data["class_names"], seed)

    X_aug = np.concatenate([data["X_train"], X_synth], axis=0).astype(np.float32)
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

    # Synth-quality metrics: only need to compute once per cancer (deterministic
    # given the trained cVAE + same generation seed). Cheapest path is to
    # compute on seed 42 only.
    mmd = corr = None
    if do_quality:
        mmd = mmd_rbf(data["X_train"], X_synth)
        corr = correlation_score(data["X_train"], X_synth)

    if do_tsne:
        plot_tsne_overlay(
            data["X_train"], X_synth, data["y_train"], y_synth, data["class_names"],
            save_path=FIG_DIR / f"{cancer}_tsne_overlay",
            title=f"{cancer.upper()} t-SNE overlay (real train vs cVAE synthetic)",
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
        print(f"\n=== {cancer.upper()} | cvae ===")
        for i, seed in enumerate(SEEDS):
            print(f" seed {seed}:", end=" ")
            rows.append(
                run_one(
                    cancer,
                    seed,
                    do_tsne=(i == 0),       # one t-SNE per cancer
                    do_quality=(i == 0),    # one quality eval per cancer
                )
            )
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
