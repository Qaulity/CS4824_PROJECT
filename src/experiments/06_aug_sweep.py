"""Augmentation sweep on BRCA — headline figure for the report.

For each method ∈ {smote, cvae, wgan_gp, mixup} × N_synth ∈ {100, 500, 1000, 2000, 3000}
× seed ∈ {42, 43, 44}, build augmented train (real + N_synth synth), train MLP, eval test.
Append to master CSV under method "{base}_sweep" with `n_synth` set to the count.

Per spec the headline plot is `accuracy + macro_f1 vs N_synth, lines per method, error bars from seeds`.

Run: `python -m src.experiments.06_aug_sweep`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.augmentation.cvae import cvae_checkpoint_path, load_cvae
from src.augmentation.mixup import mixup_augment
from src.augmentation.traditional import augment as traditional_augment
from src.augmentation.wgan_gp import (
    generate as wgan_generate,
    load_generator,
    wgan_checkpoint_path,
)
from src.classifier.mlp import MLP, evaluate_mlp, train_mlp
from src.config import get_device, set_seed
from src.data import load_processed
from src.experiments._common import append_rows, metrics_to_row

CANCERS = ("brca",)  # spec says BRCA only; extend if time permits
METHODS = ("smote", "cvae", "wgan_gp", "mixup")
N_SYNTH_VALUES = (100, 500, 1000, 2000, 3000)
SEEDS = (42, 43, 44, 45, 46)


def _generate_with_proportional_class_counts(
    n_total: int, train_class_counts: dict[str, int], class_names: list[str]
) -> dict[int, int]:
    """Spread n_total synthetic samples across classes proportionally to train counts."""
    counts = np.array([train_class_counts[c] for c in class_names], dtype=np.float64)
    proportions = counts / counts.sum()
    n_per_class = np.floor(proportions * n_total).astype(int)
    # distribute remainder to the largest class to hit n_total exactly
    remainder = n_total - n_per_class.sum()
    if remainder > 0:
        n_per_class[counts.argmax()] += remainder
    return {i: int(n_per_class[i]) for i in range(len(class_names))}


def make_synth(method: str, cancer: str, data: dict, n_total: int, seed: int):
    """Return (X_synth, y_synth) of length ~n_total."""
    if method == "smote":
        # SMOTE doesn't have a "produce N synth" knob; we rebalance to a target
        # majority size such that ~n_total synth points are added.
        # Easiest: oversample to majority * factor and truncate.
        max_class = max(data["train_class_counts"].values())
        # Target counts: majority * factor where total synth ≈ n_total.
        # Simpler: use sampling_strategy as a callable returning per-class targets.
        K = len(data["class_names"])
        n_per_class = _generate_with_proportional_class_counts(
            n_total, data["train_class_counts"], data["class_names"]
        )
        target_counts = {
            i: int(data["train_class_counts"][data["class_names"][i]]) + n_per_class[i]
            for i in range(K)
        }
        # imblearn supports dict targeting; per-class n must be >= current
        X_aug, y_aug = traditional_augment(
            data["X_train"], data["y_train"],
            method="smote",
            sampling_strategy=target_counts,
            seed=seed,
        )
        n_orig = len(data["X_train"])
        return X_aug[n_orig:], y_aug[n_orig:]

    if method == "mixup":
        return mixup_augment(data["X_train"], data["y_train"], n_total, seed=seed)

    if method == "cvae":
        cvae = load_cvae(cvae_checkpoint_path(cancer)).to(get_device()).eval()
        n_per_class = _generate_with_proportional_class_counts(
            n_total, data["train_class_counts"], data["class_names"]
        )
        return cvae.generate(n_per_class)

    if method == "wgan_gp":
        G = load_generator(wgan_checkpoint_path(cancer)).to(get_device()).eval()
        n_per_class = _generate_with_proportional_class_counts(
            n_total, data["train_class_counts"], data["class_names"]
        )
        return wgan_generate(G, n_per_class)

    raise ValueError(f"unknown sweep method {method}")


def run_one(cancer: str, method: str, n_total: int, seed: int) -> dict | None:
    set_seed(seed)
    data = load_processed(cancer)
    try:
        X_synth, y_synth = make_synth(method, cancer, data, n_total, seed)
    except Exception as e:
        print(f"  !! {method} n={n_total} failed: {e}")
        return None

    X_aug = np.concatenate([data["X_train"], X_synth]).astype(np.float32)
    y_aug = np.concatenate([data["y_train"], y_synth]).astype(np.int64)

    model = MLP(input_dim=X_aug.shape[1], n_classes=len(data["class_names"]))
    history = train_mlp(
        model, X_aug, y_aug, data["X_val"], data["y_val"],
    )
    metrics = evaluate_mlp(model, data["X_test"], data["y_test"], data["class_names"])

    print(
        f"    n={n_total:5d} seed={seed}  acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  added={len(X_synth)}"
    )

    return metrics_to_row(
        cancer=cancer,
        method=f"{method}_sweep",
        n_synth=len(X_synth),
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=history["train_time_seconds"],
    )


def main() -> None:
    rows: list[dict] = []
    for cancer in CANCERS:
        for method in METHODS:
            print(f"\n=== {cancer.upper()} | {method} ===")
            for n_total in N_SYNTH_VALUES:
                for seed in SEEDS:
                    r = run_one(cancer, method, n_total, seed)
                    if r is not None:
                        rows.append(r)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
