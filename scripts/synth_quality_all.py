"""Compute MMD + correlation score for every generative method on every cancer.

Generates synthetic samples (1:1 per class, seed 42) for SMOTE, ADASYN,
BorderlineSMOTE, Random, Mixup, cVAE, WGAN-GP. Computes mmd_rbf and
correlation_score against the real train data.

Output: results/tables/synth_quality.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentation.cvae import cvae_checkpoint_path, load_cvae
from src.augmentation.mixup import mixup_augment
from src.augmentation.traditional import augment as traditional_augment
from src.augmentation.wgan_gp import (
    generate as wgan_generate,
    load_generator,
    wgan_checkpoint_path,
)
from src.config import CANCERS, RESULTS_TABLES, get_device, set_seed
from src.data import load_processed
from src.evaluation.synth_quality import correlation_score, mmd_rbf

OUT = RESULTS_TABLES / "synth_quality.csv"
SEED = 42


def synth_for_method(method: str, cancer: str, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_synth, y_synth) at 1:1 per class. Returns empty arrays on failure."""
    set_seed(SEED)
    n_per_class = {
        i: int(data["train_class_counts"][n])
        for i, n in enumerate(data["class_names"])
    }

    if method in {"random", "smote", "adasyn", "borderline"}:
        try:
            kwargs = {"sampling_strategy": "minority"} if method != "random" else {"sampling_strategy": "auto"}
            X_aug, y_aug = traditional_augment(
                data["X_train"], data["y_train"], method=method, seed=SEED, **kwargs
            )
            n_orig = len(data["X_train"])
            return X_aug[n_orig:], y_aug[n_orig:]
        except Exception as e:
            print(f"  {method} failed: {e}")
            return np.zeros((0, data["X_train"].shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if method == "mixup":
        return mixup_augment(
            data["X_train"], data["y_train"], len(data["X_train"]), seed=SEED
        )

    if method == "cvae":
        try:
            cvae = load_cvae(cvae_checkpoint_path(cancer)).to(get_device()).eval()
            return cvae.generate(n_per_class)
        except Exception as e:
            print(f"  cvae failed: {e}")
            return np.zeros((0, data["X_train"].shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if method == "wgan_gp":
        try:
            G = load_generator(wgan_checkpoint_path(cancer)).to(get_device()).eval()
            return wgan_generate(G, n_per_class)
        except Exception as e:
            print(f"  wgan_gp failed: {e}")
            return np.zeros((0, data["X_train"].shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    raise ValueError(f"unknown method {method}")


def main() -> None:
    methods = ["random", "smote", "adasyn", "borderline", "mixup", "cvae", "wgan_gp"]
    rows = []
    for cancer in CANCERS:
        print(f"\n=== {cancer.upper()} ===")
        data = load_processed(cancer)
        for method in methods:
            X_synth, y_synth = synth_for_method(method, cancer, data)
            if len(X_synth) < 5:
                print(f"  {method:>10}: skipped (synth too small: {len(X_synth)})")
                rows.append({
                    "cancer": cancer, "method": method,
                    "n_synth": int(len(X_synth)),
                    "mmd": float("nan"), "correlation_score": float("nan"),
                })
                continue
            mmd = mmd_rbf(data["X_train"], X_synth)
            corr = correlation_score(data["X_train"], X_synth)
            print(f"  {method:>10}: n={len(X_synth):4d}  mmd={mmd:.4f}  corr={corr:.4f}")
            rows.append({
                "cancer": cancer, "method": method,
                "n_synth": int(len(X_synth)),
                "mmd": round(float(mmd), 4),
                "correlation_score": round(float(corr), 4),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
