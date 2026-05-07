"""Hyperparameter sweep for the 4 traditional augmentation methods.

Per PROJECT_PLAN.md Day 4 (Ishaan): sweep `k_neighbors ∈ {3,5,7,10}` × `sampling_strategy ∈ {"auto","minority","not majority"}`,
pick the best config per (cancer × method) on the validation set, then evaluate
that config × 3 seeds on the test set and append to the master CSV under method
names `<base>_tuned`.

Selection: mean val macro_f1 over the 3 seeds (averaging reduces seed noise in
config selection). Configs that error out for every seed are skipped.

The chosen configs are also saved to `results/tables/traditional_tuning.json`.

Run: `python -m src.experiments.04_traditional_tuned`
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.augmentation.traditional import augment
from src.classifier.mlp import MLP, checkpoint_path_for, evaluate_mlp, train_mlp
from src.config import CANCERS, RESULTS_FIGURES, RESULTS_TABLES, set_seed
from src.data import load_processed
from src.evaluation.metrics import supervised_metrics
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)

K_VALUES = (3, 5, 7, 10)
STRATEGIES = ("auto", "minority", "not majority")

METHODS_WITH_K = ("smote", "adasyn", "borderline")
METHOD_RANDOM = "random"

FIG_DIR = RESULTS_FIGURES / "traditional_tuned"
CONFIG_PATH = RESULTS_TABLES / "traditional_tuning.json"


def configs_for(method: str) -> list[dict]:
    if method == METHOD_RANDOM:
        return [{"sampling_strategy": s} for s in STRATEGIES]
    return [
        {"k_neighbors": k, "sampling_strategy": s}
        for k, s in product(K_VALUES, STRATEGIES)
    ]


def evaluate_on_val(
    cancer: str, method: str, config: dict, seed: int
) -> dict | None:
    """Train MLP on augmented train, evaluate on VAL, return metrics dict (or None on failure)."""
    set_seed(seed)
    data = load_processed(cancer)
    try:
        X_aug, y_aug = augment(
            data["X_train"], data["y_train"], method=method, seed=seed, **config
        )
    except Exception:
        return None
    model = MLP(input_dim=X_aug.shape[1], n_classes=len(data["class_names"]))
    train_mlp(
        model,
        X_aug, y_aug,
        data["X_val"], data["y_val"],
    )
    # Re-evaluate on val using the metrics function for consistency
    import torch
    from torch.nn import functional as F

    from src.config import get_device

    device = get_device()
    model.eval().to(device)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(np.asarray(data["X_val"], dtype=np.float32)).to(device)
        )
        proba = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = proba.argmax(axis=1)
    return supervised_metrics(data["y_val"], y_pred, proba, data["class_names"])


def evaluate_on_test_final(
    cancer: str, method_label: str, config: dict, seed: int
) -> dict:
    """Train final model with chosen config and evaluate on TEST, append CSV row."""
    set_seed(seed)
    data = load_processed(cancer)
    base_method = method_label.replace("_tuned", "")
    X_aug, y_aug = augment(
        data["X_train"], data["y_train"], method=base_method, seed=seed, **config
    )
    n_added = len(X_aug) - len(data["X_train"])

    model = MLP(input_dim=X_aug.shape[1], n_classes=len(data["class_names"]))
    history = train_mlp(
        model,
        X_aug, y_aug,
        data["X_val"], data["y_val"],
        checkpoint_path=checkpoint_path_for(cancer, method_label, seed),
    )
    metrics = evaluate_mlp(model, data["X_test"], data["y_test"], data["class_names"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data["class_names"],
        title=f"{cancer.upper()} {method_label} (seed {seed}, {config})",
        save_path=FIG_DIR / f"{cancer}_{method_label}_seed{seed}",
    )

    print(
        f"    seed {seed}: +{n_added:>4d} synth | "
        f"acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"roc_auc={metrics['roc_auc_ovr']:.3f}"
    )

    return metrics_to_row(
        cancer=cancer,
        method=method_label,
        n_synth=n_added,
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=history["train_time_seconds"],
    )


def tune_one(cancer: str, method: str) -> tuple[dict, float] | None:
    """Sweep configs, return (best_config, best_mean_val_macro_f1) or None if all fail."""
    configs = configs_for(method)
    best = None
    print(f"\n=== {cancer.upper()} | {method} (sweeping {len(configs)} configs) ===")
    for cfg in configs:
        seed_scores = []
        for seed in SEEDS:
            m = evaluate_on_val(cancer, method, cfg, seed)
            if m is not None:
                seed_scores.append(m["macro_f1"])
        if not seed_scores:
            continue
        mean_f1 = float(np.mean(seed_scores))
        print(f"  cfg={cfg}  val_macro_f1={mean_f1:.3f}  (n_seeds_ok={len(seed_scores)})")
        if best is None or mean_f1 > best[1]:
            best = (cfg, mean_f1)
    return best


def main() -> None:
    chosen: dict[str, dict[str, dict]] = {c: {} for c in CANCERS}
    rows: list[dict] = []

    for cancer in CANCERS:
        for method in (METHOD_RANDOM,) + METHODS_WITH_K:
            result = tune_one(cancer, method)
            if result is None:
                print(f"  !! all configs failed for {cancer} {method}")
                continue
            best_cfg, best_val = result
            chosen[cancer][method] = {
                "best_config": best_cfg,
                "best_val_macro_f1": best_val,
            }
            print(f"  --> chose {best_cfg} (val macro_f1={best_val:.3f})")

            method_label = f"{method}_tuned"
            print(f"  evaluating {method_label} on test:")
            for seed in SEEDS:
                rows.append(
                    evaluate_on_test_final(cancer, method_label, best_cfg, seed)
                )

    append_rows(rows)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"\nAppended {len(rows)} tuned rows to results/tables/all_results.csv")
    print(f"Wrote chosen configs to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
