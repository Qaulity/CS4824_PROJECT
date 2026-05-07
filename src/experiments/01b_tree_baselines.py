"""Random Forest + XGBoost baselines, no-aug and SMOTE-augmented.

Per PROJECT_PLAN.md Day 3 (Ishaan): "Add RF + XGBoost baselines (no aug + with
SMOTE) — proposal mentions these". Method names in the master CSV:
    rf_none, rf_smote, xgb_none, xgb_smote

Run: `python -m src.experiments.01b_tree_baselines`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.augmentation.traditional import augment
from src.classifier.tree_models import fit_eval_rf, fit_eval_xgb
from src.config import CANCERS, RESULTS_FIGURES, set_seed
from src.data import load_processed
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)
FIG_DIR = RESULTS_FIGURES / "tree_baselines"


def run_one(cancer: str, model_name: str, use_smote: bool, seed: int) -> dict:
    set_seed(seed)
    data = load_processed(cancer)
    if use_smote:
        X_train, y_train = augment(
            data["X_train"], data["y_train"], method="smote", seed=seed
        )
    else:
        X_train, y_train = data["X_train"], data["y_train"]
    n_synth = len(X_train) - len(data["X_train"])

    fit_fn = fit_eval_rf if model_name == "rf" else fit_eval_xgb
    metrics, train_time = fit_fn(
        X_train, y_train, data["X_test"], data["y_test"], data["class_names"], seed=seed
    )

    method = f"{model_name}_{'smote' if use_smote else 'none'}"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data["class_names"],
        title=f"{cancer.upper()} {method} (seed {seed})",
        save_path=FIG_DIR / f"{cancer}_{method}_seed{seed}",
    )

    print(
        f"  acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"roc_auc_ovr={metrics['roc_auc_ovr']:.3f}  "
        f"train_time={train_time:.2f}s"
    )

    return metrics_to_row(
        cancer=cancer,
        method=method,
        n_synth=n_synth,
        seed=seed,
        metrics=metrics,
        class_names=data["class_names"],
        training_time_seconds=train_time,
    )


def main() -> None:
    rows: list[dict] = []
    for cancer in CANCERS:
        for model_name in ("rf", "xgb"):
            for use_smote in (False, True):
                method = f"{model_name}_{'smote' if use_smote else 'none'}"
                print(f"\n=== {cancer.upper()} | {method} ===")
                for seed in SEEDS:
                    print(f" seed {seed}:", end=" ")
                    rows.append(run_one(cancer, model_name, use_smote, seed))
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
