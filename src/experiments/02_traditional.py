"""Traditional augmentation (random / SMOTE / ADASYN / Borderline-SMOTE) × MLP.

Default hyperparameters per TECH §5: k_neighbors=5, sampling_strategy="auto"
(balance every class up to majority size). Day-4 will sweep these.

Run: `python -m src.experiments.02_traditional`
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.augmentation.traditional import augment
from src.classifier.mlp import MLP, checkpoint_path_for, evaluate_mlp, train_mlp
from src.config import CANCERS, RESULTS_FIGURES, set_seed
from src.data import load_processed
from src.evaluation.viz import plot_confusion_matrix
from src.experiments._common import append_rows, metrics_to_row

SEEDS = (42, 43, 44, 45, 46)
METHODS = ("random", "smote", "adasyn", "borderline")
FIG_DIR = RESULTS_FIGURES / "traditional"


def run_one(cancer: str, method: str, seed: int) -> dict | None:
    set_seed(seed)
    data = load_processed(cancer)

    try:
        X_aug, y_aug = augment(
            data["X_train"], data["y_train"], method=method, seed=seed
        )
    except Exception as e:
        print(f"  !! {method} failed for {cancer} seed {seed}: {e}")
        traceback.print_exc()
        return None

    n_added = len(X_aug) - len(data["X_train"])

    model = MLP(input_dim=X_aug.shape[1], n_classes=len(data["class_names"]))
    history = train_mlp(
        model,
        X_aug, y_aug,
        data["X_val"], data["y_val"],
        checkpoint_path=checkpoint_path_for(cancer, method, seed),
    )
    metrics = evaluate_mlp(model, data["X_test"], data["y_test"], data["class_names"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        data["class_names"],
        title=f"{cancer.upper()} {method} (seed {seed})",
        save_path=FIG_DIR / f"{cancer}_{method}_seed{seed}",
    )

    print(
        f"  +{n_added:>4d} synth | "
        f"acc={metrics['accuracy']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"roc_auc_ovr={metrics['roc_auc_ovr']:.3f}  "
        f"epochs={history['best_epoch']}  "
        f"train_time={history['train_time_seconds']:.1f}s"
    )

    return metrics_to_row(
        cancer=cancer,
        method=method,
        n_synth=n_added,
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
            for seed in SEEDS:
                print(f" seed {seed}:", end=" ")
                row = run_one(cancer, method, seed)
                if row is not None:
                    rows.append(row)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/tables/all_results.csv")


if __name__ == "__main__":
    main()
