"""Run the full preprocessing pipeline for all three cancers.

For each cancer:
  1. Load expression + subtype labels (samples x genes, label per sample).
  2. Drop missing labels and classes with < 5 samples.
  3. Stratified 70/15/15 split (seeded).
  4. Select top-500 most-variable genes on TRAIN, apply to val/test.
  5. Z-score normalize using TRAIN scaler.
  6. Encode labels as int 0..K-1 in alphabetical class order.
  7. Save data/processed/{cancer}.npz + {cancer}_meta.json.

Run: `python scripts/prepare_data.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CANCERS,
    DATA_PROCESSED,
    N_TOP_GENES,
    SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    set_seed,
)
from src.data.load_xena import load
from src.data.preprocess import (
    encode_labels,
    filter_samples,
    select_top_variable_genes,
    stratified_split,
    zscore_normalize,
)


def prepare(cancer: str) -> dict:
    print(f"\n=== {cancer.upper()} ===")

    X, y = load(cancer)
    print(f"  loaded: {X.shape[0]} samples × {X.shape[1]} genes; "
          f"{y.nunique()} classes")

    X, y = filter_samples(X, y)
    print(f"  after filter: {X.shape[0]} samples; "
          f"class counts {y.value_counts().to_dict()}")

    splits = stratified_split(
        X, y, train=SPLIT_TRAIN, val=SPLIT_VAL, test=SPLIT_TEST, seed=SEED
    )
    print(f"  split sizes: train {len(splits.X_train)}, "
          f"val {len(splits.X_val)}, test {len(splits.X_test)}")

    gene_names = select_top_variable_genes(splits.X_train, n=N_TOP_GENES)
    Xtr = splits.X_train[gene_names].to_numpy(dtype=np.float32)
    Xva = splits.X_val[gene_names].to_numpy(dtype=np.float32)
    Xte = splits.X_test[gene_names].to_numpy(dtype=np.float32)
    print(f"  selected top {len(gene_names)} variable genes")

    z = zscore_normalize(Xtr, Xva, Xte)
    y_tr, y_va, y_te, class_names = encode_labels(
        splits.y_train, splits.y_val, splits.y_test
    )

    npz_path = DATA_PROCESSED / f"{cancer}.npz"
    meta_path = DATA_PROCESSED / f"{cancer}_meta.json"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    np.savez(
        npz_path,
        X_train=z.X_train.astype(np.float32),
        X_val=z.X_val.astype(np.float32),
        X_test=z.X_test.astype(np.float32),
        y_train=y_tr,
        y_val=y_va,
        y_test=y_te,
    )

    train_class_counts = {
        name: int((y_tr == i).sum()) for i, name in enumerate(class_names)
    }
    val_class_counts = {
        name: int((y_va == i).sum()) for i, name in enumerate(class_names)
    }
    test_class_counts = {
        name: int((y_te == i).sum()) for i, name in enumerate(class_names)
    }
    meta = {
        "cancer": cancer,
        "seed": SEED,
        "n_genes": len(gene_names),
        "gene_names": gene_names,
        "class_names": class_names,
        "scaler_mean": z.scaler.mean_.astype(float).tolist(),
        "scaler_scale": z.scaler.scale_.astype(float).tolist(),
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "split": {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wrote {npz_path.relative_to(npz_path.parent.parent.parent)}")
    print(f"  wrote {meta_path.relative_to(meta_path.parent.parent.parent)}")
    print(f"  train class counts: {train_class_counts}")
    return meta


def main() -> None:
    set_seed(SEED)
    for cancer in CANCERS:
        prepare(cancer)
    print("\nDone.")


if __name__ == "__main__":
    main()
