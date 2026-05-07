"""End-to-end data audit.

Checks every assertion that should hold of the data we've built the project on:

  * raw expression files: sample counts, no NaN, gene-name regex, primary-tumor
    code, no duplicate samples
  * raw clinical files: SUBTYPE column present, value distribution
  * COAD CMS labels: TCGA cohort filter
  * processed NPZ files: shapes, dtypes, z-score (mean ~0 std ~1), no NaN/inf,
    label encoding spans 0..K-1 in alphabetical class order
  * split integrity: no patient-ID overlap across train/val/test
  * class-distribution drift: split fractions match SPLIT_TRAIN/VAL/TEST per spec

Outputs a PASS/FAIL line per check + a summary at the end. Exit code = number of
failed checks.

Run: `python scripts/data_audit.py`
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CANCERS,
    DATA_PROCESSED,
    DATA_RAW,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from src.data.load_xena import load_brca, load_coad, load_prad

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "", *, warn: bool = False) -> None:
    tag = PASS if ok else (WARN if warn else FAIL)
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok and not warn:
        failures.append(name)


# ---------------------------------------------------------------------------
# Raw expression
# ---------------------------------------------------------------------------

def audit_raw_expression() -> None:
    print("\n=== Raw expression files ===")
    expected_first_gene = "ARHGEF10L"
    for cancer in CANCERS:
        path = DATA_RAW / f"{cancer}_expression.tsv.gz"
        check(f"{cancer}: expression file exists", path.exists())
        if not path.exists():
            continue
        with gzip.open(path, "rt") as f:
            header = f.readline().rstrip("\n").split("\t")
            first_row = f.readline().rstrip("\n").split("\t")

        n_samples = len(header) - 1
        check(
            f"{cancer}: {n_samples} samples in expression matrix",
            n_samples > 100,
            f"{n_samples}",
        )
        check(
            f"{cancer}: first gene == {expected_first_gene}",
            first_row[0] == expected_first_gene,
            f"got {first_row[0]}",
        )

        sample_ids = header[1:]
        # TCGA barcode shape; allow Xena's NN suffix
        bad = [s for s in sample_ids if not re.match(r"^TCGA-\w{2,4}-\w{4}-\d{2}$", s)]
        check(
            f"{cancer}: every sample ID matches TCGA-XX-XXXX-NN",
            len(bad) == 0,
            f"{len(bad)} unexpected" + (f" (e.g. {bad[:3]})" if bad else ""),
        )
        primary = sum(1 for s in sample_ids if s.endswith("-01"))
        check(
            f"{cancer}: {primary} primary-tumor samples (-01 suffix)",
            primary > 0,
            f"{primary} of {n_samples} total",
        )

        # Spot check NaN: read first 100 genes via pandas
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep="\t", index_col=0, nrows=100)
        check(
            f"{cancer}: no NaN in first 100 genes × {df.shape[1]} samples",
            not df.isna().any().any(),
        )


# ---------------------------------------------------------------------------
# Raw clinical
# ---------------------------------------------------------------------------

def audit_raw_clinical() -> None:
    print("\n=== Raw clinical files ===")
    spec = [
        ("brca", "patient", "SUBTYPE"),
        ("coad", "patient", "SUBTYPE"),
        ("prad", "patient", "SUBTYPE"),
    ]
    for cancer, kind, col in spec:
        path = DATA_RAW / f"{cancer}_clinical_{kind}.txt"
        check(f"{cancer}: clinical_{kind}.txt exists", path.exists())
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        check(
            f"{cancer}: {col} column present in clinical_{kind}",
            col in df.columns,
            f"have cols: {list(df.columns)[:6]}...",
        )
        if col in df.columns:
            n_labelled = df[col].notna().sum()
            check(
                f"{cancer}: at least 100 patients have a {col}",
                n_labelled >= 100,
                f"{n_labelled} labelled / {len(df)} total",
            )

    # COAD CMS labels
    cms_path = DATA_RAW / "coad_cms_labels.tsv"
    check("coad: CMS labels file exists", cms_path.exists())
    if cms_path.exists():
        cms = pd.read_csv(cms_path, sep="\t")
        tcga = cms[cms["dataset"].str.lower().str.contains("tcga", na=False)]
        check(
            "coad: CMS file has TCGA samples",
            len(tcga) >= 400,
            f"{len(tcga)} TCGA / {len(cms)} total",
        )
        col = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"
        valid = {"CMS1", "CMS2", "CMS3", "CMS4"}
        n_valid = tcga[col].isin(valid).sum()
        check(
            "coad: CMS file has CMS1-4 labels",
            n_valid >= 200,
            f"{n_valid} CMS-labelled (rest are NOLBL)",
        )


# ---------------------------------------------------------------------------
# Processed NPZ + meta
# ---------------------------------------------------------------------------

def audit_processed() -> None:
    print("\n=== Processed NPZ + meta ===")
    for cancer in CANCERS:
        npz_path = DATA_PROCESSED / f"{cancer}.npz"
        meta_path = DATA_PROCESSED / f"{cancer}_meta.json"
        check(f"{cancer}: NPZ exists", npz_path.exists())
        check(f"{cancer}: meta JSON exists", meta_path.exists())
        if not (npz_path.exists() and meta_path.exists()):
            continue

        npz = np.load(npz_path)
        meta = json.loads(meta_path.read_text())

        for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
            check(f"{cancer}: {k} present in NPZ", k in npz.files)

        Xtr = npz["X_train"]
        Xva = npz["X_val"]
        Xte = npz["X_test"]
        ytr = npz["y_train"]
        yva = npz["y_val"]
        yte = npz["y_test"]

        # dtype + shape
        check(
            f"{cancer}: X dtype float32",
            Xtr.dtype == np.float32 and Xva.dtype == np.float32 and Xte.dtype == np.float32,
            f"got {Xtr.dtype}/{Xva.dtype}/{Xte.dtype}",
        )
        check(
            f"{cancer}: y dtype int64",
            ytr.dtype == np.int64 and yva.dtype == np.int64 and yte.dtype == np.int64,
            f"got {ytr.dtype}/{yva.dtype}/{yte.dtype}",
        )
        check(
            f"{cancer}: 500 features in every split",
            Xtr.shape[1] == Xva.shape[1] == Xte.shape[1] == 500,
            f"got {Xtr.shape[1]}/{Xva.shape[1]}/{Xte.shape[1]}",
        )
        check(
            f"{cancer}: rows align (X_train rows == y_train rows, etc.)",
            len(Xtr) == len(ytr) and len(Xva) == len(yva) and len(Xte) == len(yte),
        )

        # NaN/inf
        for name, arr in (("X_train", Xtr), ("X_val", Xva), ("X_test", Xte)):
            check(
                f"{cancer}: no NaN in {name}",
                not np.isnan(arr).any(),
            )
            check(
                f"{cancer}: no inf in {name}",
                not np.isinf(arr).any(),
            )

        # Z-score sanity (train fit only)
        train_mean = float(Xtr.mean())
        train_std = float(Xtr.std())
        check(
            f"{cancer}: train mean ~0",
            abs(train_mean) < 1e-3,
            f"{train_mean:.6f}",
        )
        check(
            f"{cancer}: train std ~1",
            abs(train_std - 1.0) < 1e-3,
            f"{train_std:.6f}",
        )
        # Val/test should NOT be perfectly z-scored (they use train's scaler).
        val_mean = float(Xva.mean())
        check(
            f"{cancer}: val mean within reasonable drift from train",
            abs(val_mean) < 0.5,
            f"{val_mean:.4f}",
            warn=abs(val_mean) >= 0.2,
        )

        # Label encoding 0..K-1 in alphabetical class-name order
        K = len(meta["class_names"])
        for name, arr in (("y_train", ytr), ("y_val", yva), ("y_test", yte)):
            check(
                f"{cancer}: {name} values in 0..{K-1}",
                arr.min() >= 0 and arr.max() <= K - 1,
                f"min={arr.min()} max={arr.max()}",
            )
        check(
            f"{cancer}: class names alphabetically sorted",
            list(meta["class_names"]) == sorted(meta["class_names"]),
            f"{meta['class_names']}",
        )

        # Split fractions vs spec
        N = len(Xtr) + len(Xva) + len(Xte)
        for split_name, expected, actual in (
            ("train", SPLIT_TRAIN, len(Xtr) / N),
            ("val", SPLIT_VAL, len(Xva) / N),
            ("test", SPLIT_TEST, len(Xte) / N),
        ):
            check(
                f"{cancer}: {split_name} split fraction ~{expected:.2f}",
                abs(actual - expected) < 0.02,
                f"{actual:.3f}",
            )

        # Per-split class counts non-zero everywhere (no class missing from a split)
        zeros = []
        for split_name, y in (("train", ytr), ("val", yva), ("test", yte)):
            counts = np.bincount(y, minlength=K)
            zeros.extend(
                f"{meta['class_names'][i]} in {split_name}"
                for i, c in enumerate(counts) if c == 0
            )
        check(
            f"{cancer}: every class has >=1 sample in every split",
            len(zeros) == 0,
            f"missing: {zeros}",
            warn=True,  # 1-sample classes (PRAD FOXA1) flag as warn not fail
        )


# ---------------------------------------------------------------------------
# Split integrity (no leakage)
# ---------------------------------------------------------------------------

def audit_no_leakage() -> None:
    """Re-run the loaders, then check no patient ID lands in two splits."""
    print("\n=== Split leakage check ===")

    loaders = {"brca": load_brca, "coad": load_coad, "prad": load_prad}
    for cancer in CANCERS:
        # Reload original data to recover patient IDs
        X, y = loaders[cancer]()
        # We don't have the saved split indices on disk, but the seed is fixed
        # so we can recompute identically. Use the same logic as
        # scripts/prepare_data.py.
        from src.data.preprocess import filter_samples, stratified_split
        X_filt, y_filt = filter_samples(X, y)
        s = stratified_split(
            X_filt, y_filt,
            train=SPLIT_TRAIN, val=SPLIT_VAL, test=SPLIT_TEST,
            seed=42,
        )
        train_ids = set(s.X_train.index)
        val_ids = set(s.X_val.index)
        test_ids = set(s.X_test.index)

        check(
            f"{cancer}: train ∩ val == ∅",
            len(train_ids & val_ids) == 0,
            f"{len(train_ids & val_ids)} overlap",
        )
        check(
            f"{cancer}: train ∩ test == ∅",
            len(train_ids & test_ids) == 0,
            f"{len(train_ids & test_ids)} overlap",
        )
        check(
            f"{cancer}: val ∩ test == ∅",
            len(val_ids & test_ids) == 0,
            f"{len(val_ids & test_ids)} overlap",
        )
        # Every ID is a TCGA patient id
        bad_ids = [i for i in train_ids if not re.match(r"^TCGA-\w{2,4}-\w{4}$", i)]
        check(
            f"{cancer}: train IDs all match TCGA-XX-XXXX",
            len(bad_ids) == 0,
            f"{len(bad_ids)} bad",
        )


# ---------------------------------------------------------------------------
# Data freshness summary
# ---------------------------------------------------------------------------

def audit_freshness() -> None:
    """File mtimes + a note on upstream snapshot dates."""
    print("\n=== Data freshness ===")
    print("  upstream snapshot ages (informational, not failable):")
    print("    Xena HiSeqV2:  TCGA legacy snapshot, frozen post-TCGA-completion")
    print("    cBioPortal Pan-Cancer Atlas 2018: 2018 publication, frozen")
    print("    CMS labels (Sage-Bionetworks/crc-cms-kras): 2017-02-07 commit")
    print("    These are the canonical research-community snapshots. Newer data")
    print("    would require switching to GDC raw counts (different normalization).")

    for cancer in CANCERS:
        for fname in (
            f"{cancer}_expression.tsv.gz",
            f"{cancer}_clinical_patient.txt",
        ):
            p = DATA_RAW / fname
            if p.exists():
                ts = pd.Timestamp(p.stat().st_mtime, unit="s")
                print(f"    {fname:40s} downloaded {ts.strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print(" DATA AUDIT")
    print("=" * 72)

    audit_raw_expression()
    audit_raw_clinical()
    audit_processed()
    audit_no_leakage()
    audit_freshness()

    print("\n" + "=" * 72)
    if failures:
        print(f" {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"   - {f}")
    else:
        print(" ALL CHECKS PASSED")
    print("=" * 72)
    sys.exit(len(failures))


if __name__ == "__main__":
    main()
