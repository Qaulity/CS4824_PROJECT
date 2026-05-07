"""
Verify that all six raw data files parse and contain expected counts.

Adapted from md/DATA_ACQUISITION.md - reads cBioPortal clinical files directly
as .txt (downloaded from GitHub LFS mirror) instead of extracting from S3 tarballs,
because the cBioPortal S3 datahub bucket returns 403 as of 2026-04-27.
See data/raw/MANIFEST.md for source URLs.
"""
import gzip
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def check_expression(cancer: str) -> None:
    path = RAW / f"{cancer}_expression.tsv.gz"
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        first_row = f.readline().rstrip("\n").split("\t")
    n_samples = len(header) - 1
    print(
        f"{cancer.upper()} expression: {n_samples} samples, "
        f"first gene = {first_row[0]}"
    )


def check_clinical(cancer: str) -> None:
    """Read both data_clinical_sample.txt and data_clinical_patient.txt
    so we can find a SUBTYPE column wherever it lives."""
    for kind in ("sample", "patient"):
        path = RAW / f"{cancer}_clinical_{kind}.txt"
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        subtype_cols = [c for c in df.columns if "SUBTYPE" in c.upper()]
        cms_cols = [c for c in df.columns if "CMS" in c.upper()]
        relevant = subtype_cols + cms_cols
        print(
            f"{cancer.upper()} clinical_{kind}: {len(df)} rows, "
            f"subtype/CMS column(s): {relevant or '(none)'}"
        )
        for col in relevant:
            counts = df[col].value_counts(dropna=False).to_dict()
            print(f"  {col}: {counts}")


def check_coad_cms() -> None:
    path = RAW / "coad_cms_labels.tsv"
    df = pd.read_csv(path, sep="\t")
    tcga = df[df["dataset"].str.lower().str.contains("tcga", na=False)]
    cms_col = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"
    print(
        f"COAD CMS labels: {len(df)} total rows, {len(tcga)} TCGA samples, "
        f"{cms_col}:"
    )
    print(f"  {tcga[cms_col].value_counts(dropna=False).to_dict()}")


def main() -> None:
    print("=" * 70)
    print("EXPRESSION FILES")
    print("=" * 70)
    for cancer in ("brca", "coad", "prad"):
        check_expression(cancer)

    print()
    print("=" * 70)
    print("CLINICAL FILES")
    print("=" * 70)
    for cancer in ("brca", "coad", "prad"):
        check_clinical(cancer)
        print()

    print("=" * 70)
    print("COAD CMS LABELS (Sage-Bionetworks/crc-cms-kras mirror)")
    print("=" * 70)
    check_coad_cms()


if __name__ == "__main__":
    main()
