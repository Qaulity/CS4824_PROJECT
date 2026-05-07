"""Load TCGA expression + subtype labels from local files in data/raw/.

Source URLs and provenance live in data/raw/MANIFEST.md.

Each `load_*` function returns:
    (X: pd.DataFrame, y: pd.Series)

with rows = primary-tumor TCGA samples (indexed by patient barcode TCGA-XX-XXXX),
columns = HGNC gene symbols, and y[label] aligned to X.index.

The expression matrices on Xena are stored as `genes x samples`. We transpose to
`samples x genes` and trim to the patient-level barcode (first 12 chars) so we can
join against the cBioPortal / CMS clinical files which are patient-keyed.
"""
from __future__ import annotations

from pathlib import Path

import gzip
import pandas as pd

from src.config import DATA_RAW

# ---------------------------------------------------------------------------
# Subtype-label normalization tables
# ---------------------------------------------------------------------------

# BRCA: cBioPortal SUBTYPE strings -> short class names used in PAM50 literature.
BRCA_SUBTYPE_MAP = {
    "BRCA_LumA": "LumA",
    "BRCA_LumB": "LumB",
    "BRCA_Basal": "Basal",
    "BRCA_Her2": "HER2",
    "BRCA_Normal": "Normal",
}

# PRAD Option C collapse (per PROJECT_PLAN.md and TECHNICAL_SPEC.md §2):
#   keep ERG, SPOP, FOXA1; collapse ETV1/ETV4/FLI1/IDH1/8-other -> "Other".
PRAD_SUBTYPE_MAP = {
    "1-ERG": "ERG",
    "5-SPOP": "SPOP",
    "6-FOXA1": "FOXA1",
    "2-ETV1": "Other",
    "3-ETV4": "Other",
    "4-FLI1": "Other",
    "7-IDH1": "Other",
    "8-other": "Other",
}

# COAD CMS: keep CMS1..CMS4, drop NOLBL (61 samples in the pooled file).
COAD_VALID_CMS = {"CMS1", "CMS2", "CMS3", "CMS4"}


# ---------------------------------------------------------------------------
# Expression-matrix helpers
# ---------------------------------------------------------------------------


def _read_expression(path: Path) -> pd.DataFrame:
    """Read a Xena HiSeqV2 gz expression matrix and return samples x genes.

    File layout: first column is the gene symbol, header is the list of TCGA
    sample barcodes. Values are log2(RSEM normalized count + 1).
    """
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    df.index.name = "gene"
    df.columns.name = "sample"
    return df.T  # samples x genes


def _filter_primary_tumor(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only primary-tumor barcodes (TCGA sample-type code 01)."""
    keep = X.index.to_series().str.match(r"^TCGA-[^-]+-[^-]+-01$")
    return X.loc[keep]


def _to_patient_id(X: pd.DataFrame) -> pd.DataFrame:
    """Trim sample barcode TCGA-XX-XXXX-NN to patient-level TCGA-XX-XXXX.

    If two samples map to the same patient (rare in primary-tumor-only filter),
    keep the first.
    """
    pid = X.index.str[:12]
    X = X.copy()
    X.index = pid
    X.index.name = "patient"
    return X.loc[~X.index.duplicated(keep="first")]


def _join_to_labels(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join expression rows on `y.index`, drop rows whose label is missing."""
    y = y.dropna()
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


# ---------------------------------------------------------------------------
# Cancer-specific loaders
# ---------------------------------------------------------------------------


def load_brca(raw_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.Series]:
    X = _read_expression(raw_dir / "brca_expression.tsv.gz")
    X = _filter_primary_tumor(X)
    X = _to_patient_id(X)

    clin = pd.read_csv(
        raw_dir / "brca_clinical_patient.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )
    y = (
        clin.set_index("PATIENT_ID")["SUBTYPE"]
        .map(BRCA_SUBTYPE_MAP)  # silently drops anything unmapped (incl. NaN)
        .dropna()
    )
    return _join_to_labels(X, y)


def load_coad(raw_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.Series]:
    X = _read_expression(raw_dir / "coad_expression.tsv.gz")
    X = _filter_primary_tumor(X)
    X = _to_patient_id(X)

    cms = pd.read_csv(raw_dir / "coad_cms_labels.tsv", sep="\t")
    cms = cms[cms["dataset"].str.lower().str.contains("tcga", na=False)]
    cms_col = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"
    cms = cms[cms[cms_col].isin(COAD_VALID_CMS)]
    y = cms.set_index("sample")[cms_col].rename("SUBTYPE")
    return _join_to_labels(X, y)


def load_prad(raw_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.Series]:
    X = _read_expression(raw_dir / "prad_expression.tsv.gz")
    X = _filter_primary_tumor(X)
    X = _to_patient_id(X)

    clin = pd.read_csv(
        raw_dir / "prad_clinical_patient.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )
    y = (
        clin.set_index("PATIENT_ID")["SUBTYPE"]
        .map(PRAD_SUBTYPE_MAP)  # Option C collapse; unmapped → NaN → dropped
        .dropna()
    )
    return _join_to_labels(X, y)


LOADERS = {"brca": load_brca, "coad": load_coad, "prad": load_prad}


def load(cancer: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatch to the per-cancer loader."""
    return LOADERS[cancer]()
