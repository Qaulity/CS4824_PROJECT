# Raw data manifest

Downloaded 2026-04-27. Verified with `scripts/verify_data.py`.

## Expression matrices (UCSC Xena Hub, no auth)

Format: gzipped TSV. Rows = genes (HGNC symbols), columns = TCGA sample barcodes. Values = log2(RSEM normalized count + 1). Genes × samples; transpose to samples × genes before training.

| Local file | Source URL | Samples | First gene |
|---|---|---|---|
| `brca_expression.tsv.gz` | `https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz` | 1218 | ARHGEF10L |
| `coad_expression.tsv.gz` | `https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz` | 329  | ARHGEF10L |
| `prad_expression.tsv.gz` | `https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz` | 550  | ARHGEF10L |

## Clinical / subtype labels (cBioPortal datahub via GitHub LFS mirror)

The cBioPortal S3 datahub bucket (`cbioportal-datahub.s3.amazonaws.com`) returns 403 Forbidden across every tested path as of 2026-04-27. The datahub is still mirrored on GitHub with Git-LFS, so each `data_clinical_sample.txt` and `data_clinical_patient.txt` was pulled directly from `media.githubusercontent.com/media/cBioPortal/datahub/master/public/<study>/`. Content is identical to the S3 tarball contents.

| Local file | Source study | URL | Notes |
|---|---|---|---|
| `brca_clinical_sample.txt`  | `brca_tcga_pan_can_atlas_2018`  | …/brca_tcga_pan_can_atlas_2018/data_clinical_sample.txt  | 1084 rows, no SUBTYPE col |
| `brca_clinical_patient.txt` | `brca_tcga_pan_can_atlas_2018`  | …/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt | 1084 rows, **SUBTYPE col is here**: LumA 499, LumB 197, Basal 171, Her2 78, Normal 36, NaN 103 |
| `coad_clinical_sample.txt`  | `coadread_tcga_pan_can_atlas_2018` | …/coadread_tcga_pan_can_atlas_2018/data_clinical_sample.txt  | 594 rows |
| `coad_clinical_patient.txt` | `coadread_tcga_pan_can_atlas_2018` | …/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt | 594 rows, SUBTYPE col is integrative subtypes (CIN/GS/MSI/POLE for COAD+READ) — **NOT** CMS. See COAD CMS labels file below. |
| `prad_clinical_sample.txt`  | `prad_tcga_pub` (TCGA 2015 Cell)  | …/prad_tcga_pub/data_clinical_sample.txt  | 333 rows. Promoted as canonical because Pan-Cancer Atlas SUBTYPE col contained only the literal cancer acronym `"PRAD"`. |
| `prad_clinical_patient.txt` | `prad_tcga_pub` (TCGA 2015 Cell)  | …/prad_tcga_pub/data_clinical_patient.txt | 333 rows. SUBTYPE col matches the spec: 1-ERG 152, 8-other 86, 5-SPOP 37, 2-ETV1 28, 3-ETV4 14, 6-FOXA1 9, 4-FLI1 4, 7-IDH1 3. Apply Option C collapse (keep ERG/SPOP/FOXA1, collapse the rest to "Other"). |
| `prad_pancan_clinical_sample.txt`  | `prad_tcga_pan_can_atlas_2018` | same path pattern, kept for traceability | 494 rows, **not used for subtyping** |
| `prad_pancan_clinical_patient.txt` | `prad_tcga_pan_can_atlas_2018` | same path pattern, kept for traceability | 494 rows, SUBTYPE col is `"PRAD"` only |

## COAD CMS labels (CRC Subtyping Consortium, GitHub mirror)

Spec's primary source is Synapse `syn4978511` (account required). Used Fallback B from `md/DATA_ACQUISITION.md`: a public mirror of the same `cms_labels_public_all.txt` file lives in the Sage-Bionetworks `crc-cms-kras` GitHub repo (no auth).

| Local file | Source URL |
|---|---|
| `coad_cms_labels.tsv` | `https://raw.githubusercontent.com/Sage-Bionetworks/crc-cms-kras/master/020717/cms_labels_public_all.txt` |

Content: 3397 total rows pooled across multiple CRC cohorts, 573 of which are TCGA samples. Match by TCGA barcode (first 12 chars of patient ID). Use the column `CMS_final_network_plus_RFclassifier_in_nonconsensus_samples` as the canonical CMS label.

TCGA-only CMS distribution: CMS1 76, CMS2 220, CMS3 72, CMS4 144, NOLBL 61. NOLBL = "no consensus label assigned"; drop these per spec's class-count safeguard (`< 5 → drop`, but here it's 61 unlabelled samples — drop them).

## Deviations from `md/DATA_ACQUISITION.md` (flag in report)

1. **Clinical files came from GitHub LFS mirror, not cBioPortal S3.** S3 datahub bucket is 403 across every tested path. No content difference; one less indirection (no tarball extraction needed in `load_xena.py`).
2. **PRAD clinical promoted from `prad_tcga_pub` instead of `prad_tcga_pan_can_atlas_2018`.** Pan-Cancer Atlas study's SUBTYPE column did not contain molecular subtypes — only the cancer acronym. The 2015 Cell-paper study has the spec-expected subtype distribution exactly. Pan-Cancer-Atlas versions kept side-by-side as `prad_pancan_*` for traceability.
3. **COAD CMS labels via Fallback B (GitHub mirror), not Fallback A (Synapse).** Same upstream file (`cms_labels_public_all.txt`), no Synapse account needed.
