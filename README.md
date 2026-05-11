# Benchmarking Data Augmentation for Cancer Subtype Classification

**Course:** CS 4824 — Machine Learning (Spring 2026)
**Authors:** Damien Stone (lead), Ishaan Gadkari
**Submission date:** May 8, 2026

This repository benchmarks **seven data-augmentation methods** against a no-augmentation baseline for molecular subtype classification on three TCGA cancer cohorts (BRCA, COAD, PRAD), using a fixed two-layer MLP classifier (Polepalli et al., 2023). Each (cancer × method) cell is run for **5 seeds** (42–46) on a stratified 70/15/15 train/val/test split. We additionally measure the synthetic data's **distribution match to real data** (MMD with RBF kernel, Pearson gene–gene correlation per Lacan et al., 2023 eq. 6) and ask whether distribution match predicts downstream classifier benefit.

**Headline finding.** Across 21 (cancer × method) cells, distribution-matching quality (low MMD, high gene–gene correlation) is *inversely* correlated with downstream macro-F1 lift. The deep generative methods (cVAE, WGAN-GP) produce the most realistic-looking synthetic samples but never significantly improve a downstream MLP; the simplest interpolation methods (Mixup, BorderlineSMOTE) are the only ones that ever help. See `report/Final_Report.pdf` for the full write-up.

---

## 1. Author contributions

- **Damien Stone** — built the technical project end-to-end. This includes data acquisition from UCSC Xena Hub and cBioPortal, the preprocessing pipeline (top-500 variable-gene selection, train-only z-scoring, stratified 70/15/15 split), the Polepalli MLP classifier and tree-classifier baselines (Random Forest, XGBoost), all four traditional augmentation methods (Random / SMOTE / ADASYN / BorderlineSMOTE) including their 351-config hyperparameter sweep, and all three deep-learning augmentation methods (cVAE with KL annealing, Lacan-style WGAN-GP, Mixup). Also wrote the synthetic-quality module (RBF-MMD + Pearson gene–gene correlation per Lacan eq. 6), the statistical-significance pipeline (paired t-test + Wilcoxon over 5 seeds), every figure and table in `results/`, and the data-audit script that gates the pipeline on 113 PASS integrity checks.

- **Ishaan Gadkari** — co-author on the final report and this README. Contributed to writing, editing, and review of the LaTeX manuscript in `report/`, including the framing of the headline finding, the discussion of limitations, and the references list. Reviewed the reproduction instructions and the contributions split below for accuracy. Ishaan did not contribute project code; the technical scope was originally split per `md/PROJECT_PLAN.md` §4 but, due to availability constraints during the build phase, Damien absorbed the full implementation workload.

---

## 2. Reproducing the results from scratch

This section is written so a reviewer can clone the repo and reproduce every number and figure in `report/Final_Report.pdf`. Total wall-clock on CPU: **about 60–90 minutes**, dominated by WGAN-GP training.

### 2.0. Prerequisites

- **Python 3.11 or newer.** We tested on 3.13.13. (`python --version` to check.)
- **Git** (for cloning the repo).
- ~250 MB of free disk for raw data + processed splits + model checkpoints.
- A working network connection for the data-download step. **No GPU required**; the published numbers come from a CPU-only run.

### 2.1. Clone and set up the environment

```powershell
git clone <repo-url> CS4824_PROJECT
cd CS4824_PROJECT

# (recommended) create a virtual environment so pinned versions don't fight system packages
python -m venv .venv
.venv\Scripts\Activate.ps1            # PowerShell on Windows
# source .venv/bin/activate            # bash on macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the eight third-party packages to the exact versions used to produce the checked-in `results/`. A CUDA build of PyTorch will work in place of the pinned CPU build with no code changes (`get_device()` in `src/config.py` auto-detects).

### 2.2. Download the six raw data files

The repo's `.gitignore` excludes the large expression matrices (`data/raw/*_expression.tsv.gz`), so you must fetch them yourself. All six files are public, no auth required, and total ~150 MB.

The simplest path is to run the commands below from the project root. They reproduce the directory layout that `prepare_data.py` expects.

```powershell
# Expression matrices (UCSC Xena Hub) — ~30–80 MB each
curl -L -o data/raw/brca_expression.tsv.gz https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
curl -L -o data/raw/coad_expression.tsv.gz https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz
curl -L -o data/raw/prad_expression.tsv.gz https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz

# Clinical / subtype labels (cBioPortal datahub via GitHub LFS mirror — S3 returns 403)
$BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
curl -L -o data/raw/brca_clinical_patient.txt "$BASE/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
curl -L -o data/raw/brca_clinical_sample.txt  "$BASE/brca_tcga_pan_can_atlas_2018/data_clinical_sample.txt"
curl -L -o data/raw/coad_clinical_patient.txt "$BASE/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
curl -L -o data/raw/coad_clinical_sample.txt  "$BASE/coadread_tcga_pan_can_atlas_2018/data_clinical_sample.txt"
curl -L -o data/raw/prad_clinical_patient.txt "$BASE/prad_tcga_pub/data_clinical_patient.txt"
curl -L -o data/raw/prad_clinical_sample.txt  "$BASE/prad_tcga_pub/data_clinical_sample.txt"

# COAD CMS labels (Sage-Bionetworks GitHub mirror of CRC Subtyping Consortium)
curl -L -o data/raw/coad_cms_labels.tsv https://raw.githubusercontent.com/Sage-Bionetworks/crc-cms-kras/master/020717/cms_labels_public_all.txt
```

(Three of these source URLs differ from what the original data plan specified — the cBioPortal S3 bucket returns 403, the Pan-Cancer-Atlas PRAD study has no real subtype column, and the canonical COAD CMS labels live on Synapse behind an account. See §4 below for the full deviation list.)

Now verify everything parses:

```powershell
python scripts/verify_data.py
```

You should see "BRCA expression: 1218 samples", "COAD expression: 329 samples", "PRAD expression: 550 samples", and the three SUBTYPE columns each printed with their value counts. If any line fails, fix that file before proceeding.

### 2.3. Build the processed splits

```powershell
python scripts/prepare_data.py
```

This writes `data/processed/{brca,coad,prad}.npz` (the train/val/test arrays) and `{brca,coad,prad}_meta.json` (gene names, class names, scaler params). It is fully deterministic at `seed=42`. Wall-clock: < 30 s.

Sanity-check the result before training anything on it:

```powershell
python scripts/data_audit.py
```

This runs **113 integrity checks** — no test/val sample appears in train; the train z-score has mean ≈ 0 and std ≈ 1; only primary-tumor barcodes (`NN==01`) survive the filter; the label encoder is consistent across splits; etc. **Every check must say PASS before any results are trusted.**

### 2.4. Run the experiment suite

Each script appends rows to a single master CSV, `results/tables/all_results.csv`. They can be run in any order (they are independent), but the order below matches the dependency between the cVAE/WGAN-GP train and eval halves. Use `-u` so the long runs flush their progress lines in real time.

```powershell
python -u src/experiments/01_baselines.py            # MLP no-aug, all 3 cancers, 5 seeds
python -u src/experiments/01b_tree_baselines.py      # Random Forest + XGBoost (no-aug + SMOTE)
python -u src/experiments/02_traditional.py          # 4 traditional methods, default hyperparams
python -u src/experiments/04_traditional_tuned.py    # 351 val-set configs → tuned per-cell hyperparams
python -u src/experiments/03_cvae_train.py           # train + checkpoint cVAE per cancer
python -u src/experiments/03_cvae_eval.py            # generate 1:1 synth, retrain MLP, log
python -u src/experiments/04_wgan_gp_train.py        # 400-epoch WGAN-GP per cancer (slowest step)
python -u src/experiments/04_wgan_gp_eval.py         # downstream MLP on real + WGAN-GP synth
python -u src/experiments/05_mixup.py                # input-space Mixup (α=0.2)
python -u src/experiments/06_aug_sweep.py            # BRCA-only N_synth ∈ {100,500,1000,2000,3000}
```

Approximate per-script wall-clocks on the reference CPU machine: baselines & tree models < 1 min each; traditional + tuned < 5 min combined; cVAE train+eval ~5 min; WGAN-GP train+eval ~30–45 min (this is the bulk of the budget); Mixup and sweep ~5 min combined.

**Resumability.** All scripts are deterministic given `(cancer, method, seed)`. If a run is interrupted, just re-run the script — it will overwrite previous rows for the same `(cancer, method, seed)` and pick up the missing ones. Model checkpoints are saved to `results/models/` per seed.

### 2.5. Aggregate, test, and produce all figures

```powershell
python scripts/aggregate_results.py        # summary_main.{csv,md}, per-class heatmaps, per-cancer bar charts
python scripts/synth_quality_all.py        # MMD + Pearson correlation per (cancer × method) → synth_quality.csv
python scripts/stat_sig.py                 # paired t-test + Wilcoxon vs no-aug → stat_sig.csv (124 comparisons)
python scripts/plot_synth_vs_utility.py    # CORE FINDING: MMD vs Δmacro-F1 scatter
python scripts/plot_grids.py               # 3×6 confusion-matrix grid + 3-panel ROC overlays (re-loads seed-42 checkpoints)
python scripts/plot_sweep.py               # BRCA augmentation sweep figure
```

All figures are written to `results/figures/` as both `.pdf` (for the LaTeX report) and `.png` at 300 DPI.

### 2.6. What to look for when verifying

After step 2.5, the master CSV should have **330 rows** (3 cancers × ~22 method variants × 5 seeds), and the figures listed in §6 below should match the ones embedded in `report/Final_Report.pdf`.

### 2.7. Rebuilding the PDF report (optional)

The PDF is checked into the repo, so this step is only needed if you edit the LaTeX. Any standard TeX distribution works (TeX Live, MiKTeX):

```powershell
cd report
pdflatex final_report.tex
pdflatex final_report.tex          # second pass for cross-references
```

The figures referenced by `final_report.tex` live in `results/figures/` and are regenerated by §2.5.

### 2.8. Common issues

- **`UnicodeEncodeError` on Windows for `±` or `≈`** — already worked around (we write `+/-` and `~=` to stdout). If it surfaces in a custom edit, set `PYTHONIOENCODING=utf-8`.
- **Tcl/Tk crash from matplotlib + xgboost parallel jobs** — `src/evaluation/viz.py` and the plot scripts force the Agg backend before importing pyplot, so no display is needed. Don't switch backends.
- **WGAN-GP COAD G-loss prints `nan` from epoch 1** — expected. See §8. The eval script still runs and the resulting cell is honestly reported in the master CSV.
- **`prepare_data.py` complains about a missing file** — re-run `scripts/verify_data.py` to identify which of the six raw downloads is missing or zero-byte, then re-download just that one.

---

## 3. Hardware and software environment

| | Used for the published numbers |
|---|---|
| OS | Windows 11 Home 26200 |
| Python | 3.13.13 |
| PyTorch | 2.9.1 (CPU-only build) |
| GPU | None (CPU-only run) |
| Wall-clock for full §2 run | ~60–90 min |
| Disk | ~150 MB raw data, ~20 MB processed, ~50 MB results |

**Key constraint:** the published WGAN-GP numbers use **400 epochs** rather than the **800** in Lacan et al. (2023), to keep the full re-run inside ~90 min on CPU. This deviation is documented in `report/Final_Report.pdf` §3.4. BRCA and PRAD still converge cleanly; COAD WGAN-GP G-loss diverges from epoch 1 (BatchNorm + only 4 batches/epoch — honest finding, not patched).

A CUDA build of PyTorch will work without any code changes (`get_device()` in `src/config.py` auto-detects).

---

## 4. Data-acquisition deviations

Three notes on where the actual data sources differ from what a naive read of the proposal might suggest. All three are also recorded in `data/raw/MANIFEST.md` and flagged in `report/Final_Report.pdf` §3.1.

1. **Clinical files come from the GitHub LFS mirror, not the cBioPortal S3 bucket** (S3 returns 403 across every tested path as of 2026-04-27). Content is byte-identical.
2. **PRAD clinical is from `prad_tcga_pub`** (TCGA 2015 Cell paper), not `prad_tcga_pan_can_atlas_2018` — the Pan-Cancer-Atlas study's `SUBTYPE` column contained only the literal string `"PRAD"` rather than molecular subtypes. The Pan-Cancer files are kept side-by-side as `prad_pancan_*.txt` for traceability.
3. **COAD CMS labels via the Sage-Bionetworks GitHub mirror**, not Synapse `syn4978511` (which would require an account). Same upstream `cms_labels_public_all.txt` file.

---

## 5. Repository structure

```
CS4824_PROJECT/
├── README.md                       ← you are here
├── requirements.txt                ← pinned versions
├── .gitignore
│
├── papers/                         ← five reference PDFs cited in the report
│
├── data/
│   ├── raw/                        ← downloaded files (see §2.2) + MANIFEST.md
│   └── processed/                  ← {brca,coad,prad}.npz + meta.json (produced by prepare_data.py)
│
├── src/
│   ├── config.py                   ← SEED=42, paths, set_seed, get_device
│   ├── data/
│   │   ├── load_xena.py            ← per-cancer (X, y) loaders
│   │   ├── preprocess.py           ← top-500 genes, z-score, stratified split, encode_labels
│   │   └── __init__.py             ← load_processed cache reader
│   ├── classifier/
│   │   ├── mlp.py                  ← Polepalli MLP + train/eval + checkpoint helpers
│   │   └── tree_models.py          ← RF + XGBoost wrappers
│   ├── augmentation/
│   │   ├── traditional.py          ← imbalanced-learn façade
│   │   ├── cvae.py                 ← class-conditional VAE + KL annealing
│   │   ├── wgan_gp.py              ← Lacan-style WGAN-GP
│   │   └── mixup.py                ← input-space mixup
│   ├── evaluation/
│   │   ├── metrics.py              ← supervised_metrics(...)
│   │   ├── viz.py                  ← plot_confusion_matrix (matplotlib Agg backend)
│   │   └── synth_quality.py        ← MMD, correlation_score, plot_tsne_overlay
│   └── experiments/
│       ├── _common.py              ← metrics_to_row, append_rows
│       ├── 01_baselines.py         ← MLP no-aug
│       ├── 01b_tree_baselines.py   ← RF + XGBoost no-aug + SMOTE
│       ├── 02_traditional.py       ← 4 traditional methods, default hyperparams
│       ├── 04_traditional_tuned.py ← 351-config validation sweep + chosen configs
│       ├── 03_cvae_train.py        ← train + checkpoint cVAE per cancer
│       ├── 03_cvae_eval.py         ← generate 1:1 synth, retrain MLP, log
│       ├── 04_wgan_gp_train.py     ← 400-epoch WGAN-GP training
│       ├── 04_wgan_gp_eval.py      ← evaluate WGAN-GP synthetic samples downstream
│       ├── 05_mixup.py             ← Mixup
│       └── 06_aug_sweep.py         ← BRCA N_synth ∈ {100,500,1000,2000,3000} sweep
│
├── scripts/
│   ├── prepare_data.py             ← runs the §2.3 preprocessing pipeline
│   ├── verify_data.py              ← confirms six raw files parse
│   ├── data_audit.py               ← 113 integrity checks (no leakage, z-score, etc.)
│   ├── eda.py                      ← class distributions + raw-data t-SNE
│   ├── aggregate_results.py        ← summary_main.{csv,md} + bar charts + heatmaps
│   ├── synth_quality_all.py        ← MMD + correlation per (cancer × method)
│   ├── stat_sig.py                 ← paired t-test + Wilcoxon vs no-aug
│   ├── plot_grids.py               ← 3×6 CM grid + 3-panel ROC
│   ├── plot_sweep.py               ← BRCA N_synth sweep figure
│   └── plot_synth_vs_utility.py    ← MMD vs Δmacro-F1 scatter (core finding)
│
├── results/
│   ├── tables/
│   │   ├── all_results.csv         ← master table, 330 rows
│   │   ├── summary_main.{csv,md}   ← per (cancer × method) mean ± std macro F1
│   │   ├── summary_per_class.csv   ← per-class F1 per (cancer × method)
│   │   ├── synth_quality.csv       ← MMD + corr per (cancer × method)
│   │   ├── stat_sig.csv            ← 124 paired comparisons
│   │   └── traditional_tuning.json ← chosen hyperparams per (cancer × method)
│   ├── figures/                    ← all PDF + PNG (300 DPI)
│   └── models/                     ← MLP/cVAE/WGAN-GP checkpoints (per seed)
│
└── report/
    ├── final_report.tex            ← LaTeX source
    └── Final_Report.pdf            ← submission deliverable
```

---

## 6. What gets produced where

| Artifact | Path | Produced by |
|---|---|---|
| Per-run metrics (330 rows) | `results/tables/all_results.csv` | every `src/experiments/*.py` |
| Per-cancer mean ± std macro-F1 | `results/tables/summary_main.{csv,md}` | `aggregate_results.py` |
| Per-class F1 bar charts | `results/figures/per_class_f1_{brca,coad,prad}.{pdf,png}` | `aggregate_results.py` |
| Mean accuracy / macro-F1 bar charts | `results/figures/summary_{accuracy,macro_f1}.{pdf,png}` | `aggregate_results.py` |
| MMD + correlation per (cancer × method) | `results/tables/synth_quality.csv` | `synth_quality_all.py` |
| Paired t-test + Wilcoxon vs baseline | `results/tables/stat_sig.csv` | `stat_sig.py` |
| **Core finding scatter** (MMD vs Δmacro-F1) | `results/figures/synth_quality_vs_delta_f1.{pdf,png}` | `plot_synth_vs_utility.py` |
| 3×6 confusion-matrix grid | `results/figures/cm_grid.{pdf,png}` | `plot_grids.py` |
| 3-panel macro-OvR ROC overlays | `results/figures/roc_grid.{pdf,png}` | `plot_grids.py` |
| BRCA N_synth sweep figure | `results/figures/sweep_brca.{pdf,png}` | `plot_sweep.py` |
| Per-experiment confusion matrices | `results/figures/{baselines,traditional,…}/` | individual experiment scripts |
| Model checkpoints (per seed) | `results/models/{cancer}/{method}_seed{N}.pt` | individual experiment scripts |

---

## 7. Determinism notes

- All random sources (Python `random`, NumPy, PyTorch CPU + CUDA, `PYTHONHASHSEED`) are seeded by `src.config.set_seed(seed)` at the start of every experiment script. cuDNN is set to deterministic.
- The five reproducibility seeds are `(42, 43, 44, 45, 46)`. Seed 42 is also the canonical seed used for figure generation that needs a single representative model (e.g., `plot_grids.py`).
- Train-fit-only operations: top-500 variable-gene selection, `StandardScaler`, `LabelEncoder`. Validation and test partitions never see fit-time statistics.
- The two-step stratified split (70/15/15) is computed once in `scripts/prepare_data.py` and saved into `data/processed/{cancer}.npz`; all downstream scripts read the same split.

---

## 8. Known limitations

- **WGAN-GP COAD G-loss diverges from epoch 1.** Attributed to BatchNorm in the Generator combined with only ~4 batches/epoch on the small COAD training partition. Documented as an honest finding rather than patched, since faithfulness to the Lacan reference architecture takes precedence over chasing a working number on this single cell.
- **400-epoch WGAN-GP, not Lacan's 800.** Wall-clock budget on CPU. Tested and converges for BRCA and PRAD.
- **PRAD FOXA1 subtype has only 9 patients pre-split.** After 70/15/15 it has 6/1/2 samples. We retained it and report `class_FOXA1_f1` honestly — for several method × seed combinations no FOXA1 sample is ever predicted (F1 = 0).
- **No CUDA build was tested.** All numbers come from CPU PyTorch. A CUDA build will produce numerically equivalent results up to floating-point reordering.

---

## 9. Reference papers

Driving methodology — full PDFs in `papers/`:

1. **Polepalli et al., 2023** — "Deep learning approach for cancer subtype classification using gene expression data." Provides the 500→256→128 MLP and preprocessing recipe used as our classifier and baseline.
2. **Lacan et al., 2023** — "GAN-based data augmentation for transcriptomics." Provides the WGAN-GP architecture, training hyperparameters (n_critic=5, λ_gp=10, Adam betas (0.5, 0.9), 800 epochs), and the gene–gene correlation score (eq. 6).
3. **Guinney et al., 2015** — CRC Subtyping Consortium CMS labels.
4. **The Cancer Genome Atlas Network, 2012** — TCGA BRCA paper (PAM50).
5. **The Cancer Genome Atlas Research Network, 2015** — TCGA PRAD molecular taxonomy (Cell 2015) — source of the SPOP/ERG/FOXA1 subtypes.

Full citations are in `report/Final_Report.pdf` §References.
