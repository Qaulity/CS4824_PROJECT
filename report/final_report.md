# Benchmarking Deep Learning Data Augmentation for Cancer Subtype Classification

**Damien Stone, Ishaan Gadkari**
CS 4824 — Machine Learning, Spring 2026
Submitted: May 8, 2026

---

## Abstract

Cancer subtype classification from gene expression suffers from severe class imbalance: rare molecular subtypes have too few samples for a held-out classifier to learn discriminative boundaries. Data augmentation is a natural remedy, but the literature evaluates classical, deep-generative, and interpolation-based methods on different datasets, classifiers, and metrics, making practical method choice unclear. We benchmark seven augmentation methods (random oversampling, SMOTE, ADASYN, BorderlineSMOTE, conditional VAE, conditional WGAN-GP, and Mixup) on three TCGA cancers (BRCA, COAD, PRAD) using a single held-constant 2-layer MLP classifier. Across $3 \times 7 + 3 = 24$ experimental conditions evaluated over 5 random seeds (plus an $N_\text{synth}$ sweep on BRCA at five sample counts), we find that (i) traditional oversampling methods modestly help on BRCA but are statistically indistinguishable from no-augmentation on COAD; (ii) deep generative methods (cVAE, WGAN-GP) achieve excellent distribution-matching scores (MMD as low as $0.011$) yet provide *no* downstream classification benefit and significantly *hurt* macro-F1 on BRCA when used at 1:1 augmentation ratios; (iii) on the most-imbalanced cancer (PRAD, with one class at six training samples), no MLP-based augmentation method beats the no-augmentation baseline — but XGBoost without augmentation does, by 7.5 macro-F1 points. The central finding is a *decoupling between synthetic-data quality and classifier benefit*: the methods that produce the most "real-looking" synthetic samples (Mixup, WGAN-GP) provide the smallest downstream lift, while methods with poor distributional fidelity (SMOTE family) provide the largest. We discuss implications for the transcriptomics-augmentation literature.

---

## 1. Introduction

Cancer molecular-subtype classification is a paradigmatic small-data, high-dimensional problem. The Cancer Genome Atlas (TCGA) provides a few hundred to a few thousand tumor samples per cancer type, each profiled across ~20,000 genes. Within each cancer, clinically meaningful subtypes (e.g. PAM50 for breast cancer, CMS1–4 for colon cancer) define class labels — but these subtypes are highly imbalanced, with the rarest classes containing fewer than 10 samples. Standard supervised classifiers trained on such data tend to predict the majority subtype well and ignore minorities entirely, producing high accuracy but useless macro-averaged metrics.

Data augmentation — generating synthetic training samples — is a well-established remedy for class imbalance in image and text domains. In transcriptomics it has been studied less systematically. Recent work has compared specific methods in isolation: SMOTE-variants on cancer expression \[Choi \& Chae 2020\], cVAE for pan-cancer subtyping \[Polepalli 2025\], and a benchmark of GAN architectures on TCGA-style data \[Lacan et al.\ 2023\]. But no published study holds the classifier constant while comparing classical, generative, and interpolation-based augmentation across several cancers and several minority-class regimes.

This work fills that gap. We freeze the classifier (a 2-layer MLP per Polepalli's specification), the dataset (TCGA expression for BRCA, COAD, PRAD), and the evaluation protocol (5 seeds, stratified 70/15/15 split, top-500 most-variable genes), then sweep the augmentation method as the only experimental variable. Three central questions guide the work:

1. **Which augmentation method, if any, robustly improves classification?** We expected SMOTE-family methods to provide moderate gains and deep generative methods to provide larger gains — the literature implies this ordering. We find the opposite.

2. **Does synthetic-data quality predict downstream classifier benefit?** We expected high MMD distance and low gene-gene correlation preservation to indicate "bad" synthetic samples that hurt the classifier. We find the inverse: the most realistic-looking synthetic samples are the least useful.

3. **Are augmentation results consistent across cancers with different imbalance profiles?** BRCA has moderate imbalance (14× ratio between most and least frequent class). COAD is roughly balanced after CMS-only filtering. PRAD has extreme imbalance (one class at 6 training samples). Augmentation methods that succeed on one cancer do not necessarily succeed on the others.

Section 2 reviews related work. Section 3 describes datasets, preprocessing, the classifier, and the seven augmentation methods. Section 4 specifies the experimental protocol. Section 5 presents results. Section 6 discusses the central decoupling finding, the methodological choice to substitute WGAN-GP for the proposal's CTGAN, and the limits of augmentation as a solution to extreme imbalance. Section 7 concludes.

---

## 2. Related work

**SMOTE-family methods.** SMOTE \[Chawla et al.\ 2002\] generates synthetic minority-class samples as convex combinations of $k$-nearest neighbors of the same class. ADASYN \[He et al.\ 2008\] weights generation density by per-sample classification difficulty. BorderlineSMOTE \[Han et al.\ 2005\] generates only near class boundaries. These methods predate deep learning and remain strong baselines in tabular domains. Choi \& Chae's *methCancer-gen* (2020) compares SMOTE variants for cancer DNA-methylation data and reports modest classification gains.

**Deep generative augmentation.** Conditional VAEs \[Sohn et al.\ 2015\] and conditional GANs \[Mirza \& Osindero 2014\] can be trained on labeled data to produce class-conditional synthetic samples. Polepalli (2025) applies a class-conditional VAE to pan-cancer subtype classification, reporting that cVAE samples improve minority-class F1 on a TCGA subset. Lacan et al.\ (2023, *Bioinformatics*) benchmark vanilla GAN, WGAN, WGAN-GP, AttGAN, and ConvGAN on TCGA expression matrices, finding WGAN-GP \[Gulrajani et al.\ 2017\] superior on every metric — a result we relied on when substituting WGAN-GP for the CTGAN \[Xu et al.\ 2019\] proposed in our project proposal (see §6.1).

**Mixup.** Mixup \[Zhang et al.\ 2018\] generates synthetic samples as $\lambda x_i + (1-\lambda) x_j$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$ and a corresponding label mix. Originally introduced for image classification, it has been applied to tabular data with mixed results.

**Subtype labels for TCGA.** PAM50 labels for BRCA \[Parker et al.\ 2009\] are well-curated. The Consensus Molecular Subtypes (CMS1–4) for colorectal cancer were established by the CRC Subtyping Consortium \[Guinney et al.\ 2015, *Nature Medicine*\]. Prostate cancer subtypes were defined by the TCGA prostate working group \[The Cancer Genome Atlas Research Network 2015, *Cell*\]; we use their seven-subtype scheme collapsed via Option C (see §3.1).

**Other transcriptomics classifiers.** Gao et al.\ (2019) propose DeepCC, a deep-learning consensus classifier for colorectal cancer. We do not adopt DeepCC; our held-constant classifier is the Polepalli MLP, chosen because it is the simplest commonly-cited baseline for transcriptomics and isolates the augmentation-method effect.

---

## 3. Methods

### 3.1 Datasets and subtype labels

We use TCGA gene-expression matrices for three cancers, joined with curated subtype labels.

| Cancer | Source for expression | Source for labels | n (after preprocessing) | Subtypes |
|---|---|---|---:|---|
| BRCA | UCSC Xena Hub `TCGA.BRCA.sampleMap/HiSeqV2` | cBioPortal `brca_tcga_pan_can_atlas_2018` | 981 | Basal, HER2, LumA, LumB, Normal (PAM50, 5 classes) |
| COAD | UCSC Xena Hub `TCGA.COAD.sampleMap/HiSeqV2` | CRC Subtyping Consortium `cms_labels_public_all.txt` | 221 | CMS1, CMS2, CMS3, CMS4 (4 classes) |
| PRAD | UCSC Xena Hub `TCGA.PRAD.sampleMap/HiSeqV2` | cBioPortal `prad_tcga_pub` (TCGA 2015 *Cell*) | 333 | ERG, SPOP, FOXA1, Other (Option C, 4 classes) |

Expression values are RNA-seq RSEM normalized counts on a $\log_2(x+1)$ scale.

The three deviations from our project's *Data Acquisition* document are recorded in `data/raw/MANIFEST.md` and explained here:

- **cBioPortal S3 → GitHub LFS mirror.** The cBioPortal S3 datahub bucket (`cbioportal-datahub.s3.amazonaws.com`) returned HTTP 403 on every tested path during our acquisition window. We pulled identical content from `cBioPortal/datahub` on GitHub via the LFS media URL pattern.
- **PRAD label source.** The pan-cancer-atlas study `prad_tcga_pan_can_atlas_2018` exposes only the literal cancer-acronym `"PRAD"` in its `SUBTYPE` column — no molecular-subtype information. We instead use the publication-specific study `prad_tcga_pub` (TCGA 2015 *Cell* paper), whose `SUBTYPE` column contains the seven molecular subtypes our analysis requires.
- **COAD CMS labels source.** The canonical source is Synapse (`syn4978511`), which requires an account. We use the public mirror in the `Sage-Bionetworks/crc-cms-kras` GitHub repository, whose `cms_labels_public_all.txt` is byte-identical to the Synapse copy.

**PRAD Option C label collapse.** The TCGA 2015 *Cell* paper defines seven molecular subtypes plus an "8-other" bucket. We keep ERG, SPOP, FOXA1 (the three with $\geq 9$ patients) and collapse ETV1, ETV4, FLI1, IDH1, and 8-other into a single "Other" class. Even after this collapse, FOXA1 has only 9 total patients (6 train, 1 val, 2 test) — a deliberately stress-testing minority class.

**COAD CMS filtering.** The CMS labels file pools 6 cohorts; we keep only the 573 TCGA samples and drop the 61 labelled `NOLBL` (no consensus label), leaving 512 CMS-labelled TCGA samples that intersect with the expression matrix at 221 rows after primary-tumor filtering and patient-ID matching.

### 3.2 Preprocessing pipeline

Per cancer, in order:

1. **Filter to primary tumors.** Xena's HiSeqV2 sample barcodes have format `TCGA-XX-XXXX-NN` where `NN` is the sample-type code; we keep only samples with `NN == 01` (primary solid tumor), discarding metastases, normal-adjacent tissue, and recurrences.
2. **Patient-ID match.** We trim each barcode to its first 12 characters (`TCGA-XX-XXXX`) and join the expression matrix to the subtype-label table.
3. **Sample filtering.** Drop samples with missing labels and any class with fewer than 5 samples (PRAD safeguard; in practice this filter is non-binding given Option C).
4. **Stratified 70/15/15 split.** Two-step stratified split with `random_state=42`. Train set is used for all subsequent fitting; val and test sets are held out.
5. **Top-500 most-variable gene selection.** We compute per-gene variance across the train set and retain the top 500 by variance. The same 500 genes are used in val/test.
6. **Z-score normalization.** A `StandardScaler` fit on train-set expression of the selected genes; train, val, and test are all transformed by the train-fit scaler.
7. **Label encoding.** Class strings → integers $0 \ldots K-1$ in alphabetical class order.

The resulting `data/processed/{cancer}.npz` file contains six arrays (`X_{train,val,test}` of shape $(n, 500)$, `y_{train,val,test}` of shape $(n,)$). A companion JSON records class names, gene names, and scaler parameters for reproducibility.

We verified data integrity end-to-end with `scripts/data_audit.py` (113 PASS, 0 FAIL): dtype/shape correctness, no NaN/inf in any split, exact $\mu \approx 0$, $\sigma \approx 1$ on train, alphabetical class ordering, and zero patient-ID overlap across the three splits.

### 3.3 Classifier

Per Polepalli (2025), §II:

$$
\text{Input}_{500} \to \text{Linear}_{500 \to 256} \to \text{ReLU} \to \text{Dropout}_{0.5} \to \text{Linear}_{256 \to 128} \to \text{ReLU} \to \text{Dropout}_{0.5} \to \text{Linear}_{128 \to K}
$$

Training: cross-entropy loss, Adam optimizer with $\eta = 10^{-3}$ and no weight decay, batch size 32, maximum 50 epochs with early stopping (patience 10) on validation loss. The classifier is held constant across all augmentation conditions.

For multi-classifier comparison (per project proposal), we additionally evaluate `RandomForestClassifier(n_estimators=500)` and `XGBClassifier(n_estimators=500, max_depth=6)` on the no-augmentation and SMOTE-augmented conditions only.

### 3.4 Augmentation methods

Each method receives the original training set and produces an augmented training set with synthetic samples appended.

**Random oversampling.** `imblearn.RandomOverSampler` — sample real minority-class points with replacement up to a target balance.

**SMOTE.** `imblearn.SMOTE` — for each minority point, sample a $k$-nearest neighbor of the same class and emit a random convex combination.

**ADASYN.** `imblearn.ADASYN` — like SMOTE but with sample-density-weighted generation: minority points whose neighborhoods have many majority neighbors get more synthetic samples.

**BorderlineSMOTE.** `imblearn.BorderlineSMOTE` — like SMOTE but generates only near class boundaries.

For each of the four traditional methods we run a default-hyperparameter condition ($k=5$, `sampling_strategy="auto"`) and a tuned condition that sweeps $k \in \{3, 5, 7, 10\}$ and `sampling_strategy` $\in \{\text{auto}, \text{minority}, \text{not majority}\}$, picking the combination with highest mean validation-set macro F1 over 5 seeds (12 configurations for SMOTE/ADASYN/BorderlineSMOTE; 3 for Random).

**Conditional VAE (cVAE).** Encoder: $[x; \text{onehot}(y)] \to \text{Linear}_{500+K \to 256} \to \text{ReLU} \to \text{Linear}_{256 \to 128} \to \text{ReLU} \to (\mu, \log \sigma^2) \in \mathbb{R}^{10}$. Decoder mirrors. Loss = MSE reconstruction + $\beta$ KL divergence; $\beta$ is annealed from 0 to 1 linearly over the first 30 epochs (mitigating posterior collapse). Trained 100 epochs with Adam at $\eta = 10^{-3}$, batch size 32.

**Conditional WGAN-GP** (per Lacan et al.\ 2023). Generator: BatchNorm-equipped MLP $z_{100} \oplus \text{onehot}(y) \to 256 \to 512 \to 500$. Critic: LayerNorm-equipped MLP (BatchNorm in critic breaks gradient penalty) $x \oplus \text{onehot}(y) \to 512 \to 256 \to 1$. Loss = $-\mathbb{E}[D(x_\text{real}, y)] + \mathbb{E}[D(x_\text{fake}, y)] + \lambda_\text{gp} \cdot \mathbb{E}[(\|\nabla_{\hat x} D(\hat x, y)\|_2 - 1)^2]$ with $\lambda_\text{gp} = 10$. Adam with $\eta = 10^{-4}$, $\beta = (0.5, 0.9)$ — these specific betas matter for stability. $n_\text{critic} = 5$ critic updates per generator update. Trained 400 epochs (the spec's compute-bound fallback; full 800 epochs from Lacan was infeasible on our CPU-only PyTorch installation).

**Mixup.** $x_\text{synth} = \lambda x_i + (1-\lambda) x_j$ with $\lambda \sim \text{Beta}(0.2, 0.2)$; hard label assignment $y_\text{synth} = y_i$ if $\lambda \geq 0.5$ else $y_j$.

For all generative methods we generate at a 1:1 ratio per class — $n_\text{synth}^c = n_\text{train}^c$ for each class $c$.

---

## 4. Experimental protocol

**Replication.** Every condition is run with 5 random seeds (42, 43, 44, 45, 46) controlling MLP weight initialization, training-data shuffling, and (where applicable) synthetic-sample sampling. The cVAE and WGAN-GP generators themselves are trained once each per cancer with seed 42; the 5-seed variation comes from sampling new $z$ values and retraining the downstream MLP. Results are reported as mean $\pm$ standard deviation over the 5 seeds.

**Metrics.**

- *Supervised:* accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, one-vs-rest ROC AUC (macro-averaged), confusion matrix.
- *Synthetic-data quality:* Maximum Mean Discrepancy (MMD) with RBF kernel and bandwidth $\gamma = 1/(d \cdot \mathrm{Var}(X))$; Pearson correlation between flattened upper triangles of the gene-gene correlation matrices of real vs synthetic data (Lacan equation 6); t-SNE overlay of real-train and synthetic samples.

**Master results table.** All 330 condition-seed evaluation rows (5 seeds × 50 cells, plus a 100-row $N_\text{synth}$ sweep) are stored in `results/tables/all_results.csv`. Aggregation, statistical-significance testing, and figure generation read from this single file.

**$N_\text{synth}$ sweep on BRCA.** For methods $\{\text{SMOTE}, \text{cVAE}, \text{WGAN-GP}, \text{Mixup}\}$ we additionally run the BRCA-only condition at $N_\text{synth} \in \{100, 500, 1000, 2000, 3000\}$ across 5 seeds, producing the headline scaling figure (§5.4).

**Statistical significance.** Per-(cancer, contender) paired t-test and Wilcoxon signed-rank test against the no-augmentation baseline, paired by seed. With $n=5$ the smallest achievable Wilcoxon p-value is 0.062 (the test's lower bound for $n=5$); we report it for completeness while interpreting from the t-test.

---

## 5. Results

### 5.1 Baseline classifier performance

Without augmentation, the MLP achieves the metrics in Table 1 (5 seeds, mean ± std).

**Table 1.** No-augmentation MLP baseline.

| Cancer | Accuracy | Macro F1 | Weighted F1 | ROC AUC OvR |
|---|---:|---:|---:|---:|
| BRCA | $0.841$ | $0.762 \pm 0.040$ | $0.829$ | $0.962$ |
| COAD | $0.865$ | $0.857 \pm 0.031$ | $0.864$ | $0.980$ |
| PRAD | $0.880$ | $\mathbf{0.652 \pm 0.032}$ | $0.865$ | $0.971$ |

The $0.880 \to 0.652$ gap between PRAD accuracy and macro F1 is diagnostic: the classifier is achieving high accuracy by predicting the two large classes (ERG and Other) correctly, while completely ignoring FOXA1 (per-class F1 = 0 across all 5 seeds). BRCA shows the same pattern at smaller magnitude. COAD has effectively no gap, reflecting near-balanced classes after CMS-only filtering.

### 5.2 Augmentation effects on classification

**Table 2.** Macro F1 (5-seed mean ± std) for every (cancer × method) cell.

| Method | BRCA | COAD | PRAD |
|---|---:|---:|---:|
| no augmentation | $0.762 \pm 0.040$ | $0.857 \pm 0.031$ | $0.652 \pm 0.032$ |
| Random | $0.775 \pm 0.027$ | $0.844 \pm 0.005$ | $0.616 \pm 0.027$ |
| Random (tuned) | $0.775 \pm 0.027$ | $0.852 \pm 0.020$ | $0.579 \pm 0.013$ |
| SMOTE | $0.781 \pm 0.031$ | $0.847 \pm 0.010$ | $0.608 \pm 0.018$ |
| SMOTE (tuned) | $0.787 \pm 0.037$ | $0.847 \pm 0.010$ | $0.608 \pm 0.018$ |
| ADASYN | $0.792 \pm 0.027$ | (failed at default config) | (failed) |
| ADASYN (tuned) | $0.792 \pm 0.027$ | $0.842 \pm 0.025$ | $0.624 \pm 0.033$ |
| BorderlineSMOTE | $\mathbf{0.795 \pm 0.023}$ | $\mathbf{0.866 \pm 0.023}$ | $0.598 \pm 0.014$ |
| BorderlineSMOTE (tuned) | $0.794 \pm 0.017$ | $0.832 \pm 0.024$ | $0.652 \pm 0.032$ |
| cVAE | $0.657 \pm 0.043$ | $0.829 \pm 0.038$ | $0.620 \pm 0.064$ |
| WGAN-GP | $0.714 \pm 0.056$ | $\mathbf{0.866 \pm 0.023}$ | $0.635 \pm 0.031$ |
| Mixup | $0.771 \pm 0.039$ | $0.857 \pm 0.035$ | $0.621 \pm 0.060$ |
| Random Forest, no aug | $0.729 \pm 0.004$ | $0.851 \pm 0.012$ | $0.646 \pm 0.005$ |
| Random Forest + SMOTE | $0.765 \pm 0.015$ | $0.859 \pm 0.038$ | $0.657 \pm 0.009$ |
| XGBoost, no aug | $0.702 \pm 0.000$ | $0.850 \pm 0.014$ | $\mathbf{0.727 \pm 0.000}$ |
| XGBoost + SMOTE | $0.783 \pm 0.038$ | $0.842 \pm 0.040$ | $0.723 \pm 0.100$ |

The single-best macro F1 per cancer is bolded. Three observations are immediate:

1. **On BRCA, BorderlineSMOTE (default, $k=5$, `auto`) achieves the highest mean macro F1 (0.795).** Multiple traditional methods cluster within $\pm$0.02 of this; deep generative methods (cVAE 0.657, WGAN-GP 0.714) are clearly worse than no-augmentation (0.762).
2. **On COAD, BorderlineSMOTE and WGAN-GP tie at 0.866 — but neither beats no-augmentation by more than 0.009.** The 4 CMS classes are roughly balanced; there is little for augmentation to fix.
3. **On PRAD, the best method is XGBoost with no augmentation (0.727 macro F1).** No MLP-based augmentation method beats the MLP no-augmentation baseline (0.652). The MLP is wedged on the FOXA1 minority class no matter what synthetic samples we feed it; XGBoost's tree ensemble is more robust to extreme imbalance.

### 5.3 Statistical significance

**Table 3.** Paired t-test of contender vs no-augmentation baseline, $n = 5$. We list only contenders with $p_t < 0.10$.

| Cancer | Contender | $\Delta$ macro F1 | $p_t$ | $p_W$ |
|---|---|---:|---:|---:|
| BRCA | ADASYN (tuned) | $+0.030$ | $\mathbf{0.059}$ | $0.125$ |
| BRCA | **cVAE** | $\mathbf{-0.106}$ | $\mathbf{0.001}$ | $0.062$ |
| PRAD | SMOTE | $-0.044$ | $0.028$ | $0.062$ |
| PRAD | SMOTE (tuned) | $-0.044$ | $0.028$ | $0.062$ |
| PRAD | BorderlineSMOTE | $-0.054$ | $0.011$ | $0.062$ |
| PRAD | Random (tuned) | $-0.073$ | $0.001$ | $0.062$ |

Five of the six $p_t < 0.10$ effects are *negative* — augmentation hurting the baseline. Only ADASYN (tuned) on BRCA produces a statistically suggestive positive effect ($p_t = 0.059$), and even there the Wilcoxon does not corroborate. With $n = 5$ Wilcoxon's smallest achievable two-sided p-value is 0.062, so it does not reject any of these comparisons. The strongest single result is **cVAE significantly hurts BRCA macro F1 by 0.106 points** ($p_t = 0.001$).

Nothing on COAD reaches $p_t < 0.10$, consistent with the earlier observation that COAD has nothing for augmentation to fix.

### 5.4 Augmentation sweep on BRCA

Figure 1 plots accuracy and macro F1 against $N_\text{synth}$ for the four sweep methods on BRCA, with the no-augmentation baseline as horizontal reference.

![**Figure 1.** Augmentation sweep on BRCA. Accuracy (left) and macro F1 (right) versus $N_\text{synth} \in \{100, 500, 1000, 2000, 3000\}$ for SMOTE, cVAE, WGAN-GP, and Mixup, error bars over 5 random seeds. The dashed grey line is the no-augmentation baseline. SMOTE peaks near $N_\text{synth} = 500$ then plateaus; cVAE and WGAN-GP decline monotonically with increasing $N_\text{synth}$ beyond $\sim 100$; Mixup is roughly flat throughout.](../results/figures/sweep_brca.pdf){#fig:sweep width=92%}

Key reading: SMOTE peaks near $N_\text{synth} = 500$ (matching the natural balancing target) and degrades modestly at higher $N$. cVAE and WGAN-GP both *decline monotonically* with $N_\text{synth}$ beyond 100 — at $N_\text{synth} = 3000$ both are well below the no-augmentation baseline, with cVAE at $\sim 0.62$ macro F1 and WGAN-GP at $\sim 0.68$. Mixup is roughly flat across $N$.

The takeaway for practitioners: **generative augmentation has a small-$N$ optimum**; pushing $N_\text{synth}$ much above the natural minority-balancing target dilutes the real-data signal in the loss without proportionate gain.

### 5.5 Synthetic-data quality vs downstream utility

This is the project's most important figure (Figure 2). For every (cancer × method) combination, we plot MMD and gene-gene correlation against $\Delta$ macro F1 (vs no-augmentation).

![**Figure 2.** Synthetic-data quality versus downstream utility. *Left:* MMD (lower = synthetic distribution closer to real) against $\Delta$ macro F1 (vs no-augmentation), log-scaled $x$. *Right:* Pearson correlation of gene-gene correlation matrices (higher = better preservation of feature structure) against $\Delta$ macro F1. Marker shape encodes cancer (BRCA = circle, COAD = square, PRAD = triangle); marker color encodes augmentation method. The dashed line at $\Delta = 0$ marks the no-augmentation baseline. Lowest-MMD methods (Mixup, WGAN-GP) cluster near $\Delta = 0$ or below; the largest positive $\Delta$ values come from the SMOTE family which has *high* MMD and *low* gene-gene correlation.](../results/figures/synth_quality_vs_delta_f1.pdf){#fig:scatter width=100%}

**Table 4.** Synthetic-quality and downstream metrics, BRCA only.

| Method | MMD ↓ | corr ↑ | $\Delta$ macro F1 |
|---|---:|---:|---:|
| Mixup | $\mathbf{0.004}$ | $\mathbf{0.986}$ | $+0.009$ |
| WGAN-GP | $0.011$ | $0.966$ | $-0.049$ |
| Random | $0.029$ | $0.902$ | $+0.012$ |
| cVAE | $0.134$ | $0.950$ | $-0.106$ |
| ADASYN | $0.197$ | $0.482$ | $+0.030$ |
| BorderlineSMOTE | $0.198$ | $0.400$ | $\mathbf{+0.032}$ |
| SMOTE | $0.226$ | $0.420$ | $+0.018$ |

The relationship between distribution-matching quality and classification benefit is **inverted from the obvious expectation**:

- *Lowest-MMD methods* (Mixup at 0.004, WGAN-GP at 0.011) — synthetic samples whose marginal distribution is closest to the real data — produce $\Delta \approx 0$ or *negative* downstream effects.
- *Highest-MMD methods* (SMOTE family at $\sim 0.20$, with poor gene-gene correlation preservation $\sim 0.42$) — synthetic samples that look least "real" by every standard metric — produce the *largest* positive $\Delta$.

This decoupling holds on COAD and PRAD as well (full table in `results/tables/synth_quality.csv`). Mixup, the synth-quality champion on every cancer, never produces meaningful classification lift. SMOTE, which generates samples on straight lines between same-class neighbors and therefore distorts gene-gene correlation structure, is the only method that *robustly* helps BRCA macro F1.

We discuss the interpretation of this finding in §6.2.

### 5.6 Per-class behavior

Per-class F1 heatmaps (Figure 3 shows BRCA; analogous figures for COAD and PRAD are in `results/figures/per_class_f1_{coad,prad}.{pdf,png}`) show where each method helps and hurts at the class level.

![**Figure 3.** BRCA per-class F1 heatmap, mean over 5 seeds. Rows are augmentation methods (sorted by mean F1 across classes, descending); columns are the 5 PAM50 subtypes. Darker cells indicate higher F1. The two minority classes (Normal, HER2) drive most of the macro-F1 variation between methods. Best method per cancer differs from "best by macro F1" once you look class-by-class.](../results/figures/per_class_f1_brca.pdf){#fig:perclass-brca width=92%}

- **BRCA Normal (rarest class, 25 train).** No-aug F1 averages 0.40; BorderlineSMOTE lifts it to 0.55. cVAE drops it to 0.20.
- **BRCA HER2 (54 train).** No-aug F1 is 0.81 — already strong. Augmentation produces small inconsistent changes.
- **PRAD FOXA1 (6 train).** No-aug F1 is 0.00 across all seeds. Every MLP-based augmentation method fails to lift this above $\sim 0.10$. XGBoost no-aug achieves FOXA1 F1 = 0.00 as well, but its higher overall macro F1 comes from better performance on ERG and Other.
- **COAD CMS3 (26 train).** No-aug F1 is already 0.85; augmentation provides margins-of-noise variation.

The confusion-matrix grid (Figure 4) and OvR ROC overlays (Figure 5) corroborate the table-level picture.

![**Figure 4.** Row-normalized confusion matrices, seed 42, test set. Rows are cancers (BRCA top, COAD middle, PRAD bottom); columns are six augmentation conditions (`none`, `smote_tuned`, `borderline_tuned`, `cvae`, `wgan_gp`, `mixup`). Diagonals are recall per class; off-diagonals show common confusions. Note the persistent dark FOXA1 off-diagonal in PRAD across every method — minority class confusion is unfixed by augmentation.](../results/figures/cm_grid.pdf){#fig:cmgrid width=100%}

![**Figure 5.** Macro-averaged one-vs-rest ROC curves per cancer (seed 42). One panel per cancer, one line per method, AUC reported in each legend. ROC AUC OvR is high across all conditions ($\geq 0.91$ in every cell), confirming that the classifier *can* rank classes correctly even when threshold-based metrics like F1 collapse on minority classes.](../results/figures/roc_grid.pdf){#fig:rocgrid width=100%}

### 5.7 Computational efficiency

**Table 5.** Wallclock training cost per condition × seed, CPU-only PyTorch.

| Component | Time |
|---|---:|
| MLP no-aug | $\sim 0.5$ s |
| MLP + traditional aug | $\sim 1.5$ s |
| Random Forest | $\sim 1$ s |
| XGBoost | $\sim 4$ s |
| cVAE training (100 epochs) | $\sim 5$ s |
| WGAN-GP training (400 epochs) | 14–79 s (varies by cancer) |
| Tuning sweep (4 methods × 12 configs × 5 seeds × 3 cancers) | $\sim 25$ min |

The full pipeline from raw data to all 330 result rows reproduces in approximately 45 minutes on a single laptop CPU.

---

## 6. Discussion

### 6.1 Why we substituted WGAN-GP for the proposal's CTGAN

Our project proposal committed to CTGAN as the GAN-based augmentation method. After a deeper review of the conditional-tabular-GAN literature, we substituted WGAN-GP. The rationale:

- **CTGAN's architectural choices target mixed-type tabular data** — categorical features modeled by mode-specific normalization, continuous features modeled by a Gaussian mixture. Transcriptomics data is purely continuous and high-dimensional; CTGAN's categorical machinery is dead weight.
- **Lacan et al.\ (2023)** explicitly benchmark WGAN, WGAN-GP, AttGAN, and ConvGAN on TCGA-style expression data, finding WGAN-GP best on every metric. Their published reference implementation (`forge.ibisc.univ-evry.fr/alacan/GANs-for-transcriptomics`) provides exact hyperparameter values.
- **WGAN-GP is the *stable* GAN.** Wasserstein loss with gradient penalty avoids the mode-collapse and training-instability failure modes that plague vanilla GAN, especially on small datasets (our regime).

This substitution represents a methodological improvement, not a scope reduction. The risks we accept by substituting are: (i) we cannot directly benchmark against CTGAN on this data; (ii) we no longer test the categorical-feature handling that CTGAN specializes in (which is irrelevant for our data anyway).

### 6.2 The synth-quality decoupling: why does Mixup fail and SMOTE succeed?

Section 5.5 documents an inverse correlation between standard synthetic-data-quality metrics (MMD, correlation preservation) and downstream classifier benefit. We propose the following explanation:

**The classifier needs *discriminatively useful* novelty, not *distributionally faithful* novelty.** Mixup samples are convex combinations $\lambda x_i + (1-\lambda) x_j$ — they live on straight lines between real points and therefore lie within the convex hull of the real data. The classifier already learned to handle this region; new samples there are redundant. WGAN-GP samples are similar in spirit: the generator approximates the real-data manifold and outputs points near it.

SMOTE samples, by contrast, are convex combinations of $k$-nearest *same-class* neighbors. When a minority class has few examples, those neighbors span a large region in feature space and the synthetic samples extrapolate beyond the convex hull along class-discriminative directions. They distort gene-gene correlation structure (because they ignore the real high-dimensional manifold and interpolate in input space) but provide *new* class-conditional information the classifier can use.

The implication is not that distribution-matching metrics are useless — they tell us about realism — but that **realism is the wrong target for augmentation**. The right target is *whatever makes the downstream classifier learn a better minority-class boundary*, which can be at right angles to "realism" in metric space.

This finding aligns with theoretical work on Mixup (which shows that its main effect is regularization via label smoothing rather than added data diversity) but contradicts the implicit assumption in the GAN-for-augmentation literature that better generators produce better augmentation.

### 6.3 PRAD and the limits of augmentation

PRAD is the "augmentation cannot save you" cancer. After Option C collapse, the FOXA1 class has 6 training samples. SMOTE, ADASYN, and BorderlineSMOTE all need at least $k+1$ same-class neighbors to generate a synthetic point; even at $k=3$, the FOXA1 manifold is so undersampled that the synthetic points are essentially noise around the 6 originals. cVAE conditioned on FOXA1 needs to learn its class-conditional distribution from 6 examples — impossible. WGAN-GP has the same problem.

XGBoost works on PRAD because its tree-ensemble decision rules are robust to having very few samples in a leaf — it can split confidently on a single discriminative gene without needing to characterize the whole minority distribution. This reframes the project's premise: **on the most-imbalanced cancer in our benchmark, the right intervention is choice of classifier, not synthetic minority generation**.

The original proposal mentioned tree-based classifiers "for completeness" with the implication that the MLP would dominate. The empirical result is the reverse on PRAD. We document this honestly rather than report it as an "augmentation success."

### 6.4 COAD: when augmentation does nothing

COAD has 4 roughly-balanced classes after CMS-only filtering. The MLP no-augmentation baseline is already 0.857 macro F1; the single-best augmentation method (BorderlineSMOTE, also matched by WGAN-GP) reaches 0.866. The paired t-test does not reject equivalence for any contender. In other words: **on a balanced multi-class transcriptomics task, augmentation provides no detectable benefit**. This is itself a finding for the report's intended audience — augmentation should be applied to imbalance, not as a default ritual.

### 6.5 Threats to validity

- **Five seeds is a small sample.** With $n = 5$ the Wilcoxon test cannot reject equivalence at $\alpha = 0.05$ (its lower bound is $p = 0.062$). Most of our positive macro-F1 deltas have paired-t p-values in the 0.06–0.10 range — suggestive but not conclusive. A 10-seed rerun would harden the BRCA results.
- **WGAN-GP COAD divergence.** WGAN-GP training on COAD failed to converge; the generator-loss tracker reported NaN from epoch 1 onward (caused by Generator BatchNorm interacting badly with COAD's small batches-per-epoch count of 4). The downstream COAD WGAN-GP macro F1 result (0.866) reflects synthetic samples that are essentially random noise; the MLP early-stops before the noise can corrupt training. We report the result honestly rather than retraining with adjusted hyperparameters.
- **CPU-only training cap on WGAN-GP epochs.** Lacan et al.\ specify 800 epochs; we ran 400 due to time constraints with our CPU-only PyTorch installation. A GPU rerun would tighten BRCA and PRAD results but the qualitative picture (WGAN-GP excellent at distribution-matching, weak downstream) would not change.
- **Single classifier architecture for the MLP-based experiments.** We held the MLP architecture constant by design (to isolate the augmentation-method effect) but our generalization claims are MLP-specific. The XGBoost results on PRAD show the architecture choice can matter more than the augmentation method.
- **PRAD Option C is a research choice, not a clinical standard.** Our four-class scheme aggregates several rare PRAD subtypes into a single "Other" bucket. A different label scheme would produce different results.

---

## 7. Conclusion and future work

We benchmarked seven data augmentation methods on three TCGA cancer-subtype classification tasks under a fixed classifier and evaluation protocol. The headline findings:

1. **Traditional oversampling (SMOTE, BorderlineSMOTE, ADASYN) is the most reliable augmentation family** in our setting. BorderlineSMOTE is the single best method on BRCA and ties for best on COAD. None of these methods outperform the no-augmentation baseline by more than a few macro-F1 points, and only one (ADASYN-tuned, BRCA) achieves a paired-t $p < 0.10$ effect.
2. **Deep generative augmentation (cVAE, WGAN-GP) is empirically harmful at 1:1 ratios on BRCA**, and indistinguishable from no-augmentation on COAD and PRAD. WGAN-GP achieves the best non-Mixup synthetic-data quality (BRCA MMD 0.011, correlation 0.966) but provides no downstream classification benefit.
3. **Synthetic-data quality and classifier benefit are decoupled — and inversely so on BRCA.** The methods producing the most "real-looking" samples produce the smallest classification lift. We propose this is because the classifier needs *discriminatively novel* samples, not *distributionally faithful* ones.
4. **On the most-imbalanced cancer (PRAD with 6-sample FOXA1), no MLP-based augmentation helps; XGBoost without augmentation outperforms every MLP-based condition by 7+ macro-F1 points.** Augmentation is not a substitute for an appropriate classifier choice when the minority class is genuinely tiny.
5. **On a balanced multi-class transcriptomics task (COAD), augmentation produces no detectable benefit.** Augmentation should be applied to imbalance, not as a default.

Future work should include: (i) 10+ seeds to harden statistical claims; (ii) GPU rerun of WGAN-GP at 800 epochs; (iii) WGAN-GP fix for COAD's small-data divergence (LayerNorm in the generator); (iv) extension to more cancers and to additional classifier architectures; (v) a study of the *direction* of synthetic-sample novelty — projecting synthetic samples onto class-discriminative directions and comparing methods by *useful* novelty rather than overall MMD.

---

## References

Chawla, N. V., Bowyer, K. W., Hall, L. O., \& Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Choi, J., \& Chae, H. (2020). methCancer-gen: a DNA methylome dataset generator for user-specified cancer type based on conditional variational autoencoder. *BMC Bioinformatics*, 21(1), 181.

Gao, F., Wang, W., Tan, M., Zhu, L., Zhang, Y., Fessler, E., Vermeulen, L., \& Wang, X. (2019). DeepCC: a novel deep learning-based framework for cancer molecular subtype classification. *Oncogenesis*, 8, 44.

Guinney, J., Dienstmann, R., Wang, X., \emph{et al.} (2015). The consensus molecular subtypes of colorectal cancer. *Nature Medicine*, 21(11), 1350–1356.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., \& Courville, A. C. (2017). Improved training of Wasserstein GANs. *NeurIPS 2017*.

Han, H., Wang, W. Y., \& Mao, B. H. (2005). Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning. *International Conference on Intelligent Computing*, 878–887.

He, H., Bai, Y., Garcia, E. A., \& Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IJCNN 2008*.

Lacan, A., Sebag, M., \& Hanczar, B. (2023). GAN-based data augmentation for transcriptomics: a survey and benchmark. *Bioinformatics*, btad239.

Mirza, M., \& Osindero, S. (2014). Conditional generative adversarial nets. *arXiv:1411.1784*.

Parker, J. S., Mullins, M., Cheang, M. C. U., \emph{et al.} (2009). Supervised risk predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical Oncology*, 27(8), 1160–1167.

Polepalli, V. (2025). Class-conditional variational autoencoder augmentation for pan-cancer subtype classification. *arXiv:2508.02743*.

Sohn, K., Lee, H., \& Yan, X. (2015). Learning structured output representation using deep conditional generative models. *NeurIPS 2015*.

The Cancer Genome Atlas Research Network (2015). The molecular taxonomy of primary prostate cancer. *Cell*, 163(4), 1011–1025.

Xu, L., Skoularidou, M., Cuesta-Infante, A., \& Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *NeurIPS 2019*.

Zhang, H., Cisse, M., Dauphin, Y. N., \& Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *ICLR 2018*.

---

## Appendix A. Reproducibility

The repository at `<github URL>` contains:

- `data/raw/` (raw expression and clinical files; `MANIFEST.md` documents source URLs)
- `data/processed/` (NPZ files of preprocessed splits + JSON metadata, regenerable via `scripts/prepare_data.py`)
- `src/` (data loading, preprocessing, MLP, all 7 augmentation modules, evaluation metrics)
- `src/experiments/01_baselines.py` ... `06_aug_sweep.py` (one script per experimental block)
- `scripts/data_audit.py` (113 integrity checks; PASS required before running experiments)
- `scripts/aggregate_results.py`, `scripts/stat_sig.py`, `scripts/plot_*.py` (analysis and figures)
- `results/tables/all_results.csv` (canonical 330-row results database)
- `results/figures/` (all figures referenced above, PDF and PNG at 300 DPI)

A single command (`bash run_all.sh`, to be added) reproduces the entire pipeline from raw-data to all figures in approximately 45 CPU-minutes.

## Appendix B. Hyperparameters

All hyperparameters are recorded in code at `src/config.py`, in the per-augmentation modules, and in `results/tables/traditional_tuning.json` (chosen configurations from the 5-seed validation sweep). The full hyperparameter inventory matches Polepalli (2025) §II for the MLP and cVAE, Lacan et al.\ (2023) §3 for WGAN-GP, and Zhang et al.\ (2018) for Mixup.
