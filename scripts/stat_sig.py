"""Statistical-significance comparison: best aug method vs no-aug baseline.

Per Day 8 (Ishaan): "Statistical significance: paired t-tests or Wilcoxon
signed-rank between baseline and best aug method per cancer".

Pairs test runs are matched by seed. With only 3 seeds the test has very low power,
but it's the spec'd analysis. We report both paired t-test and Wilcoxon.

Output: results/tables/stat_sig.csv with columns:
    cancer, baseline, contender, metric, baseline_mean, contender_mean,
    delta, t_stat, t_pvalue, w_stat, w_pvalue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_TABLES

ALL_RESULTS = RESULTS_TABLES / "all_results.csv"
OUT = RESULTS_TABLES / "stat_sig.csv"

BASELINE = "none"
METRICS = ["accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr"]


def main() -> None:
    df = pd.read_csv(ALL_RESULTS)

    # Methods to test (skip baseline + sweep methods + tree models for now)
    methods = sorted(
        m for m in df["method"].unique()
        if m != BASELINE
        and not m.endswith("_sweep")
        and not m.startswith("rf_")
        and not m.startswith("xgb_")
    )

    rows = []
    for cancer in sorted(df["cancer"].unique()):
        base = df[(df["cancer"] == cancer) & (df["method"] == BASELINE)].sort_values("seed")
        if len(base) < 2:
            continue
        for method in methods:
            sub = df[(df["cancer"] == cancer) & (df["method"] == method)].sort_values("seed")
            if len(sub) != len(base):
                continue
            for metric in METRICS:
                a = base[metric].to_numpy(dtype=float)
                b = sub[metric].to_numpy(dtype=float)
                # paired t-test (b - a)
                t_stat, t_p = stats.ttest_rel(b, a, nan_policy="omit")
                # Wilcoxon — needs non-zero diffs
                diffs = b - a
                if np.allclose(diffs, 0):
                    w_stat = float("nan")
                    w_p = 1.0
                else:
                    try:
                        w_stat, w_p = stats.wilcoxon(b, a, zero_method="wilcox")
                    except ValueError:
                        # too few samples or all zeros after zero-method handling
                        w_stat, w_p = float("nan"), float("nan")
                rows.append({
                    "cancer": cancer,
                    "baseline": BASELINE,
                    "contender": method,
                    "metric": metric,
                    "baseline_mean": round(float(np.mean(a)), 4),
                    "contender_mean": round(float(np.mean(b)), 4),
                    "delta": round(float(np.mean(b) - np.mean(a)), 4),
                    "t_stat": round(float(t_stat), 3) if np.isfinite(t_stat) else float("nan"),
                    "t_pvalue": round(float(t_p), 3) if np.isfinite(t_p) else float("nan"),
                    "w_stat": round(float(w_stat), 3) if np.isfinite(w_stat) else float("nan"),
                    "w_pvalue": round(float(w_p), 3) if np.isfinite(w_p) else float("nan"),
                })
    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out_df)} rows)")
    print()
    sig = out_df[
        (out_df["metric"] == "macro_f1") &
        (out_df["t_pvalue"] < 0.10)  # liberal cut given n=3
    ].sort_values(["cancer", "delta"], ascending=[True, False])
    if len(sig):
        print("Macro-F1 differences with t_pvalue < 0.10 (liberal):")
        print(sig.to_string(index=False))
    else:
        print("No (cancer, method) pair has macro_f1 t_pvalue < 0.10 — expected with n=3 seeds.")


if __name__ == "__main__":
    main()
