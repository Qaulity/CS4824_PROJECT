"""Aggregate the master CSV into report-ready tables and figures.

Inputs:
  - results/tables/all_results.csv (every experiment row)
  - data/processed/{cancer}_meta.json (class names for positional class_X_f1 cols)

Outputs:
  - results/tables/summary_main.csv      : per (cancer, method): mean ± std of metrics
  - results/tables/summary_per_class.csv : per (cancer, method, class): mean F1
  - results/tables/summary_main.md       : the summary_main as a markdown table
  - results/figures/summary_macro_f1.{pdf,png} : per-cancer bar chart of macro F1
  - results/figures/summary_accuracy.{pdf,png}

Run: `python scripts/aggregate_results.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CANCERS, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES

ALL_RESULTS = RESULTS_TABLES / "all_results.csv"
HEADLINE_METRICS = ["accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr"]


def load_class_names() -> dict[str, list[str]]:
    out = {}
    for cancer in CANCERS:
        meta = json.loads((DATA_PROCESSED / f"{cancer}_meta.json").read_text())
        out[cancer] = meta["class_names"]
    return out


def main() -> None:
    df = pd.read_csv(ALL_RESULTS)
    print(f"loaded {len(df)} rows from {ALL_RESULTS}")

    # ---- 1. Headline summary: mean ± std per (cancer, method) ----
    grouped = df.groupby(["cancer", "method"])
    means = grouped[HEADLINE_METRICS].mean().round(3)
    stds = grouped[HEADLINE_METRICS].std(ddof=1).round(3).fillna(0.0)
    n = grouped.size().rename("n")

    summary = means.copy()
    for col in HEADLINE_METRICS:
        summary[col] = (
            means[col].astype(str) + " +/- " + stds[col].astype(str)
        )
    summary["n"] = n.values
    summary_path = RESULTS_TABLES / "summary_main.csv"
    summary.to_csv(summary_path)
    print(f"wrote {summary_path}")

    # ---- 2. Markdown version of the headline table ----
    md_rows = ["| cancer | method | accuracy | macro_f1 | weighted_f1 | roc_auc_ovr | n |",
               "|---|---|---|---|---|---|---|"]
    for (cancer, method), row in summary.iterrows():
        md_rows.append(
            f"| {cancer} | {method} | "
            f"{row['accuracy']} | {row['macro_f1']} | "
            f"{row['weighted_f1']} | {row['roc_auc_ovr']} | {int(row['n'])} |"
        )
    md_path = RESULTS_TABLES / "summary_main.md"
    md_path.write_text("\n".join(md_rows) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")

    # ---- 3. Per-class F1 (mean over seeds) ----
    class_names = load_class_names()
    per_class_rows = []
    f1_cols = [c for c in df.columns if c.startswith("class_") and c.endswith("_f1")]
    for (cancer, method), sub in grouped:
        names = class_names[cancer]
        for i, cname in enumerate(names):
            col = f"class_{i}_f1"
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    per_class_rows.append(
                        {
                            "cancer": cancer,
                            "method": method,
                            "class": cname,
                            "f1_mean": float(vals.mean()),
                            "f1_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                            "n": int(len(vals)),
                        }
                    )
    per_class_df = pd.DataFrame(per_class_rows).round(3)
    per_class_path = RESULTS_TABLES / "summary_per_class.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"wrote {per_class_path}")

    # ---- 4. Per-cancer comparison bar charts (macro F1 + accuracy) ----
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    for metric in ("macro_f1", "accuracy"):
        fig, axes = plt.subplots(1, len(CANCERS), figsize=(5 * len(CANCERS), 4.5))
        if len(CANCERS) == 1:
            axes = [axes]
        for ax, cancer in zip(axes, CANCERS):
            sub = df[df["cancer"] == cancer]
            agg = sub.groupby("method")[metric].agg(["mean", "std"]).fillna(0)
            agg = agg.sort_values("mean", ascending=False)
            ax.barh(
                range(len(agg)),
                agg["mean"].values,
                xerr=agg["std"].values,
                color="tab:blue",
                alpha=0.85,
            )
            ax.set_yticks(range(len(agg)), agg.index, fontsize=8)
            ax.set_xlabel(metric)
            ax.set_title(f"{cancer.upper()} — {metric} (mean ± std over seeds)")
            ax.grid(axis="x", linestyle=":", alpha=0.5)
            ax.invert_yaxis()
        fig.tight_layout()
        stem = RESULTS_FIGURES / f"summary_{metric}"
        fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.{{pdf,png}}")

    # ---- 5. Per-class F1 heatmap per cancer ----
    for cancer in CANCERS:
        sub = per_class_df[per_class_df["cancer"] == cancer]
        if sub.empty:
            continue
        pivot = sub.pivot(index="method", columns="class", values="f1_mean")
        # Sort methods by mean F1 across classes
        pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(4 + 0.4 * len(pivot.columns), 0.3 * len(pivot) + 2))
        im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > 0.55 else "black")
        ax.set_title(f"{cancer.upper()} per-class F1 (mean over seeds)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        stem = RESULTS_FIGURES / f"per_class_f1_{cancer}"
        fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.{{pdf,png}}")

    print("\nDone.")


if __name__ == "__main__":
    main()
