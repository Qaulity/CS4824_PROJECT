"""Synth-quality vs downstream-utility scatter — visualizes the project's core finding.

X axis: MMD (lower = synth distribution closer to real)
Y axis: Δ macro F1 vs no-aug baseline (positive = augmentation helps)

Each point is (cancer × method); marker shape encodes cancer, color encodes method.
A reference line at Δ=0 marks no-aug parity.

Reads results/tables/synth_quality.csv + the master CSV. Output:
    results/figures/synth_quality_vs_delta_f1.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_FIGURES, RESULTS_TABLES

ALL_RESULTS = RESULTS_TABLES / "all_results.csv"
SYNTH_QUALITY = RESULTS_TABLES / "synth_quality.csv"

CANCER_MARKER = {"brca": "o", "coad": "s", "prad": "^"}
METHOD_COLOR = {
    "random": "tab:gray",
    "smote": "tab:blue",
    "adasyn": "tab:cyan",
    "borderline": "tab:purple",
    "mixup": "tab:olive",
    "cvae": "tab:orange",
    "wgan_gp": "tab:red",
}
PRETTY = {
    "random": "Random", "smote": "SMOTE", "adasyn": "ADASYN",
    "borderline": "BorderlineSMOTE", "mixup": "Mixup",
    "cvae": "cVAE", "wgan_gp": "WGAN-GP",
}


def main() -> None:
    runs = pd.read_csv(ALL_RESULTS)
    sq = pd.read_csv(SYNTH_QUALITY)

    # Per (cancer, method) mean macro F1 from runs
    runs_agg = runs.groupby(["cancer", "method"])["macro_f1"].mean().reset_index()
    base = runs_agg[runs_agg["method"] == "none"].set_index("cancer")["macro_f1"]

    # Build the joined frame: each (cancer, method) gets MMD, corr, Δmacro_f1
    rows = []
    for _, sq_row in sq.iterrows():
        cancer = sq_row["cancer"]
        method = sq_row["method"]
        mmd = sq_row["mmd"]
        corr = sq_row["correlation_score"]
        if pd.isna(mmd) or method not in METHOD_COLOR:
            continue
        # Find macro_f1 for this method (or its _tuned twin if base method had no run)
        m = runs_agg[(runs_agg["cancer"] == cancer) & (runs_agg["method"] == method)]
        if m.empty:
            continue
        f1 = float(m["macro_f1"].iloc[0])
        delta = f1 - float(base.get(cancer, 0))
        rows.append({
            "cancer": cancer, "method": method, "mmd": mmd, "corr": corr,
            "delta_macro_f1": delta, "macro_f1": f1,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (xcol, xlabel, xscale) in enumerate(
        [("mmd", "MMD vs real (lower = closer)", "log"),
         ("corr", "Pearson corr of gene-gene matrices", "linear")]
    ):
        ax = axes[ax_idx]
        for _, r in df.iterrows():
            ax.scatter(
                r[xcol], r["delta_macro_f1"],
                marker=CANCER_MARKER[r["cancer"]],
                color=METHOD_COLOR[r["method"]],
                s=110, edgecolors="black", linewidth=0.7, alpha=0.9,
            )
            ax.annotate(
                f"{r['cancer'].upper()}/{PRETTY[r['method']]}",
                (r[xcol], r["delta_macro_f1"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, label="no-aug baseline")
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ macro F1 (vs no-aug)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(
            ("Distribution match (MMD)" if xcol == "mmd" else "Correlation preservation") +
            r" vs downstream $\Delta$F1"
        )

    # Two custom legends: cancer (markers) + method (colors)
    cancer_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
                   markeredgecolor="black", markersize=10, label=c.upper())
        for c, m in CANCER_MARKER.items()
    ]
    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=9, label=PRETTY[m])
        for m, c in METHOD_COLOR.items()
    ]
    leg1 = axes[0].legend(handles=cancer_handles, title="cancer",
                          loc="upper right", fontsize=8)
    axes[0].add_artist(leg1)
    axes[1].legend(handles=method_handles, title="method",
                   loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(
        "Synth-quality vs downstream utility — distribution-matching does not predict classifier benefit",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    stem = RESULTS_FIGURES / "synth_quality_vs_delta_f1"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png}}")
    print()
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
