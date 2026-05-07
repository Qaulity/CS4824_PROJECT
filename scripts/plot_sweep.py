"""Plot the augmentation sweep on BRCA — the report's headline figure.

For each method ∈ {smote, cvae, wgan_gp, mixup}, plot accuracy and macro-F1
vs N_synth with error bars from 3 seeds. Adds the no-aug baseline as a
horizontal reference line.

Output: results/figures/sweep_brca.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_FIGURES, RESULTS_TABLES

ALL_RESULTS = RESULTS_TABLES / "all_results.csv"
N_VALUES = (100, 500, 1000, 2000, 3000)
METHODS = ("smote_sweep", "cvae_sweep", "wgan_gp_sweep", "mixup_sweep")
PRETTY = {
    "smote_sweep": "SMOTE",
    "cvae_sweep": "cVAE",
    "wgan_gp_sweep": "WGAN-GP",
    "mixup_sweep": "Mixup",
}


def main() -> None:
    df = pd.read_csv(ALL_RESULTS)
    df = df[df["cancer"] == "brca"]

    base = df[df["method"] == "none"]
    base_acc = base["accuracy"].mean()
    base_f1 = base["macro_f1"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")

    for ax, metric, base_val in (
        (axes[0], "accuracy", base_acc),
        (axes[1], "macro_f1", base_f1),
    ):
        for i, method in enumerate(METHODS):
            sub = df[df["method"] == method]
            xs, means, stds = [], [], []
            for n in N_VALUES:
                rows = sub[(sub["n_synth"] >= n - 50) & (sub["n_synth"] <= n + 50)]
                if rows.empty:
                    continue
                xs.append(n)
                means.append(rows[metric].mean())
                stds.append(rows[metric].std(ddof=1))
            ax.errorbar(
                xs, means, yerr=stds,
                marker="o", capsize=4, label=PRETTY[method], color=cmap(i),
            )
        ax.axhline(base_val, color="grey", linestyle="--", linewidth=1.0,
                   label=f"no-aug baseline ({base_val:.3f})")
        ax.set_xlabel(r"$N_{\mathrm{synth}}$")
        ax.set_ylabel(metric)
        ax.set_title(f"BRCA — {metric} vs N_synth")
        ax.set_xscale("log")
        ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)

    fig.tight_layout()
    stem = RESULTS_FIGURES / "sweep_brca"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
