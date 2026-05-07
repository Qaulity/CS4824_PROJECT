"""Train conditional WGAN-GP per cancer (Day 5–6 Damien).

Uses 400 epochs (the spec's compute-bound fallback) since CPU-only torch.
Saves generator checkpoint to results/models/wgan_gp_{cancer}.pt and Wasserstein-
distance + GP curves to results/figures/wgan_gp/{cancer}_curves.{pdf,png}.

Run: `python -m src.experiments.04_wgan_gp_train`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt

from src.augmentation.wgan_gp import train_wgan_gp, wgan_checkpoint_path
from src.config import CANCERS, RESULTS_FIGURES, SEED, set_seed
from src.data import load_processed

EPOCHS = 400  # Spec compute-bound fallback; 800 would be ideal on GPU.
FIG_DIR = RESULTS_FIGURES / "wgan_gp"


def plot_curves(cancer: str, history: dict) -> None:
    epochs = list(range(1, len(history["d_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["wasserstein"], label="W ≈ E[D(real)] − E[D(fake)]", color="tab:blue")
    axes[0].axhline(0, color="grey", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Wasserstein estimate")
    axes[0].set_title(f"{cancer.upper()} WGAN-GP — Wasserstein distance")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, history["d_loss"], label="critic loss", color="tab:red")
    axes[1].plot(epochs, history["g_loss"], label="generator loss", color="tab:green")
    axes[1].plot(epochs, history["gp"], label="gradient penalty", color="grey", linestyle=":")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title(f"{cancer.upper()} WGAN-GP — D / G / GP")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / f"{cancer}_curves"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    for cancer in CANCERS:
        print(f"\n=== {cancer.upper()} ===")
        data = load_processed(cancer)
        n_classes = len(data["class_names"])
        print(
            f"  training WGAN-GP on {len(data['X_train'])} samples x "
            f"{data['X_train'].shape[1]} genes, K={n_classes}, epochs={EPOCHS}"
        )
        G, history = train_wgan_gp(
            data["X_train"],
            data["y_train"],
            n_classes=n_classes,
            epochs=EPOCHS,
            checkpoint_path=wgan_checkpoint_path(cancer),
            verbose=True,
        )
        plot_curves(cancer, history)
        print(
            f"  done in {history['train_time_seconds']:.1f}s -- "
            f"final D={history['d_loss'][-1]:+.3f} "
            f"G={history['g_loss'][-1]:+.3f} "
            f"W~={history['wasserstein'][-1]:+.3f}"
        )


if __name__ == "__main__":
    main()
