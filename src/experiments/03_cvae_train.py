"""Train cVAE per cancer (Day 4 Damien).

Saves checkpoint to results/models/cvae_{cancer}.pt and a training-curves figure
to results/figures/cvae/{cancer}_curves.{pdf,png}.

Run: `python -m src.experiments.03_cvae_train`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt

from src.augmentation.cvae import cvae_checkpoint_path, train_cvae
from src.config import CANCERS, RESULTS_FIGURES, SEED, set_seed
from src.data import load_processed

FIG_DIR = RESULTS_FIGURES / "cvae"


def plot_curves(cancer: str, history: dict) -> None:
    epochs = list(range(1, len(history["recon_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["recon_loss"], label="recon (MSE / batch)")
    axes[0].plot(epochs, history["kl_loss"], label="KL / batch", color="tab:orange")
    ax2 = axes[0].twinx()
    ax2.plot(epochs, history["beta"], color="grey", linestyle=":", label="β (annealed)")
    ax2.set_ylabel("β")
    ax2.set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{cancer.upper()} cVAE training (recon + KL + β)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1].plot(epochs, history["total_loss"], color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("recon + β·KL")
    axes[1].set_title(f"{cancer.upper()} cVAE total loss")
    axes[1].grid(True, linestyle=":", alpha=0.5)

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
        print(f"  training cVAE on {len(data['X_train'])} samples × "
              f"{data['X_train'].shape[1]} genes, K={n_classes}")
        model, history = train_cvae(
            data["X_train"],
            data["y_train"],
            n_classes=n_classes,
            checkpoint_path=cvae_checkpoint_path(cancer),
            verbose=True,
        )
        plot_curves(cancer, history)
        print(
            f"  done in {history['train_time_seconds']:.1f}s — "
            f"final recon={history['recon_loss'][-1]:.3f} "
            f"kl={history['kl_loss'][-1]:.3f}"
        )


if __name__ == "__main__":
    main()
