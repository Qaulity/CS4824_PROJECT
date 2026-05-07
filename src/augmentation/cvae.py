"""Class-conditional VAE for transcriptomics (Polepalli architecture).

Architecture (TECHNICAL_SPEC.md §6):

    Encoder:
        [x; one_hot(y)]  →  Linear(500 + K, 256) → ReLU
                         →  Linear(256, 128)     → ReLU
                         →  parallel Linear(128, 10) for mu and logvar

    Reparameterize:  z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, I)

    Decoder:
        [z; one_hot(y)]  →  Linear(10 + K, 128)  → ReLU
                         →  Linear(128, 256)     → ReLU
                         →  Linear(256, 500)     (no final activation; data is z-scored)

Loss:
    recon = MSE(x_recon, x).sum() / batch
    kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch
    loss  = recon + beta * kl

KL annealing: beta linearly 0→1 over epochs 1..30, then beta=1.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULTS_MODELS, get_device

LATENT_DIM = 10
KL_ANNEAL_EPOCHS = 30


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 500,
        n_classes: int = 4,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = nn.Linear(input_dim + n_classes, 256)
        self.enc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim + n_classes, 128)
        self.dec2 = nn.Linear(128, 256)
        self.dec3 = nn.Linear(256, input_dim)

    # ------------------------------------------------------------------ utility

    def _one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=self.n_classes).float()

    # ------------------------------------------------------------------ forward

    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor):
        h = F.relu(self.enc1(torch.cat([x, y_onehot], dim=1)))
        h = F.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(torch.cat([z, y_onehot], dim=1)))
        h = F.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_oh = self._one_hot(y)
        mu, logvar = self.encode(x, y_oh)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y_oh)
        return x_recon, mu, logvar

    # ------------------------------------------------------------------ generate

    @torch.no_grad()
    def generate(self, n_per_class: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Sample z ~ N(0, I), decode conditioned on class y, return (X_synth, y_synth)."""
        device = next(self.parameters()).device
        Xs, ys = [], []
        for cls, n in n_per_class.items():
            if n <= 0:
                continue
            z = torch.randn(n, self.latent_dim, device=device)
            y = torch.full((n,), cls, dtype=torch.long, device=device)
            y_oh = self._one_hot(y)
            x = self.decode(z, y_oh).cpu().numpy()
            Xs.append(x)
            ys.append(np.full(n, cls, dtype=np.int64))
        if not Xs:
            return (
                np.zeros((0, self.input_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


# ---------------------------------------------------------------------- training


def _kl_beta(epoch: int, anneal_epochs: int = KL_ANNEAL_EPOCHS) -> float:
    if epoch >= anneal_epochs:
        return 1.0
    return float(epoch) / float(anneal_epochs)


def _vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recon_per_batch, kl_per_batch). Per-sample sums divided by batch."""
    batch = x.size(0)
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return recon, kl


def train_cvae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    *,
    latent_dim: int = LATENT_DIM,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    anneal_epochs: int = KL_ANNEAL_EPOCHS,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> tuple[CVAE, dict]:
    """Train cVAE; return (model, history). History has recon_loss/kl_loss/beta per epoch."""
    device = get_device()
    model = CVAE(
        input_dim=X_train.shape[1], n_classes=n_classes, latent_dim=latent_dim
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = {"recon_loss": [], "kl_loss": [], "beta": [], "total_loss": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        beta = _kl_beta(epoch, anneal_epochs)
        model.train()
        n_batches = 0
        ep_recon = 0.0
        ep_kl = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            x_recon, mu, logvar = model(xb, yb)
            recon, kl = _vae_loss(x_recon, xb, mu, logvar)
            loss = recon + beta * kl
            loss.backward()
            optim.step()
            n_batches += 1
            ep_recon += recon.item()
            ep_kl += kl.item()
        ep_recon /= max(1, n_batches)
        ep_kl /= max(1, n_batches)
        history["recon_loss"].append(ep_recon)
        history["kl_loss"].append(ep_kl)
        history["beta"].append(beta)
        history["total_loss"].append(ep_recon + beta * ep_kl)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d}  beta={beta:.2f}  "
                f"recon={ep_recon:.3f}  kl={ep_kl:.3f}"
            )

    history["train_time_seconds"] = time.time() - t0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "n_classes": model.n_classes,
                "latent_dim": model.latent_dim,
            },
            checkpoint_path,
        )

    return model, history


def load_cvae(checkpoint_path: Path) -> CVAE:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CVAE(
        input_dim=ck["input_dim"],
        n_classes=ck["n_classes"],
        latent_dim=ck["latent_dim"],
    )
    model.load_state_dict(ck["state_dict"])
    return model


def cvae_checkpoint_path(cancer: str) -> Path:
    """Per spec: results/models/cvae_{cancer}.pt"""
    return RESULTS_MODELS / f"cvae_{cancer}.pt"
