"""Class-conditional WGAN-GP for transcriptomics (Lacan et al. 2023).

Architecture per TECHNICAL_SPEC.md §7.

Generator (BatchNorm OK in G):
    [z (100); one_hot(y)]  → Linear(100+K, 256) → BN → ReLU
                           → Linear(256, 512)   → BN → ReLU
                           → Linear(512, 500)   (no final activation)

Critic / discriminator (LayerNorm — BatchNorm breaks gradient penalty):
    [x; one_hot(y)]  → Linear(500+K, 512) → LN → LeakyReLU(0.2)
                     → Linear(512, 256)   → LN → LeakyReLU(0.2)
                     → Linear(256, 1)     (no final activation)

Training (do not change without flagging):
    Adam(lr=1e-4, betas=(0.5, 0.9))     # betas matter — defaults break stability
    n_critic = 5     λ_gp = 10     latent_dim = 100     batch = 32
    epochs = 800 (Lacan); 400 acceptable when compute-bound.
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

LATENT_DIM = 100
N_CRITIC = 5
LAMBDA_GP = 10
DEFAULT_EPOCHS = 800


# ---------------------------------------------------------------------- modules


class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, output_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def _onehot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=self.n_classes).float()

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self._onehot(y)], dim=1))


class Critic(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim + n_classes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def _onehot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=self.n_classes).float()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self._onehot(y)], dim=1))


def gradient_penalty(
    critic: Critic,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    y: torch.Tensor,
    lambda_gp: float = LAMBDA_GP,
) -> torch.Tensor:
    """WGAN-GP gradient penalty (Lacan eq. 2)."""
    batch_size = x_real.size(0)
    alpha = torch.rand(batch_size, 1, device=x_real.device)
    x_interp = alpha * x_real + (1 - alpha) * x_fake
    x_interp.requires_grad_(True)
    d_interp = critic(x_interp, y)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=x_interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ---------------------------------------------------------------------- training


def train_wgan_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    *,
    latent_dim: int = LATENT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 32,
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.5, 0.9),
    n_critic: int = N_CRITIC,
    lambda_gp: float = LAMBDA_GP,
    checkpoint_path: Path | None = None,
    log_every: int = 50,
    verbose: bool = False,
) -> tuple[Generator, dict]:
    """Train conditional WGAN-GP. Returns (generator, history)."""
    device = get_device()
    output_dim = X_train.shape[1]

    G = Generator(latent_dim, n_classes, output_dim).to(device)
    D = Critic(output_dim, n_classes).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {
        "d_loss": [],
        "g_loss": [],
        "wasserstein": [],  # E[D(real)] - E[D(fake)]
        "gp": [],
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        d_losses = []
        g_losses = []
        w_dists = []
        gps = []

        critic_step = 0
        for x_real, y_real in loader:
            x_real = x_real.to(device)
            y_real = y_real.to(device)
            bs = x_real.size(0)

            # ---- Train Critic ----
            G.eval()
            D.train()
            z = torch.randn(bs, latent_dim, device=device)
            with torch.no_grad():
                x_fake = G(z, y_real)
            d_real = D(x_real, y_real)
            d_fake = D(x_fake, y_real)
            gp = gradient_penalty(D, x_real, x_fake, y_real, lambda_gp)
            d_loss = -d_real.mean() + d_fake.mean() + gp
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())
            w_dists.append((d_real.mean() - d_fake.mean()).item())
            gps.append(gp.item())

            critic_step += 1
            if critic_step % n_critic == 0:
                # ---- Train Generator ----
                G.train()
                D.eval()
                z = torch.randn(bs, latent_dim, device=device)
                x_fake = G(z, y_real)
                g_loss = -D(x_fake, y_real).mean()
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()
                g_losses.append(g_loss.item())

        history["d_loss"].append(float(np.mean(d_losses)))
        history["g_loss"].append(float(np.mean(g_losses)) if g_losses else float("nan"))
        history["wasserstein"].append(float(np.mean(w_dists)))
        history["gp"].append(float(np.mean(gps)))

        if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                f"  epoch {epoch:4d}  D={history['d_loss'][-1]:+.3f}  "
                f"G={history['g_loss'][-1]:+.3f}  "
                f"W~={history['wasserstein'][-1]:+.3f}  "
                f"GP={history['gp'][-1]:.3f}"
            )

    history["train_time_seconds"] = time.time() - t0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": G.state_dict(),
                "latent_dim": G.latent_dim,
                "n_classes": G.n_classes,
                "output_dim": G.output_dim,
            },
            checkpoint_path,
        )

    return G, history


# ---------------------------------------------------------------------- generation


@torch.no_grad()
def generate(generator: Generator, n_per_class: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Sample z ~ N(0, I) per class, decode through G."""
    generator.eval()
    device = next(generator.parameters()).device
    Xs, ys = [], []
    for cls, n in n_per_class.items():
        if n <= 0:
            continue
        z = torch.randn(n, generator.latent_dim, device=device)
        y = torch.full((n,), cls, dtype=torch.long, device=device)
        Xs.append(generator(z, y).cpu().numpy())
        ys.append(np.full(n, cls, dtype=np.int64))
    if not Xs:
        return (
            np.zeros((0, generator.output_dim), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def load_generator(checkpoint_path: Path) -> Generator:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    G = Generator(
        latent_dim=ck["latent_dim"],
        n_classes=ck["n_classes"],
        output_dim=ck["output_dim"],
    )
    G.load_state_dict(ck["state_dict"])
    return G


def wgan_checkpoint_path(cancer: str) -> Path:
    """Per spec: results/models/wgan_gp_{cancer}.pt"""
    return RESULTS_MODELS / f"wgan_gp_{cancer}.pt"
