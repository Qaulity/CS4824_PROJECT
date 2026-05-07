"""Polepalli-style MLP classifier (TECHNICAL_SPEC.md §4).

Architecture: 500 → 256 (ReLU, dropout 0.5) → 128 (ReLU, dropout 0.5) → K.
Optimizer: Adam, lr=1e-3, no weight decay. Loss: CrossEntropyLoss.
Batch size 32, max 50 epochs, early stopping on val loss with patience 10.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULTS_MODELS, get_device
from src.evaluation.metrics import supervised_metrics


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 10,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> dict:
    """Train with early stopping on val loss.

    Returns history with `train_loss`, `val_loss`, `train_acc`, `val_acc` per epoch
    plus `best_epoch` and `train_time_seconds`. Mutates `model` in place to hold
    the best (lowest val-loss) weights.
    """
    import time

    device = get_device()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        n_seen = 0
        loss_sum = 0.0
        correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            n_seen += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        train_loss = loss_sum / n_seen
        train_acc = correct / n_seen

        model.eval()
        n_seen = 0
        loss_sum = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                n_seen += yb.size(0)
                loss_sum += loss.item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss = loss_sum / n_seen
        val_acc = correct / n_seen

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"  epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["train_time_seconds"] = time.time() - t0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
    return history


def evaluate_mlp(
    model: MLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Evaluate on test set and return the spec metrics dict."""
    device = get_device()
    model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device))
        proba = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = proba.argmax(axis=1)
    return supervised_metrics(y_test, y_pred, proba, class_names)


def checkpoint_path_for(cancer: str, method: str, seed: int) -> Path:
    """Canonical checkpoint path: results/models/mlp_{cancer}_{method}_seed{seed}.pt"""
    return RESULTS_MODELS / f"mlp_{cancer}_{method}_seed{seed}.pt"
