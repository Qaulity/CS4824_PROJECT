"""Project-wide constants, paths, and seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_MODELS = RESULTS / "models"

CANCERS: tuple[str, ...] = ("brca", "coad", "prad")

N_TOP_GENES = 500
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if available) PyTorch + cuDNN deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """Return torch.device for CUDA if available else CPU. Lazy-imports torch."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
