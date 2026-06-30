import os
import random

import numpy as np
import torch


class SeedManager:
    """Seed every RNG backend AutoSampler touches for reproducible runs.

    Covers Python, NumPy, and PyTorch (CPU + CUDA), and — when available —
    PyTorch Lightning (used by deep-TICA/LDA) via ``seed_everything``. cuDNN is
    put in deterministic mode. Note that exact bitwise reproducibility across
    different hardware/MD engines is not guaranteed (floating-point and
    nondeterministic CUDA kernels); seeding makes runs as deterministic as the
    backends allow.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def set_seed(self) -> None:
        """Initialise random seeds globally across all relevant backends."""
        # 1. Standard Python library
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # 2. NumPy backend
        np.random.seed(self.seed)

        # 3. PyTorch (CPU & GPU)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Request deterministic algorithm implementations where available. Some
        # CUDA kernels need this workspace setting to be deterministic; warn_only
        # avoids hard-failing on the rare op that lacks a deterministic variant.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - older torch without warn_only support
            pass

        # 4. PyTorch Lightning (deep-TICA / deep-LDA), if installed.
        self._seed_lightning()

    def _seed_lightning(self) -> None:
        for module in ("lightning.pytorch", "pytorch_lightning", "lightning"):
            try:
                mod = __import__(module, fromlist=["seed_everything"])
            except ImportError:
                continue
            seed_everything = getattr(mod, "seed_everything", None)
            if callable(seed_everything):
                seed_everything(self.seed, workers=True)
                return
