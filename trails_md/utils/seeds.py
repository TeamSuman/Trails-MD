import os
import random

import numpy as np

# torch is imported lazily inside set_seed so that constructing a SeedManager
# (and importing lightweight utils) does not require torch on, e.g., a CPU-only
# analysis/login node running the path/log CLIs.


class SeedManager:
    """Seed every RNG backend Trails-MD touches for reproducible runs.

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

        # 3. PyTorch (CPU & GPU) — optional; skip cleanly if torch is absent.
        try:
            import torch
        except ImportError:
            return

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


def capture_rng_state() -> dict:
    """Snapshot the global RNG state (Python / NumPy / torch) for checkpointing.

    Restoring this on resume makes a resumed run reproduce the RNG stream — and
    therefore the spawn choices and CV-training draws — of an uninterrupted run,
    which seeding-once-at-startup does not (the uninterrupted run has consumed
    RNG through the completed iterations). torch state is stored as plain NumPy
    arrays so the checkpoint remains unpicklable-without-torch-free.
    """
    import random

    import numpy as np

    state: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state().cpu().numpy()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                s.cpu().numpy() for s in torch.cuda.get_rng_state_all()
            ]
    except Exception:  # noqa: BLE001 - torch optional / no CUDA
        pass
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore a snapshot from :func:`capture_rng_state` (best-effort)."""
    if not state:
        return
    import logging
    import random

    import numpy as np

    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except (ValueError, TypeError) as exc:
        logging.warning("Could not restore Python/NumPy RNG state on resume: %s", exc)
    if state.get("torch") is None:
        return
    try:
        import torch

        torch.set_rng_state(
            torch.from_numpy(np.asarray(state["torch"], dtype=np.uint8))
        )
        cuda = state.get("torch_cuda")
        if cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.from_numpy(np.asarray(s, dtype=np.uint8)) for s in cuda]
            )
    except Exception as exc:  # noqa: BLE001 - torch optional
        logging.warning("Could not restore torch RNG state on resume: %s", exc)
