import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch

# On-disk checkpoint format version. Bump when the layout changes; ``load``
# tolerates older checkpoints (a missing version file is treated as v1).
CHECKPOINT_FORMAT_VERSION = 2

# CV methods whose projection network lives in ``space_model.fitted`` as a
# torch module and is additionally snapshotted as ``model.pt``.
_TORCH_ENCODER_MODES = ("tvae", "vampnet", "spib")


def _atomic_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path`` atomically (write tmp, then os.replace).

    Prevents a crash mid-write (e.g. an HPC walltime kill) from leaving a
    truncated file — important for delta history, where every checkpoint's
    ``history.pkl`` is needed to reconstruct the full history on resume.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle)
    os.replace(tmp, path)


def reconstruct_history(checkpoint_root: Path, iteration: int) -> dict[Any, Any]:
    """Merge the per-checkpoint delta ``history.pkl`` files (for all iterations
    ``<= iteration``) into the full cumulative history.

    Each key normally lives in exactly one delta; on overlap the newer checkpoint
    wins. Unreadable deltas (truncated by a crash) are skipped with a warning
    rather than aborting the whole restore.
    """
    iters = sorted(
        int(path.name.removeprefix("iter_"))
        for path in checkpoint_root.glob("iter_*")
        if path.is_dir()
        and path.name.removeprefix("iter_").isdigit()
        and int(path.name.removeprefix("iter_")) <= iteration
    )
    full: dict[Any, Any] = {}
    for it in iters:
        hist_file = checkpoint_root / f"iter_{it}" / "history.pkl"
        if not hist_file.exists():
            continue
        try:
            with open(hist_file, "rb") as handle:
                part = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - tolerate a corrupt delta
            logging.warning("Skipping unreadable history delta %s: %s", hist_file, exc)
            continue
        if isinstance(part, dict):
            full.update(part)
    return full


class CheckpointManager:
    """Handles serialization and reconstruction of the sampler state to allow exact deterministic restarts."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        iteration: int,
        space_model: Any,
        scaler: Any,
        bin_state: dict[str, Any],
        history: dict[str, Any],
        sampler_state: dict[str, Any] | None = None,
    ) -> None:
        """Save a complete state snapshot."""
        iter_dir = self.checkpoint_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # 1. Save Space Model (TVAE or TICA)
        if space_model is not None:
            _atomic_pickle(space_model, iter_dir / "space_model.pkl")

        if hasattr(space_model, "type"):
            if (
                space_model.type in _TORCH_ENCODER_MODES
                and getattr(space_model, "fitted", None) is not None
            ):
                tmp = iter_dir / "model.pt.tmp"
                torch.save(space_model.fitted.state_dict(), tmp)
                os.replace(tmp, iter_dir / "model.pt")
            elif (
                space_model.type == "tica"
                and getattr(space_model, "model", None) is not None
            ):
                _atomic_pickle(space_model.model, iter_dir / "model.pkl")

        # 2. Save Feature Scaler
        _atomic_pickle(scaler, iter_dir / "scaler.pkl")

        # 3. Save Bins & Spawn History
        _atomic_pickle(bin_state, iter_dir / "bin_state.pkl")

        # Delta checkpointing: each file stores only the history since the last
        # checkpoint. load()/reconstruct_history() merge the deltas back into the
        # full history. Writes are atomic so a crash can't truncate a delta and
        # break the chain.
        last_ckpt = self._get_latest_checkpoint_before(iteration)
        delta_history = {
            k: v for k, v in history.items() if k > last_ckpt and k <= iteration
        }
        _atomic_pickle(delta_history, iter_dir / "history.pkl")

        if sampler_state is not None:
            _atomic_pickle(sampler_state, iter_dir / "sampler_state.pkl")

        # Write the format marker last: its presence signals a complete checkpoint.
        tmp = iter_dir / "format_version.tmp"
        tmp.write_text(str(CHECKPOINT_FORMAT_VERSION))
        os.replace(tmp, iter_dir / "format_version")

    def load(
        self, iteration: int, space_model: Any = None
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Restore the state exactly as it was at the specified iteration."""
        iter_dir = self.checkpoint_dir / f"iter_{iteration}"
        if not iter_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint for iteration {iteration} not found at {iter_dir}"
            )
        self._check_format_version(iter_dir)

        # 1. Load model weights
        if (iter_dir / "space_model.pkl").exists():
            with open(iter_dir / "space_model.pkl", "rb") as f:
                space_model = pickle.load(f)
        elif hasattr(space_model, "type"):
            if space_model.type in _TORCH_ENCODER_MODES:
                if not (iter_dir / "model.pt").exists():
                    raise FileNotFoundError(
                        f"{space_model.type} model checkpoint missing in {iter_dir}"
                    )
                space_model.fitted.load_state_dict(torch.load(iter_dir / "model.pt"))
                space_model.fitted.eval()
            elif space_model.type == "tica":
                if not (iter_dir / "model.pkl").exists():
                    raise FileNotFoundError(
                        f"TICA model checkpoint missing in {iter_dir}"
                    )
                with open(iter_dir / "model.pkl", "rb") as f:
                    space_model.model = pickle.load(f)

        # 2. Load scaler
        with open(iter_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # 3. Load bins & reconstruct the full (delta-checkpointed) history.
        with open(iter_dir / "bin_state.pkl", "rb") as f:
            bin_state = pickle.load(f)

        history = reconstruct_history(self.checkpoint_dir, iteration)

        state_path = iter_dir / "sampler_state.pkl"
        if state_path.exists():
            with open(state_path, "rb") as f:
                sampler_state = pickle.load(f)
        else:
            sampler_state = {}

        return space_model, scaler, bin_state, history, sampler_state

    @staticmethod
    def _check_format_version(iter_dir: Path) -> None:
        version_file = iter_dir / "format_version"
        version = 1
        if version_file.exists():
            try:
                version = int(version_file.read_text().strip())
            except ValueError:
                version = 1
        if version > CHECKPOINT_FORMAT_VERSION:
            logging.warning(
                "Checkpoint %s was written with format version %d, newer than the "
                "supported version %d; restore may be incomplete.",
                iter_dir,
                version,
                CHECKPOINT_FORMAT_VERSION,
            )

    def latest_iteration(self) -> int:
        checkpoint_dirs = [
            path
            for path in self.checkpoint_dir.glob("iter_*")
            if path.is_dir() and path.name.removeprefix("iter_").isdigit()
        ]
        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"No checkpoints found under {self.checkpoint_dir}"
            )
        return max(int(path.name.removeprefix("iter_")) for path in checkpoint_dirs)

    def _get_latest_checkpoint_before(self, iteration: int) -> int | float:
        checkpoint_dirs = [
            path
            for path in self.checkpoint_dir.glob("iter_*")
            if path.is_dir() and path.name.removeprefix("iter_").isdigit()
        ]
        previous_iters = sorted([
            int(path.name.removeprefix("iter_"))
            for path in checkpoint_dirs
            if int(path.name.removeprefix("iter_")) < iteration
        ], reverse=True)
        return previous_iters[0] if previous_iters else -float('inf')
