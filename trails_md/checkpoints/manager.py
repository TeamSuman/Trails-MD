import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

# NOTE: ``torch`` is imported lazily inside the methods that need it so that
# checkpoint bookkeeping (scanning for the resume target, reconstructing the
# delta history) works in a light environment without torch — e.g. a CPU-only
# login node running ``trails-md-path`` — and only the neural-CV save/load paths
# pull it in.

# On-disk checkpoint format version. Bump when the layout changes; ``load``
# tolerates older checkpoints (a missing version file is treated as v1).
CHECKPOINT_FORMAT_VERSION = 2

# Marker file written last by ``save``; its presence is the sole signal that a
# checkpoint directory is complete and safe to resume from.
_COMPLETION_MARKER = "format_version"

# CV methods whose projection network lives in ``space_model.fitted`` as a
# torch module and is additionally snapshotted as ``model.pt``.
_TORCH_ENCODER_MODES = ("tvae", "vampnet", "spib")


def _atomic_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path`` atomically (write tmp, fsync, then os.replace).

    Prevents a crash mid-write (e.g. an HPC walltime kill) from leaving a
    truncated file — important for delta history, where every checkpoint's
    ``history.pkl`` is needed to reconstruct the full history on resume. The
    ``fsync`` before the rename means the bytes are durable even across a node
    power-loss / networked-FS client failover, not just a process kill.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _is_complete_checkpoint(iter_dir: Path) -> bool:
    """A checkpoint dir is resumable only once its completion marker exists."""
    return (iter_dir / _COMPLETION_MARKER).exists()


def _iter_number(path: Path) -> int:
    return int(path.name.removeprefix("iter_"))


def _complete_checkpoint_iters(checkpoint_root: Path) -> list[int]:
    """Iteration numbers of *complete* checkpoint directories, ascending."""
    return sorted(
        _iter_number(path)
        for path in checkpoint_root.glob("iter_*")
        if path.is_dir()
        and path.name.removeprefix("iter_").isdigit()
        and _is_complete_checkpoint(path)
    )


def reconstruct_history(checkpoint_root: Path, iteration: int) -> dict[Any, Any]:
    """Merge the per-checkpoint delta ``history.pkl`` files (for all iterations
    ``<= iteration``) into the full cumulative history.

    Each key normally lives in exactly one delta; on overlap the newer checkpoint
    wins. Incomplete (torn) checkpoint directories and unreadable deltas
    (truncated by a crash) are skipped with a warning rather than aborting the
    whole restore.
    """
    iters = [it for it in _complete_checkpoint_iters(checkpoint_root) if it <= iteration]

    # Integrity check: each checkpoint records the delta chain it depends on. If a
    # required delta was pruned/lost after the fact, the reconstructed history is
    # incomplete — surface that LOUDLY instead of silently dropping keys.
    chain_file = checkpoint_root / f"iter_{iteration}" / "history_chain.json"
    if chain_file.exists():
        try:
            required = [int(x) for x in json.loads(chain_file.read_text())]
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            required = []
        missing = [
            r
            for r in required
            if not (checkpoint_root / f"iter_{r}" / "history.pkl").exists()
        ]
        if missing:
            logging.error(
                "Delta-checkpoint history chain is broken: history deltas for "
                "iteration(s) %s are missing (pruned or lost). The reconstructed "
                "history will be INCOMPLETE and lineage/MSM over those iterations "
                "is unreliable — resume from a self-contained/earlier checkpoint.",
                missing,
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
                import torch

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

        # Record the delta chain this checkpoint depends on (all prior complete
        # checkpoints plus this one). reconstruct_history() uses it to detect and
        # loudly report a broken chain (a delta pruned/lost after the fact) rather
        # than silently returning an incomplete history.
        chain = sorted(set(_complete_checkpoint_iters(self.checkpoint_dir)) | {iteration})
        tmp = iter_dir / "history_chain.json.tmp"
        tmp.write_text(json.dumps(chain))
        os.replace(tmp, iter_dir / "history_chain.json")

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
        if not _is_complete_checkpoint(iter_dir):
            raise FileNotFoundError(
                f"Checkpoint {iter_dir} is incomplete (no '{_COMPLETION_MARKER}' "
                "marker) — it was likely truncated by a crash mid-save. Resume "
                "from an earlier complete checkpoint or remove this directory."
            )
        self._check_format_version(iter_dir)

        # 1. Load model weights
        if (iter_dir / "space_model.pkl").exists():
            with open(iter_dir / "space_model.pkl", "rb") as f:
                space_model = pickle.load(f)
        elif hasattr(space_model, "type"):
            if space_model.type in _TORCH_ENCODER_MODES:
                import torch

                if not (iter_dir / "model.pt").exists():
                    raise FileNotFoundError(
                        f"{space_model.type} model checkpoint missing in {iter_dir}"
                    )
                # map_location="cpu" so a GPU-trained checkpoint can be restored
                # on a CPU-only node (analysis / login node) or one with a
                # different GPU count; the module is moved back to its device by
                # the caller if needed.
                space_model.fitted.load_state_dict(
                    torch.load(iter_dir / "model.pt", map_location="cpu")
                )
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
        # Only *complete* checkpoints are resumable: an incomplete ``iter_N``
        # (crash mid-save, missing the completion marker) must never be selected
        # as the resume target, otherwise ``load`` reads torn/absent files.
        complete = _complete_checkpoint_iters(self.checkpoint_dir)
        if not complete:
            raise FileNotFoundError(
                f"No complete checkpoints found under {self.checkpoint_dir}"
            )
        return complete[-1]

    def _get_latest_checkpoint_before(self, iteration: int) -> int | float:
        previous = [
            it
            for it in _complete_checkpoint_iters(self.checkpoint_dir)
            if it < iteration
        ]
        return previous[-1] if previous else -float("inf")
