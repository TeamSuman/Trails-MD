import logging
import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Tuple

# On-disk checkpoint format version. Bump when the layout changes; ``load``
# tolerates older checkpoints (a missing version file is treated as v1).
CHECKPOINT_FORMAT_VERSION = 2

# CV methods whose projection network lives in ``space_model.fitted`` as a
# torch module and is additionally snapshotted as ``model.pt``.
_TORCH_ENCODER_MODES = ("tvae", "vampnet", "spib")


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
        bin_state: Dict[str, Any],
        history: Dict[str, Any],
        sampler_state: Dict[str, Any] | None = None,
    ) -> None:
        """Save a complete state snapshot."""
        iter_dir = self.checkpoint_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        (iter_dir / "format_version").write_text(str(CHECKPOINT_FORMAT_VERSION))

        # 1. Save Space Model (TVAE or TICA)
        if space_model is not None:
            with open(iter_dir / "space_model.pkl", "wb") as f:
                pickle.dump(space_model, f)

        if hasattr(space_model, "type"):
            if (
                space_model.type in _TORCH_ENCODER_MODES
                and getattr(space_model, "fitted", None) is not None
            ):
                torch.save(space_model.fitted.state_dict(), iter_dir / "model.pt")
            elif (
                space_model.type == "tica"
                and getattr(space_model, "model", None) is not None
            ):
                with open(iter_dir / "model.pkl", "wb") as f:
                    pickle.dump(space_model.model, f)

        # 2. Save Feature Scaler
        with open(iter_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        # 3. Save Bins & Spawn History
        with open(iter_dir / "bin_state.pkl", "wb") as f:
            pickle.dump(bin_state, f)
            
        # Delta Checkpointing: Only save history since the last checkpoint
        last_ckpt = self._get_latest_checkpoint_before(iteration)
        delta_history = {
            k: v for k, v in history.items() if k > last_ckpt and k <= iteration
        }
        with open(iter_dir / "history.pkl", "wb") as f:
            pickle.dump(delta_history, f)
            
        if sampler_state is not None:
            with open(iter_dir / "sampler_state.pkl", "wb") as f:
                pickle.dump(sampler_state, f)

    def load(
        self, iteration: int, space_model: Any = None
    ) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
            
        # 3. Load bins & history
        with open(iter_dir / "bin_state.pkl", "rb") as f:
            bin_state = pickle.load(f)
            
        with open(iter_dir / "history.pkl", "rb") as f:
            history = pickle.load(f)

        # Reconstruct full history for Delta Checkpointing
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
        
        for prev_iter in previous_iters:
            prev_hist_file = self.checkpoint_dir / f"iter_{prev_iter}" / "history.pkl"
            if prev_hist_file.exists():
                with open(prev_hist_file, "rb") as f:
                    part_hist = pickle.load(f)
                    if isinstance(part_hist, dict):
                        for k, v in part_hist.items():
                            if k not in history:
                                history[k] = v

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
