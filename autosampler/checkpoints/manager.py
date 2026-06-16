import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Tuple


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
        
        # 1. Save Space Model (TVAE or TICA)
        if space_model is not None:
            with open(iter_dir / "space_model.pkl", "wb") as f:
                pickle.dump(space_model, f)

        if hasattr(space_model, "type"):
            if (
                space_model.type == "tvae"
                and getattr(space_model, "fited", None) is not None
            ):
                torch.save(space_model.fited.state_dict(), iter_dir / "model.pt")
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
        with open(iter_dir / "history.pkl", "wb") as f:
            pickle.dump(history, f)
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

        # 1. Load model weights
        if (iter_dir / "space_model.pkl").exists():
            with open(iter_dir / "space_model.pkl", "rb") as f:
                space_model = pickle.load(f)
        elif hasattr(space_model, "type"):
            if space_model.type == "tvae":
                if not (iter_dir / "model.pt").exists():
                    raise FileNotFoundError(
                        f"TVAE model checkpoint missing in {iter_dir}"
                    )
                space_model.fited.load_state_dict(torch.load(iter_dir / "model.pt"))
                space_model.fited.eval()
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

        state_path = iter_dir / "sampler_state.pkl"
        if state_path.exists():
            with open(state_path, "rb") as f:
                sampler_state = pickle.load(f)
        else:
            sampler_state = {}

        return space_model, scaler, bin_state, history, sampler_state

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
