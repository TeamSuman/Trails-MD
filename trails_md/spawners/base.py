from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Spawner(ABC):
    """Abstract Strategy interface for spawning outliers from explored space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        self.seed = int(seed) if seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def sample(self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None) -> list[int]:
        """Select top_n points from the set of explored points.

        Args:
            points: numpy array of shape (n_points, n_features)
            top_n: number of points to select
            history: optional sampler history used by spawners that need
                cumulative state.

        Returns:
            List of indices of the selected points.
        """
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state, including RNG state."""
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore serializable state, including RNG state."""
        if state and state.get("rng_state") is not None:
            try:
                self.rng.bit_generator.state = state["rng_state"]
            except Exception:
                pass

# Factory Registry
class SpawnerFactory:
    _spawners = {}

    @classmethod
    def register(cls, name: str, spawner_cls):
        """Register a new spawner implementation."""
        cls._spawners[name] = spawner_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> Spawner:
        """Instantiate a spawner by name."""
        if name not in cls._spawners:
            raise ValueError(f"Unknown spawner method: {name}")
        return cls._spawners[name](**kwargs)
