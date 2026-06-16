from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional

class Spawner(ABC):
    """Abstract Strategy interface for spawning outliers from explored space."""
    
    @abstractmethod
    def sample(self, points: np.ndarray, top_n: int, history: Optional[Dict[int, Any]] = None) -> List[int]:
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
