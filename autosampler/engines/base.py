import os
from abc import ABC, abstractmethod
from pathlib import Path


def md_subprocess_timeout() -> float | None:
    """Timeout (seconds) for external MD subprocesses, or None for no limit.

    Configured via the ``AUTOSAMPLER_MD_TIMEOUT`` environment variable so it
    propagates cleanly to walker worker processes without threading through
    engine constructors. Guards against hung ``gmx``/``pmemd`` invocations.
    """
    raw = os.environ.get("AUTOSAMPLER_MD_TIMEOUT")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class MDEngine(ABC):
    """Abstract Strategy interface for molecular dynamics execution."""

    @abstractmethod
    def prepare(self, conf: Path, top: Path, system_file: Path | None = None) -> None:
        """Prepare the MD environment, e.g., setup system, topology, forces."""
        pass

    @abstractmethod
    def run_production(self, run_index: int, start_coords: Path, steps: int,
                       traj_out: Path, stride: int, device_index: int) -> bool:
        """Execute a production run from start_coords for a given number of steps."""
        pass

# Factory Registry
class EngineFactory:
    _engines = {}

    @classmethod
    def register(cls, name: str, engine_cls):
        """Register a new engine implementation."""
        cls._engines[name] = engine_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> MDEngine:
        """Instantiate an engine by name."""
        if name not in cls._engines:
            raise ValueError(f"Unknown MD engine: {name}")
        return cls._engines[name](**kwargs)
