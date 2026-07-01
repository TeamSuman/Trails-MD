import os
from abc import ABC, abstractmethod
from pathlib import Path


def md_subprocess_timeout() -> float | None:
    """Timeout (seconds) for external MD subprocesses, or None for no limit.

    Configured via the ``TRAILS_MD_TIMEOUT`` environment variable so it
    propagates cleanly to walker worker processes without threading through
    engine constructors. Guards against hung ``gmx``/``pmemd`` invocations.
    """
    raw = os.environ.get("TRAILS_MD_TIMEOUT")
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
    _lazy: dict = {}  # name -> (module_path, class_name)

    @classmethod
    def register(cls, name: str, engine_cls):
        """Register a new engine implementation."""
        cls._engines[name] = engine_cls

    @classmethod
    def register_lazy(cls, name: str, module_path: str, class_name: str):
        """Register an engine that is imported only when first requested.

        Keeps heavy optional backends (OpenMM, GROMACS, Amber) out of the import
        path of ``import trails_md`` so the base install need not pull them in.
        """
        cls._lazy[name] = (module_path, class_name)

    @classmethod
    def get(cls, name: str, **kwargs) -> MDEngine:
        """Instantiate an engine by name (importing its backend on first use)."""
        if name not in cls._engines and name in cls._lazy:
            import importlib

            module_path, class_name = cls._lazy[name]
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                raise ImportError(
                    f"MD engine {name!r} needs an optional dependency that is not "
                    f"installed ({exc}). Install it, e.g. "
                    f"`pip install 'trails-md[{name}]'` or via conda."
                ) from exc
            cls._engines[name] = getattr(module, class_name)
        if name not in cls._engines:
            raise ValueError(f"Unknown MD engine: {name}")
        return cls._engines[name](**kwargs)

    @classmethod
    def available(cls) -> list:
        return sorted(set(cls._engines) | set(cls._lazy))
