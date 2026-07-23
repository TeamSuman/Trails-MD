import os
from abc import ABC, abstractmethod
from pathlib import Path


def box_vectors_to_abc_angles(box_vectors):
    """Convert OpenMM periodic box vectors (nm) to ``[a, b, c, α, β, γ]`` in
    Angstrom / degrees, **correctly for triclinic cells**.

    Amber and GROMACS solvation almost always uses a truncated octahedron or
    rhombic dodecahedron (a triclinic box with large off-diagonal components).
    The previous per-engine converters read only the box-vector diagonal and
    hard-coded 90° angles, silently rewriting such a box as a wrong orthorhombic
    cell — corrupting volume/density, the minimum image, and pressure coupling.
    This helper derives the true cell edges and angles from the full vectors.

    Returns ``None`` for a non-periodic system (``box_vectors is None``) or a
    degenerate (zero-length) box.
    """
    import numpy as np

    if box_vectors is None:
        return None
    try:
        from openmm.unit import is_quantity, nanometer  # type: ignore

        def _strip(vec):
            return vec.value_in_unit(nanometer) if is_quantity(vec) else vec
    except ImportError:  # OpenMM absent: assume raw nm values

        def _strip(vec):
            return vec

    rows = []
    for vec in box_vectors:
        vec = _strip(vec)
        rows.append([float(vec[0]), float(vec[1]), float(vec[2])])
    vectors = np.asarray(rows, dtype=float) * 10.0  # nm → Å
    a_vec, b_vec, c_vec = vectors[0], vectors[1], vectors[2]
    a, b, c = (float(np.linalg.norm(v)) for v in (a_vec, b_vec, c_vec))
    if min(a, b, c) <= 0.0:
        return None
    alpha = float(np.degrees(np.arccos(np.clip(b_vec @ c_vec / (b * c), -1.0, 1.0))))
    beta = float(np.degrees(np.arccos(np.clip(a_vec @ c_vec / (a * c), -1.0, 1.0))))
    gamma = float(np.degrees(np.arccos(np.clip(a_vec @ b_vec / (a * b), -1.0, 1.0))))
    return np.array([a, b, c, alpha, beta, gamma], dtype=float)


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

    # Whether a persistent worker may cache a prepared instance of this engine and
    # re-arm it per walker instead of rebuilding. In-process engines that hold a
    # reusable context (OpenMM) override this to True; subprocess engines (GROMACS,
    # Amber) leave it False and always run fresh.
    supports_warm_reuse = False

    def rearm_for_walker(self, seed: int | None) -> None:
        """Re-point a cached engine at the next walker (persistent-worker mode).

        Only warm-reusable engines are ever cached, so the base implementation is
        never called; it exists so the attribute is always present."""
        raise NotImplementedError

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
