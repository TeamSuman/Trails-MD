"""MD engines.

Engine backends are registered *lazily*: each engine module (and its heavy,
optional dependency — OpenMM / GROMACS / Amber) is imported only when that engine
is first requested via ``EngineFactory.get``. This keeps ``import trails_md``
free of the MD backends, so the base ``pip install trails-md`` does not need
OpenMM (install it with ``pip install 'trails-md[openmm]'`` or via conda).
"""

from .base import EngineFactory, MDEngine

EngineFactory.register_lazy("openmm", "trails_md.engines.openmm", "OpenMMEngine")
EngineFactory.register_lazy("amber", "trails_md.engines.amber", "AmberEngine")
EngineFactory.register_lazy("gromacs", "trails_md.engines.gromacs", "GromacsEngine")

__all__ = ["EngineFactory", "MDEngine"]


def __getattr__(name):
    # Backwards-compatible lazy access to the engine classes.
    _classes = {
        "OpenMMEngine": ("trails_md.engines.openmm", "OpenMMEngine"),
        "AmberEngine": ("trails_md.engines.amber", "AmberEngine"),
        "GromacsEngine": ("trails_md.engines.gromacs", "GromacsEngine"),
    }
    if name in _classes:
        import importlib

        module_path, class_name = _classes[name]
        return getattr(importlib.import_module(module_path), class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
