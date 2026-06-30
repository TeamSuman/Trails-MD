"""MD engines.

Engine backends are registered *lazily*: each engine module (and its heavy,
optional dependency — OpenMM / GROMACS / Amber) is imported only when that engine
is first requested via ``EngineFactory.get``. This keeps ``import autosampler``
free of the MD backends, so the base ``pip install autosampler`` does not need
OpenMM (install it with ``pip install 'autosampler[openmm]'`` or via conda).
"""

from .base import EngineFactory, MDEngine

EngineFactory.register_lazy("openmm", "autosampler.engines.openmm", "OpenMMEngine")
EngineFactory.register_lazy("amber", "autosampler.engines.amber", "AmberEngine")
EngineFactory.register_lazy("gromacs", "autosampler.engines.gromacs", "GromacsEngine")

__all__ = ["EngineFactory", "MDEngine"]


def __getattr__(name):
    # Backwards-compatible lazy access to the engine classes.
    _classes = {
        "OpenMMEngine": ("autosampler.engines.openmm", "OpenMMEngine"),
        "AmberEngine": ("autosampler.engines.amber", "AmberEngine"),
        "GromacsEngine": ("autosampler.engines.gromacs", "GromacsEngine"),
    }
    if name in _classes:
        import importlib

        module_path, class_name = _classes[name]
        return getattr(importlib.import_module(module_path), class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
