"""
Trails-MD Adaptive Sampling Framework

A modular, extensible, and scalable framework for enhanced molecular dynamics sampling.
"""

__version__ = "2.0.0"
__all__ = ["TrailsMDConfig", "TrailsMDCore"]


def __getattr__(name):
    if name == "TrailsMDConfig":
        from .config import TrailsMDConfig

        return TrailsMDConfig
    if name == "TrailsMDCore":
        from .core import TrailsMDCore

        return TrailsMDCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
