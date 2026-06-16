"""
AutoSampler Adaptive Sampling Framework

A modular, extensible, and scalable framework for enhanced molecular dynamics sampling.
"""

__version__ = "2.0.0"
__all__ = ["AutoSamplerConfig", "AutoSamplerCore"]


def __getattr__(name):
    if name == "AutoSamplerConfig":
        from .config import AutoSamplerConfig

        return AutoSamplerConfig
    if name == "AutoSamplerCore":
        from .core import AutoSamplerCore

        return AutoSamplerCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
