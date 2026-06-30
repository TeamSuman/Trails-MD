"""Spaces subpackage: CV / dimensionality-reduction models and feature extraction.

Heavy or optional dependencies (MDAnalysis, torch, deeptime) are imported
**lazily** so that lightweight modules such as
:mod:`autosampler.spaces.registry` can be imported without them — e.g. in
minimal CI environments or when only the CV-method metadata is needed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "TrajectoryScaler",
    "TVAEBottleneckEncoder",
    "TVAEBottleneckDecoder",
    "FeatureExtractor",
    "AdaptiveSpaceModel",
]

_LAZY = {
    "TrajectoryScaler": ".scalers",
    "TVAEBottleneckEncoder": ".tvae",
    "TVAEBottleneckDecoder": ".tvae",
    "FeatureExtractor": ".features",
    "AdaptiveSpaceModel": ".model",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import hints for type checkers only
    from .features import FeatureExtractor
    from .model import AdaptiveSpaceModel
    from .scalers import TrajectoryScaler
    from .tvae import TVAEBottleneckDecoder, TVAEBottleneckEncoder
