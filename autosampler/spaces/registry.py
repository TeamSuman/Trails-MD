"""Registry of collective-variable (CV) / dimensionality-reduction methods.

This is the single source of truth for which adaptive CV methods AutoSampler
supports, what backend each needs, and whether each is available in the current
environment. It lets new cutting-edge CV methods be added in one place and keeps
the rest of the codebase (e.g. ``core.py``) free of hard-coded method lists.

Supported methods
-----------------
- ``pca``       : linear PCA baseline (scikit-learn).
- ``tica``      : time-lagged independent component analysis (deeptime).
- ``tvae``      : time-lagged variational autoencoder (deeptime + torch).
- ``vampnet``   : VAMPNet deep CV via the variational approach for Markov
                  processes (deeptime + torch).
- ``spib``      : State Predictive Information Bottleneck, Wang & Tiwary 2021
                  (built-in torch implementation, no extra dependency).
- ``deep-tica`` : deep (nonlinear) TICA via mlcolvar (optional).
- ``deep-lda``  : deep linear discriminant analysis, supervised, via mlcolvar
                  (optional; requires state labels).

``space_mode: fixed`` (user-provided CVs through a project file) is handled
outside this registry.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

FIXED_MODE = "fixed"


@dataclass(frozen=True)
class CVMethod:
    """Metadata describing one CV method."""

    name: str
    backend: str  # 'sklearn' | 'deeptime' | 'mlcolvar' | 'builtin'
    time_lagged: bool  # uses a lag time (dynamics-aware)
    supervised: bool  # requires per-frame state labels
    requires: tuple[str, ...]  # importable module names needed at runtime
    description: str
    optional: bool = False  # needs an optional / extra dependency


_METHODS: dict[str, CVMethod] = {
    "pca": CVMethod(
        "pca", "sklearn", False, False, ("sklearn",),
        "Linear PCA baseline.",
    ),
    "tica": CVMethod(
        "tica", "deeptime", True, False, ("deeptime",),
        "Time-lagged independent component analysis (linear, dynamics-aware).",
    ),
    "tvae": CVMethod(
        "tvae", "deeptime", True, False, ("deeptime", "torch"),
        "Time-lagged variational autoencoder (nonlinear bottleneck).",
    ),
    "vampnet": CVMethod(
        "vampnet", "deeptime", True, False, ("deeptime", "torch"),
        "VAMPNet: deep CVs trained with the variational approach for Markov "
        "processes (VAMP-2 score).",
    ),
    "spib": CVMethod(
        "spib", "builtin", True, False, ("torch",),
        "State Predictive Information Bottleneck (Wang & Tiwary, 2021): a "
        "variational information-bottleneck CV that predicts the future state.",
    ),
    "deep-tica": CVMethod(
        "deep-tica", "mlcolvar", True, False, ("mlcolvar", "lightning", "torch"),
        "Deep (nonlinear) TICA via mlcolvar.", True,
    ),
    "deep-lda": CVMethod(
        "deep-lda", "mlcolvar", False, True, ("mlcolvar", "lightning", "torch"),
        "Deep linear discriminant analysis (supervised) via mlcolvar; needs "
        "per-frame state labels.", True,
    ),
}

# Install hints for optional backends, surfaced when a method is unavailable.
_INSTALL_HINTS = {
    "mlcolvar": 'pip install "autosampler[deep-tica]"  # installs mlcolvar + lightning',
    "lightning": 'pip install "autosampler[deep-tica]"',
    "deeptime": "pip install deeptime",
    "torch": "pip install torch",
    "sklearn": "pip install scikit-learn",
}


def all_methods() -> dict[str, CVMethod]:
    """Return a copy of the full method registry."""
    return dict(_METHODS)


def adaptive_modes() -> tuple[str, ...]:
    """Names of all adaptive (learned) CV methods."""
    return tuple(_METHODS)


def is_adaptive_space(mode: str) -> bool:
    """True if ``mode`` is a learned CV method (i.e. not ``fixed``)."""
    return mode in _METHODS


def get_method(mode: str) -> CVMethod:
    if mode not in _METHODS:
        raise ValueError(
            f"Unknown CV space_mode {mode!r}. Valid options: "
            f"{(FIXED_MODE,) + adaptive_modes()}."
        )
    return _METHODS[mode]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def is_available(mode: str) -> bool:
    """True if every backend dependency of ``mode`` is importable."""
    method = get_method(mode)
    return all(_module_available(req) for req in method.requires)


def ensure_available(mode: str) -> None:
    """Raise an informative ImportError if ``mode``'s backend is missing."""
    method = get_method(mode)
    missing = [req for req in method.requires if not _module_available(req)]
    if missing:
        hints = "; ".join(_INSTALL_HINTS.get(m, f"pip install {m}") for m in missing)
        raise ImportError(
            f"CV method {mode!r} requires missing package(s): {', '.join(missing)}. "
            f"Install via: {hints}."
        )


def register_method(method: CVMethod) -> None:
    """Register a custom CV method (extension point for plugins)."""
    _METHODS[method.name] = method
