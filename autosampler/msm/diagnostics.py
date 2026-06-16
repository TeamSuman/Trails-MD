"""Serializable diagnostic containers for Markov State Model estimation.

These dataclasses hold the per-iteration MSM outputs (timescales, stationary
distribution, scores, implied-timescale sweeps, metastable decomposition) in a
form that is cheap to checkpoint (``to_dict`` / ``from_dict`` round-trip through
plain Python / NumPy objects) and convenient to feed into convergence checks,
logging and plotting.

The module deliberately avoids importing ``deeptime`` at import time; estimation
lives in :mod:`autosampler.msm.estimator`. Here we only describe the *results*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


@dataclass
class ITSResult:
    """Implied-timescale sweep across a set of lag times.

    Attributes
    ----------
    lagtimes:
        Lag times (in frames) at which MSMs were estimated.
    timescales:
        Array of shape ``(n_lagtimes, n_processes)`` with the implied
        timescales of the slowest processes at each lag time. ``NaN`` entries
        indicate a process that could not be resolved at that lag time.
    """

    lagtimes: np.ndarray
    timescales: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "lagtimes": np.asarray(self.lagtimes).tolist(),
            "timescales": np.asarray(self.timescales).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ITSResult:
        return cls(
            lagtimes=np.asarray(data["lagtimes"]),
            timescales=np.asarray(data["timescales"], dtype=float),
        )


@dataclass
class MSMResult:
    """Outputs of a single MSM estimation over the current CV/latent space.

    All fields are plain NumPy / Python objects so the result serialises
    cleanly into ``iter_*/msm.npz`` and into the run checkpoint.
    """

    lagtime: int
    n_microstates: int
    n_states_active: int
    timescales: np.ndarray
    stationary_distribution: np.ndarray
    transition_matrix: np.ndarray
    cluster_centers: np.ndarray
    counts_per_state: np.ndarray
    vamp2_score: float | None = None
    estimator: str = "mle"
    iteration: int | None = None
    n_metastable: int | None = None
    metastable_assignments: np.ndarray | None = None
    metastable_populations: np.ndarray | None = None
    its: ITSResult | None = None
    # Bayesian statistical errors on the slowest timescales (std over posterior).
    timescale_errors: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slowest_timescale(self) -> float | None:
        ts = np.asarray(self.timescales, dtype=float)
        finite = ts[np.isfinite(ts)]
        return float(finite[0]) if finite.size else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict recurses into the ITSResult dataclass; normalise to its dict.
        if self.its is not None:
            data["its"] = self.its.to_dict()
        for key in (
            "timescales",
            "stationary_distribution",
            "transition_matrix",
            "cluster_centers",
            "counts_per_state",
            "metastable_assignments",
            "metastable_populations",
            "timescale_errors",
        ):
            value = getattr(self, key)
            data[key] = None if value is None else np.asarray(value).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MSMResult:
        data = dict(data)
        its = data.get("its")
        if isinstance(its, dict):
            data["its"] = ITSResult.from_dict(its)
        for key in (
            "timescales",
            "stationary_distribution",
            "transition_matrix",
            "cluster_centers",
            "counts_per_state",
        ):
            data[key] = _to_array(data.get(key))
        for key in (
            "metastable_assignments",
            "metastable_populations",
            "timescale_errors",
        ):
            value = data.get(key)
            data[key] = None if value is None else _to_array(value)
        # Drop any unexpected keys so the dataclass stays forward-compatible.
        allowed = set(cls.__dataclass_fields__)
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)

    def summary(self) -> str:
        ts = np.asarray(self.timescales, dtype=float)
        ts_str = ", ".join(f"{t:.1f}" if np.isfinite(t) else "nan" for t in ts[:3])
        score_str = "n/a" if self.vamp2_score is None else f"{self.vamp2_score:.3f}"
        return (
            f"MSM(lag={self.lagtime}, active_states={self.n_states_active}/"
            f"{self.n_microstates}, t2..={ts_str}, VAMP2={score_str})"
        )
