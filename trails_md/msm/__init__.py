"""Markov State Model subsystem for Trails-MD.

This package adds the MSM-building and MSM-based convergence capability that the
adaptive loop uses to decide when sampling is complete:

- :class:`~trails_md.msm.estimator.MSMEstimator` -- clustering, transition
  counting, MLE/Bayesian MSM, implied timescales, VAMP-2, PCCA+.
- :class:`~trails_md.msm.convergence.ConvergenceMonitor` -- pluggable
  convergence criteria (ITS stability, VAMP-2 plateau, stationary-distribution
  drift, Bayesian statistical error).
- :class:`~trails_md.msm.diagnostics.MSMResult` -- serialisable per-iteration
  result container.
"""

from .convergence import (
    ConvergenceCriterion,
    ConvergenceMonitor,
    ImpliedTimescaleCriterion,
    StationaryDistributionCriterion,
    StatisticalErrorCriterion,
    TransitionMatrixCriterion,
    VAMP2Criterion,
    build_criterion,
)
from .diagnostics import ITSResult, MSMResult
from .estimator import MSMEstimator, MSMEstimatorFactory

__all__ = [
    "MSMEstimator",
    "MSMEstimatorFactory",
    "MSMResult",
    "ITSResult",
    "ConvergenceMonitor",
    "ConvergenceCriterion",
    "ImpliedTimescaleCriterion",
    "VAMP2Criterion",
    "StationaryDistributionCriterion",
    "StatisticalErrorCriterion",
    "TransitionMatrixCriterion",
    "build_criterion",
]


def build_monitor_from_config(msm_config) -> "ConvergenceMonitor":
    """Construct a :class:`ConvergenceMonitor` from an ``MSMConfig`` object."""
    criteria = []
    for spec in msm_config.convergence_criteria:
        name = spec["name"] if isinstance(spec, dict) else spec.name
        kwargs = dict(spec.get("params", {})) if isinstance(spec, dict) else dict(
            getattr(spec, "params", {}) or {}
        )
        criteria.append(build_criterion(name, **kwargs))
    return ConvergenceMonitor(
        criteria,
        mode=msm_config.convergence_mode,
        patience=msm_config.convergence_patience,
    )
