"""MSM-based convergence detection for the adaptive sampling loop.

A :class:`ConvergenceMonitor` holds a list of pluggable
:class:`ConvergenceCriterion` objects. Each iteration it is fed the latest
:class:`~autosampler.msm.diagnostics.MSMResult`; it records per-criterion state
and reports convergence once the configured combination of criteria
(``"all"`` / ``"any"``) has held for ``patience`` consecutive iterations.

The criteria here operate on quantities that are invariant to microstate
relabelling (slow implied timescales, VAMP-2 score, sorted metastable
populations), so they remain valid even though the clustering changes from one
iteration to the next.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .diagnostics import MSMResult

logger = logging.getLogger(__name__)


@dataclass
class CriterionStatus:
    name: str
    satisfied: bool
    value: Optional[float]
    detail: str


class ConvergenceCriterion(ABC):
    """Base class for a single convergence test fed one MSMResult per call."""

    name: str = "criterion"

    @abstractmethod
    def update(self, result: MSMResult) -> CriterionStatus:
        """Record ``result`` and report whether the test is currently satisfied."""

    def reset(self) -> None:  # pragma: no cover - trivial default
        """Clear accumulated history (used when restarting a monitor)."""


def _relative_change(prev: float, curr: float) -> float:
    denom = max(abs(prev), abs(curr), 1e-12)
    return abs(curr - prev) / denom


class ImpliedTimescaleCriterion(ConvergenceCriterion):
    """Satisfied when the slowest ``k`` implied timescales stop changing.

    Compares the current slow timescales against the previous iteration; the
    test passes when the maximum relative change is below ``tol``.
    """

    name = "implied_timescales"

    def __init__(self, tol: float = 0.1, n_timescales: int = 2) -> None:
        self.tol = float(tol)
        self.n_timescales = int(n_timescales)
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, result: MSMResult) -> CriterionStatus:
        ts = np.asarray(result.timescales, dtype=float)
        ts = ts[np.isfinite(ts)][: self.n_timescales]
        if ts.size == 0:
            return CriterionStatus(self.name, False, None, "no finite timescales")
        if self._prev is None or self._prev.size != ts.size:
            self._prev = ts
            return CriterionStatus(self.name, False, None, "baseline established")
        changes = [_relative_change(p, c) for p, c in zip(self._prev, ts)]
        max_change = float(max(changes))
        self._prev = ts
        satisfied = max_change < self.tol
        return CriterionStatus(
            self.name,
            satisfied,
            max_change,
            f"max rel. change {max_change:.3f} (tol {self.tol})",
        )


class VAMP2Criterion(ConvergenceCriterion):
    """Satisfied when the VAMP-2 score plateaus (relative change below ``tol``)."""

    name = "vamp2"

    def __init__(self, tol: float = 0.05) -> None:
        self.tol = float(tol)
        self._prev: Optional[float] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, result: MSMResult) -> CriterionStatus:
        score = result.vamp2_score
        if score is None:
            return CriterionStatus(self.name, False, None, "no VAMP-2 score")
        if self._prev is None:
            self._prev = score
            return CriterionStatus(self.name, False, score, "baseline established")
        change = _relative_change(self._prev, score)
        self._prev = score
        satisfied = change < self.tol
        return CriterionStatus(
            self.name, satisfied, change, f"rel. change {change:.3f} (tol {self.tol})"
        )


class StationaryDistributionCriterion(ConvergenceCriterion):
    """Satisfied when metastable populations stabilise across iterations.

    Uses the PCCA+ metastable populations (sorted, so the test is invariant to
    macrostate relabelling) and measures L1 drift between consecutive
    iterations. Falls back to inactivity when no metastable decomposition is
    available.
    """

    name = "stationary_distribution"

    def __init__(self, tol: float = 0.05) -> None:
        self.tol = float(tol)
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, result: MSMResult) -> CriterionStatus:
        pops = result.metastable_populations
        if pops is None:
            return CriterionStatus(
                self.name, False, None, "no metastable populations (set n_metastable)"
            )
        pops = np.sort(np.asarray(pops, dtype=float))[::-1]
        if self._prev is None or self._prev.size != pops.size:
            self._prev = pops
            return CriterionStatus(self.name, False, None, "baseline established")
        drift = float(np.abs(pops - self._prev).sum())
        self._prev = pops
        satisfied = drift < self.tol
        return CriterionStatus(
            self.name, satisfied, drift, f"L1 drift {drift:.3f} (tol {self.tol})"
        )


class StatisticalErrorCriterion(ConvergenceCriterion):
    """Satisfied when the Bayesian relative error on the slowest timescale is low.

    Requires ``estimator: bayesian`` so that ``timescale_errors`` is populated.
    """

    name = "statistical_error"

    def __init__(self, tol: float = 0.1) -> None:
        self.tol = float(tol)

    def update(self, result: MSMResult) -> CriterionStatus:
        errors = result.timescale_errors
        ts = result.slowest_timescale
        if errors is None or ts is None or not np.isfinite(ts) or ts == 0:
            return CriterionStatus(
                self.name, False, None, "no Bayesian errors (set estimator=bayesian)"
            )
        rel_err = float(np.asarray(errors, dtype=float)[0] / abs(ts))
        satisfied = rel_err < self.tol
        return CriterionStatus(
            self.name, satisfied, rel_err, f"rel. error {rel_err:.3f} (tol {self.tol})"
        )


_CRITERION_REGISTRY = {
    ImpliedTimescaleCriterion.name: ImpliedTimescaleCriterion,
    VAMP2Criterion.name: VAMP2Criterion,
    StationaryDistributionCriterion.name: StationaryDistributionCriterion,
    StatisticalErrorCriterion.name: StatisticalErrorCriterion,
}


def build_criterion(name: str, **kwargs) -> ConvergenceCriterion:
    if name not in _CRITERION_REGISTRY:
        raise ValueError(
            f"Unknown convergence criterion {name!r}; "
            f"available: {sorted(_CRITERION_REGISTRY)}"
        )
    return _CRITERION_REGISTRY[name](**kwargs)


class ConvergenceMonitor:
    """Aggregate pluggable criteria into a single converged / not-converged signal.

    Parameters
    ----------
    criteria:
        List of :class:`ConvergenceCriterion` instances.
    mode:
        ``"all"`` (default) requires every criterion satisfied simultaneously;
        ``"any"`` requires at least one.
    patience:
        Number of *consecutive* iterations the combination must hold before the
        monitor reports convergence.
    """

    def __init__(
        self,
        criteria: List[ConvergenceCriterion],
        mode: str = "all",
        patience: int = 2,
    ) -> None:
        if not criteria:
            raise ValueError("ConvergenceMonitor requires at least one criterion.")
        if mode not in ("all", "any"):
            raise ValueError("mode must be 'all' or 'any'.")
        self.criteria = criteria
        self.mode = mode
        self.patience = int(patience)
        self.streak = 0
        self.converged = False
        self.reason: Optional[str] = None
        self.last_statuses: List[CriterionStatus] = []

    def update(self, result: MSMResult) -> bool:
        """Feed a new MSMResult; return whether convergence is now declared."""
        statuses = [c.update(result) for c in self.criteria]
        self.last_statuses = statuses
        flags = [s.satisfied for s in statuses]
        combined = all(flags) if self.mode == "all" else any(flags)

        if combined:
            self.streak += 1
        else:
            self.streak = 0

        if not self.converged and self.streak >= self.patience:
            self.converged = True
            detail = "; ".join(f"{s.name}: {s.detail}" for s in statuses)
            self.reason = (
                f"MSM convergence: criteria ({self.mode}) satisfied for "
                f"{self.streak} consecutive iteration(s). [{detail}]"
            )
            logger.info(self.reason)
        return self.converged

    def status_line(self) -> str:
        parts = [
            f"{s.name}={'ok' if s.satisfied else 'no'}"
            + (f"({s.value:.3f})" if s.value is not None else "")
            for s in self.last_statuses
        ]
        return f"streak={self.streak}/{self.patience} " + " ".join(parts)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "patience": self.patience,
            "streak": self.streak,
            "converged": self.converged,
            "reason": self.reason,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.streak = int(state.get("streak", 0))
        self.converged = bool(state.get("converged", False))
        self.reason = state.get("reason")
