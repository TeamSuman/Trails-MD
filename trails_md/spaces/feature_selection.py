"""VAMP-2 based input-feature selection and optimisation.

The quality of an MSM/CV is bounded by the input features fed to it. VAMP-2 is a
variational score for how well a feature set captures the slow dynamics: higher
is better, and it can be compared across *different* feature sets on the same
trajectories (Wu & Noé, 2017; Scherer et al., 2019).

This module provides:

- :func:`vamp2_score` — a dependency-light VAMP-2 score from time-lagged
  covariances of feature trajectories.
- :func:`rank_candidates` — rank named candidate feature sets by VAMP-2.
- :func:`greedy_vamp_selection` — greedy forward selection of the feature
  *columns* (or column groups) that maximise VAMP-2, i.e. an optimisation
  protocol that picks the best subset of features.
- :class:`FeatureSelector` — thin orchestrator used by the adaptive loop to
  choose and periodically update the input features.

All functions operate on plain arrays, so they are testable without MD inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _lagged_pairs(trajs: list[np.ndarray], lagtime: int):
    """Stack instantaneous/time-lagged frame pairs across trajectories."""
    inst, lagged = [], []
    for traj in trajs:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
        if len(traj) > lagtime:
            inst.append(traj[:-lagtime])
            lagged.append(traj[lagtime:])
    if not inst:
        raise ValueError(
            f"No trajectory is longer than the lag time ({lagtime}); "
            "cannot compute a VAMP score."
        )
    return np.vstack(inst), np.vstack(lagged)


def _whiten(cov: np.ndarray, epsilon: float) -> np.ndarray:
    """Return cov^{-1/2} via symmetric eigendecomposition, dropping tiny modes."""
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > epsilon * max(vals.max(), 1e-12)
    vals, vecs = vals[keep], vecs[:, keep]
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def vamp2_score(
    trajs: list[np.ndarray],
    lagtime: int,
    dim: int | None = None,
    epsilon: float = 1e-6,
) -> float:
    """VAMP-2 score of feature ``trajs`` at ``lagtime``.

    Defined as the sum of squared singular values of the whitened time-lagged
    correlation (Koopman) matrix ``C00^{-1/2} C0t C11^{-1/2}`` on mean-free
    features. Larger means the features resolve more, slower kinetic variance.
    ``dim`` optionally caps the number of singular values retained.
    """
    inst, lagged = _lagged_pairs(trajs, lagtime)
    mean = 0.5 * (inst.mean(axis=0) + lagged.mean(axis=0))
    inst = inst - mean
    lagged = lagged - mean
    n = len(inst)

    c00 = inst.T @ inst / n
    c11 = lagged.T @ lagged / n
    c0t = inst.T @ lagged / n

    koopman = _whiten(c00, epsilon) @ c0t @ _whiten(c11, epsilon)
    singular_values = np.linalg.svd(koopman, compute_uv=False)
    if dim is not None:
        singular_values = singular_values[:dim]
    # Clip for numerical noise; true singular values of the Koopman op are <= 1.
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(np.sum(singular_values**2))


def rank_candidates(
    candidates: dict[str, list[np.ndarray]],
    lagtime: int,
    dim: int | None = None,
) -> list[tuple[str, float]]:
    """Rank named candidate feature sets by VAMP-2 (best first)."""
    scored: list[tuple[str, float]] = []
    for name, trajs in candidates.items():
        try:
            scored.append((name, vamp2_score(trajs, lagtime, dim=dim)))
        except (ValueError, np.linalg.LinAlgError):
            scored.append((name, float("-inf")))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def greedy_vamp_selection(
    trajs: list[np.ndarray],
    lagtime: int,
    groups: list[list[int]] | None = None,
    max_groups: int | None = None,
    dim: int | None = None,
    min_gain: float = 1e-4,
) -> list[int]:
    """Greedy forward selection of feature columns maximising VAMP-2.

    Starting from an empty set, repeatedly add the column group whose inclusion
    most increases the VAMP-2 score, stopping when no group improves the score
    by more than ``min_gain`` (or ``max_groups`` are selected). Returns the
    sorted list of selected column indices.
    """
    n_features = np.asarray(trajs[0]).reshape(len(trajs[0]), -1).shape[1]
    if groups is None:
        groups = [[i] for i in range(n_features)]
    remaining = list(range(len(groups)))
    chosen: list[int] = []
    selected_cols: list[int] = []
    best_score = 0.0
    limit = max_groups if max_groups is not None else len(groups)

    while remaining and len(chosen) < limit:
        best_gain, best_g = min_gain, None
        for g in remaining:
            cols = sorted(selected_cols + groups[g])
            score = vamp2_score([t[:, cols] for t in trajs], lagtime, dim=dim)
            if score - best_score > best_gain:
                best_gain, best_g, best_cols = score - best_score, g, cols
        if best_g is None:
            break
        chosen.append(best_g)
        remaining.remove(best_g)
        selected_cols = best_cols
        best_score += best_gain

    return sorted(selected_cols) if selected_cols else list(range(n_features))


@dataclass
class FeatureSelection:
    """Outcome of a feature-selection step (serialisable for checkpoints)."""

    columns: list[int]
    score: float
    method: str

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "score": self.score, "method": self.method}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSelection:
        return cls(
            columns=list(data["columns"]),
            score=float(data["score"]),
            method=str(data.get("method", "greedy_vamp")),
        )


class FeatureSelector:
    """Choose the best input-feature columns by VAMP-2 optimisation.

    Used by the adaptive loop when ``feature_selection.enabled`` is set. Operates
    on a feature matrix reshaped into per-walker trajectories.
    """

    def __init__(
        self,
        lagtime: int = 10,
        method: str = "greedy_vamp",
        max_features: int | None = None,
        dim: int | None = None,
        min_gain: float = 1e-4,
    ):
        self.lagtime = int(lagtime)
        self.method = method
        self.max_features = max_features
        self.dim = dim
        self.min_gain = float(min_gain)

    def select(self, trajs: list[np.ndarray]) -> FeatureSelection:
        if self.method == "greedy_vamp":
            cols = greedy_vamp_selection(
                trajs,
                self.lagtime,
                max_groups=self.max_features,
                dim=self.dim,
                min_gain=self.min_gain,
            )
        elif self.method == "all":
            cols = list(range(np.asarray(trajs[0]).reshape(len(trajs[0]), -1).shape[1]))
        else:
            raise ValueError(f"Unknown feature-selection method: {self.method!r}")
        score = vamp2_score([t[:, cols] for t in trajs], self.lagtime, dim=self.dim)
        return FeatureSelection(columns=cols, score=score, method=self.method)
