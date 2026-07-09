"""Shared history-pooling logic for spawners and the core sampling loop.

Both the spawners (which build the cumulative candidate-point pool) and the core
loop (which maps a chosen spawn index back to a trajectory frame and its lineage
record) must agree, frame-for-frame, on *which* historical iterations enter the
pool and in what order. If they disagree -- for example one drops a
dimension-mismatched iteration while the other keeps it -- a spawn index selects
the wrong conformation, silently. This module is the single source of truth for
that decision so the two callers can never fall out of sync.

The dimension filter matters when the projection dimensionality changes across
history, most commonly when ``system.initial_trajectory`` injects a 2-D
physical-CV projection at ``iteration -1`` while the adaptive space projects to a
different latent dimension from iteration 0 onward.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def projection_dim(projection: Any) -> int:
    """Feature dimension of a stored projection (1-D arrays count as dimension 1)."""
    proj = np.asarray(projection)
    return 1 if proj.ndim == 1 else int(proj.shape[1])


def pooled_history_iterations(
    history: dict[int, Any] | None, target_dim: int | None
) -> list[int]:
    """Sorted history iterations whose projection joins the cumulative pool.

    An iteration is included when it stores a non-empty projection whose feature
    dimension equals ``target_dim`` (the dimension of the current projection being
    spawned from). ``target_dim=None`` includes every stored projection.

    The returned order (ascending iteration) is the canonical concatenation order
    for the cumulative point cloud, the trajectory list, and the frame-record
    list, guaranteeing index ``i`` refers to the same frame in all three.
    """
    if not history:
        return []
    included: list[int] = []
    for iteration in sorted(history):
        entry = history[iteration]
        if not isinstance(entry, dict):
            continue
        projection = entry.get("projection")
        if projection is None:
            continue
        if target_dim is not None and projection_dim(projection) != target_dim:
            continue
        included.append(iteration)
    return included
