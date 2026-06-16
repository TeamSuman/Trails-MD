"""Tests for VAMP-2 input-feature selection."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.config import FeatureSelectionConfig  # noqa: E402
from autosampler.spaces.feature_selection import (  # noqa: E402
    FeatureSelector,
    greedy_vamp_selection,
    rank_candidates,
    vamp2_score,
)


def _slow_plus_noise(n_steps=8000, seed=0, p_escape=0.02, n_noise=4):
    """Column 0 carries slow 2-state dynamics; remaining columns are noise."""
    rng = np.random.default_rng(seed)
    state = 0
    states = np.empty(n_steps, dtype=int)
    for i in range(n_steps):
        if rng.random() < p_escape:
            state = 1 - state
        states[i] = state
    slow = np.where(states == 0, -1.0, 1.0) + rng.normal(scale=0.1, size=n_steps)
    noise = rng.normal(size=(n_steps, n_noise))
    return np.column_stack([slow, noise]).astype(np.float64)


def test_vamp2_score_prefers_slow_feature():
    traj = _slow_plus_noise()
    slow_only = [traj[:, [0]]]
    noise_only = [traj[:, 1:]]
    assert vamp2_score(slow_only, lagtime=10) > vamp2_score(noise_only, lagtime=10)


def test_vamp2_score_raises_on_short_traj():
    with pytest.raises(ValueError):
        vamp2_score([np.zeros((5, 2))], lagtime=10)


def test_rank_candidates_orders_by_score():
    traj = _slow_plus_noise()
    ranked = rank_candidates(
        {"slow": [traj[:, [0]]], "noise": [traj[:, 1:]]}, lagtime=10
    )
    assert ranked[0][0] == "slow"
    assert ranked[0][1] >= ranked[1][1]


def test_greedy_selection_is_parsimonious():
    traj = _slow_plus_noise()
    cols = greedy_vamp_selection([traj], lagtime=10, min_gain=1e-3)
    # Picks the informative column and rejects the noise columns (VAMP-2 is
    # monotonic in #features, so parsimony comes from the min_gain threshold).
    assert cols == [0]
    assert len(cols) < traj.shape[1]
    # The parsimonious subset retains essentially all of the kinetic variance.
    assert vamp2_score([traj[:, cols]], 10) >= 0.95 * vamp2_score([traj], 10)


def test_feature_selector_select_and_serialise():
    traj = _slow_plus_noise()
    selector = FeatureSelector(lagtime=10, method="greedy_vamp", min_gain=1e-3)
    selection = selector.select([traj])
    assert 0 in selection.columns
    from autosampler.spaces.feature_selection import FeatureSelection

    restored = FeatureSelection.from_dict(selection.to_dict())
    assert restored.columns == selection.columns


def test_feature_selector_method_all():
    traj = _slow_plus_noise(n_noise=3)
    selector = FeatureSelector(method="all")
    selection = selector.select([traj])
    assert selection.columns == [0, 1, 2, 3]


def test_feature_selection_config_validation():
    from pydantic import ValidationError

    cfg = FeatureSelectionConfig(enabled=True, method="greedy_vamp", lagtime=5)
    assert cfg.enabled and cfg.cadence == 5
    with pytest.raises(ValidationError):
        FeatureSelectionConfig(method="banana")
    with pytest.raises(ValidationError):
        FeatureSelectionConfig(lagtime=0)


def test_candidate_feature_types_validation():
    from pydantic import ValidationError

    cfg = FeatureSelectionConfig(
        enabled=True, candidate_feature_types=["distances", "phi_psi"]
    )
    assert cfg.candidate_feature_types == ["distances", "phi_psi"]
    with pytest.raises(ValidationError):
        FeatureSelectionConfig(candidate_feature_types=["distances", "nope"])


def test_rank_candidates_selects_best_feature_type():
    # Simulate two feature *types*: one resolves the slow process, one is noise.
    traj = _slow_plus_noise()
    ranked = rank_candidates(
        {"distances": [traj[:, [0]]], "fitted_coords": [traj[:, 1:]]}, lagtime=10
    )
    assert ranked[0][0] == "distances"
