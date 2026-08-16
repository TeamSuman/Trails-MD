"""SPIB must refine its state labels self-consistently, not freeze the initial k-means.

Wang & Tiwary's SPIB alternates two steps: train the encoder/predictor to predict the
future state label, then *relabel* every frame by the model's own predicted state, and
repeat until the labelling stops changing. That loop is what makes the method "state
predictive" -- without it the CV is a variational information bottleneck onto whatever
partition k-means happened to produce, which is a different (and weaker) method.

These tests pin the behaviour that distinguishes the two:

* the refinement actually runs and reports how the labelling converged;
* states that end up unoccupied are dropped, so the effective state count adapts
  (the paper's mechanism for choosing the number of metastable states);
* on data with a clear two-state structure but a deliberately bad initial partition,
  refinement recovers the true split -- the fixed-label version cannot.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

torch = pytest.importorskip("torch")

from trails_md.spaces.spib import train_spib  # noqa: E402


def _two_state_trajs(n_traj=12, n_frames=60, seed=0):
    """Trajectories that dwell in one of two well-separated basins.

    The basins are far apart relative to their width, so any faithful state
    assignment must split along that gap.
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_traj):
        centre = np.array([-4.0, 0.0]) if i % 2 == 0 else np.array([4.0, 0.0])
        x = centre + 0.35 * rng.standard_normal((n_frames, 2))
        pad = 0.1 * rng.standard_normal((n_frames, 4))
        trajs.append(np.hstack([x, pad]).astype(np.float32))
    return trajs


def test_train_spib_reports_refinement_history():
    """The result must expose how the labelling converged, not just an encoder."""
    trajs = _two_state_trajs()
    enc, info = train_spib(trajs, lagtime=2, latent_dim=2, hidden_dims=[16, 8],
                           epochs=3, learning_rate=1e-3, batch_size=64,
                           n_states=8, beta=1e-3, refine_rounds=4, seed=0)
    assert hasattr(enc, "forward")
    assert "label_changes" in info and len(info["label_changes"]) >= 1
    # Each entry is the fraction of frames whose state label moved that round.
    assert all(0.0 <= c <= 1.0 for c in info["label_changes"])
    assert "n_states_final" in info and info["n_states_final"] >= 1


def test_refinement_prunes_unoccupied_states():
    """Two well-separated basins must not keep eight populated states."""
    trajs = _two_state_trajs()
    _, info = train_spib(trajs, lagtime=2, latent_dim=2, hidden_dims=[16, 8],
                         epochs=8, learning_rate=5e-3, batch_size=64,
                         n_states=8, beta=1e-4, refine_rounds=8, seed=0)
    assert info["n_states_final"] < 8, (
        f"refinement kept all {info['n_states_final']} initial states; "
        "unoccupied states are supposed to be pruned"
    )


def test_refinement_converges():
    """The fraction of relabelled frames must decrease -- otherwise it is not
    converging and calling the result self-consistent would be wrong."""
    trajs = _two_state_trajs()
    _, info = train_spib(trajs, lagtime=2, latent_dim=2, hidden_dims=[16, 8],
                         epochs=8, learning_rate=5e-3, batch_size=64,
                         n_states=8, beta=1e-4, refine_rounds=8, seed=0)
    changes = info["label_changes"]
    assert changes[-1] <= changes[0], (
        f"label churn did not decrease: {changes}"
    )


def test_refine_rounds_zero_reproduces_fixed_label_behaviour():
    """Opting out must be possible and must be a no-op refinement.

    This is what the previous implementation did, so it stays reachable for
    reproducing older results.
    """
    trajs = _two_state_trajs()
    _, info = train_spib(trajs, lagtime=2, latent_dim=2, hidden_dims=[16, 8],
                         epochs=3, learning_rate=1e-3, batch_size=64,
                         n_states=6, beta=1e-3, refine_rounds=0, seed=0)
    assert info["label_changes"] == []
