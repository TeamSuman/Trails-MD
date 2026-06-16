"""Tests for MSMConfig defaults and the MSM-guided spawner."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.config import AutoSamplerConfig, MSMConfig  # noqa: E402
from autosampler.spawners import SpawnerFactory  # noqa: E402
import autosampler.spawners.msm  # noqa: E402,F401  (ensures registration)


def _base_config(**overrides):
    cfg = {
        "system": {"conf_file": "a.gro", "top_file": "a.top"},
        "engine": {"md_engine": "openmm"},
        "spawning": {"spawn_scheme": "density"},
    }
    cfg.update(overrides)
    return cfg


def test_msm_disabled_by_default():
    cfg = AutoSamplerConfig(**_base_config())
    assert cfg.msm.enabled is False
    # Default convergence criteria present so enabling needs no extra config.
    names = {c["name"] for c in cfg.msm.convergence_criteria}
    assert "implied_timescales" in names and "vamp2" in names


def test_msm_config_validation():
    with pytest.raises(Exception):
        MSMConfig(cluster_method="banana")
    with pytest.raises(Exception):
        MSMConfig(estimator="frequentist")
    with pytest.raises(Exception):
        MSMConfig(lagtime=0)
    good = MSMConfig(enabled=True, estimator="bayesian", n_metastable=4)
    assert good.estimator == "bayesian"


def test_msm_spawner_registered():
    spawner = SpawnerFactory.get("msm", n_clusters=20, seed=1)
    assert spawner is not None


def test_msm_spawner_prefers_undersampled_regions():
    """Frames in sparse microstates should be selected far more often.

    Averaged over several seeds with a realistic walker count, the sparse
    region (1% of all frames) should attract a large share of restarts.
    """
    rng = np.random.default_rng(0)
    # 990 frames densely packed near origin, 10 frames in a far, sparse region.
    dense = rng.normal(scale=0.1, size=(990, 2))
    sparse = rng.normal(loc=[10.0, 10.0], scale=0.1, size=(10, 2))
    points = np.vstack([dense, sparse])

    fractions = []
    for seed in range(8):
        spawner = SpawnerFactory.get("msm", n_clusters=20, seed=seed)
        picks = np.asarray(spawner.sample(points, top_n=10, history=None))
        fractions.append(np.mean(picks >= 990))
    mean_frac = float(np.mean(fractions))
    # Population share of the sparse region is 0.01; least-counts must massively
    # oversample it.
    assert mean_frac > 0.15


def test_msm_spawner_indices_in_range():
    points = np.random.default_rng(1).normal(size=(100, 2))
    spawner = SpawnerFactory.get("msm", n_clusters=10, seed=3)
    picks = spawner.sample(points, top_n=10, history=None)
    assert all(0 <= i < 100 for i in picks)
    assert len(picks) == 10
