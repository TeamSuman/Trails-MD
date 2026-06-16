"""Tests for weighted-ensemble resampling: split/merge weight conservation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.binning.we import WeightedEnsemble  # noqa: E402
from autosampler.spawners import SpawnerFactory  # noqa: E402


def test_split_low_population_bin():
    we = WeightedEnsemble(target_per_bin=4)
    # One walker in a single bin, weight 1.0 -> split into 4 of 0.25.
    res = we.resample([1.0], [0], rng=np.random.default_rng(0))
    assert len(res) == 4
    assert all(p == 0 for p in res.parents)
    assert sum(res.weights) == pytest.approx(1.0)
    assert all(w == pytest.approx(0.25) for w in res.weights)


def test_merge_high_population_bin():
    we = WeightedEnsemble(target_per_bin=1)
    res = we.resample(
        [0.1, 0.1, 0.1, 0.7], [0, 0, 0, 0], rng=np.random.default_rng(0)
    )
    assert len(res) == 1
    assert sum(res.weights) == pytest.approx(1.0)  # weight conserved


def test_weight_conserved_across_multiple_bins():
    we = WeightedEnsemble(target_per_bin=3)
    rng = np.random.default_rng(1)
    weights = rng.random(20)
    labels = rng.integers(0, 4, size=20)
    res = we.resample(weights, labels, rng=rng)
    # Total weight conserved and each occupied bin has exactly target walkers.
    assert sum(res.weights) == pytest.approx(weights.sum())
    parent_labels = labels[np.asarray(res.parents)]
    for b in np.unique(labels):
        assert int(np.sum(parent_labels == b)) == 3


def test_no_change_when_already_at_target():
    we = WeightedEnsemble(target_per_bin=2)
    res = we.resample([0.5, 0.5], [0, 0], rng=np.random.default_rng(0))
    assert len(res) == 2
    assert sorted(res.parents) == [0, 1]
    assert sum(res.weights) == pytest.approx(1.0)


def test_invalid_target():
    with pytest.raises(ValueError):
        WeightedEnsemble(target_per_bin=0)
    we = WeightedEnsemble(target_per_bin=2)
    with pytest.raises(ValueError):
        we.resample([1.0, 1.0], [0], rng=np.random.default_rng(0))  # length mismatch


def test_we_spawner_returns_valid_indices():
    spawner = SpawnerFactory.get(
        "we", n_bins=[5, 5], target_per_bin=3, seed=0
    )
    rng = np.random.default_rng(0)
    points = rng.normal(size=(60, 2))
    idx = spawner.sample(points, top_n=8)
    assert len(idx) == 8
    assert all(0 <= i < len(points) for i in idx)
    # Weights are tracked and normalised for the next iteration.
    assert spawner.weights is not None
    assert spawner.weights.sum() == pytest.approx(1.0)


def test_we_spawner_state_roundtrip():
    spawner = SpawnerFactory.get("we", n_bins=[5, 5], target_per_bin=2, seed=0)
    points = np.random.default_rng(1).normal(size=(40, 2))
    spawner.sample(points, top_n=5)
    state = spawner.state_dict()
    other = SpawnerFactory.get("we", n_bins=[5, 5], target_per_bin=2, seed=0)
    other.load_state_dict(state)
    np.testing.assert_allclose(other.weights, spawner.weights)
