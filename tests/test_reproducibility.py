"""Reproducibility: seed plumbing and deterministic CV training."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_seed_manager_enables_deterministic_mode():
    from trails_md.utils.seeds import SeedManager

    SeedManager(123).set_seed()  # must not raise even with deterministic algos on
    a = np.random.rand(5)
    SeedManager(123).set_seed()
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_adaptive_model_carries_seed():
    torch = pytest.importorskip("torch")
    pytest.importorskip("deeptime")
    from trails_md.spaces.model import AdaptiveSpaceModel

    m = AdaptiveSpaceModel(space_mode="vampnet", seed=7)
    assert m.seed == 7

    # Default seed is defined (round-trips through ensure_config_defaults).
    m2 = AdaptiveSpaceModel(space_mode="vampnet")
    m2.ensure_config_defaults()
    assert isinstance(m2.seed, int)
    del torch


def test_same_seed_gives_identical_vampnet_projection():
    pytest.importorskip("torch")
    pytest.importorskip("deeptime")
    from trails_md.spaces.model import AdaptiveSpaceModel

    rng = np.random.default_rng(0)
    # Two metastable blobs, ordered by walker then time (2 walkers).
    feats = np.vstack(
        [rng.normal(-1, 0.1, (50, 4)), rng.normal(1, 0.1, (50, 4))]
    ).astype(np.float32)

    def project():
        m = AdaptiveSpaceModel(
            space_mode="vampnet", seed=42, lagtime=2, latent_dim=2, epochs=3
        )
        m.fit(feats, walker_length=50, n_walkers=2)
        return np.asarray(m.project(feats))

    np.testing.assert_allclose(project(), project(), rtol=1e-5, atol=1e-5)


def test_spawner_rng_checkpoint_continuation():
    from trails_md.spawners.base import SpawnerFactory

    # Generate test point cloud
    points = np.random.default_rng(123).normal(0, 1, (100, 2))

    # Instantiate spawner with a specific seed
    spawner1 = SpawnerFactory.get("density", seed=999)
    res1 = spawner1.sample(points, top_n=5)
    
    # Save checkpoint state after first sample
    state = spawner1.state_dict()
    res2 = spawner1.sample(points, top_n=5)

    # Instantiate a fresh spawner, load state, and sample
    spawner2 = SpawnerFactory.get("density", seed=999)
    spawner2.load_state_dict(state)
    res2_restored = spawner2.sample(points, top_n=5)

    assert res2 == res2_restored
    # Check that sampling advances the RNG state (res1 and res2 should differ on average)
    assert res1 != res2

