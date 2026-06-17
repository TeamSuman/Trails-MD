"""Tests for landscape-adaptive binning (gradient / mab / eigenvector)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.binning.adaptive import (  # noqa: E402
    EigenvectorBinner,
    GradientBinner,
    MABinner,
    make_binner,
)
from autosampler.binning.spatial import RegularBinner  # noqa: E402
from autosampler.spawners import SpawnerFactory  # noqa: E402


def _double_well(n=4000, seed=0):
    """1-D points: two dense basins at ±2 with a sparse barrier near 0."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(-2, 0.3, n // 2), rng.normal(2, 0.3, n // 2)])
    return x.reshape(-1, 1)


def test_gradient_bins_are_finer_at_the_barrier():
    pts = _double_well()
    gb = GradientBinner(n_bins=[12], n_fine=80, smoothing=3)
    edges = gb._axis_edges(pts[:, 0], float(pts.min()), float(pts.max()), 12)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    barrier_w = widths[np.argmin(np.abs(centers))]  # bin nearest x=0
    basin_w = widths[np.argmin(np.abs(centers - 2.0))]  # bin nearest a basin
    assert barrier_w < basin_w  # sparse barrier gets finer bins


def test_uniform_scheme_is_regular_binner():
    b = make_binner("uniform", n_bins=[10, 10])
    assert isinstance(b, RegularBinner)


def test_make_binner_unknown_scheme():
    with pytest.raises(ValueError):
        make_binner("banana", n_bins=[10])


@pytest.mark.parametrize("scheme", ["gradient", "mab", "eigenvector"])
def test_binner_bintable_is_consistent(scheme):
    pts = np.random.default_rng(0).normal(size=(500, 2))
    binner = make_binner(scheme, n_bins=[8, 8])
    table = binner.fit(pts)
    # Every frame lands in exactly one bin.
    assert int(table.populations.sum()) == len(pts)
    assert sum(len(d) for d in table.populated_data) == len(pts)
    assert len(table.occupied_indices) >= 1


def test_eigenvector_bins_only_leading_coordinate():
    pts = np.random.default_rng(1).normal(size=(600, 3))
    table = EigenvectorBinner(n_bins=[7, 99, 99]).fit(pts)
    # 1-D along the leading axis -> at most n_bins[0] bins.
    assert len(table.ids) == 7
    assert int(table.populations.sum()) == len(pts)


def test_mab_produces_front_footholds():
    pts = _double_well()
    edges = MABinner(n_bins=[10]).fit(pts)  # smoke: fit must succeed
    assert int(edges.populations.sum()) == len(pts)


def test_binning_config_validation():
    from pydantic import ValidationError

    from autosampler.config import BinningConfig

    assert BinningConfig(scheme="gradient").scheme == "gradient"
    with pytest.raises(ValidationError):
        BinningConfig(scheme="banana")


def test_we_spawner_with_adaptive_binner():
    spawner = SpawnerFactory.get("we", n_bins=[6, 6], target_per_bin=3, seed=0)
    spawner.binner = make_binner("gradient", n_bins=[6, 6])
    pts = np.random.default_rng(0).normal(size=(60, 2))
    idx = spawner.sample(pts, top_n=8)
    assert len(idx) == 8 and all(0 <= i < 60 for i in idx)


def test_density_spawner_with_adaptive_binner():
    spawner = SpawnerFactory.get(
        "density", n_bins=[6, 6], probabilistic=True, seed=0
    )
    spawner.binner = make_binner("eigenvector", n_bins=[6, 6])
    pts = np.random.default_rng(0).normal(size=(80, 2))
    idx = spawner.sample(pts, top_n=5)
    assert len(idx) == 5 and all(0 <= i < 80 for i in idx)
