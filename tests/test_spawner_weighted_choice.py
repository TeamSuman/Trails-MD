"""Regression: density/voronoi weighted choice must not crash when most weights
are zero (target mode zeroes out distant bins)."""

from __future__ import annotations

import numpy as np

from trails_md.spawners.density import _weighted_choice


def test_weighted_choice_with_mostly_zero_weights():
    rows = np.arange(4)
    weights = np.array([1.0, 0.0, 0.0, 0.0])  # only one nonzero
    out = _weighted_choice(rows, weights, top_n=3)  # would raise pre-fix
    assert len(out) == 3
    assert set(out).issubset(set(rows.tolist()))


def test_weighted_choice_all_zero_falls_back_to_uniform():
    rows = np.arange(5)
    out = _weighted_choice(rows, np.zeros(5), top_n=2)
    assert len(out) == 2 and set(out).issubset(set(rows.tolist()))
