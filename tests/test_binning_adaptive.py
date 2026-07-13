"""Regression tests for the landscape-adaptive binning schemes.

These encode two bugs found by benchmarking on a real steep-barrier landscape
(explicit-solvent proline, whose cumulative population decays
176124 -> 67107 -> 10353 -> 775 -> 40 -> 1 frames per 10 deg up the barrier):

* `gradient` placed its edges over the FULL configured domain, where the unvisited
  region has zero density so 1/P diverges and swallowed the whole edge budget --
  collapsing the occupied range into ~2 giant bins and DILUTING the frontier, the
  exact opposite of the intent.
* `mab` had no bottleneck detection and no dedicated bin for the leading frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.binning.adaptive import make_binner
from trails_md.binning.spatial import bucket_frames


def barrier_landscape(seed: int = 0):
    """Boltzmann-decaying population up a barrier, with ONE lone leading frame."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([
        rng.normal(0, 6, 20000),     # basin
        rng.normal(20, 6, 3000),
        rng.normal(35, 4, 400),
        rng.normal(48, 3, 30),
        np.array([57.0]),            # the frontier: a single frame
    ])
    y = rng.normal(0, 5, len(x))
    return np.column_stack([x, y])


@pytest.mark.parametrize("scheme", ["uniform", "gradient", "mab"])
def test_frontier_frame_is_not_diluted(scheme):
    """The lone leading frame must sit ALONE in its bin, so density weight 1/n_b is maximal."""
    pts = barrier_landscape()
    lead = int(np.argmax(pts[:, 0]))
    table = make_binner(scheme, n_bins=[36, 12],
                        min_values=[-20, -30], max_values=[200, 30]).fit(pts)
    row = next(i for i, d in enumerate(table.populated_data)
               if lead in np.asarray(d, dtype=int))
    assert table.populations[row] == 1, (
        f"{scheme}: the frontier frame was diluted into a bin of "
        f"{table.populations[row]} frames"
    )


def test_gradient_spends_its_bins_on_the_occupied_range():
    """Regression: gradient must not collapse the occupied range into a couple of bins."""
    pts = barrier_landscape()
    table = make_binner("gradient", n_bins=[36, 12],
                        min_values=[-20, -30], max_values=[200, 30]).fit(pts)
    assert int((table.populations > 0).sum()) > 10


def test_mab_gives_the_leading_frame_a_narrow_dedicated_bin():
    """MAB brackets the moving front with a narrow bin.

    Only the LEADING frame can be genuinely isolated: the trailing frame sits inside
    the dense basin, where any finite-width bracket necessarily catches neighbours.
    Isolating the frontier is the point -- that is where flux is being lost.
    """
    pts = barrier_landscape()
    table = make_binner("mab", n_bins=[36, 12],
                        min_values=[-20, -30], max_values=[200, 30]).fit(pts)

    lead = int(np.argmax(pts[:, 0]))
    row = next(i for i, d in enumerate(table.populated_data)
               if lead in np.asarray(d, dtype=int))
    assert table.populations[row] == 1

    # and MAB should resolve the sparse barrier far more finely than a uniform grid
    uniform = make_binner("uniform", n_bins=[36, 12],
                          min_values=[-20, -30], max_values=[200, 30]).fit(pts)
    sparse_mab = int(((table.populations > 0) & (table.populations <= 5)).sum())
    sparse_uni = int(((uniform.populations > 0) & (uniform.populations <= 5)).sum())
    assert sparse_mab >= sparse_uni


def test_bucket_frames_matches_the_naive_loop():
    """The vectorized bucketing must be identical to the old per-frame Python loop."""
    rng = np.random.default_rng(3)
    nbin = [7, 5]
    cells = np.column_stack([rng.integers(0, nbin[0], 5000),
                             rng.integers(0, nbin[1], 5000)])
    pops, data = bucket_frames(cells, nbin)

    ids = list(np.ndindex(*nbin))
    id_to_row = {t: i for i, t in enumerate(ids)}
    exp_pops = np.zeros(len(ids), dtype=int)
    exp_data: list[list[int]] = [[] for _ in ids]
    for frame, cell in enumerate(map(tuple, cells)):
        r = id_to_row[cell]
        exp_pops[r] += 1
        exp_data[r].append(frame)

    assert np.array_equal(pops, exp_pops)
    for got, exp in zip(data, exp_data):
        assert np.array_equal(np.asarray(got, dtype=int), np.asarray(exp, dtype=int))


def test_bucket_frames_handles_empty_input():
    pops, data = bucket_frames(np.empty((0, 2), dtype=int), [3, 3])
    assert pops.sum() == 0 and len(data) == 9
