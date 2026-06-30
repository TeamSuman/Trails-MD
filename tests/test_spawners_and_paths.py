"""Coverage for the previously-untested spawners (voronoi/lof/fps) and the
trajectory-lineage helpers in autosampler.paths."""

from __future__ import annotations

import numpy as np
import pytest

from autosampler.spawners.base import SpawnerFactory


def _cloud(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(-2, 0.3, (n // 2, 2)), rng.normal(2, 0.3, (n // 2, 2))])


@pytest.mark.parametrize(
    "scheme,kwargs",
    [
        ("fps", {}),
        ("lof", {"n_neighbors": 8}),
        ("voronoi", {"n_clusters": 10}),
    ],
)
def test_spawner_explore_returns_valid_indices(scheme, kwargs):
    if scheme == "voronoi":
        pytest.importorskip("shapely")
    if scheme == "lof":
        pytest.importorskip("sklearn")
    points = _cloud()
    spawner = SpawnerFactory.get(scheme, mode="explore", **kwargs)
    idx = spawner.sample(points, top_n=6)
    assert len(idx) == 6
    assert all(0 <= int(i) < len(points) for i in idx)


@pytest.mark.parametrize("scheme", ["fps", "lof", "voronoi"])
def test_spawner_target_mode(scheme):
    if scheme == "voronoi":
        pytest.importorskip("shapely")
    if scheme == "lof":
        pytest.importorskip("sklearn")
    points = _cloud()
    kwargs = {"target": [2.0, 2.0]}
    if scheme == "lof":
        kwargs["n_neighbors"] = 8
    if scheme == "voronoi":
        kwargs["n_clusters"] = 10
    spawner = SpawnerFactory.get(scheme, mode="target", **kwargs)
    idx = spawner.sample(points, top_n=5)
    assert len(idx) == 5 and all(0 <= int(i) < len(points) for i in idx)


def test_spawner_factory_unknown_scheme():
    with pytest.raises((ValueError, KeyError)):
        SpawnerFactory.get("nope")


# ── paths / lineage helpers ─────────────────────────────────────────────────
from autosampler import paths  # noqa: E402


def test_frame_key_and_frameref_roundtrip():
    assert paths.frame_key(3, 1, 4) == "3:1:4"
    ref = paths.FrameRef(
        iteration=1, walker=0, frame=2, trajectory="t.xtc", cv=(0.1, 0.2), parent=None
    )
    again = paths.FrameRef.from_dict(ref.to_dict())
    assert again == ref
    assert again.key == "1:0:2"


def _records():
    # 2 walkers x 2 frames, non-xtc so expected_frames is used (no real I/O).
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    return paths.build_frame_records(
        iteration=0,
        trajectories=["w0.dcd", "w1.dcd"],
        points=pts,
        walker_parents=[None, None],
        expected_frames=2,
    )


def test_build_frame_records_and_lineage():
    recs = _records()
    assert len(recs) == 4
    # within-walker parent links: frame 1 points at frame 0 of the same walker.
    assert recs[1]["parent"] == paths.frame_key(0, 0, 0)
    assert recs[0]["parent"] is None

    mapped = paths.map_global_frame(recs, 2)
    assert mapped["walker"] == 1 and mapped["frame"] == 0
    with pytest.raises(IndexError):
        paths.map_global_frame(recs, 99)


def test_history_records_and_nearest():
    history = {0: {"frames": _records()}}
    refs = paths.history_records(history)
    assert len(refs) == 4
    near = paths.nearest_record(refs, np.array([2.9, 2.9]))
    assert near.cv == (3.0, 3.0)
    with pytest.raises(ValueError):
        paths.nearest_record(refs, np.array([0.0, 0.0, 0.0]))  # dim mismatch


def test_history_records_empty_raises():
    with pytest.raises(ValueError):
        paths.history_records({0: {"frames": []}})
