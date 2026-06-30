"""Regression tests for delta-checkpointed history (save/reconstruct/resume).

History is written incrementally: each ``iter_*/history.pkl`` holds only the
entries since the previous checkpoint, and ``load`` / ``paths.load_history``
reconstruct the full history by merging the deltas. These tests cover the
round-trip, checkpoint-frequency gaps, corruption tolerance, and atomicity.
"""

from __future__ import annotations

import pickle

import pytest

from autosampler.checkpoints.manager import CheckpointManager


def test_delta_history_roundtrip(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    history: dict = {}
    for it in range(4):
        history[it] = {"frames": [{"iter": it}]}
        mgr.save(it, None, {"s": it}, {"b": it}, dict(history))

    # Each delta holds exactly the one new key for that iteration.
    for it in range(4):
        with open(tmp_path / f"iter_{it}" / "history.pkl", "rb") as f:
            assert set(pickle.load(f)) == {it}

    _, scaler, bin_state, full, _ = mgr.load(3)
    assert set(full) == {0, 1, 2, 3}
    assert scaler == {"s": 3} and bin_state == {"b": 3}


def test_delta_history_with_checkpoint_gaps(tmp_path):
    # checkpoint_freq > 1: only iters 0 and 2 are written, but history accrued
    # every iteration must be fully reconstructed.
    mgr = CheckpointManager(str(tmp_path))
    mgr.save(0, None, {}, {}, {0: "a"})
    mgr.save(2, None, {}, {}, {0: "a", 1: "b", 2: "c"})

    with open(tmp_path / "iter_2" / "history.pkl", "rb") as f:
        assert set(pickle.load(f)) == {1, 2}  # delta covers the gap

    _, _, _, full, _ = mgr.load(2)
    assert full == {0: "a", 1: "b", 2: "c"}


def test_delta_history_tolerates_corrupt_delta(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    mgr.save(0, None, {}, {}, {0: "a"})
    mgr.save(1, None, {}, {}, {0: "a", 1: "b"})
    # Simulate a crash-truncated earlier delta.
    (tmp_path / "iter_0" / "history.pkl").write_bytes(b"\x80\x04truncated")

    _, _, _, full, _ = mgr.load(1)  # must not raise
    assert 1 in full  # the newer, intact delta is still recovered


def test_checkpoint_writes_are_atomic(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    mgr.save(0, None, {"s": 0}, {"b": 0}, {0: "a"})
    # No leftover temp files from the atomic write+replace.
    assert not list(tmp_path.glob("iter_*/*.tmp"))


def test_paths_load_history_reconstructs_full_history(tmp_path):
    from autosampler.paths import load_history

    mgr = CheckpointManager(str(tmp_path / "checkpoints"))
    mgr.save(0, None, {}, {}, {0: {"frames": []}})
    mgr.save(1, None, {}, {}, {0: {"frames": []}, 1: {"frames": []}})

    # Reads the whole run dir; must merge deltas, not return the last window only.
    assert set(load_history(tmp_path)) == {0, 1}
    assert set(load_history(tmp_path, checkpoint=1)) == {0, 1}

    with pytest.raises(FileNotFoundError):
        load_history(tmp_path, checkpoint=99)
