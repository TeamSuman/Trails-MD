"""Tests for Phase 2 hardening: reporter, checkpoint versioning, trajectory
validation, MD subprocess timeout, and the fited->fitted compatibility shim."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from trails_md.engines.base import md_subprocess_timeout  # noqa: E402
from trails_md.reporting import IterationReporter  # noqa: E402


def test_reporter_plain_and_colored():
    reporter = IterationReporter(width=40)
    plain = reporter.format_summary(3, 1.2, 0.5, "10/100", color=False)
    assert "Iteration: 3" in plain
    assert "\033[" not in plain  # no ANSI escapes when color disabled
    assert "╔" in plain and "╝" in plain

    colored = reporter.format_summary(3, 1.2, 0.5, "10/100", color=True)
    assert "\033[96m" in colored  # cyan applied


def test_md_subprocess_timeout(monkeypatch):
    monkeypatch.delenv("TRAILS_MD_TIMEOUT", raising=False)
    assert md_subprocess_timeout() is None
    monkeypatch.setenv("TRAILS_MD_TIMEOUT", "120")
    assert md_subprocess_timeout() == 120.0
    monkeypatch.setenv("TRAILS_MD_TIMEOUT", "0")
    assert md_subprocess_timeout() is None  # non-positive ignored
    monkeypatch.setenv("TRAILS_MD_TIMEOUT", "not-a-number")
    assert md_subprocess_timeout() is None


def test_checkpoint_format_version(tmp_path):
    pytest.importorskip("torch")
    from trails_md.checkpoints.manager import (
        CHECKPOINT_FORMAT_VERSION,
        CheckpointManager,
    )

    mgr = CheckpointManager(str(tmp_path))
    mgr.save(
        iteration=0,
        space_model=None,
        scaler={"k": 1},
        bin_state={},
        history={},
        sampler_state={"x": 1},
    )
    version_file = tmp_path / "iter_0" / "format_version"
    assert version_file.exists()
    assert int(version_file.read_text()) == CHECKPOINT_FORMAT_VERSION

    # Round-trips without raising and reads the stored state back.
    _, scaler, _, _, sampler_state = mgr.load(0)
    assert scaler == {"k": 1}
    assert sampler_state == {"x": 1}


def test_fited_to_fitted_pickle_shim():
    pytest.importorskip("torch")
    pytest.importorskip("deeptime")
    from trails_md.spaces.model import AdaptiveSpaceModel

    model = AdaptiveSpaceModel(space_mode="pca")
    # Simulate a legacy pickle state that used the misspelled attribute.
    state = dict(model.__dict__)
    state["fited"] = state.pop("fitted")
    restored = AdaptiveSpaceModel.__new__(AdaptiveSpaceModel)
    restored.__setstate__(state)
    assert hasattr(restored, "fitted")
    assert "fited" not in restored.__dict__


def test_validate_trajectory_files(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("MDAnalysis")
    from trails_md.core import TrailsMDCore

    good = tmp_path / "ok.xtc"
    good.write_bytes(b"\x00\x01\x02")
    empty = tmp_path / "empty.xtc"
    empty.write_bytes(b"")
    missing = tmp_path / "missing.xtc"

    # All good -> no error.
    TrailsMDCore._validate_trajectory_files([str(good)])
    # Empty or missing -> RuntimeError listing the offenders.
    with pytest.raises(RuntimeError, match="empty"):
        TrailsMDCore._validate_trajectory_files([str(good), str(empty)])
    with pytest.raises(RuntimeError, match="missing"):
        TrailsMDCore._validate_trajectory_files([str(missing)])


def test_no_weresampler_import():
    # The dead WEResampler stub and its export were removed in Phase 2.
    import trails_md.binning as binning

    assert not hasattr(binning, "WEResampler")
