"""Tests for MSM analysis data utilities and plotting."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.analysis import data  # noqa: E402


def _make_run(tmp_path, n_iters=4):
    """Build a synthetic run dir with msm.npz + cvs.npz per iteration."""
    rng = np.random.default_rng(0)
    for it in range(n_iters):
        d = tmp_path / f"iter_{it}"
        d.mkdir()
        np.savez_compressed(
            d / "msm.npz",
            lagtime=np.asarray(10),
            timescales=np.asarray([100.0 + it, 50.0 + it], dtype=float),
            stationary_distribution=np.array([0.5, 0.3, 0.2]),
            transition_matrix=np.eye(3),
            cluster_centers=np.zeros((3, 2)),
            vamp2_score=np.asarray([2.0 + 0.1 * it]),
            metastable_populations=np.array([0.6, 0.4]),
            its_lagtimes=np.array([1.0, 2.0, 5.0]),
            its_timescales=np.array([[90.0, 40.0], [95.0, 45.0], [100.0, 50.0]]),
        )
        np.savez_compressed(d / "cvs.npz", cvs=rng.normal(size=(50, 2)))
    return tmp_path


def test_load_msm_series(tmp_path):
    series = data.load_msm_series(_make_run(tmp_path))
    assert list(series["iterations"]) == [0, 1, 2, 3]
    assert series["vamp2"][0] == pytest.approx(2.0)
    assert series["timescales"].shape == (4, 2)
    assert series["timescales"][3, 0] == pytest.approx(103.0)


def test_load_latest_and_cv_points(tmp_path):
    run = _make_run(tmp_path)
    latest = data.load_latest_msm(run)
    assert "its_lagtimes" in latest and "transition_matrix" in latest
    points = data.load_cv_points(run)
    assert points.shape == (200, 2)  # 4 iters x 50 frames


def test_load_msm_series_empty(tmp_path):
    series = data.load_msm_series(tmp_path)
    assert series["iterations"].size == 0


def test_free_energy_from_populations():
    f = data.free_energy_from_populations([0.5, 0.25, 0.25], temperature=300.0)
    assert f.min() == pytest.approx(0.0)
    # Less populated states have higher free energy.
    assert f[1] > f[0] and f[2] > f[0]


def test_free_energy_surface(tmp_path):
    pts = np.random.default_rng(0).normal(size=(2000, 2))
    f, xe, ye = data.free_energy_surface(pts, bins=20)
    assert f.shape == (20, 20)
    assert np.nanmin(f) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        data.free_energy_surface(np.zeros((10, 1)))


def test_plots_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from autosampler.analysis import plots

    run = _make_run(tmp_path)
    out = plots.plot_convergence_report(run)
    assert out.exists()
    # Individual plotters return an Axes without raising.
    series = data.load_msm_series(run)
    latest = data.load_latest_msm(run)
    assert plots.plot_vamp2_convergence(series) is not None
    assert plots.plot_implied_timescales(
        latest["its_lagtimes"], latest["its_timescales"]
    ) is not None
    assert plots.plot_msm_network(
        latest["transition_matrix"], latest["stationary_distribution"]
    ) is not None


def test_analysis_cli_errors_without_msm(tmp_path):
    from autosampler.analysis_cli import main

    with pytest.raises(SystemExit):
        main(["--run-dir", str(tmp_path)])
