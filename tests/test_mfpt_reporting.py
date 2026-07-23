"""User-facing MFPT reporting: the shared estimator, the run-dir loaders, and the
`trails-md-analyze` CLI path for a weighted-ensemble kinetics run.

The MFPT itself is validated for correctness elsewhere (test_we_rate_exactness.py,
against a brute-force reference); here we test that the *reporting* surfaces are wired
up: the estimator returns the right number and diagnostics, the flux series and tau are
recovered from a run directory, and the analyze CLI prints the rate and writes a plot.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from trails_md.analysis import data as adata
from trails_md.spawners.we import WESpawner, steady_state_mfpt


def test_steady_state_mfpt_matches_hill_relation():
    # constant flux f per tau -> MFPT = tau / f (converted ps -> ns)
    flux = [0.01] * 100
    r = steady_state_mfpt(flux, tau_ps=2.0, discard_fraction=0.5)
    assert r.mfpt_ns == pytest.approx(2.0 / 0.01 / 1000.0)  # 0.2 ns
    assert r.n_iterations == 100
    assert r.n_flux_events == 100
    assert r.plateau_ratio == pytest.approx(1.0)
    assert r.converged


def test_transient_is_discarded():
    # zero flux for the first half (transient), real flux after -> discard recovers it
    flux = [0.0] * 50 + [0.02] * 50
    r = steady_state_mfpt(flux, tau_ps=2.0, discard_fraction=0.5)
    assert r.mfpt_ns == pytest.approx(2.0 / 0.02 / 1000.0)  # transient dropped
    # without discarding, the mean flux (and hence rate) would be wrong
    r0 = steady_state_mfpt(flux, tau_ps=2.0, discard_fraction=0.0)
    assert r0.mfpt_ns > r.mfpt_ns


def test_no_flux_yet_returns_none():
    assert steady_state_mfpt([], tau_ps=2.0).mfpt_ns is None
    assert steady_state_mfpt([0.0, 0.0], tau_ps=2.0).mfpt_ns is None


def test_rising_flux_flags_not_converged():
    flux = list(np.linspace(0.001, 0.02, 100))  # still climbing
    r = steady_state_mfpt(flux, tau_ps=2.0, discard_fraction=0.5)
    assert r.plateau_ratio > 1.25
    assert not r.converged


def test_spawner_method_delegates_to_shared_function():
    sp = WESpawner()
    sp.flux_history = [0.01] * 40
    assert sp.mfpt(tau_ps=2.0) == pytest.approx(
        steady_state_mfpt(sp.flux_history, 2.0).mfpt_ns
    )


def _make_run(tmp_path, flux, step=1000, dt=0.002):
    (tmp_path / "output.log").write_text(
        f"# Trails-MD run log\n# step={step}\n# dt={dt}\n# spawn_scheme=we\n"
    )
    ck = tmp_path / "checkpoints" / "iter_10"
    ck.mkdir(parents=True)
    with open(ck / "sampler_state.pkl", "wb") as fh:
        pickle.dump({"spawner": {"flux_history": list(flux)}}, fh)
    return tmp_path


def test_load_flux_history_and_meta(tmp_path):
    run = _make_run(tmp_path, [0.01] * 30)
    assert adata.load_flux_history(run) == [0.01] * 30
    meta = adata.load_run_meta(run)
    assert meta["step"] == 1000 and meta["dt"] == pytest.approx(0.002)
    # tau recoverable = step * dt
    assert meta["step"] * meta["dt"] == pytest.approx(2.0)


def test_load_flux_history_empty_for_non_kinetics(tmp_path):
    (tmp_path / "output.log").write_text("# step=1000\n# dt=0.002\n")
    assert adata.load_flux_history(tmp_path) == []


def test_analyze_cli_reports_mfpt(tmp_path, capsys):
    from trails_md.analysis_cli import main

    run = _make_run(tmp_path, [0.01] * 40)
    main(["--run-dir", str(run)])
    out = capsys.readouterr().out
    assert "MFPT estimate" in out
    assert "0.2 ns" in out  # tau 2 ps / flux 0.01 = 0.2 ns
    # the flux plot is written when matplotlib is available
    plot = run / "analysis" / "flux_convergence.png"
    try:
        import matplotlib  # noqa: F401

        assert plot.exists()
    except ImportError:
        pass


def test_analyze_cli_errors_when_nothing_present(tmp_path):
    from trails_md.analysis_cli import main

    (tmp_path / "output.log").write_text("# step=1000\n# dt=0.002\n")
    with pytest.raises(SystemExit, match="Nothing to analyze"):
        main(["--run-dir", str(tmp_path)])
