"""Velocity inheritance (WE kinetics mode).

Exploration mode redraws Maxwell-Boltzmann velocities at every respawn: walkers
are independent restarts, fine for discovery but not a continuous trajectory.
Kinetics mode instead CONTINUES each walker from its parent's endpoint velocities,
so weighted ensemble is an unbiased resampling of unperturbed dynamics and a rate
can be read from it. These tests pin the contract:

* the config guard (kinetics mode only with WE + OpenMM),
* the endpoint-State round-trip the engine writes and the orchestrator reads,
* the physics: continuing from a saved State reproduces an uninterrupted run,
  while redrawing velocities does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alanine_dipeptide"


def test_config_rejects_kinetics_without_we():
    from trails_md.config import TrailsMDConfig

    base = dict(
        system=dict(conf_file="c.pdb", top_file="t.pdb"),
        engine=dict(md_engine="openmm"),
        spawning=dict(spawn_scheme="density", inherit_velocities=True),
    )
    with pytest.raises(ValueError, match="requires spawn_scheme: we"):
        TrailsMDConfig(**base)


def test_config_rejects_kinetics_on_non_openmm():
    from trails_md.config import TrailsMDConfig

    with pytest.raises(ValueError, match="only for the OpenMM engine"):
        TrailsMDConfig(
            system=dict(conf_file="c.pdb", top_file="t.pdb"),
            engine=dict(md_engine="gromacs"),
            spawning=dict(spawn_scheme="we", inherit_velocities=True),
        )


def test_config_accepts_kinetics_with_we_openmm():
    from trails_md.config import TrailsMDConfig

    cfg = TrailsMDConfig(
        system=dict(conf_file="c.pdb", top_file="t.pdb"),
        engine=dict(md_engine="openmm"),
        spawning=dict(spawn_scheme="we", inherit_velocities=True),
    )
    assert cfg.spawning.inherit_velocities is True


def test_we_spawner_records_selected_parents():
    """Kinetics mode needs to know which live walker each spawn continues from."""
    from trails_md.spawners.we import WESpawner

    fpw = 10
    progress = np.array([2.0, 3.0, 4.0, 5.0, 40.0, 90.0, 6.0, 7.0])
    points = np.column_stack([np.repeat(progress, fpw), np.zeros(len(progress) * fpw)])
    sp = WESpawner(n_bins=[6, 1], min_values=[0, -1], max_values=[100, 1],
                   target_per_bin=2, seed=0)
    idx = sp.sample(points, top_n=len(progress))
    assert sp.selected_parents is not None
    assert len(sp.selected_parents) == len(idx)
    # every parent is a valid current-walker index
    assert all(0 <= p < len(progress) for p in sp.selected_parents)
    # A range check is not enough on its own: `selected_parents = [0, 0, ...]` -- every
    # walker inheriting walker 0's velocities -- satisfies it, and that is exactly the
    # bug this test is named for, since the orchestrator reads these to decide whose
    # endpoint State each child continues from. Pin parent to the frame index it must
    # agree with: `idx[i]` is parent p's endpoint, so p is recoverable from it.
    for i, parent in enumerate(sp.selected_parents):
        assert idx[i] == (parent + 1) * fpw - 1, (
            f"spawn {i} restarts from frame {idx[i]}, which is not the endpoint of "
            f"its recorded parent {parent} (frame {(parent + 1) * fpw - 1}) -- "
            "positions and velocities would come from different walkers"
        )
    # The ensemble must not collapse onto one parent (WE splits AND merges).
    assert len(set(sp.selected_parents)) > 1, (
        f"all spawns claim the same parent: {sp.selected_parents}"
    )


def _verlet_sim():
    """Deterministic (Verlet) integrator: positions + velocities fully determine
    the future, so a split walker that inherits both reproduces the uninterrupted
    trajectory exactly. This isolates velocity inheritance from the Langevin noise
    stream, which -- correctly -- restarts on a split and makes stochastic
    continuation unbiased but not bit-identical."""
    from openmm import Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    pdb = PDBFile(str(EXAMPLE / "structure.pdb"))
    system = XmlSerializer.deserialize((EXAMPLE / "system.xml").read_text())
    sim = Simulation(pdb.topology, system, VerletIntegrator(0.002 * unit.picoseconds),
                     Platform.getPlatformByName("CPU"), {"Threads": "1"})
    sim.context.setPositions(pdb.positions)
    return sim, pdb


def test_continuing_from_saved_state_reproduces_uninterrupted_run(monkeypatch):
    """The physics the whole mode rests on: state save + restore == no interruption."""
    from openmm import unit

    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    sim, _ = _verlet_sim()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 7)

    # Reference: 200 uninterrupted steps.
    sim.step(100)
    mid = sim.context.getState(getPositions=True, getVelocities=True)
    sim.step(100)
    ref = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Split at the midpoint: restore pos+vel into a fresh context and continue.
    sim2, _ = _verlet_sim()
    sim2.context.setPositions(mid.getPositions())
    sim2.context.setVelocities(mid.getVelocities())    # inherit velocities
    sim2.step(100)
    cont = sim2.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(cont, ref, atol=1e-5)

    # Contrast: redrawing velocities (exploration mode) must NOT reproduce it.
    sim3, _ = _verlet_sim()
    sim3.context.setPositions(mid.getPositions())
    sim3.context.setVelocitiesToTemperature(300 * unit.kelvin, 999)
    sim3.step(100)
    redrawn = sim3.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert not np.allclose(redrawn, ref, atol=1e-5)


def test_engine_writes_endstate_when_enabled(tmp_path, monkeypatch):
    """save_endstate=True must drop a readable positions+velocities+box sidecar."""
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    from openmm.app import PDBFile

    from trails_md.engines.openmm import OpenMMEngine

    eng = OpenMMEngine(platform_name="CPU", dt=0.002, seed=3, save_endstate=True)
    eng.prepare(conf=EXAMPLE / "structure.pdb", top=EXAMPLE / "structure.pdb",
                system_file=EXAMPLE / "system.py")
    traj = tmp_path / "w.xtc"
    pos = PDBFile(str(EXAMPLE / "structure.pdb")).getPositions(asNumpy=True)
    assert eng.run_production(run_index=0, start_coords=pos, steps=100,
                              traj_out=traj, stride=50, device_index=0)
    end = Path(f"{traj}.endstate.npz")
    assert end.exists()
    d = np.load(end)
    assert d["positions"].shape == d["velocities"].shape
    assert d["box"].shape == (3, 3)
    assert not np.isnan(d["velocities"]).any()
