"""Regression tests for the HPC-scale review fixes.

These cover the scheduler polling/robustness fixes, checkpoint completeness
gating, engine correctness fixes, and the new config validators. They only need
numpy + pydantic (no torch / OpenMM / deeptime), matching the lazy-import design.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trails_md.execution.base import WalkerTask
from trails_md.execution.scheduler import parse_walltime_seconds
from trails_md.execution.slurm import SlurmBackend


# ── SLURM job-state polling ─────────────────────────────────────────────────
def test_slurm_job_active_detects_running_array_elements():
    """`squeue --array` prints `<jobid>_<taskid>`; the poller must see it as active."""
    slurm = SlurmBackend()
    running = "12345_0   gpu trails-md user R 0:05 1 node01\n12345_1 gpu t u R 0:05 1 n2"
    assert slurm._job_active("12345", running, 0) is True
    # A pending bracketed range is also still active.
    assert slurm._job_active("12345", "12345_[2-9] gpu t u PD 0:00 1 (Resources)", 0)
    # A plain (non-array) id line still matches.
    assert slurm._job_active("12345", "12345 gpu t u R 0:05 1 node01", 0) is True


def test_slurm_job_active_empty_output_means_done():
    slurm = SlurmBackend()
    assert slurm._job_active("12345", "", 0) is False
    assert slurm._job_active("12345", "   \n", 0) is False
    # Non-zero return code => job gone.
    assert slurm._job_active("12345", "12345_0 ...", 1) is False


def test_slurm_job_active_does_not_match_unrelated_id():
    slurm = SlurmBackend()
    # A different job id sharing a digit prefix must not read as active.
    assert slurm._job_active("123", "9123_0 gpu t u R 0:01 1 n", 0) is False


def test_slurm_array_directive_throttles_with_max_in_flight(tmp_path):
    slurm = SlurmBackend(max_in_flight=4)
    directives = slurm._directives(100, tmp_path)
    assert any("--array=0-99%4" in line for line in directives)
    # Without a cap, no `%N` suffix.
    plain = SlurmBackend()._directives(100, tmp_path)
    assert any(line.endswith("--array=0-99") for line in plain)


def test_render_script_uses_tab_delimited_manifest(tmp_path):
    """cut must default to TAB so paths with spaces survive the field split."""
    slurm = SlurmBackend()
    script = slurm._render_script(3, tmp_path / "m.txt", tmp_path / "logs")
    assert "cut -f1" in script and "cut -f2" in script
    assert 'cut -d" "' not in script


def _fake_scheduler_runner(submit_calls):
    """A fake runner that executes each submitted manifest's tasks synchronously."""
    from types import SimpleNamespace

    import trails_md.execution.run_task as run_task

    def runner(cmd, timeout):
        if cmd[0] in ("sbatch", "qsub"):
            submit_calls.append(cmd)
            script = __import__("pathlib").Path(cmd[-1])
            manifest = None
            for line in script.read_text().splitlines():
                if line.startswith("MANIFEST="):
                    manifest = __import__("pathlib").Path(
                        line.split("=", 1)[1].strip().strip('"')
                    )
            for entry in manifest.read_text().splitlines():
                if entry.strip():
                    task_pkl, result_json = entry.split("\t")
                    run_task.main([task_pkl, result_json])
            return SimpleNamespace(returncode=0, stdout="777", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")  # squeue: done

    return runner


def test_array_chunking_splits_large_batches(tmp_path):
    """5 walkers with max_array_size=2 -> 3 sub-array submissions, all succeed."""
    submit_calls: list = []
    slurm = SlurmBackend(
        command_runner=_fake_scheduler_runner(submit_calls),
        sleep_fn=lambda s: None,
        max_array_size=2,
    )
    (tmp_path / "iter_0").mkdir(parents=True, exist_ok=True)
    # Register a trivial engine under a UNIQUE name (do not clobber the shared
    # "fake" engine used by tests/test_execution.py).
    from trails_md.engines.base import EngineFactory

    class _OkEngine:
        def __init__(self, **k):
            pass

        def prepare(self, **k):
            pass

        def run_production(self, traj_out, **k):
            __import__("pathlib").Path(traj_out).write_bytes(b"x")
            return True

    EngineFactory.register("chunk_ok", _OkEngine)
    tasks = [
        WalkerTask(
            index=i,
            engine_name="chunk_ok",
            engine_kwargs={},
            prepare_kwargs={},
            steps=10,
            stride=1,
            traj_out=str(tmp_path / "iter_0" / f"iteration_0_{i}.xtc"),
            start_coords="ok",
        )
        for i in range(5)
    ]
    results = slurm.execute(tasks)
    assert results == [True] * 5
    assert len(submit_calls) == 3  # ceil(5/2) sub-arrays


# ── Scheduler robustness ────────────────────────────────────────────────────
def test_parse_walltime_seconds():
    assert parse_walltime_seconds("01:00:00") == 3600
    assert parse_walltime_seconds("02:30:00") == 2 * 3600 + 30 * 60
    assert parse_walltime_seconds("30:00") == 30 * 60  # MM:SS
    assert parse_walltime_seconds("60") == 60 * 60  # bare integer is minutes in SLURM
    assert parse_walltime_seconds("1-00:00:00") == 86400  # SLURM D-HH:MM:SS
    assert parse_walltime_seconds("nonsense") is None



def test_scheduler_cancels_and_returns_on_wait_timeout():
    """A job that never finishes and stays 'active' must not hang forever."""
    cancelled: list[list[str]] = []

    def runner(cmd, timeout):
        from types import SimpleNamespace

        if cmd[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="999\n", stderr="")
        if cmd[0] == "scancel":
            cancelled.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        # squeue: always reports the array as still running.
        return SimpleNamespace(returncode=0, stdout="999_0 gpu t u R 0:10 1 n", stderr="")

    clock = {"t": 0.0}

    def fake_clock():
        clock["t"] += 100.0  # each read advances time so the deadline is hit fast
        return clock["t"]

    slurm = SlurmBackend(
        command_runner=runner,
        sleep_fn=lambda s: None,
        clock_fn=fake_clock,
        wait_timeout=50.0,
    )
    # No markers will ever appear; the deadline must trigger a scancel + return.
    slurm._wait_for_completion("999", [__import__("pathlib").Path("/nonexistent/marker")])
    assert cancelled, "expected the hung job to be cancelled at the wait deadline"


def test_scheduler_empty_job_id_raises(tmp_path):
    from types import SimpleNamespace

    def runner(cmd, timeout):
        if cmd[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="\n", stderr="")  # no id
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    slurm = SlurmBackend(command_runner=runner, sleep_fn=lambda s: None)
    task = WalkerTask(
        index=0,
        engine_name="fake",
        engine_kwargs={},
        prepare_kwargs={},
        steps=10,
        stride=1,
        traj_out=str(tmp_path / "iter_0" / "iteration_0_0.xtc"),
    )
    (tmp_path / "iter_0").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="no parseable job id"):
        slurm.execute([task])


# ── WalkerTask device sentinel ──────────────────────────────────────────────
def test_walker_task_device_index_defaults_to_scheduler_sentinel():
    task = WalkerTask(
        index=0,
        engine_name="openmm",
        engine_kwargs={},
        prepare_kwargs={},
        steps=1,
        stride=1,
        traj_out="x.xtc",
    )
    assert task.device_index == -1  # inherit scheduler CUDA_VISIBLE_DEVICES


# ── Checkpoint completeness gating ──────────────────────────────────────────
def test_broken_delta_chain_is_reported_loudly(tmp_path, caplog):
    import logging
    import pytest

    from trails_md.checkpoints.manager import CheckpointManager

    mgr = CheckpointManager(str(tmp_path))
    mgr.save(0, None, {}, {}, {0: "a"})
    mgr.save(1, None, {}, {}, {0: "a", 1: "b"})
    mgr.save(2, None, {}, {}, {0: "a", 1: "b", 2: "c"})
    # Simulate an operator pruning a middle checkpoint's delta after the fact.
    (tmp_path / "iter_1" / "history.pkl").unlink()

    with pytest.raises(RuntimeError):
        mgr.load(2)  # must raise by default when chain is broken

    with caplog.at_level(logging.ERROR):
        _, _, _, full, _ = mgr.load(2, ignore_missing_history=True)
    assert any("chain is broken" in r.message for r in caplog.records)
    assert 1 not in full  # the lost delta's key is genuinely absent (now flagged)


def test_incomplete_checkpoint_is_ignored(tmp_path):
    from trails_md.checkpoints.manager import CheckpointManager

    mgr = CheckpointManager(str(tmp_path))
    mgr.save(0, None, {}, {}, {0: "a"})
    mgr.save(1, None, {}, {}, {0: "a", 1: "b"})
    # Simulate a crash mid-save of iter_2: files present, but no completion marker.
    torn = tmp_path / "iter_2"
    torn.mkdir()
    (torn / "scaler.pkl").write_bytes(b"partial")

    # The torn checkpoint must not be chosen as the resume target.
    assert mgr.latest_iteration() == 1
    with pytest.raises(FileNotFoundError, match="incomplete"):
        mgr.load(2)
    # And it must be excluded from delta reconstruction.
    _, _, _, full, _ = mgr.load(1)
    assert set(full) == {0, 1}


# ── Triclinic box conversion ────────────────────────────────────────────────
def test_box_vectors_orthorhombic():
    from trails_md.engines.base import box_vectors_to_abc_angles

    # nm vectors -> Angstrom cell.
    out = box_vectors_to_abc_angles([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]])
    assert np.allclose(out, [30.0, 40.0, 50.0, 90.0, 90.0, 90.0])


def test_box_vectors_triclinic_roundtrip():
    from trails_md.engines.base import box_vectors_to_abc_angles

    # Build lower-triangular box vectors (nm) for a known triclinic cell, then
    # confirm the helper recovers the a,b,c / alpha,beta,gamma we started from.
    a, b, c = 40.0, 42.0, 45.0  # Angstrom
    alpha, beta, gamma = 70.0, 80.0, 100.0  # degrees
    la, lb, lc = a / 10, b / 10, c / 10  # nm
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    ax, ay, az = la, 0.0, 0.0
    bx, by, bz = lb * cg, lb * sg, 0.0
    cx = lc * cb
    cy = lc * (ca - cb * cg) / sg
    cz = math.sqrt(max(lc * lc - cx * cx - cy * cy, 0.0))
    out = box_vectors_to_abc_angles([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])
    assert np.allclose(out, [a, b, c, alpha, beta, gamma], atol=1e-4)


def test_box_vectors_none():
    from trails_md.engines.base import box_vectors_to_abc_angles

    assert box_vectors_to_abc_angles(None) is None


# ── Angle sin/cos encoding ──────────────────────────────────────────────────
def test_encode_angles_sincos_is_continuous_across_pi():
    from trails_md.utils.math import encode_angles_sincos

    near_plus = encode_angles_sincos(np.array([[math.pi - 1e-3]]))
    near_minus = encode_angles_sincos(np.array([[-math.pi + 1e-3]]))
    # Raw radians differ by ~2pi; the sin/cos embedding must be nearly identical.
    assert near_plus.shape == (1, 2)
    assert np.linalg.norm(near_plus - near_minus) < 1e-2


# ── Amber cold-start fix ────────────────────────────────────────────────────
def test_rng_state_roundtrip_reproduces_stream():
    """Capturing then restoring RNG state must reproduce subsequent draws."""
    import random

    from trails_md.utils.seeds import capture_rng_state, restore_rng_state

    np.random.seed(123)
    random.seed(123)
    _ = np.random.rand(10)  # advance the stream (mimic completed iterations)
    _ = [random.random() for _ in range(10)]

    snapshot = capture_rng_state()
    expected_np = np.random.rand(5)
    expected_py = [random.random() for _ in range(5)]

    # Simulate a crash + resume from a *different* base seed, then restore.
    np.random.seed(999)
    random.seed(999)
    restore_rng_state(snapshot)
    assert np.allclose(np.random.rand(5), expected_np)
    assert [random.random() for _ in range(5)] == expected_py


def test_rng_state_is_torch_independent_and_picklable():
    import pickle

    from trails_md.utils.seeds import capture_rng_state

    snap = capture_rng_state()
    # Must round-trip through pickle without requiring torch to unpickle.
    reloaded = pickle.loads(pickle.dumps(snap))
    assert "numpy" in reloaded and "python" in reloaded


def test_amber_default_input_sets_tempi(tmp_path):
    from trails_md.engines.amber import AmberEngine

    eng = AmberEngine(temperature=310.0, amber_executable="pmemd")
    out = tmp_path / "md.in"
    eng._write_input(str(out), steps=1000, stride=100, trajectory_format="netcdf")
    text = out.read_text()
    assert "tempi=310.00" in text  # velocities generated at the target T, not 0 K
    assert "temp0=310.00" in text


def test_engine_seeding(tmp_path):
    from trails_md.engines.amber import AmberEngine
    from trails_md.engines.gromacs import GromacsEngine

    gmx = GromacsEngine(temperature=300.0, seed=12345)
    gmx_out = tmp_path / "gromacs.mdp"
    gmx._write_mdp(str(gmx_out), steps=100, stride=10)
    assert "gen_seed                = 12345" in gmx_out.read_text()

    amber = AmberEngine(temperature=300.0, seed=54321)
    amber_out = tmp_path / "amber.in"
    amber._write_input(str(amber_out), steps=100, stride=10, trajectory_format="netcdf")
    assert "ig=54321," in amber_out.read_text()


def test_per_walker_task_seeding(tmp_path):
    from trails_md.execution.base import build_walker_tasks

    tasks = build_walker_tasks(
        engine_name="openmm",
        engine_kwargs={"seed": 42},
        prepare_kwargs={},
        steps=100,
        stride=10,
        outdir=tmp_path,
        iteration=2,
        walkers=["coord1.pdb", "coord2.pdb", "coord3.pdb"],
    )
    seeds = [t.engine_kwargs.get("seed") for t in tasks]
    assert len(seeds) == 3
    assert len(set(seeds)) == 3  # all unique
    assert all(isinstance(s, int) and s > 0 for s in seeds)

