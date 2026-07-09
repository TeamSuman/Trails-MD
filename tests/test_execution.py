"""Tests for the pluggable execution backends (local / SLURM / PBS).

No real scheduler is needed: a fake command-runner runs the array tasks
synchronously in-process, exercising the full submit -> poll -> collect -> retry
state machine, script rendering, and job-id parsing.
"""

from __future__ import annotations

import warnings
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

warnings.filterwarnings("ignore")

import trails_md.execution.run_task as run_task  # noqa: E402
from trails_md.engines.base import EngineFactory  # noqa: E402
from trails_md.execution import (  # noqa: E402
    ExecutionBackendFactory,
    build_walker_tasks,
    make_backend,
)
from trails_md.execution import local as local_mod  # noqa: E402
from trails_md.execution.pbs import PBSBackend  # noqa: E402
from trails_md.execution.slurm import SlurmBackend  # noqa: E402


# ── fake engine: directive carried via start_coords ─────────────────────────
class FakeEngine:
    calls: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def prepare(self, **kwargs):
        pass

    def run_production(
        self, run_index, start_coords, steps, traj_out, stride, device_index
    ):
        FakeEngine.calls[run_index] = FakeEngine.calls.get(run_index, 0) + 1
        Path(traj_out).parent.mkdir(parents=True, exist_ok=True)
        Path(traj_out).write_bytes(b"x")
        if start_coords == "fail":
            return False
        if start_coords == "transient":  # fail first attempt, succeed on retry
            return FakeEngine.calls[run_index] >= 2
        return True


EngineFactory.register("fake", FakeEngine)


@pytest.fixture(autouse=True)
def _reset_fake_calls():
    FakeEngine.calls.clear()
    yield


def _tasks(tmp_path, walkers):
    return build_walker_tasks(
        engine_name="fake",
        engine_kwargs={},
        prepare_kwargs={},
        walkers=walkers,
        steps=10,
        stride=1,
        outdir=tmp_path / "iter_0",
        iteration=0,
    )


# ── task building ───────────────────────────────────────────────────────────
def test_build_walker_tasks_names_and_payload(tmp_path):
    tasks = _tasks(tmp_path, ["ok", "ok"])
    assert [t.index for t in tasks] == [0, 1]
    assert tasks[0].traj_out.endswith("iter_0/iteration_0_0.xtc")
    assert tasks[1].start_coords == "ok"
    assert tasks[0].run_kwargs()["steps"] == 10


# ── local backend (synchronous executor to avoid spawn pickling) ────────────
class _SyncExecutor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut


def test_local_backend_runs_all_walkers(tmp_path, monkeypatch):
    monkeypatch.setattr(local_mod, "ProcessPoolExecutor", _SyncExecutor)
    backend = ExecutionBackendFactory.get("local", gpu_ids=[0], max_workers=2)
    results = backend.execute(_tasks(tmp_path, ["ok", "fail", "ok"]))
    assert results == [True, False, True]


class _HangingExecutor:
    """Hands out futures that never complete, to simulate a hung walker."""

    def __init__(self, *a, **k):
        self._processes = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        return Future()  # never set → never done


def test_local_backend_walker_timeout(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(local_mod, "ProcessPoolExecutor", _HangingExecutor)
    backend = ExecutionBackendFactory.get(
        "local", gpu_ids=[0], max_workers=2, walker_timeout=0.3
    )
    start = time.monotonic()
    results = backend.execute(_tasks(tmp_path, ["hang", "hang"]))
    elapsed = time.monotonic() - start
    assert results == [False, False]  # timed-out batch reported as failed
    assert elapsed < 5.0  # and it did not hang


# ── fake scheduler command runner ───────────────────────────────────────────
def _fake_runner(submit_id="4242"):
    def runner(cmd, timeout):
        prog = cmd[0]
        if prog in ("sbatch", "qsub"):
            script = Path(cmd[-1])
            manifest = None
            for line in script.read_text().splitlines():
                if line.startswith("MANIFEST="):
                    manifest = Path(line.split("=", 1)[1].strip().strip('"'))
            assert manifest is not None
            for entry in manifest.read_text().splitlines():
                entry = entry.strip()
                if entry:
                    task_pkl, result_json = entry.split()
                    run_task.main([task_pkl, result_json])
            return SimpleNamespace(returncode=0, stdout=submit_id, stderr="")
        # squeue / qstat: report no active jobs (markers already written).
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


@pytest.mark.parametrize("backend_cls", [SlurmBackend, PBSBackend])
def test_scheduler_backend_runs_and_collects(tmp_path, backend_cls):
    backend = backend_cls(
        command_runner=_fake_runner(),
        sleep_fn=lambda s: None,
        max_retries=0,
        python_executable="python",
    )
    results = backend.execute(_tasks(tmp_path, ["ok", "fail", "ok"]))
    assert results == [True, False, True]
    # Job artifacts written beside the iteration outputs.
    assert (tmp_path / "iter_0" / "_jobs" / "manifest_attempt0.txt").exists()
    assert (tmp_path / "iter_0" / "_jobs" / "submit_attempt0.sh").exists()


def test_scheduler_backend_retries_transient_failure(tmp_path):
    backend = SlurmBackend(
        command_runner=_fake_runner(),
        sleep_fn=lambda s: None,
        max_retries=1,
        python_executable="python",
    )
    results = backend.execute(_tasks(tmp_path, ["ok", "transient"]))
    assert results == [True, True]  # transient succeeds on the retry attempt
    # Retry produced a second attempt manifest.
    assert (tmp_path / "iter_0" / "_jobs" / "manifest_attempt1.txt").exists()


def test_scheduler_submit_failure_raises(tmp_path):
    def failing_runner(cmd, timeout):
        if cmd[0] in ("sbatch", "qsub"):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    backend = SlurmBackend(command_runner=failing_runner, sleep_fn=lambda s: None)
    with pytest.raises(RuntimeError, match="submission failed"):
        backend.execute(_tasks(tmp_path, ["ok"]))


# ── script rendering & job-id parsing ───────────────────────────────────────
def test_slurm_script_directives(tmp_path):
    backend = SlurmBackend(
        partition="gpu", account="proj", gpus_per_task=1, walltime="02:00:00"
    )
    script = backend._render_script(4, tmp_path / "m.txt", tmp_path / "logs")
    assert "#SBATCH --array=0-3" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gpus-per-task=1" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "trails_md.execution.run_task" in script


def test_pbs_script_directives(tmp_path):
    backend = PBSBackend(cpus_per_task=4, gpus_per_task=2, memory="8G")
    script = backend._render_script(3, tmp_path / "m.txt", tmp_path / "logs")
    assert "#PBS -V" in script
    assert "#PBS -J 0-2" in script
    assert "ncpus=4" in script and "ngpus=2" in script and "mem=8G" in script
    assert "PBS_ARRAY_INDEX" in script


def test_torque_script_directives(tmp_path):
    backend = ExecutionBackendFactory.get("torque", cpus_per_task=2)
    script = backend._render_script(3, tmp_path / "m.txt", tmp_path / "logs")
    assert "#PBS -V" in script
    assert "#PBS -t 0-2" in script
    assert "PBS_ARRAYID" in script




def test_slurm_parse_job_id():
    backend = SlurmBackend()
    assert backend._parse_job_id("123456\n") == "123456"
    assert backend._parse_job_id("123456;cluster\n") == "123456"


def test_pbs_parse_job_id():
    backend = PBSBackend()
    assert backend._parse_job_id("789[].pbs01\n") == "789[].pbs01"


# ── make_backend dispatch ───────────────────────────────────────────────────
def test_make_backend_dispatch():
    from trails_md.config import ExecutionConfig

    assert make_backend(None).__class__.__name__ == "LocalProcessBackend"
    assert (
        make_backend(ExecutionConfig(backend="local")).__class__.__name__
        == "LocalProcessBackend"
    )
    slurm = make_backend(ExecutionConfig(backend="slurm", partition="gpu"))
    assert slurm.__class__.__name__ == "SlurmBackend"
    assert slurm.partition == "gpu"
