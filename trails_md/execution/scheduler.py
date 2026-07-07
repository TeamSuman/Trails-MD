"""Base class for HPC scheduler execution backends (SLURM, PBS).

Each iteration's walkers are dispatched as one *array job*. The flow is:

1. Pickle each pending :class:`WalkerTask` and write a manifest line
   ``<task.pkl> <result.json>`` per task.
2. Render a scheduler script whose array elements read their manifest line and
   invoke :mod:`trails_md.execution.run_task`.
3. Submit, then poll until every result marker appears (or the job leaves the
   queue). Missing/failed markers are resubmitted up to ``max_retries`` times.

Completion is driven by **filesystem result markers**, not scheduler accounting,
which makes the logic portable and unit-testable: the only external seam is
``command_runner`` (a callable wrapping ``subprocess.run``), which tests replace
with a fake scheduler that runs tasks synchronously.
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
import time
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path

from .base import ExecutionBackend, WalkerTask

CommandRunner = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]

logger = logging.getLogger(__name__)


def _default_command_runner(cmd: list[str], timeout: float):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def parse_walltime_seconds(walltime: str) -> float | None:
    """Best-effort conversion of a scheduler walltime string to seconds.

    Accepts ``HH:MM:SS``, ``MM:SS``, ``D-HH:MM:SS`` (SLURM), or a bare integer
    number of seconds/minutes. Returns ``None`` when the format is not
    recognised, so callers can fall back to an unbounded wait.
    """
    if not walltime:
        return None
    text = str(walltime).strip()
    days = 0
    if "-" in text:  # SLURM D-HH:MM:SS
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, s = 0, 0, nums[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


class SchedulerBackend(ExecutionBackend):
    """Shared submit/poll/retry machinery for array-job schedulers."""

    def __init__(
        self,
        *,
        partition: str | None = None,
        account: str | None = None,
        walltime: str = "01:00:00",
        cpus_per_task: int = 1,
        gpus_per_task: int = 0,
        memory: str | None = None,
        max_retries: int = 1,
        poll_interval: float = 30.0,
        submit_timeout: float = 60.0,
        max_in_flight: int | None = None,
        wait_timeout: float | None = None,
        marker_grace: float = 30.0,
        module_loads: list[str] | None = None,
        extra_directives: list[str] | None = None,
        job_name: str = "trails-md",
        command_runner: CommandRunner | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock_fn: Callable[[], float] | None = None,
        python_executable: str | None = None,
        **_,
    ):
        self.partition = partition
        self.account = account
        self.walltime = walltime
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        self.memory = memory
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.submit_timeout = submit_timeout
        # Cap concurrently-running array elements (SLURM ``%N``); ``None`` = no cap.
        self.max_in_flight = max_in_flight
        # Overall ceiling (seconds) on how long to wait for one array job before
        # cancelling it and treating unfinished walkers as failed. ``None``
        # derives a generous bound from ``walltime`` so a held / never-scheduled
        # job cannot hang the campaign forever.
        self.wait_timeout = wait_timeout
        # How long to keep re-checking result markers after the job leaves the
        # queue, to absorb shared-filesystem (NFS/Lustre) metadata lag.
        self.marker_grace = marker_grace
        self.module_loads = list(module_loads or [])
        self.extra_directives = list(extra_directives or [])
        self.job_name = job_name
        self._run_command = command_runner or _default_command_runner
        self._sleep = sleep_fn or time.sleep
        self._clock = clock_fn or time.monotonic
        import sys

        self.python_executable = python_executable or sys.executable

    # ── scheduler-specific hooks ────────────────────────────────────────────
    @property
    @abstractmethod
    def array_index_var(self) -> str:
        """Env var holding the array index (e.g. ``SLURM_ARRAY_TASK_ID``)."""

    @abstractmethod
    def _directives(self, n_tasks: int, logdir: Path) -> list[str]:
        """Scheduler directive lines (``#SBATCH`` / ``#PBS``)."""

    @abstractmethod
    def _submit_command(self, script_path: Path) -> list[str]:
        ...

    @abstractmethod
    def _parse_job_id(self, stdout: str) -> str:
        ...

    @abstractmethod
    def _poll_command(self, job_id: str) -> list[str]:
        ...

    @abstractmethod
    def _job_active(self, job_id: str, poll_stdout: str, returncode: int) -> bool:
        """True while any array element is still queued/running."""

    def _cancel_command(self, job_id: str) -> list[str] | None:
        """Command to cancel a running job (``scancel`` / ``qdel``).

        Returns ``None`` when cancellation is unsupported. Overridden by the
        concrete backends.
        """
        return None

    # ── core flow ───────────────────────────────────────────────────────────
    def execute(self, tasks: list[WalkerTask]) -> list[bool]:
        if not tasks:
            return []

        iter_dir = Path(tasks[0].traj_out).parent
        jobdir = iter_dir / "_jobs"
        jobdir.mkdir(parents=True, exist_ok=True)

        task_files: dict[int, Path] = {}
        result_files: dict[int, Path] = {}
        for task in tasks:
            tf = jobdir / f"task_{task.index}.pkl"
            with open(tf, "wb") as handle:
                pickle.dump(task, handle)
            task_files[task.index] = tf
            result_files[task.index] = jobdir / f"result_{task.index}.json"

        success: dict[int, bool] = {}
        for attempt in range(self.max_retries + 1):
            pending = [t.index for t in tasks if not success.get(t.index, False)]
            if not pending:
                break
            self._dispatch_attempt(attempt, pending, task_files, result_files, jobdir)
            for idx in pending:
                success[idx] = self._read_success(result_files[idx])

        return [success.get(task.index, False) for task in tasks]

    def _dispatch_attempt(
        self,
        attempt: int,
        pending: list[int],
        task_files: dict[int, Path],
        result_files: dict[int, Path],
        jobdir: Path,
    ) -> None:
        # Clear stale markers for the indices we are about to (re)run.
        for idx in pending:
            result_files[idx].unlink(missing_ok=True)

        manifest = jobdir / f"manifest_attempt{attempt}.txt"
        manifest.write_text(
            "\n".join(f"{task_files[i]} {result_files[i]}" for i in pending) + "\n"
        )
        logdir = jobdir / f"logs_attempt{attempt}"
        logdir.mkdir(exist_ok=True)
        script = self._render_script(len(pending), manifest, logdir)
        script_path = jobdir / f"submit_attempt{attempt}.sh"
        script_path.write_text(script)

        try:
            proc = self._run_command(
                self._submit_command(script_path), self.submit_timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{type(self).__name__} submission timed out after "
                f"{self.submit_timeout}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"{type(self).__name__} submission failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        job_id = self._parse_job_id(proc.stdout)
        if not job_id:
            raise RuntimeError(
                f"{type(self).__name__} submission returned no parseable job id; "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        self._wait_for_completion(job_id, [result_files[i] for i in pending])

    def _render_script(self, n_tasks: int, manifest: Path, logdir: Path) -> str:
        lines = ["#!/bin/bash"]
        lines += self._directives(n_tasks, logdir)
        lines.append("set -euo pipefail")
        lines += self.module_loads
        lines.append(f'MANIFEST="{manifest}"')
        lines.append(f'LINE=$(sed -n "$((${self.array_index_var}+1))p" "$MANIFEST")')
        lines.append('TASK_PKL=$(echo "$LINE" | cut -d" " -f1)')
        lines.append('RESULT_JSON=$(echo "$LINE" | cut -d" " -f2)')
        lines.append(
            f'"{self.python_executable}" -m trails_md.execution.run_task '
            '"$TASK_PKL" "$RESULT_JSON"'
        )
        return "\n".join(lines) + "\n"

    def _resolve_wait_timeout(self) -> float | None:
        if self.wait_timeout is not None:
            return self.wait_timeout
        # Derive a generous ceiling from the requested walltime: the job cannot
        # legitimately run longer than its walltime, so allow that plus queueing
        # slack. Unrecognised walltime strings fall back to an unbounded wait.
        base = parse_walltime_seconds(self.walltime)
        if base is None:
            return None
        return base * 2.0 + 3600.0

    def _wait_for_completion(self, job_id: str, expected: list[Path]) -> None:
        deadline_budget = self._resolve_wait_timeout()
        start = self._clock()
        while True:
            if all(path.exists() for path in expected):
                return
            if deadline_budget is not None and self._clock() - start > deadline_budget:
                logger.error(
                    "%s job %s exceeded wait_timeout=%.0fs; cancelling and treating "
                    "unfinished walkers as failed.",
                    type(self).__name__,
                    job_id,
                    deadline_budget,
                )
                self._cancel_job(job_id)
                return
            try:
                proc = self._run_command(
                    self._poll_command(job_id), self.submit_timeout
                )
            except subprocess.TimeoutExpired:
                # A slow scheduler poll (common at scale) must not crash the
                # campaign — the filesystem markers remain the source of truth.
                logger.warning(
                    "%s poll for job %s timed out after %.0fs; retrying next cycle.",
                    type(self).__name__,
                    job_id,
                    self.submit_timeout,
                )
                self._sleep(self.poll_interval)
                continue
            if not self._job_active(job_id, proc.stdout, proc.returncode):
                # Job left the queue; re-check markers over a grace window to
                # absorb shared-filesystem metadata lag before giving up.
                self._await_markers_after_exit(expected)
                return
            self._sleep(self.poll_interval)

    def _await_markers_after_exit(self, expected: list[Path]) -> None:
        waited = 0.0
        step = min(self.poll_interval, 5.0) or 1.0
        while waited < self.marker_grace:
            if all(path.exists() for path in expected):
                return
            self._sleep(step)
            waited += step

    def _cancel_job(self, job_id: str) -> None:
        cmd = self._cancel_command(job_id)
        if not cmd:
            return
        try:
            self._run_command(cmd, self.submit_timeout)
        except (subprocess.TimeoutExpired, OSError) as exc:  # best-effort
            logger.warning("Failed to cancel job %s: %s", job_id, exc)

    @staticmethod
    def _read_success(result_file: Path) -> bool:
        if not result_file.exists():
            return False
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return bool(data.get("success", False))
