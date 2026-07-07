"""SLURM array-job execution backend."""

from __future__ import annotations

import re
from pathlib import Path

from .base import ExecutionBackendFactory
from .scheduler import SchedulerBackend


class SlurmBackend(SchedulerBackend):
    array_index_var = "SLURM_ARRAY_TASK_ID"

    def _directives(self, n_tasks: int, logdir: Path) -> list[str]:
        array_spec = f"0-{n_tasks - 1}"
        if self.max_in_flight and self.max_in_flight > 0:
            # ``%N`` caps how many array elements run concurrently, so a large
            # walker batch does not flood the scheduler / hit submit-rate limits.
            array_spec += f"%{self.max_in_flight}"
        d = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --array={array_spec}",
            f"#SBATCH --time={self.walltime}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --output={logdir}/%A_%a.out",
            f"#SBATCH --error={logdir}/%A_%a.err",
        ]
        if self.partition:
            d.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            d.append(f"#SBATCH --account={self.account}")
        if self.gpus_per_task > 0:
            d.append(f"#SBATCH --gpus-per-task={self.gpus_per_task}")
        if self.memory:
            d.append(f"#SBATCH --mem={self.memory}")
        d += [line for line in self.extra_directives]
        return d

    def _submit_command(self, script_path: Path) -> list[str]:
        return ["sbatch", "--parsable", str(script_path)]

    def _parse_job_id(self, stdout: str) -> str:
        # `sbatch --parsable` prints just the job id (optionally `id;cluster`).
        token = stdout.strip().splitlines()[-1] if stdout.strip() else ""
        return token.split(";")[0].strip()

    def _poll_command(self, job_id: str) -> list[str]:
        return ["squeue", "--job", job_id, "--noheader", "--array"]

    def _cancel_command(self, job_id: str) -> list[str]:
        return ["scancel", job_id]

    def _job_active(self, job_id: str, poll_stdout: str, returncode: int) -> bool:
        # ``squeue --array`` prints one line per still-active array element with
        # the id formatted as ``<jobid>_<taskid>`` (e.g. ``12345_0``). A prior
        # implementation matched ``\b<jobid>\b`` — but ``_`` is a regex word
        # character, so ``\b`` never occurs between the digits and the ``_`` and
        # the pattern silently failed to match *running* arrays, making the
        # poller believe the job had already left the queue. We therefore match
        # the id at the start of a line, optionally followed by ``_<taskid>``.
        if returncode != 0:
            return False
        pattern = re.compile(rf"^\s*{re.escape(job_id)}(?:_|\b)", re.MULTILINE)
        return bool(pattern.search(poll_stdout))


ExecutionBackendFactory.register("slurm", SlurmBackend)
