"""SLURM array-job execution backend."""

from __future__ import annotations

import re
from pathlib import Path

from .base import ExecutionBackendFactory
from .scheduler import SchedulerBackend


class SlurmBackend(SchedulerBackend):
    array_index_var = "SLURM_ARRAY_TASK_ID"

    def _directives(self, n_tasks: int, logdir: Path) -> list[str]:
        d = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --array=0-{n_tasks - 1}",
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

    def _job_active(self, job_id: str, poll_stdout: str, returncode: int) -> bool:
        # squeue lists one line per still-active array element; empty => done.
        if returncode != 0:
            return False
        return bool(re.search(rf"\b{re.escape(job_id)}\b", poll_stdout))


ExecutionBackendFactory.register("slurm", SlurmBackend)
