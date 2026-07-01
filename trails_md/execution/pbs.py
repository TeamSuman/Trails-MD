"""PBS / Torque (PBS Pro) array-job execution backend."""

from __future__ import annotations

from pathlib import Path

from .base import ExecutionBackendFactory
from .scheduler import SchedulerBackend


class PBSBackend(SchedulerBackend):
    array_index_var = "PBS_ARRAY_INDEX"

    def _directives(self, n_tasks: int, logdir: Path) -> list[str]:
        select = f"select=1:ncpus={self.cpus_per_task}"
        if self.gpus_per_task > 0:
            select += f":ngpus={self.gpus_per_task}"
        if self.memory:
            select += f":mem={self.memory}"
        d = [
            f"#PBS -N {self.job_name}",
            f"#PBS -J 0-{n_tasks - 1}",
            f"#PBS -l {select}",
            f"#PBS -l walltime={self.walltime}",
            f"#PBS -o {logdir}/",
            f"#PBS -e {logdir}/",
        ]
        if self.partition:  # PBS "queue"
            d.append(f"#PBS -q {self.partition}")
        if self.account:
            d.append(f"#PBS -A {self.account}")
        d += [line for line in self.extra_directives]
        return d

    def _submit_command(self, script_path: Path) -> list[str]:
        return ["qsub", str(script_path)]

    def _parse_job_id(self, stdout: str) -> str:
        # qsub prints the job id, e.g. `1234[].pbsserver`.
        return stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""

    def _poll_command(self, job_id: str) -> list[str]:
        return ["qstat", "-t", job_id]

    def _job_active(self, job_id: str, poll_stdout: str, returncode: int) -> bool:
        # qstat returns non-zero once the job is fully gone from the system.
        if returncode != 0:
            return False
        # Any array subjob not in state C (completed) means still active.
        active = False
        for line in poll_stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].split(".")[0].split("[")[0].isdigit():
                state = parts[-2]
                if state not in {"C", "F", "X"}:
                    active = True
        return active


ExecutionBackendFactory.register("pbs", PBSBackend)
