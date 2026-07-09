"""Tests for per-walker device/thread isolation (HPC oversubscription guards).

B1 -- GROMACS ``mdrun -ntomp`` is derived from the walker's CPU allocation when
the user does not set it explicitly.
B2 -- the OpenMM CPU/OpenCL/HIP platforms get thread/device isolation, not just
CUDA.
"""

from __future__ import annotations

import pytest

_THREAD_VARS = ("OPENMM_CPU_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS")


@pytest.fixture(autouse=True)
def _clear_thread_env(monkeypatch):
    for var in _THREAD_VARS:
        monkeypatch.delenv(var, raising=False)


def test_gromacs_ntomp_prefers_explicit_value(monkeypatch):
    from trails_md.engines.gromacs import GromacsEngine

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    engine = GromacsEngine(gromacs_mdrun_ntomp=8)
    assert engine._resolve_ntomp() == 8
    assert "-ntomp" in engine._mdrun_option_args()
    args = engine._mdrun_option_args()
    assert args[args.index("-ntomp") + 1] == "8"


def test_gromacs_ntomp_derived_from_scheduler_env(monkeypatch):
    from trails_md.engines.gromacs import GromacsEngine

    engine = GromacsEngine()  # no explicit ntomp
    assert engine._resolve_ntomp() is None
    assert "-ntomp" not in engine._mdrun_option_args()

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert engine._resolve_ntomp() == 6

    monkeypatch.setenv("OMP_NUM_THREADS", "2")  # OMP_NUM_THREADS takes precedence
    assert engine._resolve_ntomp() == 2


def test_gromacs_mdp_pins_ld_seed(tmp_path):
    """The built-in mdp must pin ld-seed (not just gen_seed) so the V-rescale
    thermostat stream is seeded and a fixed per-walker seed is reproducible."""
    from trails_md.engines.gromacs import GromacsEngine

    engine = GromacsEngine(seed=12345)
    mdp = tmp_path / "run.mdp"
    engine._write_mdp(str(mdp), steps=500, stride=100)
    text = mdp.read_text()
    assert "ld-seed                 = 12345" in text
    assert "gen_seed                = 12345" in text


openmm = pytest.importorskip("openmm")
from trails_md.engines.openmm import OpenMMEngine  # noqa: E402


def test_openmm_cpu_threads_from_env(monkeypatch):
    engine = OpenMMEngine(platform_name="CPU")
    assert engine._platform_properties(-1) == {}  # no env -> OpenMM decides
    monkeypatch.setenv("OPENMM_CPU_THREADS", "3")
    assert engine._platform_properties(-1) == {"Threads": "3"}


def test_openmm_device_isolation_per_platform():
    assert OpenMMEngine(platform_name="CUDA", precision="mixed")._platform_properties(
        2
    ) == {
        "Precision": "mixed",
        "DeviceIndex": "2",
    }
    assert OpenMMEngine(
        platform_name="OpenCL", precision="single"
    )._platform_properties(1) == {
        "OpenCLPrecision": "single",
        "OpenCLDeviceIndex": "1",
    }
    assert OpenMMEngine(platform_name="HIP")._platform_properties(0) == {
        "HipDeviceIndex": "0"
    }
    # Negative sentinel (scheduler-bound) -> no explicit device pin.
    assert OpenMMEngine(platform_name="CUDA")._platform_properties(-1) == {
        "Precision": "mixed"
    }
