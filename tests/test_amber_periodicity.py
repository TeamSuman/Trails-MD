"""Regression tests for Amber non-periodic (vacuum) system support.

The Amber engine historically hard-coded ``ntb=1`` (constant-volume periodic),
so a vacuum / implicit-solvent prmtop (no unit cell) made pmemd abort with
"Box parameters not found in inpcrd file!". The engine now detects periodicity
from the prmtop ``IFBOX`` pointer and emits ``ntb=0`` (no cutoff, no wrap) for
non-periodic systems.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from trails_md.engines.amber import AmberEngine  # noqa: E402


def _write_prmtop(path, ifbox: int) -> str:
    """Minimal prmtop with a POINTERS section whose IFBOX (28th value) is set."""
    pointers = [0] * 31
    pointers[0] = 22  # NATOM (cosmetic)
    pointers[27] = ifbox  # IFBOX
    rows = [pointers[i : i + 10] for i in range(0, 31, 10)]
    body = "\n".join("".join(f"{v:8d}" for v in row) for row in rows)
    text = (
        "%VERSION test\n"
        "%FLAG POINTERS\n"
        "%FORMAT(10I8)\n"
        f"{body}\n"
        "%FLAG ATOM_NAME\n"
    )
    path.write_text(text)
    return str(path)


def test_detect_periodic_vacuum_vs_box(tmp_path):
    vac = _write_prmtop(tmp_path / "vac.prmtop", ifbox=0)
    box = _write_prmtop(tmp_path / "box.prmtop", ifbox=1)
    assert AmberEngine._detect_periodic(vac) is False
    assert AmberEngine._detect_periodic(box) is True
    # Unparseable / missing → default to periodic (historical behaviour).
    assert AmberEngine._detect_periodic(tmp_path / "missing.prmtop") is True


def test_write_input_vacuum_uses_ntb0(tmp_path):
    eng = AmberEngine()
    eng.is_periodic = False
    mdin = tmp_path / "md.in"
    eng._write_input(str(mdin), steps=500, stride=100, trajectory_format="ascii")
    text = mdin.read_text()
    assert "ntb=0" in text
    assert "ntp=0" in text
    assert "iwrap=0" in text
    assert "cut=999" in text


def test_write_input_periodic_uses_ntb1(tmp_path):
    eng = AmberEngine()
    eng.is_periodic = True
    mdin = tmp_path / "md.in"
    eng._write_input(str(mdin), steps=500, stride=100, trajectory_format="ascii")
    text = mdin.read_text()
    assert "ntb=1" in text
    assert "cut=9.0" in text
    assert "iwrap=1" in text


def test_write_input_vacuum_npt_is_disabled(tmp_path):
    # NPT is meaningless without a box: it must be force-disabled, not emit ntb=2.
    eng = AmberEngine(npt=True)
    eng.is_periodic = False
    mdin = tmp_path / "md.in"
    eng._write_input(str(mdin), steps=500, stride=100, trajectory_format="ascii")
    text = mdin.read_text()
    assert "ntb=0" in text
    assert "ntp=0" in text
    assert "barostat" not in text
