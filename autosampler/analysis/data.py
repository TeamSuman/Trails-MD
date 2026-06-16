"""Data utilities for post-hoc MSM analysis.

Pure NumPy helpers (no matplotlib) that load the per-iteration ``msm.npz`` /
``cvs.npz`` files written during a run and derive quantities for plotting:
convergence series, free energies, and free-energy surfaces. Kept separate from
:mod:`autosampler.analysis.plots` so the numerics are testable without a
plotting backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Boltzmann constant in kJ/mol/K (free energies reported in kJ/mol).
KB_KJ_MOL = 0.00831446


def kT(temperature: float = 300.0) -> float:
    return KB_KJ_MOL * temperature


def _iter_dirs(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    dirs = [
        p
        for p in run_dir.glob("iter_*")
        if p.is_dir() and p.name.removeprefix("iter_").isdigit()
    ]
    return sorted(dirs, key=lambda p: int(p.name.removeprefix("iter_")))


def load_msm_series(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Collect per-iteration MSM scalars into aligned arrays.

    Returns a dict with ``iterations``, ``vamp2`` and ``timescales``
    (shape ``(n_iters, max_processes)``, NaN-padded). Iterations without an
    ``msm.npz`` are skipped.
    """
    iters: list[int] = []
    vamp2: list[float] = []
    timescales: list[np.ndarray] = []
    for d in _iter_dirs(run_dir):
        f = d / "msm.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as data:
            iters.append(int(d.name.removeprefix("iter_")))
            v = data["vamp2_score"]
            vamp2.append(float(np.asarray(v).ravel()[0]))
            timescales.append(np.asarray(data["timescales"], dtype=float).ravel())

    if not iters:
        return {
            "iterations": np.array([], dtype=int),
            "vamp2": np.array([], dtype=float),
            "timescales": np.zeros((0, 0), dtype=float),
        }

    width = max(len(t) for t in timescales)
    padded = np.full((len(timescales), width), np.nan)
    for i, t in enumerate(timescales):
        padded[i, : len(t)] = t
    return {
        "iterations": np.asarray(iters, dtype=int),
        "vamp2": np.asarray(vamp2, dtype=float),
        "timescales": padded,
    }


def load_latest_msm(run_dir: str | Path) -> dict[str, np.ndarray] | None:
    """Return the arrays of the most recent ``msm.npz`` (or None if absent)."""
    for d in reversed(_iter_dirs(run_dir)):
        f = d / "msm.npz"
        if f.exists():
            with np.load(f, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}
    return None


def load_cv_points(run_dir: str | Path) -> np.ndarray:
    """Stack all per-iteration CV projections (``cvs.npz``) into one array."""
    chunks: list[np.ndarray] = []
    for d in _iter_dirs(run_dir):
        f = d / "cvs.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as data:
            key = "cvs" if "cvs" in data.files else data.files[0]
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            chunks.append(arr)
    if not chunks:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(chunks)


def free_energy_from_populations(
    populations: np.ndarray, temperature: float = 300.0
) -> np.ndarray:
    """Relative free energy ``-kT ln(p)`` (kJ/mol), shifted so the min is 0."""
    p = np.asarray(populations, dtype=float)
    with np.errstate(divide="ignore"):
        f = -kT(temperature) * np.log(np.where(p > 0, p, np.nan))
    f = f - np.nanmin(f)
    return f


def free_energy_surface(
    points: np.ndarray,
    bins: int = 60,
    temperature: float = 300.0,
):
    """2D free-energy surface ``F(x,y) = -kT ln P(x,y)`` from CV points.

    Returns ``(F, xedges, yedges)`` with ``F`` shifted to a 0 minimum and
    unsampled cells set to NaN. Requires at least 2D points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("free_energy_surface needs points with >= 2 dimensions.")
    hist, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, density=True
    )
    with np.errstate(divide="ignore"):
        f = -kT(temperature) * np.log(np.where(hist > 0, hist, np.nan))
    f = f - np.nanmin(f)
    return f.T, xedges, yedges
