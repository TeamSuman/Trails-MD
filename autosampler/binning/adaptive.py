"""Landscape-adaptive binning schemes.

The default :class:`~autosampler.binning.spatial.RegularBinner` is a uniform grid:
constant bin width everywhere. Near a steep free-energy barrier a wide bin lets a
walker slide back before it can reach the next bin within the lag time, so the WE
flux across the barrier stalls; in flat basins fine bins waste replicas. These
schemes make the bins **landscape-adaptive** — finer where the landscape is steep
/ sparse, coarser where it is flat — and are recomputed every iteration.

All binners share the :class:`~autosampler.binning.spatial.RegularBinner` API
(``fit(points) -> BinTable``), so the density / WE spawners consume them
interchangeably. ``uniform`` maps straight to ``RegularBinner`` for exact
backwards compatibility.

Schemes
-------
- ``uniform``     : constant-width grid (``RegularBinner``).
- ``gradient``    : equi-resistance edges — boundaries at equal increments of
                    ``∫ exp(βF) dx ∝ ∫ 1/P dx`` so bins concentrate where the
                    sampled density is low (barriers / steep regions).
- ``mab``         : Minimal-Adaptive-Binning-style — uniform bins between the
                    occupied extremes plus narrow "foothold" bins at the moving
                    fronts.
- ``eigenvector`` : bin uniformly along the leading (slowest) CV coordinate
                    only; for a learned CV / committor proxy this is automatically
                    fine across the barrier and coarse in basins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .spatial import BinTable, RegularBinner, padded_bounds


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _dedupe_increasing(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Force strictly-increasing edges spanning [lo, hi]."""
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = lo, hi
    edges = np.maximum.accumulate(edges)
    # nudge any duplicates apart so np.searchsorted yields non-empty bins
    eps = (hi - lo) * 1e-9 + 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def _grid_bintable(coords, edges, target=None) -> BinTable:
    """Bucket ``coords`` into a variable-width grid defined by per-axis ``edges``."""
    coords = np.asarray(coords, dtype=float)
    n_axes = len(edges)
    nbin = [max(len(e) - 1, 1) for e in edges]

    idx = np.empty((len(coords), n_axes), dtype=int)
    for a in range(n_axes):
        pos = np.searchsorted(edges[a], coords[:, a], side="right") - 1
        idx[:, a] = np.clip(pos, 0, nbin[a] - 1)

    ids = list(np.ndindex(*nbin))
    id_to_row = {t: i for i, t in enumerate(ids)}
    populations = np.zeros(len(ids), dtype=int)
    populated_data: list[list[int]] = [[] for _ in ids]
    for frame, cell in enumerate(map(tuple, idx)):
        row = id_to_row[cell]
        populations[row] += 1
        populated_data[row].append(frame)

    centers = np.array(
        [
            [0.5 * (edges[a][t[a]] + edges[a][t[a] + 1]) for a in range(n_axes)]
            for t in ids
        ],
        dtype=float,
    )

    target_closeness = None
    if target is not None:
        tt = np.asarray(target, dtype=float)[:n_axes]
        dist = np.linalg.norm(centers - tt, axis=1)
        denom = dist.max() - dist.min()
        target_closeness = (
            (dist.max() - dist) / denom if denom > 1e-12 else np.ones_like(dist)
        )

    return BinTable(
        ids=ids,
        centers=centers,
        populations=populations,
        populated_data=populated_data,
        target_closeness=target_closeness,
    )


class AdaptiveBinner(ABC):
    """Base class: per-axis adaptive edges over a (padded) bounding box."""

    def __init__(
        self,
        n_bins,
        min_values=None,
        max_values=None,
        target=None,
        padding: float = 0.1,
        **_,
    ):
        self.n_bins = np.asarray(n_bins, dtype=int)
        self.min_values = None if min_values is None else np.asarray(min_values, float)
        self.max_values = None if max_values is None else np.asarray(max_values, float)
        self.target = None if target is None else np.asarray(target, float)
        self.padding = float(padding)

    def _bounds(self, points: np.ndarray):
        if self.min_values is None or self.max_values is None:
            lo, hi = padded_bounds(points, self.padding)
            lo = lo if self.min_values is None else self.min_values
            hi = hi if self.max_values is None else self.max_values
        else:
            lo, hi = self.min_values, self.max_values
        return np.asarray(lo, float), np.asarray(hi, float)

    def _coords(self, points: np.ndarray):
        """Columns of ``points`` to bin and the per-axis bin counts."""
        return points, [int(b) for b in self.n_bins]

    @abstractmethod
    def _axis_edges(self, col: np.ndarray, lo: float, hi: float, nb: int) -> np.ndarray:
        """Return ``nb``-ish increasing edges for one coordinate."""

    def fit(self, points: np.ndarray) -> BinTable:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("AdaptiveBinner expects a 2D array of points.")
        coords, nbin = self._coords(points)
        lo, hi = self._bounds(points)
        edges = []
        for a in range(coords.shape[1]):
            lo_a = float(lo[a]) if a < len(lo) else float(coords[:, a].min())
            hi_a = float(hi[a]) if a < len(hi) else float(coords[:, a].max())
            if hi_a <= lo_a:
                hi_a = lo_a + 1e-9
            edges.append(
                _dedupe_increasing(
                    self._axis_edges(coords[:, a], lo_a, hi_a, int(nbin[a])), lo_a, hi_a
                )
            )
        return _grid_bintable(coords, edges, self.target)


class GradientBinner(AdaptiveBinner):
    """Equi-resistance edges: dense where the sampled density is low (barriers)."""

    def __init__(self, *args, n_fine: int = 100, smoothing: int = 3, **kw):
        super().__init__(*args, **kw)
        self.n_fine = int(n_fine)
        self.smoothing = int(smoothing)

    def _axis_edges(self, col, lo, hi, nb):
        n_fine = max(self.n_fine, nb * 4)
        hist, fine = np.histogram(col, bins=n_fine, range=(lo, hi))
        density = _smooth(hist.astype(float), self.smoothing) + 1e-9
        resistance = 1.0 / density  # ∝ exp(βF)
        cum = np.concatenate([[0.0], np.cumsum(resistance * np.diff(fine))])
        if cum[-1] <= 0:
            return np.linspace(lo, hi, nb + 1)
        targets = np.linspace(0.0, cum[-1], nb + 1)
        return np.interp(targets, cum, fine)


class MABinner(AdaptiveBinner):
    """Minimal-Adaptive-Binning style: uniform middle + narrow front footholds."""

    def _axis_edges(self, col, lo, hi, nb):
        occ_lo, occ_hi = float(col.min()), float(col.max())
        if nb < 4 or occ_hi <= occ_lo:
            return np.linspace(lo, hi, nb + 1)
        inner = np.linspace(occ_lo, occ_hi, nb - 1)
        # Split the two outermost occupied bins to give footholds at the fronts.
        foot_lo = 0.5 * (inner[0] + inner[1])
        foot_hi = 0.5 * (inner[-2] + inner[-1])
        return np.unique(
            np.concatenate(
                [[lo], inner[:1], [foot_lo], inner[1:-1], [foot_hi], inner[-1:], [hi]]
            )
        )


class EigenvectorBinner(AdaptiveBinner):
    """Bin uniformly along the leading (slowest) CV coordinate only.

    For a learned CV / committor-proxy the slow coordinate compresses basins and
    stretches the barrier, so uniform bins in it are automatically fine across the
    barrier and coarse in basins. ``coordinate`` selects the column (default 0).
    """

    def __init__(self, *args, coordinate: int = 0, **kw):
        super().__init__(*args, **kw)
        self.coordinate = int(coordinate)

    def _coords(self, points):
        col = self.coordinate if self.coordinate < points.shape[1] else 0
        return points[:, [col]], [int(self.n_bins[0])]

    def _axis_edges(self, col, lo, hi, nb):
        return np.linspace(lo, hi, nb + 1)


class BinnerFactory:
    _binners = {
        "gradient": GradientBinner,
        "mab": MABinner,
        "eigenvector": EigenvectorBinner,
    }

    @classmethod
    def register(cls, name: str, binner_cls) -> None:
        cls._binners[name] = binner_cls

    @classmethod
    def available(cls) -> list[str]:
        return sorted(["uniform", *cls._binners])


def make_binner(
    scheme: str,
    *,
    n_bins,
    min_values=None,
    max_values=None,
    target=None,
    n_fine: int = 100,
    smoothing: int = 3,
):
    """Construct the configured binner. ``uniform`` → ``RegularBinner`` (exact)."""
    if scheme == "uniform":
        return RegularBinner(
            n_bins=n_bins, min_values=min_values, max_values=max_values, target=target
        )
    if scheme not in BinnerFactory._binners:
        raise ValueError(
            f"Unknown binning scheme {scheme!r}; available: {BinnerFactory.available()}"
        )
    kwargs = dict(
        n_bins=n_bins, min_values=min_values, max_values=max_values, target=target
    )
    if scheme == "gradient":
        kwargs.update(n_fine=n_fine, smoothing=smoothing)
    return BinnerFactory._binners[scheme](**kwargs)
