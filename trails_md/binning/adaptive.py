"""Landscape-adaptive binning schemes.

The default :class:`~trails_md.binning.spatial.RegularBinner` is a uniform grid:
constant bin width everywhere. Near a steep free-energy barrier a wide bin lets a
walker slide back before it can reach the next bin within the lag time, so the WE
flux across the barrier stalls; in flat basins fine bins waste replicas. These
schemes make the bins **landscape-adaptive** — finer where the landscape is steep
/ sparse, coarser where it is flat — and are recomputed every iteration.

All binners share the :class:`~trails_md.binning.spatial.RegularBinner` API
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

from .spatial import BinTable, RegularBinner, bucket_frames, padded_bounds


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
    populations, populated_data = bucket_frames(idx, nbin)

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
    """Equi-resistance edges: dense where the sampled density is low (barriers).

    Boundaries are placed at equal increments of the "resistance"
    ``∫ 1/P(x) dx ∝ ∫ exp(βF(x)) dx``, so bins bunch up where the sampled density is
    low — i.e. across barriers — and spread out in well-sampled basins.

    Two safeguards matter in practice and were added after benchmarking:

    * Edges are laid out across the **occupied** range only. Over the full configured
      domain the unvisited region has zero density, so ``1/P`` diverges there and
      swallows the entire edge budget: the occupied region then collapses into one or
      two enormous bins and the frontier is *diluted* rather than resolved — the exact
      opposite of the intent.
    * The resistance is **clipped** to ``max_resistance`` times its median, so a single
      near-empty slice cannot monopolise the edges either.
    """

    def __init__(self, *args, n_fine: int = 100, smoothing: int = 3,
                 max_resistance: float = 50.0, **kw):
        super().__init__(*args, **kw)
        self.n_fine = int(n_fine)
        self.smoothing = int(smoothing)
        self.max_resistance = float(max_resistance)

    def _axis_edges(self, col, lo, hi, nb):
        occ_lo, occ_hi = float(col.min()), float(col.max())
        if nb < 3 or occ_hi <= occ_lo:
            return np.linspace(lo, hi, nb + 1)

        n_fine = max(self.n_fine, nb * 4)
        hist, fine = np.histogram(col, bins=n_fine, range=(occ_lo, occ_hi))
        density = _smooth(hist.astype(float), self.smoothing)
        density = np.maximum(density, 1e-9)
        resistance = 1.0 / density                       # ∝ exp(βF)
        cap = self.max_resistance * float(np.median(resistance))
        resistance = np.minimum(resistance, cap)

        cum = np.concatenate([[0.0], np.cumsum(resistance * np.diff(fine))])
        if cum[-1] <= 0:
            return np.linspace(lo, hi, nb + 1)
        # nb-1 interior edges across the occupied range; keep the domain bounds outside
        targets = np.linspace(0.0, cum[-1], nb - 1)
        inner = np.interp(targets, cum, fine)
        return np.unique(np.concatenate([[lo], inner, [hi]]))


class MABinner(AdaptiveBinner):
    """Minimal Adaptive Binning (Torrillo, Bogetti & Chong, *J. Phys. Chem. A* 2021).

    Three ingredients, recomputed every iteration:

    1. **Dedicated boundary bins.** The single leading (front-most) and trailing
       (rear-most) frames each get their *own* narrow bin. A bin holding one frame has
       the maximum possible density weight ``1/n_b``, so the frontier is always eligible
       for respawning and can never be diluted into a populated neighbour.
    2. **Dedicated bottleneck bins.** Bottlenecks are detected with the MAB objective
       ``Z_i = log(n_i) - log(sum of n_j ahead of i)``: a slice that still holds
       population while almost nothing lies beyond it is the uphill face of a barrier.
       ``n_bottleneck`` such slices (per direction) get their own narrow bins, which
       concentrates respawning exactly where flux is being lost.
    3. **Evenly spaced bins in between**, spanning only the *occupied* range rather than
       the full configured domain, so resolution follows the walkers.

    Note that fine bins at the frontier are necessary but not sufficient: with hard
    density spawning the frontier bin is only one of ``walker`` selected bins, so only
    ~1/``walker`` of the effort attacks the barrier. Pair this binner with the WE spawner
    (``spawn_scheme: we``, which *replicates* ``we_target_per_bin`` walkers into each
    occupied bin) to obtain a genuine ratchet.
    """

    def __init__(self, *args, n_bottleneck: int = 1, n_fine: int = 60, **kw):
        super().__init__(*args, **kw)
        self.n_bottleneck = int(n_bottleneck)
        self.n_fine = int(n_fine)

    def _bottlenecks(self, col, lo, hi):
        """Positions of the MAB bottleneck slices along one axis."""
        hist, edges = np.histogram(col, bins=self.n_fine, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        occupied = hist > 0
        if occupied.sum() < 3:
            return []

        out = []
        for direction in (+1, -1):
            h = hist if direction > 0 else hist[::-1]
            c = centers if direction > 0 else centers[::-1]
            # cumulative population strictly AHEAD of slice i, in this direction
            ahead = np.cumsum(h[::-1])[::-1] - h
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log(np.where(h > 0, h, np.nan)) - np.log(np.where(ahead > 0, ahead, np.nan))
            z = np.where(np.isfinite(z), z, -np.inf)
            # ignore the extreme front slice itself (it has nothing ahead by construction)
            valid = np.where(h > 0)[0]
            if len(valid) < 3:
                continue
            valid = valid[:-1]
            if len(valid) == 0:
                continue
            best = valid[np.argsort(z[valid])[-self.n_bottleneck:]]
            out.extend(float(c[i]) for i in best)
        return out

    def _axis_edges(self, col, lo, hi, nb):
        occ_lo, occ_hi = float(col.min()), float(col.max())
        if nb < 6 or occ_hi <= occ_lo:
            return np.linspace(lo, hi, nb + 1)

        span = occ_hi - occ_lo
        eps = max(span * 0.02, (hi - lo) * 1e-4)   # width of a dedicated bin

        specials = [occ_lo, occ_hi]                      # leading + trailing frames
        specials += self._bottlenecks(col, lo, hi)       # uphill faces of barriers

        # narrow bracket around each special position
        cuts = [lo, hi]
        for s in specials:
            cuts += [s - eps, s + eps]

        # evenly spaced bins filling the remaining budget across the occupied range
        n_even = max(nb - 2 * len(specials), 2)
        cuts += list(np.linspace(occ_lo, occ_hi, n_even + 1))

        edges = np.unique(np.clip(np.asarray(cuts, dtype=float), lo, hi))
        return edges


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
    n_bottleneck: int = 1,
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
    if scheme == "mab":
        kwargs.update(n_fine=n_fine, n_bottleneck=n_bottleneck)
    return BinnerFactory._binners[scheme](**kwargs)
