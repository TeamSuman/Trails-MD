"""Weighted-ensemble spawner (Huber & Kim, 1996).

Resamples the *live* walker ensemble by split/merge, keeping ``we_target_per_bin``
walkers in every occupied bin while **conserving total statistical weight**. Unlike
the exploration-oriented spawners (density / FPS / LOF / MSM least-counts), the
weights it carries are rigorous: a walker split ``c`` ways yields ``c`` children of
weight ``w/c``, merges sum weight, and ``sum(weights) == 1`` every iteration. That
is what makes an unbiased rate (MFPT) recoverable from a WE run, and it is the whole
reason to prefer ``spawn_scheme: we`` over the cheaper exploration schemes.

Implements the standard ``sample(points, top_n, history)`` contract, returning
indices into the cumulative point cloud (repeats indicate split walkers), so it is a
drop-in ``spawn_scheme: we`` option. Walker weights are carried across iterations in
the spawner instance (and exposed via ``state_dict``).

Two invariants are easy to get backwards and both are load-bearing; see ``sample``
and ``_resample_to_budget`` for why:

* CPU is allocated **across bins**, never in proportion to weight.
* The ensemble is the **live walkers**, never an arbitrary frame from history.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trails_md.binning.spatial import RegularBinner
from trails_md.binning.we import WeightedEnsemble

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class WESpawner(Spawner):
    def __init__(
        self,
        n_bins: list[int] | None = None,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        target_per_bin: int = 4,
        seed: int = 42,
        recycle_target: list[list[float]] | None = None,
        recycle_basis_index: int = 0,
        **kwargs: Any,
    ):
        super().__init__(seed=seed, **kwargs)
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.target_per_bin = int(target_per_bin)
        self.we = WeightedEnsemble(target_per_bin=target_per_bin)
        # --- source->sink recycling (steady-state rate mode) ------------------
        # `recycle_target` is a CV-space box [[lo, hi], ...] per dimension. A walker
        # whose endpoint lands inside it is TERMINATED and its weight restarted from
        # the basis (source) frame. That drives a non-equilibrium steady state, and
        # the recycled weight per tau IS the probability flux into the target, so
        # MFPT = 1 / steady-state flux (Hill relation) -- the same estimator WESTPA
        # uses, which is what makes the two directly comparable.
        self.recycle_target = (
            np.asarray(recycle_target, dtype=float) if recycle_target else None
        )
        self.recycle_basis_index = int(recycle_basis_index)
        # Recycled weight per iteration; the flux time series the MFPT is read from
        # (after discarding the pre-steady-state transient).
        self.flux_history: list[float] = []
        # Statistical weight of each walker spawned last iteration (sums to 1).
        # These are WALKER weights, carried down the lineage -- not per-frame
        # weights over the cumulative cloud (see `sample`).
        self.weights: np.ndarray | None = None
        # Current-iteration walker indices each spawned walker continues from (set
        # per sample(); used by the orchestrator for velocity inheritance).
        self.selected_parents: list[int] | None = None
        # Optional landscape-adaptive binner (set by the orchestrator); None -> grid.
        self.binner = None

    def sample(
        self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None
    ) -> list[int]:
        """One Huber-Kim weighted-ensemble resampling step.

        Two things make this rigorous WE rather than "adaptive sampling that also
        tracks some numbers":

        **The ensemble is the current walkers, not the whole history.** WE is a
        Markov resampling of the *live* ensemble: each walker runs for tau, and the
        endpoints are then split/merged. Restarting from an arbitrary frame drawn
        out of the cumulative cloud has no well-defined statistical weight -- the
        frame's weight was already spent in the iteration that produced it -- and
        reusing it silently double-counts probability. A rate computed from such
        weights is *wrong*, which is worse than having no rate. So the ensemble
        here is exactly the endpoint of each current walker.

        **The walker budget is the resampling target.** Slots are allocated across
        occupied bins first (equal share; ties to the sparsest bin, which is the
        frontier), and each bin is then split/merged to *exactly* its slot count.
        Every resampled walker is therefore actually run, so total weight is
        conserved: sum(w) == 1 every iteration, and a child of a walker split c ways
        carries w/c. That is what makes an unbiased MFPT recoverable downstream.
        """
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            raise ValueError("Cannot run WE on an empty point cloud.")
        cumulative = _cumulative_points(points, history)
        offset = len(cumulative) - len(points)  # current frames start here

        # Endpoint frame of each live walker. Frames are laid out contiguously per
        # walker (frames_per_walker = step // stride), so the last frame of block i
        # is walker i's endpoint -- the state WE is entitled to continue from.
        n_live = max(1, min(top_n, len(points)))
        fpw = max(1, len(points) // n_live)
        ends = np.minimum(np.arange(1, n_live + 1) * fpw - 1, len(points) - 1)

        weights = self._live_weights(n_live)
        end_cvs = cumulative[offset + ends]

        # RECYCLE FIRST, THEN RESAMPLE -- the WESTPA order. A walker that reached the
        # target is terminated here: its weight is booked as flux and restarted from
        # the basis, so by the time we bin, it is *at the source* and bins there.
        recycled = self._recycle(end_cvs, weights, cumulative)

        # Bin the LIVE ENSEMBLE, not the cumulative cloud. This is not a detail: an
        # adaptive binner (MAB) gives the leading frame its own dedicated bin, and if
        # the binner is fitted to the cumulative cloud that leading frame is usually a
        # *historical* one that no live walker occupies. A bin with no live walker in
        # it gets no slots, so nothing is ever replicated into the frontier and the
        # ratchet never engages -- the run then behaves exactly like the unweighted
        # control. Binning the live walkers is also what MAB/WESTPA actually do: bins
        # are re-laid over the current ensemble every iteration.
        labels = self._bin_labels(end_cvs)

        parents, new_weights = self._resample_to_budget(
            weights, labels, top_n, self.rng
        )
        self.weights = np.asarray(new_weights, dtype=float)
        # For kinetics mode: `parents` are indices into the live-walker endpoints,
        # i.e. the *current-iteration walker index* each spawned walker continues
        # from (with repeats for splits). The orchestrator uses these to inherit
        # each parent's endpoint velocities. Exposed as an attribute rather than
        # returned, to keep the sample() contract (a list of frame indices) intact.
        # A recycled parent is marked -1: it is a NEW trajectory launched from the
        # basis, so it must draw fresh Maxwell-Boltzmann velocities rather than
        # inherit -- which is exactly what WESTPA does when it restarts a bstate.
        self.selected_parents = [-1 if recycled[p] else int(p) for p in parents]
        return [
            int(self.recycle_basis_index if recycled[p] else offset + ends[p])
            for p in parents
        ]

    def _recycle(self, end_cvs: np.ndarray, weights: np.ndarray,
                 cumulative: np.ndarray) -> np.ndarray:
        """Terminate walkers that reached the target; restart their weight at the basis.

        Returns a per-live-walker boolean mask of who was recycled. ``end_cvs`` is
        modified in place so a recycled walker is subsequently binned *at the basis*.

        The recycled weight this iteration is the probability flux into the target
        over one tau. Weight is conserved exactly -- nothing is created or destroyed,
        it is simply moved back to the source -- which is what sustains the
        non-equilibrium steady state the Hill relation needs.
        """
        n = len(end_cvs)
        if self.recycle_target is None:
            return np.zeros(n, dtype=bool)

        recycled = self._in_region(end_cvs, self.recycle_target)
        self.flux_history.append(float(weights[recycled].sum()))
        if recycled.any():
            end_cvs[recycled] = cumulative[self.recycle_basis_index]
        return recycled

    @staticmethod
    def _in_region(cvs: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Which rows of ``cvs`` lie inside the CV-space box ``region`` ([lo, hi] per dim)."""
        ndim = min(cvs.shape[1], region.shape[0])
        inside = np.ones(len(cvs), dtype=bool)
        for d in range(ndim):
            inside &= (cvs[:, d] >= region[d, 0]) & (cvs[:, d] <= region[d, 1])
        return inside

    def _live_weights(self, n_live: int) -> np.ndarray:
        """Weights of the live walkers, inherited from the previous resampling."""
        if self.weights is None or len(self.weights) != n_live:
            # First iteration (or a walker-count change): start from uniform.
            return np.full(n_live, 1.0 / n_live, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        total = w.sum()
        return w / total if total > 0 else np.full(n_live, 1.0 / n_live)

    def _resample_to_budget(
        self,
        weights: np.ndarray,
        labels: np.ndarray,
        top_n: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[float]]:
        """Split/merge each bin to exactly its allocated share of the budget.

        Allocation is bin-balanced and NEVER weight-proportional: a walker on a
        barrier top carries an exponentially small weight by construction, and that
        smallness is the result being computed, not a reason to stop simulating it.
        (Selecting with p ~ weight is what reduced this spawner to unbiased MD: on
        the proline barrier the frontier bin held weight 4e-6 and was picked once
        per ~15,900 iterations.) Weight decides nothing here except bookkeeping.
        """
        bins, inverse = np.unique(labels, return_inverse=True)
        n_occ = len(bins)

        # Population of each occupied bin among the live walkers; ascending order
        # puts the sparsest bin -- the frontier -- first in line for scarce slots.
        counts = np.bincount(inverse, minlength=n_occ)
        order = np.argsort(counts, kind="stable")

        # Every occupied bin is guaranteed at least one slot, so no bin -- and no
        # weight -- is ever dropped. That is not luck: the ensemble is the set of
        # live walker endpoints, so n_occ <= n_live <= top_n by construction. It is
        # what lets weight be conserved *exactly* rather than renormalised, and it
        # is why the rate this produces is trustworthy. Assert it rather than
        # assume it: silently dropping the densest bin (scarce slots go to the
        # sparsest) would corrupt every weight and yield a plausible, wrong MFPT.
        if n_occ > top_n:  # pragma: no cover -- unreachable by construction
            raise AssertionError(
                f"WE invariant violated: {n_occ} occupied bins > {top_n} walkers. "
                "Weights cannot be conserved; refusing to produce a corrupt ensemble."
            )

        slots = np.zeros(n_occ, dtype=int)
        for k in range(top_n):
            slots[order[k % n_occ]] += 1

        parents_out: list[int] = []
        weights_out: list[float] = []
        for b in range(n_occ):
            target = int(slots[b])
            if target == 0:
                continue
            members = np.flatnonzero(inverse == b).tolist()
            mweights = [float(weights[i]) for i in members]
            members, mweights = WeightedEnsemble._merge(members, mweights, target, rng)
            members, mweights = WeightedEnsemble._split(members, mweights, target)
            parents_out.extend(int(m) for m in members)
            weights_out.extend(float(w) for w in mweights)

        # Split/merge conserve weight exactly and every bin was served, so this only
        # mops up floating-point drift -- it is not a rescue renormalisation.
        total = float(np.sum(weights_out))
        if total > 0:
            weights_out = [w / total for w in weights_out]
        return parents_out, weights_out

    def _bin_labels(self, cumulative: np.ndarray) -> np.ndarray:
        if self.binner is not None:
            self.binner.n_bins = np.asarray(self.n_bins, dtype=int)
            table = self.binner.fit(cumulative)
        else:
            table = RegularBinner(
                n_bins=self.n_bins,
                min_values=self.min_values,
                max_values=self.max_values,
            ).fit(cumulative)
        labels = np.full(len(cumulative), -1, dtype=int)
        for row, frames in enumerate(table.populated_data):
            for frame in frames:
                labels[frame] = row
        return labels

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["weights"] = None if self.weights is None else self.weights.tolist()
        # The flux series IS the rate measurement -- checkpoint it, or a resumed run
        # silently loses the observable the whole kinetics mode exists to produce.
        state["flux_history"] = list(self.flux_history)
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if state and state.get("weights") is not None:
            self.weights = np.asarray(state["weights"], dtype=float)
        if state and state.get("flux_history") is not None:
            self.flux_history = list(state["flux_history"])

    def mfpt(self, tau_ps: float, discard_fraction: float = 0.5) -> float | None:
        """Steady-state MFPT (ns) from the recycled-flux series, via the Hill relation.

        The early iterations are a transient: the ensemble has not yet reached the
        non-equilibrium steady state, and the flux during that ramp-up systematically
        UNDERESTIMATES the rate. So the leading ``discard_fraction`` of the series is
        dropped before averaging -- reporting the un-discarded average is the single
        most common way a WE rate is wrong. Returns None if nothing has been recycled
        yet (no flux -> no rate, rather than an infinite one).
        """
        if not self.flux_history:
            return None
        n_skip = int(len(self.flux_history) * discard_fraction)
        tail = np.asarray(self.flux_history[n_skip:], dtype=float)
        if tail.size == 0 or tail.mean() <= 0:
            return None
        return float(tau_ps / tail.mean() / 1000.0)  # tau/flux -> ps -> ns



SpawnerFactory.register("we", WESpawner)
