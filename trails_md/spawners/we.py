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
    # Weighted ensemble resamples ONLY the live walker endpoints -- it never spawns
    # from a historical frame (doing so would have no well-defined weight; see
    # `sample`). So the orchestrator must not pool the whole trajectory history for
    # it: that pooling costs one file-open per past iteration on every spawn step,
    # which grew the per-iteration overhead by ~0.8 s per iteration on a real run
    # (7.9 s at iteration 5 -> 11.1 s at iteration 9, unbounded) and made converged
    # kinetics runs impossible. Exploration spawners (density/FPS/LOF) DO pick
    # historical frames and leave this True.
    uses_history = False

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
        # CV of the basis (source) state, captured on the FIRST sample() call and then
        # held fixed. It must be cached: without history pooling the point cloud holds
        # only the current iteration, so `cloud[recycle_basis_index]` would silently
        # become "wherever walker 0 happens to be now" and the recycling target would
        # drift every iteration. On the first call it IS the equilibrated start.
        self.basis_cv: np.ndarray | None = None
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
        # Which of the previous iteration's walkers survived, set by the orchestrator
        # each iteration (None -> assume all did). Needed because failed walkers are
        # dropped upstream, so the live ensemble can be a strict subset of the budget.
        self.live_walker_indices: list[int] | None = None
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
        # The live ensemble is NOT always the walker budget: the orchestrator drops
        # walkers whose trajectory failed (min_success_fraction < 1), so it hands us
        # only the survivors' frames. Inferring n_live geometrically as min(top_n,
        # len(points)) then silently misaligns everything -- with 4 walkers x 10
        # frames and one failure, len(points)=30 gives n_live=4, fpw=7, so the
        # "endpoints" land at frames 6/13/20/27 (mid-segment), one survivor is
        # counted twice and another is lost. Ask the orchestrator instead, and only
        # fall back to the geometric guess when nothing told us.
        live_indices = getattr(self, "live_walker_indices", None)
        if live_indices is not None and len(live_indices) > 0:
            n_live = min(len(live_indices), len(points))
        else:
            n_live = max(1, min(top_n, len(points)))
        fpw = max(1, len(points) // n_live)
        # Endpoint identification assumes every live walker contributed the SAME number
        # of frames -- that is the only reason `block i ends at (i+1)*fpw - 1` holds.
        # Nothing upstream enforces it: `build_frame_records` checks only that the
        # counts SUM to len(points), and `_validate_trajectory_files` accepts any
        # non-empty file while its own docstring notes a walker "can report success yet
        # leave a truncated file". One short trajectory would slide every subsequent
        # endpoint into the middle of a neighbouring walker's segment -- the same
        # silent walker<->endpoint misalignment as the failed-walker bug, from a
        # different cause. Only enforce it when the orchestrator told us the true
        # ensemble size; a direct caller passing an arbitrary cloud keeps the
        # lenient geometric guess.
        if live_indices is not None and len(points) != n_live * fpw:
            raise ValueError(
                f"WE received {len(points)} frames for {n_live} live walkers, which is "
                f"not a whole number of frames per walker ({fpw}). Walker segments must "
                "be equal length: endpoints are identified by position, so a ragged "
                "segment misaligns walker, weight and endpoint and corrupts the rate."
            )
        ends = np.minimum(np.arange(1, n_live + 1) * fpw - 1, len(points) - 1)

        weights = self._live_weights(n_live, live_indices)
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

        if self.basis_cv is None:
            # First call: the cloud still holds the run's starting structure, so this
            # is the source state. Freeze it -- see the note on `basis_cv`.
            self.basis_cv = np.asarray(
                cumulative[self.recycle_basis_index], dtype=float
            ).copy()

        recycled = self._in_region(end_cvs, self.recycle_target)
        self.flux_history.append(float(weights[recycled].sum()))
        if recycled.any():
            end_cvs[recycled] = self.basis_cv
        return recycled

    @staticmethod
    def _in_region(cvs: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Which rows of ``cvs`` lie inside the CV-space box ``region`` ([lo, hi] per dim).

        Strict on dimensionality. This used to take ``min(cvs.shape[1], region.shape[0])``
        and truncate, which left any unspecified dimension UNBOUNDED: walkers in an
        unrelated basin that happened to share the specified coordinate were recycled,
        booked as flux, and the MFPT came out fast with nothing to show for it. The
        config validator rejects a mismatch up front; this is the backstop for callers
        that build a spawner directly.
        """
        if region.shape[0] != cvs.shape[1]:
            raise ValueError(
                f"recycle_target has {region.shape[0]} dimension(s) but the CV space "
                f"has {cvs.shape[1]}. The target must be bounded in every dimension: "
                f"an unbounded dimension inflates the flux and biases the MFPT fast."
            )
        inside = np.ones(len(cvs), dtype=bool)
        for d in range(region.shape[0]):
            inside &= (cvs[:, d] >= region[d, 0]) & (cvs[:, d] <= region[d, 1])
        return inside

    def _live_weights(
        self, n_live: int, live_indices: list[int] | None = None
    ) -> np.ndarray:
        """Weights of the live walkers, inherited from the previous resampling.

        ``live_indices`` names which of last iteration's walkers actually survived.
        It matters whenever a walker fails (``min_success_fraction < 1``): the
        orchestrator drops the failed one, so the survivors are a SUBSET of the
        walkers these weights were assigned to, and walker *i* of the live ensemble
        is walker ``live_indices[i]`` of the previous one. Matching on length alone
        is not a sufficient guard -- the lengths can coincide while the mapping is
        wrong, which silently re-attached weights to the wrong trajectories while
        ``sum(w) == 1`` still held and every invariant test still passed.

        A failed walker's weight cannot be honoured (its trajectory does not exist),
        so it is dropped and the survivors renormalised. That perturbs the steady
        state slightly, which is why the loss is logged rather than hidden.
        """
        import logging

        if self.weights is None:
            return np.full(n_live, 1.0 / n_live, dtype=float)

        w = np.asarray(self.weights, dtype=float)
        if len(w) != n_live:
            if (
                live_indices is not None
                and len(live_indices) == n_live
                and len(w) > max(live_indices, default=-1)
            ):
                lost = 1.0 - float(w[list(live_indices)].sum() / max(w.sum(), 1e-300))
                logging.warning(
                    "WE: %d walker(s) failed; carrying the %d survivors' own weights "
                    "and renormalising (%.3g of the ensemble weight discarded).",
                    len(w) - n_live,
                    n_live,
                    lost,
                )
                w = w[list(live_indices)]
            else:
                # Genuine walker-count change (or no mapping available): uniform is
                # the only defensible start.
                return np.full(n_live, 1.0 / n_live, dtype=float)

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
            members, mweights = self._resample_bin(members, mweights, target, rng)
            parents_out.extend(int(m) for m in members)
            weights_out.extend(float(w) for w in mweights)

        # Split/merge conserve weight exactly, the incoming weights sum to 1, and every
        # occupied bin was served -- so `total` is already 1 up to floating-point drift.
        # ASSERT that instead of assuming it. Rescaling unconditionally makes
        # `sum(weights) == 1` a TAUTOLOGY: it holds for whatever this function returns,
        # so every "weight is conserved" test resting on it is unfalsifiable (checked:
        # replacing the weights with random garbage passes all of them), while a real
        # leak -- weight destroyed in `_recycle`, a bin dropped -- gets quietly rescaled
        # into a plausible, wrong rate. The tolerance admits FP drift and nothing else.
        total = float(np.sum(weights_out))
        if not np.isclose(total, 1.0, rtol=1e-9, atol=0.0):
            raise AssertionError(
                f"WE weight conservation violated: resampled weights sum to {total!r}, "
                "not 1. Split/merge conserve weight and every occupied bin was served, "
                "so this is a genuine leak (weight destroyed, or a bin dropped), not "
                "rounding. Refusing to renormalise it away: that would hide the leak "
                "behind a plausible, wrong rate."
            )
        weights_out = [w / total for w in weights_out]
        return parents_out, weights_out

    @staticmethod
    def _resample_bin(members, mweights, target, rng, spread_tol: float = 4.0,
                      max_rounds: int = 200):
        """Split/merge one bin to ``target`` walkers of APPROXIMATELY EQUAL weight.

        Reaching the right walker *count* is not enough -- the weights within a bin
        must also be equalised to ~bin_weight/target. Getting only the count right
        (merge the two lightest, split the heaviest, stop) leaves light walkers to
        merge with *each other* forever, never absorbed into the heavy ones. They
        survive as "zombies" carrying ~1e-21: they occupy a walker slot, cost a full
        MD segment, and contribute nothing. Measured on a real alanine run before this
        fix: 13/40 walkers below 1e-10, weights spanning 1e20 *within a single bin*,
        and an effective sample size (1/sum w^2) of **3.6 of 40 walkers**. The flux
        then decays without bound, because the walkers reaching the target are
        increasingly zombies delivering no weight -- a plausible, badly wrong rate.

        So after fixing the count we equalise: merge the two lightest and split the
        heaviest in tandem (which leaves the count unchanged) until the spread is
        within ``spread_tol``. Each round raises the minimum and lowers the maximum,
        so it converges; once the zombies are exhausted they merge into a real walker
        and their slot is freed for a split of a heavy one. Weight is conserved
        throughout -- merging sums weight and splitting halves it.
        """
        members, mweights = WeightedEnsemble._merge(members, mweights, target, rng)
        members, mweights = WeightedEnsemble._split(members, mweights, target)

        if target < 2:
            return members, mweights
        for _ in range(max_rounds):
            lo, hi = min(mweights), max(mweights)
            if lo > 0 and hi <= spread_tol * lo:
                break
            # merge the two lightest (count -> target-1), then split the heaviest
            # (count -> target): the spread shrinks, the count is unchanged.
            members, mweights = WeightedEnsemble._merge(
                members, mweights, target - 1, rng
            )
            members, mweights = WeightedEnsemble._split(members, mweights, target)
        return members, mweights

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
        # The frozen basis must survive too: a resume would otherwise re-freeze it to
        # wherever the ensemble happens to be, silently moving the source state.
        state["basis_cv"] = None if self.basis_cv is None else self.basis_cv.tolist()
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if state and state.get("weights") is not None:
            self.weights = np.asarray(state["weights"], dtype=float)
        if state and state.get("flux_history") is not None:
            self.flux_history = list(state["flux_history"])
        if state and state.get("basis_cv") is not None:
            self.basis_cv = np.asarray(state["basis_cv"], dtype=float)

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
