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
        **kwargs: Any,
    ):
        super().__init__(seed=seed, **kwargs)
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.target_per_bin = int(target_per_bin)
        self.we = WeightedEnsemble(target_per_bin=target_per_bin)
        # Statistical weight of each walker spawned last iteration (sums to 1).
        # These are WALKER weights, carried down the lineage -- not per-frame
        # weights over the cumulative cloud (see `sample`).
        self.weights: np.ndarray | None = None
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
        labels = self._bin_labels(cumulative)[offset + ends]

        parents, new_weights = self._resample_to_budget(
            weights, labels, top_n, self.rng
        )
        self.weights = np.asarray(new_weights, dtype=float)
        return [int(offset + ends[p]) for p in parents]

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
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if state and state.get("weights") is not None:
            self.weights = np.asarray(state["weights"], dtype=float)



SpawnerFactory.register("we", WESpawner)
