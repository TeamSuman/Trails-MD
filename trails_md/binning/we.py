"""Weighted-ensemble (WE) resampling.

Implements the split/merge resampling of Huber & Kim (1996): walkers carry
statistical weights and are kept at a target count per bin while **total weight
is conserved**. Under-represented bins gain walkers by *splitting* high-weight
walkers (weight divided among copies); over-represented bins lose walkers by
*merging* low-weight walkers (weights summed, one survivor chosen with
probability proportional to weight). This focuses sampling on bins/regions
without biasing the estimated probabilities.

The core operates on plain arrays (weights + bin labels), so it is independent
of the binning implementation and fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResampleResult:
    """Outcome of one WE resampling step.

    ``parents`` are indices into the *input* ensemble (a value may repeat when a
    walker was split); ``weights`` are the matching statistical weights. Total
    weight equals the input total (up to floating point).
    """

    parents: list[int]
    weights: list[float]

    def __len__(self) -> int:
        return len(self.parents)


class WeightedEnsemble:
    """Split/merge resampler that conserves probability weight.

    Parameters
    ----------
    target_per_bin:
        Desired number of walkers in each occupied bin after resampling.
    """

    def __init__(self, target_per_bin: int = 4):
        if target_per_bin < 1:
            raise ValueError("target_per_bin must be >= 1")
        self.target_per_bin = int(target_per_bin)

    def resample(
        self,
        weights,
        bin_labels,
        target_per_bin: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResampleResult:
        """Resample walkers to ``target_per_bin`` per occupied bin.

        ``weights[i]`` and ``bin_labels[i]`` describe walker ``i``. Returns the
        post-resampling ensemble as parent indices + weights.
        """
        weights = np.asarray(weights, dtype=float)
        labels = np.asarray(bin_labels)
        if weights.shape[0] != labels.shape[0]:
            raise ValueError("weights and bin_labels must have equal length")
        target = self.target_per_bin if target_per_bin is None else int(target_per_bin)
        if target < 1:
            raise ValueError("target_per_bin must be >= 1")
        rng = np.random.default_rng() if rng is None else rng

        parents_out: list[int] = []
        weights_out: list[float] = []
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            members = [int(i) for i in idx]  # parent index per current walker
            mweights = [float(weights[i]) for i in idx]
            members, mweights = self._merge(members, mweights, target, rng)
            members, mweights = self._split(members, mweights, target)
            parents_out.extend(members)
            weights_out.extend(mweights)
        return ResampleResult(parents_out, weights_out)

    @staticmethod
    def _merge(members, mweights, target, rng):
        """Merge the two lowest-weight walkers until ``target`` remain."""
        while len(members) > target:
            order = np.argsort(mweights)
            i, j = int(order[0]), int(order[1])
            combined = mweights[i] + mweights[j]
            # Survivor chosen with probability proportional to weight.
            prob_i = mweights[i] / combined if combined > 0 else 0.5
            survivor = i if rng.random() < prob_i else j
            keep_parent = members[survivor]
            members = [m for k, m in enumerate(members) if k not in (i, j)]
            mweights = [w for k, w in enumerate(mweights) if k not in (i, j)]
            members.append(keep_parent)
            mweights.append(combined)
        return members, mweights

    @staticmethod
    def _split(members, mweights, target):
        """Split the highest-weight walker until ``target`` walkers exist."""
        while 0 < len(members) < target:
            k = int(np.argmax(mweights))
            half = mweights[k] / 2.0
            mweights[k] = half
            members.append(members[k])
            mweights.append(half)
        return members, mweights
