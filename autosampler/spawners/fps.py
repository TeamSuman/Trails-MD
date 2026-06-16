from typing import Optional

import numpy as np

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class FPSSpawner(Spawner):
    """Spawns walkers using Farthest-Point Sampling to maximise spatial coverage.

    When *mode* is ``"target"`` and a *target* point is provided the greedy
    search is seeded from the frame nearest to the target rather than a random
    frame, so the resulting diverse set is anchored to the target region.
    """

    def __init__(
        self,
        mode: str = "explore",
        target: Optional[list] = None,
        **_,
    ):
        self.mode = mode
        self.target = np.asarray(target, dtype=float) if target is not None else None

    def sample(self, points: np.ndarray, top_n: int, history=None) -> list:
        """Greedily selects the farthest points, optionally anchored near *target*."""
        points = np.asarray(points, dtype=float)
        cumulative_points = _cumulative_points(points, history)
        n_points = len(cumulative_points)
        if n_points <= top_n:
            return list(range(n_points))

        # Seed selection
        if self.mode == "target" and self.target is not None:
            # Anchor the search at the frame closest to the target so the
            # diverse set covers the target region first.
            dists_to_target = np.linalg.norm(cumulative_points - self.target, axis=1)
            seed = int(np.argmin(dists_to_target))
        else:
            seed = np.random.randint(n_points)

        selected = [seed]
        distances = np.linalg.norm(cumulative_points - cumulative_points[seed], axis=1)

        while len(selected) < top_n:
            farthest = int(np.argmax(distances))
            selected.append(farthest)
            new_dist = np.linalg.norm(
                cumulative_points - cumulative_points[farthest], axis=1
            )
            distances = np.minimum(distances, new_dist)

        return selected


SpawnerFactory.register("fps", FPSSpawner)
