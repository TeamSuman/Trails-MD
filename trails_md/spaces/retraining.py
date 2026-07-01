"""Adaptive CV-retraining policy.

A learned CV can go stale as new regions of phase space are discovered. Instead
of always retraining on a fixed schedule, the ``vamp_adaptive`` policy retrains
only when the current CV's VAMP-2 score on fresh data drops relative to its
post-training reference — i.e. when the CV stops resolving the dynamics it is
now seeing. This couples retraining to sampling progress and avoids both
under- and over-training.

The controller is pure decision logic (no MD/torch), so it is fully unit-tested.
"""

from __future__ import annotations


class RetrainController:
    """Decide whether to (re)train the CV model this iteration.

    Parameters
    ----------
    policy:
        ``"fixed"`` reproduces the legacy ``iteration % retrain_freq == 0``
        schedule. ``"vamp_adaptive"`` retrains when the CV's VAMP-2 score drops
        by more than ``vamp_tol`` (relative) below its reference.
    retrain_freq:
        Cadence for the ``fixed`` policy.
    vamp_tol:
        Relative VAMP-2 drop that triggers a retrain (``vamp_adaptive``).
    min_interval / max_interval:
        Lower/upper bounds (in iterations) between retrains for the adaptive
        policy, to avoid thrashing and to guarantee periodic refreshes.
    """

    def __init__(
        self,
        policy: str = "fixed",
        retrain_freq: int = 1,
        vamp_tol: float = 0.1,
        min_interval: int = 1,
        max_interval: int | None = None,
    ):
        if policy not in {"fixed", "vamp_adaptive"}:
            raise ValueError("policy must be 'fixed' or 'vamp_adaptive'")
        self.policy = policy
        self.retrain_freq = int(retrain_freq)
        self.vamp_tol = float(vamp_tol)
        self.min_interval = int(min_interval)
        self.max_interval = None if max_interval is None else int(max_interval)
        self.reference_score: float | None = None
        self.iters_since_retrain: int = 0
        self.last_reason: str | None = None

    def should_retrain(
        self,
        iteration: int,
        has_model: bool,
        current_score: float | None = None,
    ) -> bool:
        """Return whether to retrain at ``iteration``.

        ``current_score`` is the VAMP-2 score of the *existing* CV on the current
        data (ignored when there is no model yet or for the ``fixed`` policy).
        """
        if not has_model:
            self.last_reason = "no model yet"
            return True

        if self.policy == "fixed":
            due = self.retrain_freq > 0 and iteration % self.retrain_freq == 0
            self.last_reason = "scheduled" if due else None
            return due

        # vamp_adaptive
        if self.iters_since_retrain < self.min_interval:
            self.last_reason = None
            return False
        if self.max_interval is not None and self.iters_since_retrain >= self.max_interval:
            self.last_reason = f"max interval ({self.max_interval}) reached"
            return True
        if self.reference_score is None or current_score is None:
            self.last_reason = None
            return False
        rel_drop = (self.reference_score - current_score) / max(
            abs(self.reference_score), 1e-9
        )
        if rel_drop > self.vamp_tol:
            self.last_reason = (
                f"VAMP-2 dropped {rel_drop:.1%} (>{self.vamp_tol:.1%}) "
                f"from {self.reference_score:.3f} to {current_score:.3f}"
            )
            return True
        self.last_reason = None
        return False

    def notify_retrained(self, new_score: float | None = None) -> None:
        """Record that a retrain happened, updating the reference VAMP-2 score."""
        self.iters_since_retrain = 0
        if new_score is not None:
            self.reference_score = (
                new_score
                if self.reference_score is None
                else max(self.reference_score, new_score)
            )

    def notify_skipped(self) -> None:
        self.iters_since_retrain += 1

    def state_dict(self) -> dict:
        return {
            "reference_score": self.reference_score,
            "iters_since_retrain": self.iters_since_retrain,
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.reference_score = state.get("reference_score")
        self.iters_since_retrain = int(state.get("iters_since_retrain", 0))
