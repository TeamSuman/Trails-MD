"""Tests for the adaptive CV-retraining controller and reproducible seeding."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from autosampler.spaces.retraining import RetrainController  # noqa: E402


def test_fixed_policy_matches_legacy_schedule():
    ctrl = RetrainController(policy="fixed", retrain_freq=3)
    # No model yet -> always retrain.
    assert ctrl.should_retrain(0, has_model=False) is True
    # With a model, retrain only when iteration % freq == 0.
    assert ctrl.should_retrain(3, has_model=True) is True
    assert ctrl.should_retrain(4, has_model=True) is False
    assert ctrl.should_retrain(6, has_model=True) is True


def test_fixed_policy_freq_zero_never_retrains_with_model():
    ctrl = RetrainController(policy="fixed", retrain_freq=0)
    assert ctrl.should_retrain(5, has_model=True) is False


def test_vamp_adaptive_retrains_on_score_drop():
    ctrl = RetrainController(policy="vamp_adaptive", vamp_tol=0.1, min_interval=1)
    # First training establishes the reference.
    assert ctrl.should_retrain(0, has_model=False) is True
    ctrl.notify_retrained(new_score=2.0)
    # Score holds -> no retrain.
    ctrl.notify_skipped()
    assert ctrl.should_retrain(1, has_model=True, current_score=1.95) is False
    # Score drops >10% -> retrain, with an informative reason.
    ctrl.notify_skipped()
    assert ctrl.should_retrain(2, has_model=True, current_score=1.5) is True
    assert "VAMP-2 dropped" in ctrl.last_reason


def test_vamp_adaptive_respects_min_and_max_interval():
    ctrl = RetrainController(
        policy="vamp_adaptive", vamp_tol=0.1, min_interval=2, max_interval=4
    )
    ctrl.should_retrain(0, has_model=False)
    ctrl.notify_retrained(new_score=2.0)
    # min_interval=2: a big drop is ignored until enough iterations pass.
    ctrl.notify_skipped()  # iters_since_retrain = 1
    assert ctrl.should_retrain(1, has_model=True, current_score=0.5) is False
    ctrl.notify_skipped()  # = 2
    assert ctrl.should_retrain(2, has_model=True, current_score=0.5) is True

    # max_interval=4: force a refresh even if the score is stable.
    ctrl.notify_retrained(new_score=2.0)
    for _ in range(4):
        ctrl.notify_skipped()
    assert ctrl.should_retrain(10, has_model=True, current_score=2.0) is True
    assert "max interval" in ctrl.last_reason


def test_controller_state_roundtrip():
    ctrl = RetrainController(policy="vamp_adaptive")
    ctrl.notify_retrained(new_score=1.5)
    ctrl.notify_skipped()
    state = ctrl.state_dict()
    other = RetrainController(policy="vamp_adaptive")
    other.load_state_dict(state)
    assert other.reference_score == 1.5
    assert other.iters_since_retrain == ctrl.iters_since_retrain


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        RetrainController(policy="banana")


def test_retrain_policy_config_validation():
    from pydantic import ValidationError

    from autosampler.config import AutoSamplerConfig

    base = {
        "system": {"conf_file": "a", "top_file": "b"},
        "engine": {},
        "spawning": {},
    }
    cfg = AutoSamplerConfig(**base, retrain_policy="vamp_adaptive")
    assert cfg.retrain_policy == "vamp_adaptive"
    with pytest.raises(ValidationError):
        AutoSamplerConfig(**base, retrain_policy="banana")


def test_seed_manager_is_deterministic():
    import numpy as np

    from autosampler.utils.seeds import SeedManager

    SeedManager(123).set_seed()
    a = np.random.rand(5)
    SeedManager(123).set_seed()
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)
