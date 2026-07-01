"""Tests for the CV-method registry and the new VAMPNet / SPIB CV methods."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from trails_md.spaces.registry import (  # noqa: E402
    FIXED_MODE,
    adaptive_modes,
    ensure_available,
    get_method,
    is_adaptive_space,
    is_available,
)


def test_registry_lists_expected_methods():
    modes = set(adaptive_modes())
    for expected in {"pca", "tica", "tvae", "vampnet", "spib", "deep-tica", "deep-lda"}:
        assert expected in modes
    assert not is_adaptive_space(FIXED_MODE)
    assert is_adaptive_space("vampnet")


def test_registry_metadata():
    assert get_method("deep-lda").supervised is True
    assert get_method("spib").backend == "builtin"
    assert get_method("vampnet").time_lagged is True
    # Built-in / installed methods are available in the test environment.
    assert is_available("spib") is True
    assert is_available("pca") is True


def test_ensure_available_raises_for_missing_backend():
    # mlcolvar is an optional dependency, absent in the base test environment.
    if not is_available("deep-tica"):
        with pytest.raises(ImportError):
            ensure_available("deep-tica")


def _three_state_chain(n_steps=12000, seed=0, p_escape=0.03):
    rng = np.random.default_rng(seed)
    centers = np.array([-2.0, 0.0, 2.0])
    P = np.array(
        [
            [1 - p_escape, p_escape, 0.0],
            [p_escape, 1 - 2 * p_escape, p_escape],
            [0.0, p_escape, 1 - p_escape],
        ]
    )
    state = 0
    states = np.empty(n_steps, dtype=int)
    for i in range(n_steps):
        state = rng.choice(3, p=P[state])
        states[i] = state
    # 4D features (only the first coordinate carries the slow signal).
    feats = rng.normal(scale=0.2, size=(n_steps, 4))
    feats[:, 0] += centers[states]
    return feats.astype(np.float32), states


pytest.importorskip("torch")
pytest.importorskip("deeptime")

from trails_md.spaces.model import AdaptiveSpaceModel  # noqa: E402


@pytest.mark.parametrize("mode", ["vampnet", "spib"])
def test_cv_method_trains_and_separates_states(mode):
    feats, states = _three_state_chain()
    model = AdaptiveSpaceModel(
        space_mode=mode,
        lagtime=5,
        latent_dim=1,
        epochs=8,
        batch_size=512,
        encoder_hidden_dims=[32, 16],
    )
    model.fit(feats, walker_length=len(feats), n_walkers=1)
    proj = model.project(feats)
    assert proj.shape[0] == len(feats)
    # The learned CV should correlate with the true slow state.
    cv = np.asarray(proj)[:, 0]
    corr = abs(np.corrcoef(cv, states)[0, 1])
    assert corr > 0.5, f"{mode} CV correlation with state too low: {corr:.2f}"


def test_deep_lda_raises_informative_error():
    feats, _ = _three_state_chain(n_steps=2000)
    model = AdaptiveSpaceModel(space_mode="deep-lda", lagtime=5, latent_dim=1, epochs=2)
    # Only reaches the NotImplementedError when mlcolvar is installed; otherwise
    # the availability guard raises ImportError first. Either is acceptable.
    with pytest.raises((NotImplementedError, ImportError)):
        model.fit(feats, walker_length=len(feats), n_walkers=1)
