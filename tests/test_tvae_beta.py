"""The TVAE KLD weight (beta) must be settable, recorded, and default to 1.0.

deeptime's TVAE loss is ``mse + beta * kld / n_features``. Until now Trails-MD
constructed the estimator without passing ``beta`` at all, so every reported result
silently used deeptime's default of 1.0 and no user could change it. A referee asking
"what beta did you use?" could only be answered by reading a dependency's source.

These tests pin both halves of the fix: the value reaches the estimator, and the
default is the 1.0 that all previously published Trails-MD results were produced with.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from trails_md.spaces.model import AdaptiveSpaceModel  # noqa: E402

torch = pytest.importorskip("torch")


def _features(n_walkers=2, walker_length=24, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_walkers * walker_length, n_features))


def test_tvae_beta_defaults_to_one():
    """1.0 is deeptime's default and therefore what every published run used."""
    model = AdaptiveSpaceModel(space_mode="tvae")
    assert model.tvae_beta == pytest.approx(1.0)


def test_tvae_beta_reaches_the_estimator():
    """Setting beta must change the estimator's KLD weight, not just an attribute.

    Asserting on ``model.model._beta`` after a real fit checks the wiring end to end:
    an implementation that stored the value but kept calling ``TVAE(...)`` without it
    would pass an attribute check and fail this one.
    """
    model = AdaptiveSpaceModel(
        space_mode="tvae", tvae_beta=0.25, epochs=1, lagtime=1, latent_dim=2
    )
    model.fit(_features(), walker_length=24, n_walkers=2)
    assert model.model._beta == pytest.approx(0.25)


def test_tvae_default_beta_reaches_the_estimator():
    model = AdaptiveSpaceModel(space_mode="tvae", epochs=1, lagtime=1, latent_dim=2)
    model.fit(_features(), walker_length=24, n_walkers=2)
    assert model.model._beta == pytest.approx(1.0)


def test_tvae_beta_must_be_positive():
    """A non-positive KLD weight is not a meaningful VAE objective."""
    with pytest.raises(ValueError):
        AdaptiveSpaceModel(space_mode="tvae", tvae_beta=0.0)
    with pytest.raises(ValueError):
        AdaptiveSpaceModel(space_mode="tvae", tvae_beta=-1.0)


def test_tvae_beta_survives_a_checkpoint_round_trip():
    """Restored checkpoints must keep beta; otherwise a resumed run silently
    changes its objective mid-campaign."""
    import pickle

    model = AdaptiveSpaceModel(space_mode="tvae", tvae_beta=0.4)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.tvae_beta == pytest.approx(0.4)


def test_old_checkpoint_without_beta_restores_to_one():
    """Checkpoints written before beta existed must keep reproducing their original
    behaviour, which was deeptime's default of 1.0."""
    model = AdaptiveSpaceModel(space_mode="tvae", tvae_beta=0.4)
    state = dict(model.__dict__)
    del state["tvae_beta"]
    revived = AdaptiveSpaceModel.__new__(AdaptiveSpaceModel)
    revived.__setstate__(state)
    assert revived.tvae_beta == pytest.approx(1.0)
