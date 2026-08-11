"""State Predictive Information Bottleneck (SPIB) collective variable.

A compact, self-contained PyTorch implementation of SPIB (Wang & Tiwary,
*Nat. Commun.* 2021). SPIB learns a low-dimensional CV ``z`` that retains just
enough information about the present configuration to predict the *future*
state (a time-lagged, discretised label) under a variational information
bottleneck:

    L = E[ CE(p(state_{t+tau} | z), label_{t+tau}) ] + beta * KL(q(z|x) || prior)

The encoder mean becomes the CV used for projection. Only ``torch`` is required
(no extra dependency), so SPIB is available out of the box.

This module exposes the network pieces plus :func:`train_spib`, which the
:class:`~trails_md.spaces.model.AdaptiveSpaceModel` calls.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def _make_mlp(input_size: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_size
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*layers)


class SPIBEncoder(nn.Module):
    """Encode features into a Gaussian latent (mean, log-variance)."""

    def __init__(
        self,
        input_size: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = _make_mlp(input_size, hidden_dims, dropout)
        last = hidden_dims[-1] if hidden_dims else input_size
        self.mean = nn.Linear(last, latent_dim)
        self.log_var = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.mean(h), self.log_var(h)


class SPIBPredictor(nn.Module):
    """Predict the future-state distribution from the latent CV."""

    def __init__(self, latent_dim: int, n_states: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _reparameterise(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    return mean + std * torch.randn_like(std)


def _kl_to_standard_normal(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))


def _state_labels(features: np.ndarray, n_states: int, seed: int) -> np.ndarray:
    """Initial discrete states via k-means (deeptime, then sklearn fallback)."""
    n_states = max(2, min(n_states, len(features)))
    try:
        from deeptime.clustering import KMeans

        model = KMeans(
            n_clusters=n_states, max_iter=100, fixed_seed=seed, progress=None
        ).fit_fetch(features)
        return np.asarray(model.transform(features), dtype=np.int64)
    except Exception:  # noqa: BLE001
        from sklearn.cluster import KMeans as SKMeans

        model = SKMeans(n_clusters=n_states, n_init=10, random_state=seed)
        return np.asarray(model.fit_predict(features), dtype=np.int64)


def train_spib(
    traj_list: Sequence[np.ndarray],
    lagtime: int,
    latent_dim: int,
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    n_states: int = 10,
    beta: float = 1e-3,
    dropout: float = 0.0,
    device: str | None = None,
    seed: int = 42,
    refine_rounds: int = 5,
    refine_tol: float = 0.01,
) -> tuple[SPIBEncoder, dict]:
    """Train SPIB and return ``(encoder, info)``.

    The encoder is returned on CPU in eval mode for projection; ``info`` records how
    the self-consistent relabelling converged.

    Parameters
    ----------
    traj_list:
        List of continuous per-walker feature arrays ``(n_frames_i, n_features)``
        (already scaled). Time-lagged pairs are formed within each trajectory.
    lagtime:
        Prediction lag (in frames).
    n_states:
        Number of states the bottleneck predicts *initially*. Refinement prunes
        states that end up unoccupied, so the effective count adapts to the data --
        this is how SPIB decides the number of metastable states rather than having
        it imposed.
    beta:
        Information-bottleneck weight on the KL term.
    refine_rounds:
        Rounds of self-consistent relabelling. After each round every frame is
        relabelled by the model's own predicted state and training resumes. This
        loop is what makes the method *state predictive*: without it the CV is a
        variational information bottleneck onto whatever partition k-means happened
        to produce. ``0`` disables refinement and reproduces the earlier
        fixed-label behaviour, for reproducing results generated before this
        existed.
    refine_tol:
        Stop early once fewer than this fraction of frames change label in a round.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    stacked = np.vstack([np.asarray(t, dtype=np.float32) for t in traj_list])
    input_size = stacked.shape[1]
    labels_all = _state_labels(stacked, n_states, seed)
    n_states_eff = int(labels_all.max()) + 1

    def lagged_pairs(labels: np.ndarray):
        """(x_t, label_{t+lag}) pairs formed strictly WITHIN each trajectory.

        Velocities are redrawn at every respawn, so a pair spanning two walker
        segments would relate dynamically unrelated configurations.
        """
        offset = 0
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        keep: list[np.ndarray] = []      # index of x_t in the stacked array
        for traj in traj_list:
            length = len(traj)
            if length > lagtime:
                xs.append(np.asarray(traj[:-lagtime], dtype=np.float32))
                ys.append(labels[offset + lagtime : offset + length])
                keep.append(np.arange(offset, offset + length - lagtime))
            offset += length
        if not xs:
            raise ValueError("SPIB: no trajectory longer than the lag time.")
        return np.vstack(xs), np.concatenate(ys), np.concatenate(keep)

    x_np, y_np, _ = lagged_pairs(labels_all)
    x = torch.from_numpy(x_np).to(device)
    x_all = torch.from_numpy(stacked).to(device)

    encoder = SPIBEncoder(input_size, latent_dim, hidden_dims, dropout).to(device)
    ce = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = max(1, min(batch_size, x.shape[0]))

    def train_once(y_np_local: np.ndarray, n_out: int):
        predictor = SPIBPredictor(latent_dim, n_out).to(device)
        optim = torch.optim.Adam(
            list(encoder.parameters()) + list(predictor.parameters()), lr=learning_rate
        )
        y_local = torch.from_numpy(y_np_local).long().to(device)
        n = x.shape[0]
        encoder.train()
        predictor.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=generator).to(device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                mean, log_var = encoder(x[idx])
                z = _reparameterise(mean, log_var)
                loss = ce(predictor(z), y_local[idx]) + beta * _kl_to_standard_normal(
                    mean, log_var
                )
                optim.zero_grad()
                loss.backward()
                optim.step()
        return predictor

    predictor = train_once(y_np, n_states_eff)

    # --- self-consistent relabelling ------------------------------------------
    # Relabel every frame by the model's own predicted state, drop states that end up
    # unoccupied (this is how SPIB adapts the number of metastable states), and
    # retrain. Repeat until the labelling stops moving. Skipping this loop leaves a
    # variational information bottleneck onto a fixed k-means partition, which is a
    # different method.
    label_changes: list[float] = []
    for _ in range(max(0, int(refine_rounds))):
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            mean_all, _ = encoder(x_all)
            new_labels = predictor(mean_all).argmax(dim=1).cpu().numpy().astype(np.int64)
        changed = float(np.mean(new_labels != labels_all))
        label_changes.append(changed)

        # Renumber to consecutive ids so unoccupied states disappear.
        uniq, new_labels = np.unique(new_labels, return_inverse=True)
        if len(uniq) < 2:
            # Everything collapsed to one state: nothing left to predict, so stop
            # rather than train a degenerate classifier.
            labels_all = new_labels.astype(np.int64)
            break
        labels_all = new_labels.astype(np.int64)
        n_states_eff = len(uniq)
        _, y_np, _ = lagged_pairs(labels_all)
        predictor = train_once(y_np, n_states_eff)
        if changed < refine_tol:
            break

    encoder.eval()
    info = {
        "label_changes": label_changes,
        "n_states_initial": int(n_states),
        "n_states_final": int(len(np.unique(labels_all))),
        "refine_rounds_run": len(label_changes),
    }
    return encoder.to("cpu"), info
