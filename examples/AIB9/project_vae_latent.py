"""Project AIB9 trajectories into the trained 2D VAE latent space."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    BASE_DIR
    / "training_data"
    / "aib9_lr_tda"
    / "vae_latent"
    / "aib9_vae_latent_model.pt"
)


def _load_phi_psi_projector():
    path = BASE_DIR / "project_phi_psi.py"
    spec = importlib.util.spec_from_file_location("aib9_project_phi_psi", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_all_phi_psi


class AIB9VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int = 2):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            encoder_layers.extend([nn.Linear(previous, hidden), nn.SiLU()])
            previous = hidden
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(previous, latent_dim)
        self.logvar = nn.Linear(previous, latent_dim)

        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(previous, hidden), nn.SiLU()])
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.classifier = nn.Linear(latent_dim, 2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)


def _load_model(model_path: Path) -> tuple[AIB9VAE, dict]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AIB9VAE(
        int(checkpoint["input_dim"]),
        list(checkpoint["hidden_dims"]),
        int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _vae_latent_from_phi_psi(phi_psi: np.ndarray, model_path: Path) -> np.ndarray:
    model, checkpoint = _load_model(model_path)
    torsion_sincos = np.concatenate([np.sin(phi_psi), np.cos(phi_psi)], axis=1).astype(np.float32)
    x = (torsion_sincos - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        mu, _logvar = model.encode(torch.tensor(x, dtype=torch.float32))
    return mu.cpu().numpy().astype(np.float32)


def extract_cvs(trajectories: list[str], top_file: str, conf_file: str) -> np.ndarray:
    """Return fixed 2D CVs: VAE latent mean coordinates [z0, z1]."""

    model_path = Path(os.environ.get("AIB9_VAE_MODEL", str(DEFAULT_MODEL))).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"VAE model checkpoint not found: {model_path}. "
            "Run examples/AIB9/train_vae_latent.py first."
        )

    extract_all_phi_psi = _load_phi_psi_projector()
    phi_psi = extract_all_phi_psi(
        trajectories=trajectories,
        top_file=top_file,
        conf_file=conf_file,
    )
    return _vae_latent_from_phi_psi(phi_psi, model_path)
