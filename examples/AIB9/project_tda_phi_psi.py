"""Project AIB9 trajectories to [TDA CV, one phi/psi angle].

The TDA axis uses the lightweight supervised scalar-CV baseline trained on
left/right basin samples in ``training_data/aib9_lr_tda``.  The second axis is
one selected AIB phi or psi angle in radians.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "training_data" / "aib9_lr_tda" / "tda_baseline" / "tda_baseline_model.pt"
DEFAULT_RESIDUE = 5
DEFAULT_ANGLE = "phi"


def _load_phi_psi_projector():
    path = BASE_DIR / "project_phi_psi.py"
    spec = importlib.util.spec_from_file_location("aib9_project_phi_psi", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_all_phi_psi


class TDARegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.Tanh()])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TDARegressor(checkpoint["input_dim"], list(checkpoint["hidden_dims"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _tda_cv_from_phi_psi(phi_psi: np.ndarray, model_path: Path) -> np.ndarray:
    model, checkpoint = _load_model(model_path)
    torsion_sincos = np.concatenate([np.sin(phi_psi), np.cos(phi_psi)], axis=1).astype(np.float32)
    x = (torsion_sincos - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        cv = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return cv.astype(np.float32)


def _selected_angle(phi_psi: np.ndarray) -> np.ndarray:
    residue = int(os.environ.get("AIB9_TDA_ANGLE_RESIDUE", DEFAULT_RESIDUE))
    angle = os.environ.get("AIB9_TDA_ANGLE", DEFAULT_ANGLE).strip().lower()
    if residue < 1 or residue > 9:
        raise ValueError("AIB9_TDA_ANGLE_RESIDUE must be between 1 and 9.")
    if angle not in {"phi", "psi"}:
        raise ValueError("AIB9_TDA_ANGLE must be 'phi' or 'psi'.")
    column = 2 * (residue - 1) + (0 if angle == "phi" else 1)
    return phi_psi[:, column].astype(np.float32)


def extract_cvs(trajectories: list[str], top_file: str, conf_file: str) -> np.ndarray:
    """Return fixed 2D CVs: [TDA scalar CV, selected AIB phi/psi angle]."""

    model_path = Path(os.environ.get("AIB9_TDA_MODEL", str(DEFAULT_MODEL))).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"TDA model checkpoint not found: {model_path}")

    extract_all_phi_psi = _load_phi_psi_projector()
    phi_psi = extract_all_phi_psi(
        trajectories=trajectories,
        top_file=top_file,
        conf_file=conf_file,
    )
    tda_cv = _tda_cv_from_phi_psi(phi_psi, model_path)
    angle = _selected_angle(phi_psi)
    return np.column_stack([tda_cv, angle]).astype(np.float32)
