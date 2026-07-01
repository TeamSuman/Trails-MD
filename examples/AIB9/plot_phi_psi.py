#!/usr/bin/env python
"""Plot AIB9 phi/psi exploration from Trails-MD outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from trails_md.spaces.features import FeatureExtractor


BASE_DIR = Path(__file__).resolve().parent


def _resolve(path: str | Path, base: Path = BASE_DIR) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def _default_outdir(config_path: Path) -> Path:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return _resolve(config["outdir"], config_path.parent)


def _default_topology(config_path: Path) -> Path:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    system = config.get("system", {})
    topology = (
        system.get("trajectory_topology_file")
        or system.get("top_file")
        or BASE_DIR / "aib9_equilibrated.pdb"
    )
    return _resolve(topology, config_path.parent)


def _trajectory_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"iteration_(\d+)_(\d+)\.xtc$", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    parent = re.search(r"iter_(\d+)$", path.parent.name)
    return int(parent.group(1)) if parent else -1, -1, path.name


def find_trajectories(outdir: Path) -> list[Path]:
    trajectories = sorted(outdir.glob("iter_*/iteration_*_*.xtc"), key=_trajectory_sort_key)
    if not trajectories:
        raise FileNotFoundError(f"No Trails-MD XTC files found under {outdir}")
    return trajectories


def iteration_labels(trajectories: list[Path], frames_per_traj: list[int]) -> np.ndarray:
    labels = []
    for traj_path, n_frames in zip(trajectories, frames_per_traj):
        iteration, _walker, _ = _trajectory_sort_key(traj_path)
        labels.extend([iteration] * n_frames)
    return np.asarray(labels, dtype=int)


def plot_residue(
    phi_psi: np.ndarray,
    iterations: np.ndarray,
    residue: int,
    output: Path,
) -> None:
    phi = phi_psi[:, 2 * (residue - 1)]
    psi = phi_psi[:, 2 * (residue - 1) + 1]

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    scatter = ax.scatter(phi, psi, c=iterations, cmap="viridis", s=10, alpha=0.72, linewidths=0)
    ax.set_title(f"AIB9 Residue {residue} Phi/Psi Exploration")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel(r"$\psi$ (rad)")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Trails-MD iteration")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config_phi_psi.yaml")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="Topology for trajectory analysis. Defaults to trajectory_topology_file from --config.",
    )
    parser.add_argument("--residue", type=int, default=5, choices=range(1, 10))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = _resolve(args.config, Path.cwd())
    outdir = _resolve(args.outdir, BASE_DIR) if args.outdir else _default_outdir(config_path)
    topology = _resolve(args.topology, Path.cwd()) if args.topology else _default_topology(config_path)
    output = args.output or outdir / f"aib9_residue_{args.residue}_phi_psi.png"
    output = _resolve(output, Path.cwd())
    output_npz = output.with_suffix(".npz")

    trajectories = find_trajectories(outdir)
    extractor = FeatureExtractor(topology=str(topology), selection="resname AIB")

    all_features = []
    frame_counts = []
    for traj_path in trajectories:
        features = extractor.extract_aib9_phi_psi([str(traj_path)])
        all_features.append(features)
        frame_counts.append(len(features))
    phi_psi = np.vstack(all_features)
    iterations = iteration_labels(trajectories, frame_counts)

    output.parent.mkdir(parents=True, exist_ok=True)
    plot_residue(phi_psi, iterations, args.residue, output)
    np.savez_compressed(
        output_npz,
        phi_psi=phi_psi,
        iteration=iterations,
        residue=args.residue,
        columns=[f"{name}{i}" for i in range(1, 10) for name in ("phi", "psi")],
    )

    print(f"Found {len(trajectories)} XTC files in {outdir}")
    print(f"Frames analyzed: {len(phi_psi)}")
    print(f"Saved plot: {output}")
    print(f"Saved data: {output_npz}")


if __name__ == "__main__":
    main()
