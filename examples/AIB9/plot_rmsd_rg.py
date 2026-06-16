#!/usr/bin/env python
"""Plot AIB9 exploration in RMSD/Rg space from AutoSampler XTC outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import yaml
from MDAnalysis.analysis import rms


BASE_DIR = Path(__file__).resolve().parent


def _resolve(path: str | Path, base: Path = BASE_DIR) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def _default_outdir(config_path: Path) -> Path:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return _resolve(config["outdir"], config_path.parent)


def _trajectory_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"iteration_(\d+)_(\d+)\.xtc$", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    parent = re.search(r"iter_(\d+)$", path.parent.name)
    return int(parent.group(1)) if parent else -1, -1, path.name


def find_trajectories(outdir: Path) -> list[Path]:
    trajectories = sorted(outdir.glob("iter_*/iteration_*_*.xtc"), key=_trajectory_sort_key)
    if not trajectories:
        raise FileNotFoundError(f"No AutoSampler XTC files found under {outdir}")
    return trajectories


def compute_rmsd_rg(
    topology: Path,
    reference: Path,
    trajectories: list[Path],
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref = mda.Universe(str(reference))
    ref_atoms = ref.select_atoms(selection)
    if ref_atoms.n_atoms == 0:
        raise ValueError(f"Reference selection matched no atoms: {selection!r}")

    rmsd_values: list[float] = []
    rg_values: list[float] = []
    iteration_values: list[int] = []
    walker_values: list[int] = []

    for traj_path in trajectories:
        iteration, walker, _ = _trajectory_sort_key(traj_path)
        universe = mda.Universe(str(topology), str(traj_path))
        atoms = universe.select_atoms(selection)
        if atoms.n_atoms != ref_atoms.n_atoms:
            raise ValueError(
                f"Selection atom count mismatch for {traj_path}: "
                f"{atoms.n_atoms} vs reference {ref_atoms.n_atoms}"
            )

        for _ts in universe.trajectory:
            rg_values.append(float(atoms.radius_of_gyration()))
            rmsd_values.append(
                float(
                    rms.rmsd(
                        atoms.positions,
                        ref_atoms.positions,
                        center=True,
                        superposition=True,
                    )
                )
            )
            iteration_values.append(iteration)
            walker_values.append(walker)
        universe.trajectory.close()

    return (
        np.asarray(rmsd_values, dtype=float),
        np.asarray(rg_values, dtype=float),
        np.asarray(iteration_values, dtype=int),
        np.asarray(walker_values, dtype=int),
    )


def plot_rmsd_rg(
    rmsd_values: np.ndarray,
    rg_values: np.ndarray,
    iterations: np.ndarray,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(
        rmsd_values,
        rg_values,
        c=iterations,
        cmap="viridis",
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title("AIB9 Exploration in RMSD/Rg Space")
    ax.set_xlabel("AIB heavy-atom RMSD to equilibrated structure (A)")
    ax.set_ylabel("AIB heavy-atom radius of gyration (A)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("AutoSampler iteration")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config_adaptive.yaml")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--selection",
        default="resname AIB and not name H*",
        help="MDAnalysis atom selection used for RMSD/Rg.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = _resolve(args.config, Path.cwd())
    outdir = _resolve(args.outdir, BASE_DIR) if args.outdir else _default_outdir(config_path)
    topology = _resolve(args.topology, BASE_DIR) if args.topology else BASE_DIR / "aib9_equilibrated.pdb"
    reference = _resolve(args.reference, BASE_DIR) if args.reference else BASE_DIR / "aib9_equilibrated.pdb"
    output_png = args.output if args.output else outdir / "aib9_rmsd_rg_exploration.png"
    output_png = _resolve(output_png, Path.cwd())
    output_npz = output_png.with_suffix(".npz")

    trajectories = find_trajectories(outdir)
    print(f"Found {len(trajectories)} XTC files in {outdir}")
    rmsd_values, rg_values, iterations, walkers = compute_rmsd_rg(
        topology=topology,
        reference=reference,
        trajectories=trajectories,
        selection=args.selection,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_rmsd_rg(rmsd_values, rg_values, iterations, output_png)
    np.savez_compressed(
        output_npz,
        rmsd=rmsd_values,
        rg=rg_values,
        iteration=iterations,
        walker=walkers,
        selection=args.selection,
    )
    print(f"Frames analyzed: {len(rmsd_values)}")
    print(f"RMSD range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f} A")
    print(f"Rg range: {rg_values.min():.3f} - {rg_values.max():.3f} A")
    print(f"Saved plot: {output_png}")
    print(f"Saved data: {output_npz}")


if __name__ == "__main__":
    main()
