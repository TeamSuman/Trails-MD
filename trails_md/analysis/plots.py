"""Matplotlib plotting utilities for MSM analysis.

Each function accepts an optional Axes and returns it, so plots compose into
custom figures; :func:`plot_convergence_report` assembles a standard multi-panel
summary for a run. matplotlib is an optional dependency (``trails-md[examples]``);
it is imported lazily with an actionable error if missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data as _data


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. Install with: "
            'pip install "trails-md[examples]".'
        ) from exc


def _ax(ax):
    if ax is not None:
        return ax
    return _plt().subplots(figsize=(5, 4))[1]


def plot_implied_timescales(lagtimes, timescales, ax=None):
    """Implied timescales vs lag time (the ITS convergence plot)."""
    ax = _ax(ax)
    lagtimes = np.asarray(lagtimes, dtype=float)
    timescales = np.asarray(timescales, dtype=float)
    for j in range(timescales.shape[1]):
        ax.plot(lagtimes, timescales[:, j], marker="o", label=f"t{j + 2}")
    ax.fill_between(lagtimes, lagtimes, color="0.85", label="lag time")
    ax.set_xlabel("lag time (frames)")
    ax.set_ylabel("implied timescale (frames)")
    ax.set_yscale("log")
    ax.set_title("Implied timescales")
    ax.legend(fontsize="small")
    return ax


def plot_timescale_convergence(series, ax=None):
    """Slowest implied timescales vs iteration."""
    ax = _ax(ax)
    iters = series["iterations"]
    ts = series["timescales"]
    n = min(3, ts.shape[1]) if ts.size else 0
    for j in range(n):
        ax.plot(iters, ts[:, j], marker="o", label=f"t{j + 2}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("implied timescale (frames)")
    ax.set_title("Timescale convergence")
    if n:
        ax.legend(fontsize="small")
    return ax


def plot_vamp2_convergence(series, ax=None):
    """VAMP-2 score vs iteration."""
    ax = _ax(ax)
    ax.plot(series["iterations"], series["vamp2"], marker="o", color="C3")
    ax.set_xlabel("iteration")
    ax.set_ylabel("VAMP-2 score")
    ax.set_title("VAMP-2 convergence")
    return ax


def plot_free_energy_surface(points, bins=60, temperature=300.0, ax=None):
    """Free-energy surface over the first two CV dimensions."""
    ax = _ax(ax)
    f, xedges, yedges = _data.free_energy_surface(points, bins, temperature)
    mesh = ax.pcolormesh(xedges, yedges, f, shading="auto", cmap="viridis")
    ax.figure.colorbar(mesh, ax=ax, label="free energy (kJ/mol)")
    ax.set_xlabel("CV 1")
    ax.set_ylabel("CV 2")
    ax.set_title("Free-energy surface")
    return ax


def plot_metastable_free_energy(populations, temperature=300.0, ax=None):
    """Bar chart of metastable-state free energies (from PCCA+ populations)."""
    ax = _ax(ax)
    f = _data.free_energy_from_populations(populations, temperature)
    ax.bar(np.arange(len(f)), f, color="C0")
    ax.set_xlabel("metastable state")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.set_title("Metastable free energies")
    return ax


def plot_msm_network(transition_matrix, stationary=None, ax=None, threshold=0.01):
    """Draw the MSM as a network: node size ~ stationary weight, edges ~ T_ij.

    Uses a circular layout (no networkx dependency).
    """
    ax = _ax(ax)
    T = np.asarray(transition_matrix, dtype=float)
    n = T.shape[0]
    if stationary is None:
        stationary = np.full(n, 1.0 / n)
    stationary = np.asarray(stationary, dtype=float)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    for i in range(n):
        for j in range(n):
            if i != j and T[i, j] > threshold:
                ax.annotate(
                    "",
                    xy=pos[j],
                    xytext=pos[i],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="0.6",
                        alpha=min(1.0, float(T[i, j])),
                        lw=0.5 + 2.0 * float(T[i, j]),
                    ),
                )
    sizes = 200 + 3000 * stationary / max(stationary.max(), 1e-9)
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=np.arange(n), cmap="tab20", zorder=3)
    for i in range(n):
        ax.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MSM network")
    return ax


def plot_convergence_report(run_dir, outfile=None, temperature=300.0):
    """Assemble a standard multi-panel summary figure for a run.

    Panels: VAMP-2 convergence, timescale convergence, free-energy surface, and
    (when available) the latest implied-timescale sweep / MSM network. Saves to
    ``outfile`` (default ``<run_dir>/analysis/convergence_report.png``) and
    returns the saved path.
    """
    plt = _plt()
    series = _data.load_msm_series(run_dir)
    latest = _data.load_latest_msm(run_dir)
    points = _data.load_cv_points(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_vamp2_convergence(series, ax=axes[0, 0])
    plot_timescale_convergence(series, ax=axes[0, 1])

    if points.size and points.shape[1] >= 2:
        plot_free_energy_surface(points, temperature=temperature, ax=axes[1, 0])
    else:
        axes[1, 0].set_axis_off()

    if latest is not None and "its_lagtimes" in latest:
        plot_implied_timescales(
            latest["its_lagtimes"], latest["its_timescales"], ax=axes[1, 1]
        )
    elif latest is not None and "transition_matrix" in latest:
        plot_msm_network(
            latest["transition_matrix"],
            latest.get("stationary_distribution"),
            ax=axes[1, 1],
        )
    else:
        axes[1, 1].set_axis_off()

    fig.tight_layout()
    if outfile is None:
        outfile = Path(run_dir) / "analysis" / "convergence_report.png"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_flux_convergence(flux_history, tau_ps, discard_fraction=0.5, ax=None):
    """Weighted-ensemble kinetics: recycled flux and the running MFPT vs iteration.

    Two overlaid series on twin axes: the per-iteration recycled flux (left) and the
    cumulative MFPT estimate over the retained tail (right). The shaded region is the
    discarded pre-steady-state transient. A flat flux tail = a trustworthy rate.
    """
    from ..spawners.we import steady_state_mfpt

    ax = _ax(ax)
    flux = np.asarray([] if flux_history is None else list(flux_history), dtype=float)
    n = flux.size
    it = np.arange(1, n + 1)

    ax.plot(it, flux, color="#8C5109", lw=1.2, marker="o", ms=2.5, label="recycled flux")
    ax.set_xlabel("iteration")
    ax.set_ylabel("recycled flux per τ", color="#8C5109")
    ax.tick_params(axis="y", labelcolor="#8C5109")
    n_skip = int(n * discard_fraction)
    if n_skip > 0:
        ax.axvspan(0.5, n_skip + 0.5, color="0.85", alpha=0.6, lw=0,
                   label="discarded transient")

    # running MFPT over the retained tail, as data accumulate
    ax2 = ax.twinx()
    running = [
        steady_state_mfpt(flux[: k + 1], tau_ps, discard_fraction).mfpt_ns
        for k in range(n)
    ]
    ax2.plot(it, running, color="#26456E", lw=1.6, label="MFPT estimate")
    final = steady_state_mfpt(flux, tau_ps, discard_fraction)
    if final.mfpt_ns is not None:
        ax2.axhline(final.mfpt_ns, color="#26456E", lw=0.8, ls=":")
        ax2.annotate(f"{final.mfpt_ns:.3g} ns", (n, final.mfpt_ns),
                     color="#26456E", fontsize=9, va="bottom", ha="right")
    ax2.set_ylabel("MFPT estimate (ns)", color="#26456E")
    ax2.tick_params(axis="y", labelcolor="#26456E")
    ax.set_title("Weighted-ensemble kinetics: flux & running MFPT")
    return ax


def save_flux_convergence(flux_history, tau_ps, outfile, discard_fraction=0.5):
    """Render :func:`plot_flux_convergence` to ``outfile`` and return its path."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_flux_convergence(flux_history, tau_ps, discard_fraction, ax=ax)
    fig.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
