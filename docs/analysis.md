# MSM analysis & plotting

When `msm.enabled` is set, each iteration writes an `iter_*/msm.npz` with the
MSM diagnostics (timescales, VAMP-2, stationary distribution, transition matrix,
metastable populations, and the implied-timescale sweep). Trails-MD ships
utilities to turn these into figures.

## One-command report

```bash
trails-md-analyze --run-dir runs/adaptive_msm_vampnet
# -> runs/adaptive_msm_vampnet/analysis/convergence_report.png
```

The report is a 2x2 panel: VAMP-2 convergence, slowest-timescale convergence,
the CV free-energy surface, and the latest implied-timescale sweep (or MSM
network). Options: `--outfile`, `--temperature` (for free energies in kJ/mol).

## Programmatic API

Data utilities (no matplotlib required):

```python
from trails_md.analysis import data

series = data.load_msm_series("runs/my_run")     # iterations, vamp2, timescales
latest = data.load_latest_msm("runs/my_run")     # arrays of the last msm.npz
points = data.load_cv_points("runs/my_run")      # all CV projections stacked

# Free energies (kJ/mol, min shifted to 0):
F = data.free_energy_from_populations(latest["metastable_populations"])
Fxy, xe, ye = data.free_energy_surface(points, bins=60, temperature=300.0)
```

Plotting (needs `pip install "trails-md[examples]"`); each function takes an
optional `ax` and returns it:

```python
from trails_md.analysis import plots

plots.plot_vamp2_convergence(series)
plots.plot_timescale_convergence(series)
plots.plot_implied_timescales(latest["its_lagtimes"], latest["its_timescales"])
plots.plot_free_energy_surface(points)
plots.plot_metastable_free_energy(latest["metastable_populations"])
plots.plot_msm_network(latest["transition_matrix"], latest["stationary_distribution"])

# Or the full multi-panel report:
plots.plot_convergence_report("runs/my_run", outfile="report.png")
```

## What to look for

- **Implied timescales** should plateau (become flat in lag time) — the signal
  that the MSM is Markovian at the chosen lag.
- **VAMP-2 / timescale convergence** flattening across iterations indicates the
  sampling (and the MSM) have converged — the same signals the
  [ConvergenceMonitor](msm.md) uses to stop automatically.
- The **free-energy surface** reveals basins and barriers in the CV space; the
  **MSM network** summarises metastable states and their connectivity.

## Weighted-ensemble kinetics (rate / MFPT)

If you ran in [kinetics mode](modes.md) (`spawn_scheme: we` with `inherit_velocities:
true` and a `recycle_target`), the rate is read from the recycled-flux series, not from
an MSM. `trails-md-analyze` reports it and writes a convergence plot:

```bash
trails-md-analyze --run-dir runs/my_kinetics_run
# -> MFPT estimate, tau, plateau ratio, converged/not, and
#    runs/my_kinetics_run/analysis/flux_convergence.png
```

τ (`step * dt`) is auto-detected from the run log; use `--tau-ps` / `--config` if
needed and `--discard-fraction` to change the transient cut. The **status / plateau
ratio** is the key check — a still-decaying or still-rising flux means the steady state
has not been reached and the run needs more iterations. (If a run has both a recycled
flux series and `msm.npz`, both reports are produced.)

Programmatic equivalents:

```python
from trails_md.analysis.data import load_flux_history
from trails_md.spawners.we import steady_state_mfpt

flux = load_flux_history("runs/my_kinetics_run")
r = steady_state_mfpt(flux, tau_ps=step * dt)     # r.mfpt_ns, r.plateau_ratio, r.converged
```
