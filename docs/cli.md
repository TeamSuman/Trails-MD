# CLI reference

Trails-MD installs six console commands (see `pyproject.toml [project.scripts]`).

## `trails-md` (alias `trails-md-run`)

Run the adaptive sampling loop from a config file.

```bash
trails-md --config CONFIG.yaml [--iterations N] [--resume latest|N]
          [--ignore-missing-history] [--check] [--log-level LEVEL]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | `config.yaml` | Path to the YAML input file (uses `config.yaml` in the current directory if omitted). |
| `--iterations` | `1` | Number of iterations to run (`>= 0`). |
| `--resume` | — | Resume from the latest checkpoint (`--resume`, i.e. `latest`) or a specific iteration (`--resume N`). |
| `--ignore-missing-history` | off | On resume, tolerate missing/unreadable history deltas instead of failing (reconstructs a partial history). |
| `--check` | off | Preflight only: validate inputs/engine/settings, then exit without running MD. |
| `--log-level` | `WARNING` | Python logging level (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`). |

## `trails-md-init`

Write a fully-annotated starter input file.

```bash
trails-md-init [-o OUTPUT] [--force]
```

| Flag | Default | Description |
| --- | --- | --- |
| `-o`, `--output` | `config.yaml` | Where to write the template. |
| `--force` | off | Overwrite an existing file. |

## `trails-md-analyze`

Report whichever analysis the run supports: the **weighted-ensemble MFPT** (a
[kinetics-mode](modes.md) `recycle_target` run) and/or the **MSM convergence report**
(an `msm.enabled` run). The MFPT report prints the rate + a convergence diagnostic and
writes `analysis/flux_convergence.png`; the MSM report writes the multi-panel figure.

```bash
trails-md-analyze --run-dir RUN_DIR [--outfile FILE] [--temperature K] \
                  [--config CONFIG] [--tau-ps PS] [--discard-fraction F]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--run-dir` | *(required)* | Run output directory. |
| `--outfile` | `<run-dir>/analysis/convergence_report.png` | Output path for the **MSM** report. |
| `--temperature` | `300.0` | Temperature (K) for free-energy conversion (MSM report). |
| `--config` | `None` | Run config, used to recover τ = `step * dt` for the MFPT if the run log is unavailable. |
| `--tau-ps` | `None` | Walker segment length in ps (= `step * dt`); overrides auto-detection. |
| `--discard-fraction` | `0.5` | Leading fraction of the flux series dropped as pre-steady-state transient. |

## `trails-md-log`

Write/extend an exploration log (per-iteration CV-bin occupancy) for a run.

```bash
trails-md-log --run-dir RUN_DIR [--config CONFIG] [--output FILE] \
    [--n-bins ...] [--min-values ...] [--max-values ...] [--append]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--run-dir` | *(required)* | Run output directory. |
| `--config` | — | Config to read binning bounds from. |
| `--output` | — | Output log path. |
| `--n-bins` / `--min-values` / `--max-values` | — | Override the CV grid (comma-separated lists). |
| `--append` | off | Append to an existing log. |

## `trails-md-path`

Reconstruct a connected trajectory between two CV points from a run's frame
lineage.

Single pair:

```bash
trails-md-path --run-dir RUN_DIR --topology TOP --start "x,y" --end "x,y" [--output OUT]
```

Batch (many endpoint pairs from a file):

```bash
trails-md-path --run-dir RUN_DIR --topology TOP --pairs-file PAIRS.json --output-dir DIR
```

| Flag | Default | Description |
| --- | --- | --- |
| `--run-dir` | *(required)* | Run output directory. |
| `--topology` | *(required)* | Topology for writing the connected trajectory. |
| `--start` / `--end` | — | CV coordinates (comma-separated) of the path endpoints (single-pair mode). |
| `--output` | — | Output trajectory path (single-pair mode). |
| `--pairs-file` | — | JSON or CSV file of endpoint pairs for batch extraction (requires `--output-dir`). |
| `--output-dir` | — | Directory for batch outputs when `--pairs-file` is used. |
| `--metadata` | `<output>.json` | Output path for the JSON lineage metadata (defaults to the `--output` path with `.json` appended, e.g. `path.xtc` → `path.xtc.json`). |
| `--checkpoint` | latest | Checkpoint iteration to reconstruct history from. |
| `--ignore-missing-history` | off | Tolerate missing/unreadable history deltas during reconstruction. |
