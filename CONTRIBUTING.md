# Contributing to AutoSampler

Thanks for your interest in improving AutoSampler. This guide covers the
development workflow, coding standards, and how to add new components.

## Development setup

```bash
conda env create -f env.yml
conda activate autosampler
pip install -e ".[all]"        # runtime + deep-tica + examples + test extras
pip install pre-commit && pre-commit install
```

For a lightweight setup that runs the MSM and CV-method tests without the heavy
MD backends (OpenMM, MDAnalysis):

```bash
pip install numpy scipy scikit-learn pydantic pyyaml deeptime pytest torch
```

## Running tests and linters

```bash
pytest -q                 # full test suite (MD-dependent tests self-skip)
ruff check .              # lint
ruff format .             # format (or `black .`)
```

CI runs the test suite on Python 3.10 and 3.11 (see `.github/workflows/ci.yml`).
Please add or update tests for any behaviour change; aim to keep new modules
covered.

## Coding standards

- Target Python 3.10+. Use type hints on public functions.
- Formatting and linting are enforced by ruff/black (line length 88).
- Keep presentation/IO out of core logic (see `autosampler/reporting.py`).
- Prefer the existing factory/registry patterns when adding components.

## Adding components

AutoSampler is built around small registries so new methods slot in cleanly:

- **MD engine** — subclass `MDEngine` and call `EngineFactory.register(...)`
  in `autosampler/engines/`.
- **Spawner** — subclass `Spawner` and call `SpawnerFactory.register(...)`
  in `autosampler/spawners/`.
- **CV method** — add a `CVMethod` to `autosampler/spaces/registry.py` and a
  branch in `AdaptiveSpaceModel.fit` / `.project`.
- **MSM convergence criterion** — subclass `ConvergenceCriterion` and register
  it in `autosampler/msm/convergence.py`.

## Commit / PR guidelines

- Write focused commits with descriptive messages.
- Ensure `pytest` and `ruff check` pass locally before opening a PR.
- Describe the scientific or engineering motivation in the PR description.
