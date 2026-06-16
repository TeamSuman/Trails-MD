"""Tests for the input-file template and the autosampler-init CLI."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

warnings.filterwarnings("ignore")

from autosampler.config import AutoSamplerConfig  # noqa: E402
from autosampler.templates import DEFAULT_TEMPLATE  # noqa: E402


def test_template_parses_into_config():
    cfg = AutoSamplerConfig(**yaml.safe_load(DEFAULT_TEMPLATE))
    # Spot-check a value from each major block.
    assert cfg.system.conf_file
    assert cfg.engine.md_engine in {"openmm", "gromacs", "amber"}
    assert cfg.spawning.spawn_scheme in {
        "density", "voronoi", "lof", "fps", "msm", "we"
    }
    assert cfg.execution.backend == "local"
    assert cfg.msm.enabled is False  # advanced blocks opt-in by default


def test_example_template_matches_module():
    # The committed examples/template.yaml must equal the packaged template.
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "examples" / "template.yaml").read_text()
    assert text == DEFAULT_TEMPLATE


def test_init_cli_writes_and_guards(tmp_path):
    from autosampler.init_cli import main

    out = tmp_path / "config.yaml"
    main(["-o", str(out)])
    assert out.exists()
    assert AutoSamplerConfig(**yaml.safe_load(out.read_text())) is not None

    # Refuses to overwrite without --force, succeeds with it.
    with pytest.raises(SystemExit):
        main(["-o", str(out)])
    main(["-o", str(out), "--force"])
