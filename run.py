#!/usr/bin/env python
"""Compatibility wrapper for the AutoSampler command line interface."""

from autosampler.cli import load_config, main, parse_args, resolve_config_paths, run


if __name__ == "__main__":
    main()
