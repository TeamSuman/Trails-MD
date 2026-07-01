#!/usr/bin/env python
"""Compatibility wrapper for the Trails-MD command line interface."""

from trails_md.cli import load_config, main, parse_args, resolve_config_paths, run


if __name__ == "__main__":
    main()
