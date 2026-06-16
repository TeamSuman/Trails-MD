"""Terminal run banner for AutoSampler CLI runs."""

from __future__ import annotations

from pathlib import Path
from shutil import get_terminal_size
from typing import Any


class _Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class _NoColor:
    PURPLE = ""
    CYAN = ""
    GREEN = ""
    YELLOW = ""
    RED = ""
    BOLD = ""
    UNDERLINE = ""
    END = ""

_LOGO_RAW = r"""
{c1}   ___       __      ___               __{rst}
{c2}  / _ |__ __/ /____ / __/___ ___ _ ___/ /___ ____{rst}
{c3} / __ / // / __/ _ \\ \/ _ `/  ' \/ _ \/ -_) __/{rst}
{c4}/_/ |_\_,_/\__/\___/___/\_,_/_/_/_/ .__/\__/_/{rst}
{c5}                                 /_/{rst}
{sub}       Adaptive Molecular Dynamics Exploration{rst}
          {s1}explore{rst} {arr}{arrow}{rst} {s2}score{rst} {arr}{arrow}{rst} {s3}spawn{rst} {arr}{arrow}{rst} {s4}converge{rst}
""".strip("\n")

_LOGO_COLOR = _LOGO_RAW.format(
    # Logo Gradient (Cyan to Deep Blue)
    c1="\033[38;5;51m",
    c2="\033[38;5;39m",
    c3="\033[38;5;33m",
    c4="\033[38;5;27m",
    c5="\033[38;5;21m",

    # Subtitle (Gold/Orange)
    sub="\033[38;5;214m",

    # Loop Steps (Green -> Yellow -> Orange -> Red)
    s1="\033[38;5;46m",
    s2="\033[38;5;226m",
    s3="\033[38;5;208m",
    s4="\033[38;5;196m",

    # Arrows (Dark Gray)
    arr="\033[38;5;240m",
    arrow="➔",

    # Reset
    rst="\033[0m"
)
_LOGO_PLAIN = _LOGO_RAW.format(
    c1="",
    c2="",
    c3="",
    c4="",
    c5="",
    sub="",
    s1="",
    s2="",
    s3="",
    s4="",
    arr="",
    arrow="->",
    rst="",
)

def print_run_banner(
    config: dict[str, Any],
    *,
    config_path: Path,
    iterations: int,
    color: bool = True,
) -> str:
    """Print and return a formatted AutoSampler run summary."""
    text = format_run_banner(
        config,
        config_path=config_path,
        iterations=iterations,
        color=color,
    )
    print(text)
    return text


def format_run_banner(
    config: dict[str, Any],
    *,
    config_path: Path,
    iterations: int,
    color: bool = True,
) -> str:
    colors = _Color if color else _NoColor
    width = min(max(get_terminal_size((100, 24)).columns - 4, 72), 100)
    inner_width = width - 4
    rows = _run_rows(config, config_path=config_path, iterations=iterations)
    logo = _LOGO_COLOR if color else _LOGO_PLAIN

    lines: list[str] = []
    lines.append(colors.PURPLE + colors.BOLD + _center_logo(logo, width) + colors.END)
    lines.append(
        "\t"
        + colors.GREEN
        + colors.BOLD
        + "Adaptive molecular sampling run".center(width)
        + colors.END
    )
    lines.append("")
    lines.append("\t" + colors.RED + "╔" + "═" * (width - 2) + "╗" + colors.END)
    title = " Run Information "
    lines.append(
        "\t"
        + colors.RED
        + "║"
        + colors.CYAN
        + colors.BOLD
        + title.center(width - 2)
        + colors.RED
        + "║"
        + colors.END
    )
    lines.append("\t" + colors.RED + "╠" + "═" * (width - 2) + "╣" + colors.END)

    for label, value in rows:
        if label == "__section__":
            lines.append(_format_section_row(str(value), width, colors))
            continue
        lines.append(_format_value_row(label, value, inner_width, colors))

    lines.append("\t" + colors.RED + "╚" + "═" * (width - 2) + "╝" + colors.END)
    return "\n".join(lines)


def _run_rows(
    config: dict[str, Any],
    *,
    config_path: Path,
    iterations: int,
) -> list[tuple[str, str]]:
    system = config.get("system", {})
    engine = config.get("engine", {})
    spawning = config.get("spawning", {})
    md_engine = str(engine.get("md_engine", "openmm"))

    rows = [
        ("__section__", "Sampling"),
        ("iterations", str(iterations)),
        ("engine", md_engine),
        ("space mode", str(config.get("space_mode", "fixed"))),
        ("spawner", str(spawning.get("spawn_scheme", "density"))),
        ("walkers", str(spawning.get("walker", 10))),
        ("steps / stride", f"{spawning.get('step', 10000)} / {spawning.get('stride', 100)}"),
        ("max workers", str(spawning.get("max_workers", 4))),
        ("convergence patience", str(spawning.get("convergence_patience", 0))),
        ("bins", str(config.get("n_bins", spawning.get("n_bins", [30, 30])))),
        ("__section__", "Files"),
        ("config", str(config_path)),
        ("output directory", str(config.get("outdir", "runs/sampler_output"))),
        ("topology", str(system.get("top_file", ""))),
        ("coordinates", str(system.get("conf_file", ""))),
    ]

    if md_engine == "amber":
        rows.append(("__section__", "Amber Engine"))
        rows.extend(
            [
                ("amber executable", str(engine.get("amber_executable", "pmemd"))),
                (
                    "trajectory format",
                    _amber_format_summary(
                        str(engine.get("amber_trajectory_format", "auto")),
                        str(engine.get("amber_executable", "pmemd")),
                    ),
                ),
            ]
        )
    elif md_engine == "gromacs":
        rows.append(("__section__", "GROMACS Engine"))
        rows.append(
            ("gromacs executable", str(engine.get("gromacs_executable", "gmx")))
        )
    elif md_engine == "openmm":
        rows.append(("__section__", "OpenMM Engine"))
        rows.append(("platform", str(engine.get("platform_name", "CUDA"))))

    if engine.get("gpu_ids") is not None:
        rows.append(("gpu ids", str(engine.get("gpu_ids"))))

    return rows


def _amber_format_summary(format_name: str, executable: str) -> str:
    try:
        from autosampler.engines.amber import (
            amber_trajectory_suffix,
            resolve_amber_trajectory_format,
        )

        resolved = resolve_amber_trajectory_format(format_name, executable)
        suffix = amber_trajectory_suffix(format_name, executable)
        if format_name == resolved:
            return f"{resolved} (.{suffix})"
        return f"{format_name} -> {resolved} (.{suffix})"
    except Exception:
        return format_name


def _center_logo(logo: str, width: int) -> str:
    lines = [line.rstrip() for line in logo.splitlines()]
    logo_width = max((len(line) for line in lines), default=0)
    left_pad = max((width - logo_width) // 2, 0)
    prefix = "\t" + (" " * left_pad)
    return "\n".join(prefix + line for line in lines)


def _format_section_row(label: str, width: int, colors: type[_Color]) -> str:
    body_width = width - 2
    title = f" {label.upper()} "
    left = max((body_width - len(title)) // 2, 0)
    right = max(body_width - len(title) - left, 0)
    return (
        "\t"
        + colors.RED
        + "╟"
        + "─" * left
        + colors.YELLOW
        + colors.BOLD
        + title
        + colors.RED
        + "─" * right
        + "╢"
        + colors.END
    )


def _format_value_row(
    label: str,
    value: str,
    inner_width: int,
    colors: type[_Color],
) -> str:
    left = f" {label}:"
    right = _fit_value(value, inner_width - len(left) - 1)
    padding = " " * max(inner_width - len(left) - len(right), 1)
    return (
        "\t"
        + colors.RED
        + "║ "
        + colors.GREEN
        + colors.BOLD
        + left
        + colors.CYAN
        + padding
        + right
        + " "
        + colors.RED
        + "║"
        + colors.END
    )


def _fit_value(value: str, width: int) -> str:
    if width <= 0:
        return ""
    value = str(value)
    if len(value) <= width:
        return value
    if width <= 6:
        return value[:width]
    left = max((width - 3) // 2, 1)
    right = max(width - 3 - left, 1)
    return f"{value[:left]}...{value[-right:]}"
