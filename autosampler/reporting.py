"""Terminal presentation for per-iteration progress.

Keeps the ANSI / tabular UI out of :class:`~autosampler.core.AutoSamplerCore`
so the orchestrator focuses on the sampling logic. Pure formatting: easy to
unit-test and to swap for a richer reporter later.
"""

from __future__ import annotations

from dataclasses import dataclass

_CYAN = "\033[96m"
_RED = "\033[91m"
_END = "\033[0m"


@dataclass
class IterationReporter:
    """Render the boxed per-iteration summary banner."""

    width: int = 85

    def format_summary(
        self,
        iteration: int,
        runner_time: float,
        other_time: float,
        occupancy: str,
        color: bool = True,
    ) -> str:
        raw = (
            f" Iteration: {iteration:<4} | Runner: {runner_time:<6.2f}s | "
            f"Other: {other_time:<5.2f}s | Occupancy: {occupancy:<9}"
        )
        if color:
            body = (
                f" Iteration: {_CYAN}{iteration:<4}{_END} | "
                f"Runner: {_CYAN}{runner_time:<6.2f}s{_END} | "
                f"Other: {_CYAN}{other_time:<5.2f}s{_END} | "
                f"Occupancy: {_CYAN}{occupancy:<9}{_END}"
            )
            red, end = _RED, _END
        else:
            body = raw
            red = end = ""

        left = (self.width - len(raw)) // 2
        right = self.width - len(raw) - left
        top = f"\t{red}╔" + "═" * self.width + f"╗\n{end}"
        middle = (
            f"\t{red}║{end}" + " " * left + body + " " * right + f"{red}║\n{end}"
        )
        bottom = f"\t{red}╚" + "═" * self.width + f"╝{end}"
        return top + middle + bottom

    def print_summary(self, *args, **kwargs) -> None:
        print(self.format_summary(*args, **kwargs))
