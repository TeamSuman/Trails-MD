"""Entry point executed by a single scheduler array task.

Usage::

    python -m trails_md.execution.run_task <task.pkl> <result.json>

Loads a pickled :class:`~trails_md.execution.base.WalkerTask`, runs it, and
writes a JSON result marker. The marker is the source of truth for completion
and success, so it is written even when the run fails — letting the submitting
process distinguish "failed" from "never ran" (e.g. node crash).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: run_task <task.pkl> <result.json>", file=sys.stderr)
        return 2
    task_path, result_path = argv

    result: dict = {"success": False, "error": None}
    try:
        with open(task_path, "rb") as handle:
            task = pickle.load(handle)
        result["index"] = getattr(task, "index", None)
        from .base import run_walker_task

        result["success"] = bool(run_walker_task(task))
    except Exception as exc:  # noqa: BLE001 - report any failure via the marker
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    tmp = f"{result_path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(result, handle)
    os.replace(tmp, result_path)  # atomic publish
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
