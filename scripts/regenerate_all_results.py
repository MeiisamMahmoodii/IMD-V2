from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

_LOG_PATH = Path("debug-fea841.log")


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "fea841",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _resolve_main(module_name: str, run_id: str):
    try:
        # region agent log
        _dbg(
            run_id,
            "H4",
            "scripts/regenerate_all_results.py:_resolve_main:import_abs",
            "import module absolute",
            {"module": module_name},
        )
        # endregion
        return importlib.import_module(module_name).main
    except Exception as exc:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        simple_name = module_name.split(".")[-1]
        # region agent log
        _dbg(
            run_id,
            "H5",
            "scripts/regenerate_all_results.py:_resolve_main:fallback",
            "absolute import failed, fallback via repo root + simple module",
            {
                "module": module_name,
                "simple_name": simple_name,
                "repo_root_added": str(repo_root),
                "error": repr(exc),
            },
        )
        # endregion
        return importlib.import_module(simple_name).main


def main():
    run_id = "pre-fix"
    # region agent log
    _dbg(
        run_id,
        "H1",
        "scripts/regenerate_all_results.py:main:start",
        "process context",
        {
            "cwd": os.getcwd(),
            "argv0": sys.argv[0],
            "package": __package__,
            "path_head": sys.path[:5],
        },
    )
    # endregion

    try:
        setup = _resolve_main("scripts.setup_datasets", run_id)
        run_experiments = _resolve_main("scripts.run_experiments", run_id)
        stats = _resolve_main("scripts.compute_statistics", run_id)
    except Exception as exc:
        # region agent log
        _dbg(
            run_id,
            "H2",
            "scripts/regenerate_all_results.py:main:import_fail",
            "import failed after fallback",
            {"error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]},
        )
        # endregion
        raise

    # region agent log
    _dbg(
        run_id,
        "H3",
        "scripts/regenerate_all_results.py:main:before_run",
        "imports resolved",
        {"resolved": ["setup", "run_experiments", "stats"]},
    )
    # endregion
    setup()
    run_experiments()
    stats()

if __name__ == "__main__":
    main()
