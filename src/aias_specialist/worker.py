from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .pipeline import run_pipeline
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated ASR experiment worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    result = run_pipeline(load_settings(args.config))
    write_json(
        args.result,
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "report_path": str(result.report_path),
            "metrics": result.metrics,
        },
    )


if __name__ == "__main__":
    main()
