from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings
from .training import train_whisper_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated Whisper LoRA experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run_dir = train_whisper_lora(load_settings(args.config))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "adapter_dir": metrics["training"]["adapter_dir"],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
