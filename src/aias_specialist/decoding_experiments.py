from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from .config import load_settings
from .experiments import run_model_benchmark, select_experiment_member


def run_selected_lora_decoding_sweep(
    lora_selection_path: str | Path,
    base_config_path: str | Path,
) -> Path:
    selection_path = Path(lora_selection_path).expanduser().resolve()
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not selection.get("merged_model_dir"):
        raise ValueError("Run merge-selected-lora before the decoding sweep")
    settings = load_settings(base_config_path)
    selection_id = re.sub(r"[^0-9A-Za-z._-]", "-", str(selection["selection_id"]))
    benchmark_id = f"synthetic-lora-decoding-{selection_id}"
    input_dir = settings.paths.artifacts_dir.parent / "decoding_inputs" / benchmark_id
    input_dir.mkdir(parents=True, exist_ok=True)
    merged_model = str(selection["merged_model_dir"])
    cache_root = settings.paths.artifacts_dir.parent / "models/merged-lora-ct2" / selection_id
    prompt = settings.model.initial_prompt or ""
    hotwords = settings.model.hotwords or ""
    common = {
        "backend": "faster_whisper",
        "repo_id": merged_model,
        "revision": str(selection.get("base_revision", "local")),
        "format": "transformers",
        "conversion_quantization": "float16",
        "device": "cuda",
        "compute_type": "float16",
    }
    candidates = [
        {
            **common,
            "id": "beam1-no-context",
            "local_dir": str(cache_root / "beam1-no-context"),
            "beam_size": 1,
            "initial_prompt": "",
            "hotwords": "",
            "vad_filter": True,
            "vad_min_silence_duration_ms": 500,
        },
        {
            **common,
            "id": "beam5-prompt",
            "local_dir": str(cache_root / "beam5-prompt"),
            "beam_size": 5,
            "initial_prompt": prompt,
            "hotwords": "",
            "vad_filter": True,
            "vad_min_silence_duration_ms": 500,
        },
        {
            **common,
            "id": "beam5-hotwords",
            "local_dir": str(cache_root / "beam5-hotwords"),
            "beam_size": 5,
            "initial_prompt": prompt,
            "hotwords": hotwords,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 500,
        },
        {
            **common,
            "id": "beam8-hotwords-vad300",
            "local_dir": str(cache_root / "beam8-hotwords-vad300"),
            "beam_size": 8,
            "initial_prompt": prompt,
            "hotwords": hotwords,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 300,
        },
        {
            **common,
            "id": "beam5-hotwords-vad700",
            "local_dir": str(cache_root / "beam5-hotwords-vad700"),
            "beam_size": 5,
            "initial_prompt": prompt,
            "hotwords": hotwords,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 700,
        },
    ]
    matrix = {
        "benchmark": {
            "id": benchmark_id,
            "name": "선정 LoRA 모델 Validation 디코딩·Hotword·VAD 탐색",
            "base_config": str(Path(base_config_path).expanduser().resolve()),
            "evaluation_split": "validation",
            "isolated_process": True,
            "continue_on_error": False,
            "models": candidates,
        }
    }
    matrix_path = input_dir / "decoding_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(matrix, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    result = run_model_benchmark(matrix_path)
    comparison = pd.read_csv(result.comparison_path, keep_default_na=False)
    completed = comparison.loc[comparison["status"].eq("completed")].copy()
    if completed.empty:
        raise RuntimeError("No decoding candidate completed")
    completed["rank"] = pd.to_numeric(completed["rank"], errors="coerce")
    best = completed.sort_values("rank").iloc[0]
    return select_experiment_member(
        result.group_dir,
        str(best["member_id"]),
        reviewer="AUTOMATED_SYNTHETIC_VALIDATION",
        reason=(
            "Validation-only rank 1 by manufacturing-term recall, CER, WER, and RTF. "
            "Held-out Test was not inspected."
        ),
        human_reviewed=False,
    )
