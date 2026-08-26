from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import load_settings
from .models import load_model_lock


def merge_selected_lora(
    selection_path: str | Path,
    base_config_path: str | Path,
) -> Path:
    selection_path = Path(selection_path).expanduser().resolve()
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or selection.get("source_kind") != "lora_learning_curve":
        raise ValueError("Expected lora_selection.yaml from the learning-curve experiment")
    settings = load_settings(base_config_path)
    try:
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError("Install the train dependency extra before merging LoRA") from exc
    repo_id = str(selection["repo_id"])
    lock = load_model_lock(settings)
    locked = lock.get("models", {}).get(repo_id)
    if not isinstance(locked, dict):
        raise ValueError(f"Training model is not locked: {repo_id}")
    revision = str(locked["resolved_revision"])
    output_dir = selection_path.parent / "merged_model" / str(selection["model_id"])
    weight_files = [*output_dir.glob("*.safetensors"), *output_dir.glob("*.bin")]
    if (
        (output_dir / "config.json").exists()
        and weight_files
        and all(path.stat().st_size > 0 for path in weight_files)
    ):
        return output_dir
    print(f"[lora-merge] loading {repo_id}@{revision}", flush=True)
    processor = WhisperProcessor.from_pretrained(
        repo_id, revision=revision, language=settings.model.language, task="transcribe"
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(selection["adapter_dir"]))
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    metadata: dict[str, Any] = {
        "source_selection": str(selection_path),
        "base_repo_id": repo_id,
        "base_revision": revision,
        "adapter_dir": str(selection["adapter_dir"]),
        "merged_model_dir": str(output_dir),
        "test_evaluated": False,
    }
    (output_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selection["merged_model_dir"] = str(output_dir)
    selection["base_revision"] = revision
    selection_path.write_text(
        yaml.safe_dump(selection, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return output_dir
