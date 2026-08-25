from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_settings
from .models import resolve_model_revisions
from .utils import utc_now


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = payload.get("lora_learning_curve") if isinstance(payload, dict) else None
    if not isinstance(section, dict):
        raise ValueError("LoRA spec requires a lora_learning_curve mapping")
    return section


def _root(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_a100(section: dict[str, Any]) -> None:
    if not bool(section.get("require_a100", False)):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("A100 validation requires the train dependency extra") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; select an A100 Colab runtime")
    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    minimum = float(section.get("minimum_gpu_memory_gb", 35))
    if "A100" not in name.upper() or memory_gb < minimum:
        raise RuntimeError(
            f"This experiment requires A100 with at least {minimum:g} GB; "
            f"allocated device is {name} ({memory_gb:.1f} GB)"
        )
    print(f"[lora-curve] hardware={name}; memory={memory_gb:.1f} GB", flush=True)


def _absolute_paths(
    raw: dict[str, Any], settings: Any, manifest_override: Path | None = None
) -> None:
    raw["paths"] = {
        "manifest": str(manifest_override or settings.paths.manifest),
        "domain_terms": str(settings.paths.domain_terms),
        "artifacts_dir": str(settings.paths.artifacts_dir),
        "database": str(settings.paths.database),
        "model_lock": str(settings.paths.model_lock),
    }


def _write_run_config(
    *,
    raw_base: dict[str, Any],
    settings: Any,
    model: dict[str, Any],
    stage: dict[str, Any],
    group_dir: Path,
    manifest_override: Path | None = None,
) -> Path:
    raw = deepcopy(raw_base)
    _absolute_paths(raw, settings, manifest_override)
    training = raw.setdefault("training", {})
    training.update(
        {
            "enabled": True,
            "repo_id": str(model["repo_id"]),
            "revision": str(model.get("revision", "main")),
            "model_id": str(model["id"]),
            "evaluation_split": "validation",
            "train_sample_limit": int(stage["train_samples"]),
            "max_steps": int(stage["max_steps"]),
            "warmup_steps": max(1, int(stage["max_steps"]) // 10),
            "eval_steps": max(
                1,
                int(stage.get("eval_steps", int(stage["max_steps"]) // 3)),
            ),
            "save_steps": max(
                1,
                int(stage.get("save_steps", int(stage["max_steps"]) // 3)),
            ),
            "resume_from_checkpoint": False,
            "baseline_predictions_cache": str(
                group_dir / "baseline-cache" / f"{model['id']}__validation.csv"
            ),
            "output_dir": str(
                settings.paths.artifacts_dir.parent
                / "checkpoints/lora-learning-curve"
                / str(model["id"])
                / str(stage["id"])
            ),
        }
    )
    raw["evaluation"] = {**raw.get("evaluation", {}), "split": "validation"}
    raw.setdefault("report", {})["title"] = (
        f"{model['id']} LoRA - {stage['id']} Validation"
    )
    config_path = group_dir / "configs" / f"{stage['id']}__{model['id']}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path


def _stage_learning_curve_manifest(
    section: dict[str, Any],
    settings: Any,
    stages: list[dict[str, Any]],
) -> Path | None:
    """Copy the active learning-curve subset to local SSD and return its manifest."""
    staging = section.get("local_audio_staging", {})
    if not isinstance(staging, dict) or not bool(staging.get("enabled", False)):
        return None
    cache_dir = Path(str(staging["cache_dir"])).expanduser().resolve()
    workers = int(staging.get("copy_workers", 8))
    if workers < 1:
        raise ValueError("local_audio_staging.copy_workers must be at least 1")
    manifest = pd.read_csv(settings.paths.manifest, dtype=str).fillna("")
    max_train_samples = max(int(stage["train_samples"]) for stage in stages)
    split = manifest["split"].str.lower()
    staged = pd.concat(
        [
            manifest.loc[split.eq("train")].head(max_train_samples),
            manifest.loc[split.eq("validation")],
        ],
        ignore_index=True,
    )
    if len(staged) == 0:
        raise ValueError("Local audio staging selected no train or validation samples")
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    cached_paths: list[str] = []
    for row in staged.itertuples(index=False):
        source = Path(str(row.audio_path)).expanduser()
        if not source.is_absolute():
            source = (settings.paths.manifest.parent / source).resolve()
        digest = sha256(str(source).encode("utf-8")).hexdigest()[:16]
        destination = cache_dir / "audio" / str(row.split).lower() / f"{digest}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((source, destination))
        cached_paths.append(str(destination))

    def copy_one(job: tuple[Path, Path]) -> None:
        source, destination = job
        if not source.is_file():
            raise FileNotFoundError(f"Staging source is missing: {source}")
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        partial = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, partial)
        os.replace(partial, destination)

    print(
        f"[lora-stage] copying/reusing {len(jobs)} files to {cache_dir}; workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 250 == 0 or completed == len(futures):
                print(f"[lora-stage] completed={completed}/{len(futures)}", flush=True)
    staged["audio_path"] = cached_paths
    manifest_path = cache_dir / "manifest.csv"
    staged.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rank"] = pd.NA
    for _, indexes in ranked.groupby("stage_id").groups.items():
        completed = ranked.loc[indexes].loc[ranked.loc[indexes, "status"].eq("completed")]
        order = completed.sort_values(
            ["domain_term_recall", "cer", "wer"], ascending=[False, True, True]
        )
        for rank, index in enumerate(order.index, start=1):
            ranked.loc[index, "rank"] = rank
    return ranked.sort_values(["stage_order", "rank", "model_id"], na_position="last")


def _stages_through(stages: list[dict[str, Any]], max_stage: str | None) -> list[dict[str, Any]]:
    """Return the cumulative learning-curve stages up to the requested checkpoint."""
    if max_stage is None:
        return stages
    stage_ids = [str(stage["id"]) for stage in stages]
    if max_stage not in stage_ids:
        raise ValueError(f"Unknown max_stage {max_stage!r}; choose one of {stage_ids}")
    return stages[: stage_ids.index(max_stage) + 1]


def run_lora_learning_curve(
    spec_path: str | Path,
    *,
    max_stage: str | None = None,
) -> Path:
    path = Path(spec_path).expanduser().resolve()
    section = _load_spec(path)
    root = _root(path)
    base_path = _resolve(root, str(section["base_config"]))
    settings = load_settings(base_path)
    _require_a100(section)
    models = section.get("models")
    stages = section.get("stages")
    if not isinstance(models, list) or not models or not isinstance(stages, list) or not stages:
        raise ValueError("LoRA learning curve requires non-empty models and stages")
    stages = _stages_through(stages, max_stage)
    staged_manifest = _stage_learning_curve_manifest(section, settings, stages)
    resolve_model_revisions(
        settings.paths.model_lock,
        [
            {
                "repo_id": str(model["repo_id"]),
                "revision": str(model.get("revision", "main")),
                "roles": ["lora_learning_curve"],
            }
            for model in models
        ],
    )
    group_id = str(section["id"])
    group_dir = settings.paths.artifacts_dir.parent / "training" / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    snapshot = group_dir / "lora_learning_curve.snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    result_dir = group_dir / "workers"
    rows: list[dict[str, Any]] = []
    for stage_order, stage in enumerate(stages, start=1):
        for model in models:
            member_id = f"{stage['id']}__{model['id']}"
            result_path = result_dir / f"{member_id}.json"
            if not result_path.exists():
                # Never overwrite the config belonging to an already completed member;
                # its run snapshot and worker result must remain an immutable pair.
                config_path = _write_run_config(
                    raw_base=settings.raw,
                    settings=settings,
                    model=model,
                    stage=stage,
                    group_dir=group_dir,
                    manifest_override=staged_manifest,
                )
                print(
                    f"[lora-curve] stage={stage['id']} model={model['id']} starting",
                    flush=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "aias_specialist.training_worker",
                        "--config",
                        str(config_path),
                        "--result",
                        str(result_path),
                    ],
                    check=True,
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = result["metrics"]["lora"]
            rows.append(
                {
                    "member_id": member_id,
                    "stage_id": str(stage["id"]),
                    "stage_order": stage_order,
                    "train_samples": int(stage["train_samples"]),
                    "max_steps": int(stage["max_steps"]),
                    "model_id": str(model["id"]),
                    "repo_id": str(model["repo_id"]),
                    "status": "completed",
                    "run_dir": str(result["run_dir"]),
                    "adapter_dir": str(result["adapter_dir"]),
                    "domain_term_recall": float(metrics["domain_term_recall"]),
                    "cer": float(metrics["cer"]),
                    "wer": float(metrics["wer"]),
                    "evaluation_split": "validation",
                }
            )
    comparison = _rank(pd.DataFrame(rows))
    comparison_path = group_dir / "lora_learning_curve.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    final_stage = max(int(value) for value in comparison["stage_order"])
    selected = comparison.loc[
        comparison["stage_order"].eq(final_stage) & comparison["status"].eq("completed")
    ].sort_values("rank").iloc[0]
    selection = {
        "selection_id": f"lora-selection-{utc_now().strftime('%Y%m%dT%H%M%SZ')}",
        "source_kind": "lora_learning_curve",
        "group_id": group_id,
        "evaluation_split": "validation",
        "test_evaluated": False,
        "model_id": str(selected["model_id"]),
        "repo_id": str(selected["repo_id"]),
        "adapter_dir": str(selected["adapter_dir"]),
        "run_dir": str(selected["run_dir"]),
        "validation_metrics": {
            "domain_term_recall": float(selected["domain_term_recall"]),
            "cer": float(selected["cer"]),
            "wer": float(selected["wer"]),
        },
        "next_step": (
            "Tune decoding and correction on validation, then run held-out test exactly once."
        ),
    }
    selection_path = group_dir / "lora_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump(selection, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return comparison_path
