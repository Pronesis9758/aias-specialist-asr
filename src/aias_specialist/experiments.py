from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .comparison_reporting import build_comparison_report, create_comparison_chart
from .config import QualityTargetsConfig, Settings, load_settings
from .models import resolve_model_revisions
from .pipeline import run_pipeline
from .quality_targets import quality_target_columns
from .store import ExperimentStore
from .training import train_whisper_lora
from .utils import new_run_id, utc_now, write_json

MODEL_KEYS = {
    "backend",
    "repo_id",
    "revision",
    "local_dir",
    "format",
    "conversion_quantization",
    "language",
    "device",
    "compute_type",
    "beam_size",
    "initial_prompt",
    "hotwords",
}


@dataclass(frozen=True)
class ExperimentGroupResult:
    group_id: str
    group_dir: Path
    comparison_path: Path
    report_path: Path
    status: str


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _project_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _section(spec: dict[str, Any], kind: str) -> dict[str, Any]:
    section = spec.get(kind)
    if not isinstance(section, dict):
        raise ValueError(f"Experiment spec requires a '{kind}' mapping")
    return section


def _require_identifier(value: Any, label: str) -> str:
    identifier = str(value or "").strip()
    if not identifier or any(character in identifier for character in '\\/:*?"<>|'):
        raise ValueError(f"{label} must be a non-empty filesystem-safe identifier")
    return identifier


def _base_settings(spec_path: Path, section: dict[str, Any]) -> Settings:
    root = _project_root(spec_path)
    base_config = _resolve_path(root, str(section["base_config"]))
    return load_settings(base_config)


def _snapshot_spec(spec: dict[str, Any], path: Path) -> None:
    canonical = yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
    if path.exists():
        existing_spec = _load_yaml(path)
        if _snapshot_identity(existing_spec) != _snapshot_identity(spec):
            raise ValueError(
                f"Experiment ID already exists with a different specification: {path.parent.name}. "
                "Use a new ID when models, variants, data, or settings change."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical, encoding="utf-8")


def _snapshot_identity(spec: dict[str, Any]) -> str:
    """Return the material experiment identity used for safe cache reuse.

    Quantization snapshots retain the first immutable model-selection record for
    provenance. Repeating the same decision creates a new selection event and
    therefore a new ``selection_id`` and ``selected_at`` timestamp, but neither
    value changes the model, data, or runtime settings being evaluated.
    """
    stable = deepcopy(spec)
    selected_model = stable.get("selected_model")
    if isinstance(selected_model, dict):
        selected_model.pop("selection_id", None)
        selected_model.pop("selected_at", None)
    return yaml.safe_dump(stable, allow_unicode=True, sort_keys=True)


def _absolute_base_config(raw: dict[str, Any], settings: Settings) -> dict[str, Any]:
    generated = deepcopy(raw)
    generated["paths"] = {
        "manifest": str(settings.paths.manifest),
        "domain_terms": str(settings.paths.domain_terms),
        "artifacts_dir": str(settings.paths.artifacts_dir),
        "database": str(settings.paths.database),
        "model_lock": str(settings.paths.model_lock),
    }
    return generated


def _candidate_model(
    candidate: dict[str, Any],
    settings: Settings,
    *,
    default_local_dir: Path,
) -> dict[str, Any]:
    model = {
        "backend": settings.model.backend,
        "repo_id": settings.model.repo_id,
        "revision": settings.model.revision,
        "local_dir": str(default_local_dir),
        "format": settings.model.format,
        "conversion_quantization": settings.model.conversion_quantization,
        "language": settings.model.language,
        "device": settings.model.device,
        "compute_type": settings.model.compute_type,
        "beam_size": settings.model.beam_size,
        "initial_prompt": settings.model.initial_prompt,
        "hotwords": settings.model.hotwords,
    }
    model.update({key: value for key, value in candidate.items() if key in MODEL_KEYS})
    local_dir = Path(str(model["local_dir"])).expanduser()
    if not local_dir.is_absolute():
        local_dir = settings.project_root / local_dir
    model["local_dir"] = str(local_dir.resolve())
    if not model.get("conversion_quantization"):
        model.pop("conversion_quantization", None)
    if not model.get("initial_prompt"):
        model.pop("initial_prompt", None)
    if not model.get("hotwords"):
        model.pop("hotwords", None)
    return model


def _write_member_config(
    *,
    settings: Settings,
    model: dict[str, Any],
    evaluation_split: str,
    title: str,
    path: Path,
) -> None:
    raw = _absolute_base_config(settings.raw, settings)
    raw["model"] = model
    raw["evaluation"] = {
        **raw.get("evaluation", {}),
        "split": evaluation_split,
    }
    raw["training"] = {"enabled": False, "reason": "comparison inference run"}
    raw.setdefault("report", {})["title"] = title
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _execute_config(config_path: Path, result_path: Path, isolated_process: bool) -> dict[str, Any]:
    if not isolated_process:
        result = run_pipeline(load_settings(config_path))
        return {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "report_path": str(result.report_path),
            "metrics": result.metrics,
        }

    if result_path.exists():
        result_path.unlink()
    command = [
        sys.executable,
        "-m",
        "aias_specialist.worker",
        "--config",
        str(config_path),
        "--result",
        str(result_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output_tail: list[str] = []
    if process.stdout is None:
        raise RuntimeError("Worker output stream is unavailable")
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    started = time.monotonic()
    while True:
        try:
            line = output_queue.get(timeout=30)
        except queue.Empty:
            elapsed = int(time.monotonic() - started)
            print(f"[worker] still running elapsed={elapsed}s", flush=True)
            continue
        if line is None:
            break
        print(line, end="", flush=True)
        output_tail.append(line)
        output_tail = output_tail[-50:]
    returncode = process.wait()
    if returncode != 0:
        detail = "".join(output_tail).strip()
        raise RuntimeError(detail or f"Worker exited with code {returncode}")
    if not result_path.exists():
        raise RuntimeError(f"Worker did not create a result file: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _prediction_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "predictions_baseline.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, nrows=1)
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {
        name: row[name]
        for name in [
            "model_revision",
            "runtime_device",
            "compute_type",
            "storage_quantization",
        ]
        if name in frame.columns
    }


def _completed_row(
    *,
    member: dict[str, Any],
    result: dict[str, Any],
    evaluation_split: str,
    quality_targets: QualityTargetsConfig,
) -> dict[str, Any]:
    metrics = result["metrics"]["baseline"]
    run_dir = Path(result["run_dir"])
    metadata = _prediction_metadata(run_dir)
    row = {
        "member_id": member["member_id"],
        "model_id": member["model_id"],
        "variant_id": member["variant_id"],
        "status": "completed",
        "run_id": result["run_id"],
        "run_dir": str(run_dir),
        "evaluation_split": evaluation_split,
        "repo_id": member["model"]["repo_id"],
        "model_revision": metadata.get("model_revision", ""),
        "runtime_device": metadata.get("runtime_device", member["model"].get("device", "")),
        "compute_type": metadata.get("compute_type", member["model"].get("compute_type", "")),
        "storage_quantization": metadata.get(
            "storage_quantization",
            member["model"].get("conversion_quantization", "source-default"),
        ),
        "sample_count": int(metrics["sample_count"]),
        "wer": float(metrics["wer"]),
        "cer": float(metrics["cer"]),
        "domain_term_recall": float(metrics["domain_term_recall"]),
        "domain_term_recall_applicable": bool(metrics["domain_term_recall_applicable"]),
        "mean_latency_seconds": float(metrics["mean_latency_seconds"]),
        "p95_latency_seconds": float(metrics["p95_latency_seconds"]),
        "aggregate_real_time_factor": float(metrics["aggregate_real_time_factor"]),
        "model_preparation_seconds": float(metrics["model_preparation_seconds"]),
        "model_size_bytes": int(metrics["model_size_bytes"]),
        "peak_process_memory_mb": float(metrics["peak_process_memory_mb"]),
        "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
        "error": "",
    }
    row.update(quality_target_columns(metrics, quality_targets))
    return row


def _failed_row(
    member: dict[str, Any],
    evaluation_split: str,
    error: Exception,
    quality_targets: QualityTargetsConfig,
) -> dict[str, Any]:
    row = {
        "member_id": member["member_id"],
        "model_id": member["model_id"],
        "variant_id": member["variant_id"],
        "status": "failed",
        "run_id": "",
        "run_dir": "",
        "evaluation_split": evaluation_split,
        "repo_id": member["model"]["repo_id"],
        "model_revision": "",
        "runtime_device": member["model"].get("device", ""),
        "compute_type": member["model"].get("compute_type", ""),
        "storage_quantization": member["model"].get("conversion_quantization", "source-default"),
        "sample_count": 0,
        "wer": 0.0,
        "cer": 0.0,
        "domain_term_recall": 0.0,
        "domain_term_recall_applicable": False,
        "mean_latency_seconds": 0.0,
        "p95_latency_seconds": 0.0,
        "aggregate_real_time_factor": 0.0,
        "model_preparation_seconds": 0.0,
        "model_size_bytes": 0,
        "peak_process_memory_mb": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "error": str(error),
    }
    row.update(
        quality_target_columns(
            {
                "domain_term_recall": 0.0,
                "domain_term_recall_applicable": False,
                "cer": 1.0,
                "wer": 1.0,
            },
            quality_targets,
        )
    )
    return row


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    return {str(row["member_id"]): row.to_dict() for _, row in frame.iterrows()}


def _write_comparison(
    rows: dict[str, dict[str, Any]],
    path: Path,
    reference_member: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows.values())
    if frame.empty:
        raise ValueError("Experiment group has no members")
    completed = frame["status"].eq("completed")
    frame["rank"] = pd.NA
    if completed.any():
        completed_frame = frame.loc[completed]
        targets_enabled = (
            "quality_targets_enabled" in completed_frame
            and completed_frame["quality_targets_enabled"]
            .map(lambda value: str(value).strip().lower() == "true")
            .any()
        )
        if targets_enabled:
            ranked = completed_frame.sort_values(
                [
                    "quality_target_pass",
                    "quality_target_miss_count",
                    "domain_term_recall_gap",
                    "cer_gap",
                    "wer_gap",
                    "domain_term_recall",
                    "cer",
                    "wer",
                    "aggregate_real_time_factor",
                ],
                ascending=[False, True, True, True, True, False, True, True, True],
            )
        else:
            ranked = completed_frame.sort_values(
                ["cer", "wer", "aggregate_real_time_factor"],
                ascending=[True, True, True],
            )
        for rank, index in enumerate(ranked.index, start=1):
            frame.loc[index, "rank"] = rank
    if reference_member:
        references = frame.loc[
            frame["member_id"].eq(reference_member) & frame["status"].eq("completed")
        ]
        if not references.empty:
            reference = references.iloc[0]
            frame["cer_delta_vs_reference"] = frame["cer"].astype(float) - float(reference["cer"])
            frame["wer_delta_vs_reference"] = frame["wer"].astype(float) - float(reference["wer"])
            frame["term_recall_delta_vs_reference"] = frame["domain_term_recall"].astype(
                float
            ) - float(reference["domain_term_recall"])
            current_rtf = frame["aggregate_real_time_factor"].astype(float)
            reference_rtf = float(reference["aggregate_real_time_factor"])
            frame["rtf_speedup_vs_reference"] = current_rtf.map(
                lambda value: reference_rtf / value if value > 0 else 0.0
            )
            current_size = frame["model_size_bytes"].astype(float)
            reference_size = float(reference["model_size_bytes"])
            frame["model_size_reduction_ratio"] = current_size.map(
                lambda value: 1 - value / reference_size if reference_size > 0 else 0.0
            )
    frame = frame.sort_values(["status", "rank", "member_id"], na_position="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _lock_members(settings: Settings, members: list[dict[str, Any]], role: str) -> None:
    requests: dict[str, dict[str, Any]] = {}
    for member in members:
        model = member["model"]
        if model.get("backend") == "fixture":
            continue
        repo_id = str(model["repo_id"])
        requests[repo_id] = {
            "repo_id": repo_id,
            "revision": str(model.get("revision", "main")),
            "roles": [role],
        }
    if requests:
        print(f"[model-lock] resolving {len(requests)} model revision(s)", flush=True)
        resolve_model_revisions(settings.paths.model_lock, list(requests.values()))


def _run_group(
    *,
    spec_path: Path,
    full_spec: dict[str, Any],
    section: dict[str, Any],
    kind: str,
    members: list[dict[str, Any]],
    settings: Settings,
) -> ExperimentGroupResult:
    group_id = _require_identifier(section.get("id"), f"{kind}.id")
    group_name = str(section.get("name", group_id))
    evaluation_split = str(section.get("evaluation_split", "validation")).lower()
    if evaluation_split not in {"train", "validation", "test"}:
        raise ValueError(f"{kind}.evaluation_split must be train, validation, or test")
    if kind in {"benchmark", "quantization"} and evaluation_split == "test":
        raise ValueError(
            f"{kind} selection must use validation data. Reserve test for finalize-evaluation."
        )
    isolated_process = bool(section.get("isolated_process", True))
    continue_on_error = bool(section.get("continue_on_error", True))
    reference_member = (
        str(section.get("reference_variant") or members[0]["member_id"])
        if kind == "quantization"
        else None
    )

    category = "benchmarks" if kind == "benchmark" else "quantization"
    group_dir = settings.paths.artifacts_dir.parent / category / group_id
    snapshot_path = group_dir / f"{kind}.snapshot.yaml"
    _snapshot_spec(full_spec, snapshot_path)
    configs_dir = group_dir / "configs"
    workers_dir = group_dir / "workers"
    comparison_path = group_dir / f"{kind}_comparison.csv"
    store = ExperimentStore(settings.paths.database)
    store.start_group(
        group_id=group_id,
        kind=kind,
        name=group_name,
        config_path=spec_path,
        output_dir=group_dir,
    )
    print(
        f"[{kind}] group={group_id} split={evaluation_split} candidates={len(members)}",
        flush=True,
    )
    _lock_members(settings, members, f"{kind}-candidate")

    rows = _load_rows(comparison_path)
    for member in members:
        member_id = member["member_id"]
        existing = rows.get(member_id)
        if existing and existing.get("status") == "completed":
            existing.update(quality_target_columns(existing, settings.quality_targets))
            run_dir = Path(str(existing.get("run_dir", "")))
            if run_dir.exists():
                print(
                    f"[{kind}] {member_id}: completed cache found; skipping",
                    flush=True,
                )
                continue
        config_path = configs_dir / f"{member_id}.yaml"
        result_path = workers_dir / f"{member_id}.json"
        _write_member_config(
            settings=settings,
            model=member["model"],
            evaluation_split=evaluation_split,
            title=f"{group_name} - {member_id}",
            path=config_path,
        )
        store.upsert_group_member(
            group_id=group_id,
            member_id=member_id,
            model_id=member["model_id"],
            variant_id=member["variant_id"],
            status="running",
            config_path=config_path,
        )
        print(
            f"[{kind}] {member_id}: starting "
            f"({member['model'].get('repo_id')} / "
            f"{member['model'].get('compute_type', 'auto')})",
            flush=True,
        )
        try:
            result = _execute_config(config_path, result_path, isolated_process)
            row = _completed_row(
                member=member,
                result=result,
                evaluation_split=evaluation_split,
                quality_targets=settings.quality_targets,
            )
            rows[member_id] = row
            store.upsert_group_member(
                group_id=group_id,
                member_id=member_id,
                model_id=member["model_id"],
                variant_id=member["variant_id"],
                status="completed",
                config_path=config_path,
                run_id=str(result["run_id"]),
            )
            print(
                f"[{kind}] {member_id}: completed "
                f"WER={row['wer']:.4f} CER={row['cer']:.4f} "
                f"RTF={row['aggregate_real_time_factor']:.4f}",
                flush=True,
            )
        except Exception as exc:
            rows[member_id] = _failed_row(
                member,
                evaluation_split,
                exc,
                settings.quality_targets,
            )
            store.upsert_group_member(
                group_id=group_id,
                member_id=member_id,
                model_id=member["model_id"],
                variant_id=member["variant_id"],
                status="failed",
                config_path=config_path,
                error=str(exc),
            )
            print(f"[{kind}] {member_id}: failed: {exc}", flush=True)
            _write_comparison(rows, comparison_path, reference_member)
            if not continue_on_error:
                store.finish_group(group_id, "failed", str(exc))
                raise
        _write_comparison(rows, comparison_path, reference_member)

    frame = _write_comparison(rows, comparison_path, reference_member)
    completed = frame.loc[frame["status"] == "completed"]
    if completed.empty:
        store.finish_group(group_id, "failed", "No candidates completed")
        raise RuntimeError(f"No {kind} candidates completed")

    chart_path = create_comparison_chart(
        frame,
        group_dir / "charts" / f"{kind}_comparison.png",
        f"ASR {kind.title()} Comparison | {group_id}",
    )
    report_path = build_comparison_report(
        frame,
        group_dir / "reports" / f"{kind}_report.docx",
        title=group_name,
        group_id=group_id,
        group_kind=kind,
        evaluation_split=evaluation_split,
        owner=settings.project.owner,
        chart_path=chart_path,
        reference_member=reference_member,
    )
    status = "completed" if frame["status"].eq("completed").all() else "partial"
    store.finish_group(group_id, status)
    print(
        f"[{kind}] group={group_id} status={status} report={report_path}",
        flush=True,
    )
    return ExperimentGroupResult(
        group_id=group_id,
        group_dir=group_dir,
        comparison_path=comparison_path,
        report_path=report_path,
        status=status,
    )


def lock_model_matrix(matrix_path: str | Path) -> dict[str, Any]:
    path = Path(matrix_path).expanduser().resolve()
    spec = _load_yaml(path)
    section = _section(spec, "benchmark")
    settings = _base_settings(path, section)
    models = section.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("benchmark.models must be a non-empty list")
    members = _benchmark_members(section, settings)
    _lock_members(settings, members, "benchmark-candidate")
    return _load_yaml(settings.paths.model_lock)


def _benchmark_members(
    section: dict[str, Any],
    settings: Settings,
) -> list[dict[str, Any]]:
    candidates = section.get("models")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("benchmark.models must be a non-empty list")
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("Each benchmark model must be a mapping")
        if not bool(item.get("enabled", True)):
            continue
        model_id = _require_identifier(item.get("id"), "benchmark model id")
        if model_id in seen:
            raise ValueError(f"Duplicate benchmark model id: {model_id}")
        seen.add(model_id)
        quantization = str(
            item.get(
                "conversion_quantization",
                settings.model.conversion_quantization or "source-default",
            )
        )
        default_dir = settings.model.local_dir.parent / model_id / quantization
        model = _candidate_model(item, settings, default_local_dir=default_dir)
        members.append(
            {
                "member_id": model_id,
                "model_id": model_id,
                "variant_id": str(model.get("compute_type", "auto")),
                "model": model,
            }
        )
    if not members:
        raise ValueError("No enabled benchmark models")
    return members


def run_model_benchmark(matrix_path: str | Path) -> ExperimentGroupResult:
    path = Path(matrix_path).expanduser().resolve()
    spec = _load_yaml(path)
    section = _section(spec, "benchmark")
    settings = _base_settings(path, section)
    members = _benchmark_members(section, settings)
    return _run_group(
        spec_path=path,
        full_spec=spec,
        section=section,
        kind="benchmark",
        members=members,
        settings=settings,
    )


def _selection_payload(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("model"), dict):
        raise ValueError(f"Invalid selection file: {path}")
    return selection


def _quantization_members(
    section: dict[str, Any],
    settings: Settings,
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    variants = section.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("quantization.variants must be a non-empty list")
    selected_model = deepcopy(selection["model"])
    model_id = _require_identifier(
        selection.get("model_id") or selected_model.get("id"),
        "selected model id",
    )
    cache_root_value = section.get("model_cache_dir", settings.model.local_dir.parent)
    cache_root = Path(str(cache_root_value)).expanduser()
    if not cache_root.is_absolute():
        cache_root = settings.project_root / cache_root

    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("Each quantization variant must be a mapping")
        if not bool(item.get("enabled", True)):
            continue
        variant_id = _require_identifier(item.get("id"), "quantization variant id")
        if variant_id in seen:
            raise ValueError(f"Duplicate quantization variant id: {variant_id}")
        seen.add(variant_id)
        storage = str(item.get("conversion_quantization", variant_id))
        default_dir = cache_root / model_id / storage
        candidate = {**selected_model, **item}
        candidate["local_dir"] = str(default_dir)
        model = _candidate_model(candidate, settings, default_local_dir=default_dir)
        members.append(
            {
                "member_id": variant_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "model": model,
            }
        )
    if not members:
        raise ValueError("No enabled quantization variants")
    reference = str(section.get("reference_variant", members[0]["member_id"]))
    if reference not in {member["member_id"] for member in members}:
        raise ValueError("quantization.reference_variant must be enabled")
    return members


def run_quantization_sweep(
    spec_path: str | Path,
    selection_path: str | Path,
) -> ExperimentGroupResult:
    path = Path(spec_path).expanduser().resolve()
    selected_path = Path(selection_path).expanduser().resolve()
    spec = _load_yaml(path)
    section = _section(spec, "quantization")
    if not bool(section.get("enabled", True)):
        raise ValueError(
            "quantization.enabled is false; enable it in the selected quantization spec"
        )
    settings = _base_settings(path, section)
    selection = _selection_payload(selected_path)
    members = _quantization_members(section, settings, selection)
    spec_with_selection = deepcopy(spec)
    spec_with_selection["selected_model"] = selection
    return _run_group(
        spec_path=path,
        full_spec=spec_with_selection,
        section=section,
        kind="quantization",
        members=members,
        settings=settings,
    )


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def select_experiment_member(
    group_dir: str | Path,
    member_id: str,
    *,
    reviewer: str,
    reason: str,
    human_reviewed: bool = True,
) -> Path:
    directory = Path(group_dir).expanduser().resolve()
    snapshots = list(directory.glob("*.snapshot.yaml"))
    if len(snapshots) != 1:
        raise ValueError(f"Expected one experiment snapshot in {directory}")
    spec = _load_yaml(snapshots[0])
    kind = "benchmark" if "benchmark" in spec else "quantization"
    section = _section(spec, kind)
    group_id = _require_identifier(section.get("id"), f"{kind}.id")
    settings = _base_settings(snapshots[0], section)
    if not human_reviewed and settings.governance.mode == "strict_private":
        raise ValueError(
            "Automated proxy selection is not allowed with strict_private governance. "
            "A human reviewer must select the manufacturing model or quantization variant."
        )
    comparison_path = directory / f"{kind}_comparison.csv"
    frame = pd.read_csv(comparison_path, keep_default_na=False)
    matches = frame.loc[frame["member_id"].astype(str) == member_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown member '{member_id}' in {group_id}")
    row = matches.iloc[0]
    if row["status"] != "completed":
        raise ValueError(f"Member '{member_id}' did not complete successfully")
    config_path = directory / "configs" / f"{member_id}.yaml"
    generated_config = _load_yaml(config_path)
    model = generated_config["model"]
    selection_id = f"selection-{new_run_id()}"
    output_name = "model_selection.yaml" if kind == "benchmark" else "quantization_selection.yaml"
    output_path = directory / output_name
    payload = {
        "selection": {
            "selection_id": selection_id,
            "source_kind": kind,
            "source_group_id": group_id,
            "member_id": member_id,
            "model_id": str(row["model_id"]),
            "variant_id": str(row["variant_id"]),
            "run_id": str(row["run_id"]),
            "selected_at": utc_now().isoformat(),
            "reviewer": reviewer,
            "reason": reason,
            "human_reviewed": human_reviewed,
            "selection_scope": ("human_review" if human_reviewed else "automated_public_proxy"),
            "model": model,
            "metrics": {
                key: _python_value(row[key])
                for key in [
                    "wer",
                    "cer",
                    "domain_term_recall",
                    "aggregate_real_time_factor",
                    "model_size_bytes",
                    "peak_gpu_memory_mb",
                ]
            },
            "quality_target_assessment": {
                "enabled": _python_value(row.get("quality_targets_enabled", False)),
                "overall_pass": _python_value(row.get("quality_target_pass", False)),
                "miss_count": _python_value(row.get("quality_target_miss_count", 0)),
                "domain_term_recall_gap": _python_value(row.get("domain_term_recall_gap", 0.0)),
                "cer_gap": _python_value(row.get("cer_gap", 0.0)),
                "wer_gap": _python_value(row.get("wer_gap", 0.0)),
            },
            "final_test_required": True,
        }
    }
    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    immutable_path = directory / "selections" / f"{selection_id}.yaml"
    immutable_path.parent.mkdir(parents=True, exist_ok=True)
    immutable_path.write_text(serialized, encoding="utf-8")
    output_path.write_text(serialized, encoding="utf-8")
    ExperimentStore(settings.paths.database).add_selection(
        selection_id=selection_id,
        group_id=group_id,
        member_id=member_id,
        run_id=str(row["run_id"]),
        reviewer=reviewer,
        reason=reason,
        selection_path=immutable_path,
    )
    return output_path


def run_final_evaluation(
    selection_path: str | Path,
    base_config_path: str | Path,
) -> dict[str, Any]:
    selected_path = Path(selection_path).expanduser().resolve()
    base_path = Path(base_config_path).expanduser().resolve()
    selection = _selection_payload(selected_path)
    settings = load_settings(base_path)
    raw = _absolute_base_config(settings.raw, settings)
    raw["model"] = selection["model"]
    raw["evaluation"] = {
        **raw.get("evaluation", {}),
        "split": "test",
    }
    raw["training"] = {"enabled": False, "reason": "final held-out test evaluation"}
    raw.setdefault("report", {})["title"] = (
        f"최종 Test 평가 - {selection.get('model_id')} / {selection.get('variant_id')}"
    )
    config_path = selected_path.parent / "final_test_config.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    result = run_pipeline(load_settings(config_path))
    summary = {
        "selection_path": str(selected_path),
        "config_path": str(config_path),
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "report_path": str(result.report_path),
        "metrics": result.metrics,
        "quality_gate": result.metrics["quality_gate"],
        "quality_target_pass": bool(result.metrics["quality_gate"]["overall_pass"]),
        "human_review_required": True,
    }
    write_json(selected_path.parent / "final_test_result.json", summary)
    return summary


def train_selected_whisper_lora(
    selection_path: str | Path,
    base_config_path: str | Path,
) -> dict[str, Any]:
    selected_path = Path(selection_path).expanduser().resolve()
    base_path = Path(base_config_path).expanduser().resolve()
    selection = _selection_payload(selected_path)
    if selection.get("source_kind") != "benchmark":
        raise ValueError("LoRA training requires a model_selection.yaml from a benchmark")

    settings = load_settings(base_path)
    raw = _absolute_base_config(settings.raw, settings)
    training = raw.setdefault("training", {})
    training["enabled"] = True
    selected_model_id = str(selection.get("model_id") or selection.get("member_id"))
    selected_repo_id = str(selection["model"]["repo_id"])
    follow_selected_model = bool(training.get("follow_selected_model", True))
    configured_repo_id = training.get("repo_id")
    if not follow_selected_model and not configured_repo_id:
        raise ValueError(
            "training.repo_id is required when training.follow_selected_model is false"
        )
    training_repo_id = (
        selected_repo_id if follow_selected_model else str(configured_repo_id)
    )
    training_model_id = (
        selected_model_id
        if follow_selected_model
        else str(training.get("model_id") or training_repo_id.rsplit("/", 1)[-1])
    )
    training["repo_id"] = training_repo_id
    training["source_selection_model_id"] = selected_model_id
    training["source_selection_repo_id"] = selected_repo_id
    training["model_policy"] = (
        "selected_model" if follow_selected_model else "configured_resource_candidate"
    )
    output_root = str(training.get("output_dir", "checkpoints/manufacturing-whisper-lora"))
    training["output_dir"] = f"{output_root.rstrip('/')}/{training_model_id}"
    raw.setdefault("report", {})["title"] = (
        f"Whisper LoRA 평가 - {training_model_id} "
        f"(추론 선택 모델: {selected_model_id})"
    )
    config_path = selected_path.parent / "selected_training_config.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    run_dir = train_whisper_lora(load_settings(config_path))
    result = {
        "selection_path": str(selected_path),
        "config_path": str(config_path),
        "model_id": training_model_id,
        "model_repo": training_repo_id,
        "selected_inference_model_id": selected_model_id,
        "selected_inference_model_repo": selected_repo_id,
        "model_policy": training["model_policy"],
        "run_dir": str(run_dir),
        "report_path": str(run_dir / "reports/evaluation_report.docx"),
        "human_review_required": True,
    }
    write_json(selected_path.parent / "selected_training_result.json", result)
    return result
