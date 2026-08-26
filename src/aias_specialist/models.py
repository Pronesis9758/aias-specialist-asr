from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from huggingface_hub import HfApi, snapshot_download

from .config import Settings


def resolve_model_revisions(
    lock_path: Path,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one or more Hub revisions and merge them into the project model lock."""
    api = HfApi()
    existing: dict[str, Any] = {}
    if lock_path.exists():
        loaded = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    locked_models = existing.get("models", {})
    if not isinstance(locked_models, dict):
        locked_models = {}

    legacy_baseline = locked_models.pop("baseline", None)
    if isinstance(legacy_baseline, dict) and legacy_baseline.get("repo_id"):
        locked_models[str(legacy_baseline["repo_id"])] = {
            **legacy_baseline,
            "roles": ["baseline"],
        }

    for index, request in enumerate(requests, start=1):
        repo_id = str(request["repo_id"])
        requested_revision = str(request.get("revision", "main"))
        roles = sorted({str(role) for role in request.get("roles", ["baseline"])})
        print(
            f"[model-lock] {index}/{len(requests)} {repo_id}@{requested_revision}",
            flush=True,
        )
        info = api.model_info(repo_id, revision=requested_revision)
        previous = locked_models.get(repo_id, {})
        previous_roles = previous.get("roles", []) if isinstance(previous, dict) else []
        locked_models[repo_id] = {
            "repo_id": repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": info.sha,
            "roles": sorted({*previous_roles, *roles}),
        }

    lock = {"models": dict(sorted(locked_models.items()))}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return lock


def resolve_model_lock(settings: Settings) -> dict[str, Any]:
    requests = [
        {
            "repo_id": settings.model.repo_id,
            "revision": settings.model.revision,
            "roles": ["baseline"],
        }
    ]
    training = settings.training.values or {}
    if settings.training.enabled and training.get("repo_id"):
        requests.append(
            {
                "repo_id": str(training["repo_id"]),
                "revision": str(training.get("revision", "main")),
                "roles": ["training"],
            }
        )
    distillation = settings.distillation.values or {}
    if settings.distillation.enabled:
        for role in ["teacher", "student"]:
            repo_key = f"{role}_repo_id"
            if distillation.get(repo_key):
                requests.append(
                    {
                        "repo_id": str(distillation[repo_key]),
                        "revision": str(distillation.get(f"{role}_revision", "main")),
                        "roles": [f"distillation_{role}"],
                    }
                )
    if (
        settings.correction.enabled
        and settings.correction.nearest_neighbor_enabled
        and settings.correction.nn_backend == "transformers"
        and settings.correction.nn_model_repo_id
    ):
        requests.append(
            {
                "repo_id": settings.correction.nn_model_repo_id,
                "revision": settings.correction.nn_model_revision,
                "roles": ["retrieval_embedding"],
            }
        )
    return resolve_model_revisions(settings.paths.model_lock, requests)


def load_model_lock(settings: Settings) -> dict[str, Any]:
    if not settings.paths.model_lock.exists():
        return resolve_model_lock(settings)
    lock = yaml.safe_load(settings.paths.model_lock.read_text(encoding="utf-8"))
    models = lock.get("models", {}) if isinstance(lock, dict) else {}
    if settings.model.repo_id not in models:
        return resolve_model_lock(settings)
    baseline = models[settings.model.repo_id]
    if (
        not isinstance(baseline, dict)
        or str(baseline.get("requested_revision")) != settings.model.revision
    ):
        return resolve_model_lock(settings)
    training = settings.training.values or {}
    if settings.training.enabled and training.get("repo_id"):
        training_repo = str(training["repo_id"])
        training_revision = str(training.get("revision", "main"))
        training_lock = models.get(training_repo)
        if (
            not isinstance(training_lock, dict)
            or str(training_lock.get("requested_revision")) != training_revision
        ):
            return resolve_model_lock(settings)
    distillation = settings.distillation.values or {}
    if settings.distillation.enabled:
        for role in ["teacher", "student"]:
            repo_id = str(distillation.get(f"{role}_repo_id", ""))
            revision = str(distillation.get(f"{role}_revision", "main"))
            if not repo_id:
                continue
            model_lock = models.get(repo_id)
            if (
                not isinstance(model_lock, dict)
                or str(model_lock.get("requested_revision")) != revision
            ):
                return resolve_model_lock(settings)
    retrieval_repo = settings.correction.nn_model_repo_id
    if (
        settings.correction.enabled
        and settings.correction.nearest_neighbor_enabled
        and settings.correction.nn_backend == "transformers"
        and retrieval_repo
    ):
        retrieval_lock = models.get(retrieval_repo)
        if (
            not isinstance(retrieval_lock, dict)
            or str(retrieval_lock.get("requested_revision"))
            != settings.correction.nn_model_revision
        ):
            return resolve_model_lock(settings)
    return lock


def download_baseline_model(settings: Settings) -> tuple[Path, str]:
    local_source = Path(settings.model.repo_id).expanduser()
    if local_source.is_dir():
        digest = hashlib.sha256()
        for item in sorted(local_source.rglob("*")):
            if item.is_file():
                digest.update(item.relative_to(local_source).as_posix().encode("utf-8"))
                digest.update(str(item.stat().st_size).encode("ascii"))
        revision = f"local-{digest.hexdigest()[:16]}"
        if settings.model.format == "transformers":
            required_files = [
                settings.model.local_dir / "config.json",
                settings.model.local_dir / "model.bin",
                settings.model.local_dir / "tokenizer.json",
            ]
            if all(path.exists() and path.stat().st_size > 0 for path in required_files):
                print(f"[model-cache] reuse {settings.model.local_dir}", flush=True)
                return settings.model.local_dir, revision
            return _convert_transformers_model(
                settings, revision, source_model=local_source.resolve()
            ), revision
        return local_source.resolve(), revision
    lock = load_model_lock(settings)
    baseline = lock["models"][settings.model.repo_id]
    revision = str(baseline["resolved_revision"])
    local_dir = settings.model.local_dir
    required_files = [
        local_dir / "config.json",
        local_dir / "model.bin",
        local_dir / "tokenizer.json",
    ]
    if all(path.exists() and path.stat().st_size > 0 for path in required_files):
        print(f"[model-cache] reuse {local_dir}", flush=True)
        return local_dir, revision
    if settings.model.format == "transformers":
        return _convert_transformers_model(settings, revision), revision
    local_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[model-download] {settings.model.repo_id}@{revision} -> {local_dir}",
        flush=True,
    )
    snapshot_download(
        repo_id=settings.model.repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    return local_dir, revision


def _convert_transformers_model(
    settings: Settings,
    revision: str,
    source_model: Path | None = None,
) -> Path:
    """Convert an immutable Transformers Whisper checkpoint into a CT2 runtime model."""
    quantization = settings.model.conversion_quantization or "float16"
    local_dir = settings.model.local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        preserved = local_dir.with_name(f"{local_dir.name}.incomplete-{timestamp}")
        local_dir.replace(preserved)
        print(f"[model-cache] preserved incomplete cache: {preserved}", flush=True)

    partial_dir = local_dir.with_name(f".{local_dir.name}.partial-{uuid4().hex[:8]}")
    try:
        try:
            from ctranslate2.converters import TransformersConverter
        except ImportError as exc:
            raise RuntimeError(
                "Transformers model conversion requires the train dependency extra. "
                "Run 'uv sync --extra train' locally or install '.[train]' in Colab."
            ) from exc
        source_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get(
            "HF_HOME", "Hugging Face default"
        )
        print(
            f"[model-source] loading {settings.model.repo_id}@{revision} cache={source_cache}",
            flush=True,
        )
        converter = TransformersConverter(
            str(source_model or settings.model.repo_id),
            revision=None if source_model else revision,
            copy_files=["tokenizer.json", "preprocessor_config.json"],
            low_cpu_mem_usage=True,
        )
        print(
            f"[model-convert] source ready; start {settings.model.repo_id}@{revision} "
            f"quantization={quantization}",
            flush=True,
        )
        converter.convert(
            str(partial_dir),
            quantization=quantization,
            force=False,
        )
        required = [
            partial_dir / "config.json",
            partial_dir / "model.bin",
            partial_dir / "tokenizer.json",
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            raise RuntimeError(f"Incomplete CTranslate2 conversion: {partial_dir}")
        partial_dir.replace(local_dir)
        print(f"[model-convert] completed -> {local_dir}", flush=True)
    except Exception:
        if partial_dir.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = partial_dir.with_name(f"{partial_dir.name}.failed-{timestamp}")
            partial_dir.replace(failed)
        raise
    return local_dir
