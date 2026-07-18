from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

from .config import Settings


def resolve_model_lock(settings: Settings) -> dict[str, Any]:
    api = HfApi()
    existing: dict[str, Any] = {}
    if settings.paths.model_lock.exists():
        loaded = yaml.safe_load(settings.paths.model_lock.read_text(encoding="utf-8"))
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

    info = api.model_info(settings.model.repo_id, revision=settings.model.revision)
    locked_models[settings.model.repo_id] = {
        "repo_id": settings.model.repo_id,
        "requested_revision": settings.model.revision,
        "resolved_revision": info.sha,
        "roles": ["baseline"],
    }
    training = settings.training.values or {}
    if settings.training.enabled and training.get("repo_id"):
        train_repo = str(training["repo_id"])
        train_revision = str(training.get("revision", "main"))
        train_info = api.model_info(train_repo, revision=train_revision)
        locked_models[train_repo] = {
            "repo_id": train_repo,
            "requested_revision": train_revision,
            "resolved_revision": train_info.sha,
            "roles": ["training"],
        }

    lock = {"models": dict(sorted(locked_models.items()))}

    settings.paths.model_lock.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.model_lock.write_text(
        yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return lock


def load_model_lock(settings: Settings) -> dict[str, Any]:
    if not settings.paths.model_lock.exists():
        return resolve_model_lock(settings)
    lock = yaml.safe_load(settings.paths.model_lock.read_text(encoding="utf-8"))
    models = lock.get("models", {}) if isinstance(lock, dict) else {}
    if settings.model.repo_id not in models:
        return resolve_model_lock(settings)
    training = settings.training.values or {}
    if settings.training.enabled and training.get("repo_id") not in models:
        return resolve_model_lock(settings)
    return lock


def download_baseline_model(settings: Settings) -> tuple[Path, str]:
    lock = load_model_lock(settings)
    baseline = lock["models"][settings.model.repo_id]
    revision = str(baseline["resolved_revision"])
    local_dir = settings.model.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        local_dir / "config.json",
        local_dir / "model.bin",
        local_dir / "tokenizer.json",
    ]
    if all(path.exists() and path.stat().st_size > 0 for path in required_files):
        return local_dir, revision
    snapshot_download(
        repo_id=settings.model.repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    return local_dir, revision
