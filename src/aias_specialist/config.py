from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    owner: str
    seed: int = 42


@dataclass(frozen=True)
class PathsConfig:
    manifest: Path
    domain_terms: Path
    artifacts_dir: Path
    database: Path
    model_lock: Path


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    repo_id: str
    revision: str
    local_dir: Path
    format: str = "ctranslate2"
    conversion_quantization: str | None = None
    language: str = "ko"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "test"


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = True
    case_sensitive: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    values: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReportConfig:
    title: str
    include_charts: bool = True


@dataclass(frozen=True)
class Settings:
    config_path: Path
    project_root: Path
    project: ProjectConfig
    paths: PathsConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    correction: CorrectionConfig
    training: TrainingConfig
    report: ReportConfig
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return _stringify_paths(data)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value


def _find_project_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is required and must be a mapping")
    return value


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    root = _find_project_root(path)
    project = _section(raw, "project")
    paths = _section(raw, "paths")
    model = _section(raw, "model")
    correction = _section(raw, "correction")
    training = _section(raw, "training")
    report = _section(raw, "report")

    backend = str(model.get("backend", "fixture"))
    if backend not in {"fixture", "faster_whisper"}:
        raise ValueError("model.backend must be 'fixture' or 'faster_whisper'")
    model_format = str(model.get("format", "ctranslate2"))
    if model_format not in {"ctranslate2", "transformers"}:
        raise ValueError("model.format must be 'ctranslate2' or 'transformers'")
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Config section 'evaluation' must be a mapping")
    evaluation_split = str(evaluation.get("split", "test")).strip().lower()
    if evaluation_split not in {"train", "validation", "test"}:
        raise ValueError("evaluation.split must be train, validation, or test")

    return Settings(
        config_path=path,
        project_root=root,
        project=ProjectConfig(
            name=str(project["name"]),
            owner=str(project.get("owner", "Project Owner")),
            seed=int(project.get("seed", 42)),
        ),
        paths=PathsConfig(
            manifest=_resolve(root, str(paths["manifest"])),
            domain_terms=_resolve(root, str(paths["domain_terms"])),
            artifacts_dir=_resolve(root, str(paths["artifacts_dir"])),
            database=_resolve(root, str(paths["database"])),
            model_lock=_resolve(root, str(paths["model_lock"])),
        ),
        model=ModelConfig(
            backend=backend,
            repo_id=str(model["repo_id"]),
            revision=str(model.get("revision", "main")),
            local_dir=_resolve(root, str(model["local_dir"])),
            format=model_format,
            conversion_quantization=(
                str(model["conversion_quantization"])
                if model.get("conversion_quantization")
                else None
            ),
            language=str(model.get("language", "ko")),
            device=str(model.get("device", "auto")),
            compute_type=str(model.get("compute_type", "auto")),
            beam_size=int(model.get("beam_size", 5)),
        ),
        evaluation=EvaluationConfig(split=evaluation_split),
        correction=CorrectionConfig(
            enabled=bool(correction.get("enabled", True)),
            case_sensitive=bool(correction.get("case_sensitive", False)),
        ),
        training=TrainingConfig(
            enabled=bool(training.get("enabled", False)),
            values=dict(training),
        ),
        report=ReportConfig(
            title=str(report.get("title", "ASR Evaluation Report")),
            include_charts=bool(report.get("include_charts", True)),
        ),
        raw=raw,
    )
