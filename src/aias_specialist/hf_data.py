from __future__ import annotations

import re
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .utils import utc_now, write_json


def _dataset_config(settings: Settings) -> dict[str, Any]:
    value = settings.raw.get("dataset")
    if not isinstance(value, dict):
        raise ValueError("Config section 'dataset' is required for Hugging Face data preparation")
    return value


def _positive_count(values: dict[str, Any], key: str) -> int:
    count = int(values.get(key, 0))
    if count <= 0:
        raise ValueError(f"dataset.{key} must be a positive integer")
    return count


def _safe_id(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return cleaned or "sample"


def _write_audio(audio: Any, destination_stem: Path) -> Path:
    if isinstance(audio, dict):
        source_path = str(audio.get("path") or "")
        suffix = Path(source_path).suffix.lower()
        payload = audio.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            destination = destination_stem.with_suffix(suffix or ".flac")
            destination.write_bytes(bytes(payload))
            return destination

        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if array is not None and sampling_rate:
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError("Install data dependencies with: uv sync --extra data") from exc
            destination = destination_stem.with_suffix(".wav")
            sf.write(destination, array, int(sampling_rate), subtype="PCM_16")
            return destination

        if source_path and Path(source_path).is_file():
            destination = destination_stem.with_suffix(suffix or ".audio")
            destination.write_bytes(Path(source_path).read_bytes())
            return destination

    raise ValueError(
        "Unsupported Hugging Face audio value. The dataset must expose embedded bytes or a "
        "decoded array in its audio column."
    )


def _reuse_manifest(manifest_path: Path, expected_counts: dict[str, int]) -> pd.DataFrame | None:
    if not manifest_path.exists():
        return None
    frame = pd.read_csv(manifest_path, dtype=str).fillna("")
    actual_counts = frame["split"].str.lower().value_counts().to_dict()
    if any(actual_counts.get(split, 0) != count for split, count in expected_counts.items()):
        return None
    for raw_path in frame["audio_path"]:
        audio_path = Path(raw_path)
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve()
        if not audio_path.exists():
            return None
    return frame


def _stream_split(
    repo_id: str,
    subset: str,
    revision: str,
    split: str,
    audio_column: str,
    seed: int,
    buffer_size: int,
) -> Any:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data dependencies with: uv sync --extra data") from exc

    dataset = load_dataset(
        repo_id,
        name=subset,
        revision=revision,
        split=split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset.cast_column(audio_column, Audio(decode=False))


def prepare_hf_dataset(settings: Settings, force: bool = False) -> tuple[Path, dict[str, Any]]:
    values = _dataset_config(settings)
    required = {"repo_id", "revision", "subset", "audio_column", "text_column", "license"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Dataset config is missing: {', '.join(missing)}")

    counts = {
        "train": _positive_count(values, "train_samples"),
        "validation": _positive_count(values, "validation_samples"),
        "test": _positive_count(values, "test_samples"),
    }
    manifest_path = settings.paths.manifest
    if not force:
        existing = _reuse_manifest(manifest_path, counts)
        if existing is not None:
            return manifest_path, {
                "status": "reused",
                "manifest": str(manifest_path),
                "sample_count": len(existing),
                "split_counts": counts,
            }
        if manifest_path.exists():
            raise ValueError(
                f"Existing manifest is incomplete: {manifest_path}. "
                "Re-run with --force to rebuild it."
            )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required") from exc

    repo_id = str(values["repo_id"])
    requested_revision = str(values["revision"])
    info = HfApi().dataset_info(repo_id, revision=requested_revision)
    if info.private or bool(info.gated):
        raise ValueError(
            "The configured dataset must be public and ungated for this Colab workflow"
        )
    resolved_revision = str(info.sha)
    if len(requested_revision) == 40 and requested_revision != resolved_revision:
        raise ValueError(
            "Dataset revision mismatch: "
            f"requested {requested_revision}, resolved {resolved_revision}"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = manifest_path.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    subset = str(values["subset"])
    audio_column = str(values["audio_column"])
    text_column = str(values["text_column"])
    id_column = str(values.get("id_column", "id"))
    train_split = str(values.get("train_split", "train"))
    test_split = str(values.get("test_split", "test"))
    buffer_size = int(values.get("shuffle_buffer_size", 1_000))
    source = f"huggingface:{repo_id}@{resolved_revision}"
    records: list[dict[str, str]] = []

    train_stream = _stream_split(
        repo_id,
        subset,
        resolved_revision,
        train_split,
        audio_column,
        settings.project.seed,
        buffer_size,
    )
    shuffled_train_rows = iter(train_stream.take(counts["train"] + counts["validation"]))
    test_stream = _stream_split(
        repo_id,
        subset,
        resolved_revision,
        test_split,
        audio_column,
        settings.project.seed + 1,
        buffer_size,
    )
    split_plan = [
        (train_split, "train", list(islice(shuffled_train_rows, counts["train"]))),
        (
            train_split,
            "validation",
            list(islice(shuffled_train_rows, counts["validation"])),
        ),
        (test_split, "test", list(test_stream.take(counts["test"]))),
    ]
    for source_split, target_split, rows in split_plan:
        for index, row in enumerate(rows):
            reference_text = str(row.get(text_column, "")).strip()
            if not reference_text:
                raise ValueError(f"Empty reference text in {source_split} sample {index}")
            original_id = _safe_id(row.get(id_column, f"{source_split}-{index:05d}"))
            sample_id = f"zeroth-{target_split}-{index:05d}-{original_id}"
            audio_path = _write_audio(row.get(audio_column), audio_dir / sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "audio_path": audio_path.relative_to(manifest_path.parent).as_posix(),
                    "reference_text": reference_text,
                    "split": target_split,
                    "source": source,
                    "consent_status": "public",
                    "dataset_license": str(values["license"]),
                    "dataset_split": source_split,
                    "original_id": original_id,
                }
            )

    frame = pd.DataFrame(records)
    actual_counts = frame["split"].value_counts().to_dict()
    if actual_counts != counts:
        raise RuntimeError(f"Prepared split counts do not match the request: {actual_counts}")
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    provenance = {
        "dataset_repo": repo_id,
        "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "subset": subset,
        "license": str(values["license"]),
        "prepared_at": utc_now().isoformat(),
        "seed": settings.project.seed,
        "shuffle_buffer_size": buffer_size,
        "sample_count": len(frame),
        "split_counts": counts,
        "manifest": str(manifest_path),
    }
    write_json(manifest_path.parent / "dataset_provenance.json", provenance)
    return manifest_path, {"status": "created", **provenance}
