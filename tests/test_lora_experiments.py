from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from aias_specialist.lora_experiments import (
    _stage_learning_curve_manifest,
    _stages_through,
)


def _stages() -> list[dict[str, Any]]:
    return [
        {"id": "pilot-5h"},
        {"id": "target-10h"},
        {"id": "extended-15h"},
    ]


def test_stages_through_supports_incremental_a100_learning_curve() -> None:
    assert [stage["id"] for stage in _stages_through(_stages(), "target-10h")] == [
        "pilot-5h",
        "target-10h",
    ]


def test_stages_through_rejects_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unknown max_stage"):
        _stages_through(_stages(), "unknown")


def test_learning_curve_stages_only_active_train_and_validation_audio(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = []
    for sample_id, split in [
        ("train-1", "train"),
        ("train-2", "train"),
        ("train-3", "train"),
        ("valid-1", "validation"),
        ("test-1", "test"),
    ]:
        audio = source_dir / f"{sample_id}.wav"
        audio.write_bytes(sample_id.encode("utf-8"))
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio),
                "reference_text": sample_id,
                "split": split,
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    settings = SimpleNamespace(paths=SimpleNamespace(manifest=manifest))
    section = {
        "local_audio_staging": {
            "enabled": True,
            "cache_dir": str(tmp_path / "cache"),
            "copy_workers": 2,
        }
    }

    staged_path = _stage_learning_curve_manifest(
        section,
        settings,
        [{"id": "pilot", "train_samples": 2}],
    )

    assert staged_path is not None
    staged = pd.read_csv(staged_path)
    assert staged["sample_id"].tolist() == ["train-1", "train-2", "valid-1"]
    assert set(staged["split"]) == {"train", "validation"}
    assert all(Path(path).is_file() for path in staged["audio_path"])
