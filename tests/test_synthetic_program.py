import asyncio
import sys
import types
from pathlib import Path

import pytest

from aias_specialist.synthetic_program import (
    _synthesize_one,
    build_dataset_plan,
    load_synthetic_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/data/synthetic_manufacturing_7200.yaml"


def test_full_scale_synthetic_plan_meets_research_contract() -> None:
    frame = build_dataset_plan(SPEC)

    assert frame.groupby("split").size().to_dict() == {
        "train": 6000,
        "validation": 600,
        "test": 600,
    }
    assert frame["speaker_id"].nunique() == 24
    assert frame.groupby("speaker_id")["split"].nunique().max() == 1
    assert frame["reference_text"].is_unique
    assert frame["sample_id"].is_unique
    assert set(frame["consent_status"]) == {"synthetic"}


def test_full_scale_term_coverage_is_balanced_and_test_is_dense() -> None:
    frame = build_dataset_plan(SPEC)
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    train_counts: dict[str, int] = {}
    for value in train["term_targets"]:
        for term in value.split("|"):
            train_counts[term] = train_counts.get(term, 0) + 1

    assert min(train_counts.values()) >= 100
    assert max(train_counts.values()) <= 200
    assert sum(len(value.split("|")) for value in test["term_targets"]) >= 1000
    assert test["term_targets"].str.contains(r"\|").all()


def test_full_scale_spec_records_expected_duration_targets() -> None:
    _, section = load_synthetic_spec(SPEC)
    frame = build_dataset_plan(SPEC)

    assert section["split_counts"] == {"train": 6000, "validation": 600, "test": 600}
    hours = (
        frame.assign(seconds=frame["target_duration_seconds"].astype(float))
        .groupby("split")["seconds"]
        .sum()
        .div(3600)
        .to_dict()
    )
    assert hours == {"train": 10.0, "validation": 1.0, "test": 1.0}
    assert section["request_timeout_seconds"] == 30
    assert section["progress_every"] == 10


def test_edge_tts_request_timeout_prevents_indefinite_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HangingCommunicate:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def save(self, _path: str) -> None:
            await asyncio.Event().wait()

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        types.SimpleNamespace(Communicate=HangingCommunicate),
    )
    monkeypatch.setattr("aias_specialist.synthetic_program.shutil.which", lambda _name: "ffmpeg")
    row = {
        "sample_id": "timeout-sample",
        "audio_path": "train/timeout.wav",
        "spoken_text": "체결 토크를 확인합니다",
        "tts_voice": "ko-KR-SunHiNeural",
        "tts_rate": "+0%",
        "tts_pitch": "+0Hz",
    }

    with pytest.raises(TimeoutError):
        asyncio.run(
            _synthesize_one(
                row,
                tmp_path,
                retries=1,
                request_timeout_seconds=0.01,
            )
        )

    assert not (tmp_path / "train/timeout.partial.mp3").exists()
