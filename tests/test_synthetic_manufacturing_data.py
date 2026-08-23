from __future__ import annotations

import hashlib
import json
import runpy
import wave
from pathlib import Path

from aias_specialist.data import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "sample" / "manufacturing_synthetic"
MANIFEST = DATASET_ROOT / "manifest.csv"
build_utterances = runpy.run_path(
    str(ROOT / "scripts" / "generate_synthetic_manufacturing_dataset.py")
)["build_utterances"]


def test_synthetic_manufacturing_manifest_and_audio_contract() -> None:
    frame = validate_manifest(MANIFEST, backend="faster_whisper")

    assert frame.groupby("split").size().to_dict() == {
        "test": 120,
        "train": 360,
        "validation": 120,
    }
    assert set(frame["consent_status"]) == {"synthetic"}
    assert set(frame["deidentified"]) == {"true"}
    assert frame["sample_id"].is_unique
    assert frame["reference_text"].str.strip().ne("").all()
    assert frame["reference_text"].is_unique
    assert frame["spoken_text"].str.strip().ne("").all()
    assert frame["spoken_text"].is_unique
    assert (frame["reference_text"] != frame["spoken_text"]).any()
    assert frame["reference_text"].str.contains(r"\b(?:PLC|HMI|CNC|AGV|AOI)\b").any()
    assert not frame["reference_text"].str.contains(
        "피엘씨|에이치엠아이|씨엔씨|에이지브이|에이오아이"
    ).any()
    assert not frame["reference_text"].str.contains(
        "일 호기|이 호기|삼 호기|사 호기|오 호기"
    ).any()
    assert frame["reference_text"].str.contains(r"[1-5]호기").any()
    for row in frame.itertuples(index=False):
        assert all(term in row.reference_text for term in row.term_targets.split("|"))
    assert frame.groupby(["split", "difficulty_group"]).size().to_dict() == {
        ("test", "error_prone"): 40,
        ("test", "normal"): 40,
        ("test", "term_dense"): 40,
        ("train", "error_prone"): 120,
        ("train", "normal"): 120,
        ("train", "term_dense"): 120,
        ("validation", "error_prone"): 40,
        ("validation", "normal"): 40,
        ("validation", "term_dense"): 40,
    }
    noise_counts = frame.groupby(["split", "noise_condition"]).size()
    assert set(noise_counts.loc["train"]) == {90}
    assert set(noise_counts.loc["validation"]) == {30}
    assert set(noise_counts.loc["test"]) == {30}

    speaker_split_counts = frame.groupby("speaker_id")["split"].nunique()
    assert speaker_split_counts.max() == 1

    for row in frame.itertuples(index=False):
        path = Path(row.audio_path)
        with wave.open(str(path), "rb") as reader:
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getframerate() == 16_000
            assert reader.getnframes() > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.audio_sha256


def test_synthetic_manufacturing_provenance_is_explicit() -> None:
    provenance = json.loads(
        (DATASET_ROOT / "dataset_provenance.json").read_text(encoding="utf-8")
    )

    assert provenance["human_voice_data"] is False
    assert provenance["contains_personal_information"] is False
    assert provenance["dataset_id"] == "synthetic-manufacturing-korean-asr-v3"
    assert provenance["sample_count"] == 600
    assert provenance["unique_reference_count"] == 600
    assert provenance["unique_spoken_text_count"] == 600
    assert provenance["split_counts"] == {"train": 360, "validation": 120, "test": 120}
    assert provenance["postprocessing_design"]["guaranteed_improvement"] is False
    assert "real manufacturing ASR accuracy" in provenance["prohibited_claims"]


def test_synthetic_sentence_plan_is_deterministic_and_split_safe() -> None:
    first = build_utterances()
    second = build_utterances()

    assert first == second
    assert len(first) == 600
    references = [row["reference_text"] for row in first]
    spoken_texts = [row["spoken_text"] for row in first]
    assert len(set(references)) == 600
    assert len(set(spoken_texts)) == 600
    assert any("PLC" in text for text in references)
    assert any("Q502" in text for text in references)
    assert any("1호기 CNC 선반" in text for text in references)
    assert any("피엘씨" in text for text in spoken_texts)
    assert any("큐 오백이" in text for text in spoken_texts)
    assert any("일 호기 씨엔씨 선반" in text for text in spoken_texts)
    split_references = {
        split: {row["reference_text"] for row in first if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert split_references["train"].isdisjoint(split_references["validation"])
    assert split_references["train"].isdisjoint(split_references["test"])
    assert split_references["validation"].isdisjoint(split_references["test"])
