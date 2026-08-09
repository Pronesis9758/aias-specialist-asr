from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from aias_specialist.data import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "sample" / "manufacturing_synthetic"
MANIFEST = DATASET_ROOT / "manifest.csv"


def test_synthetic_manufacturing_manifest_and_audio_contract() -> None:
    frame = validate_manifest(MANIFEST, backend="faster_whisper")

    assert frame.groupby("split").size().to_dict() == {
        "test": 6,
        "train": 18,
        "validation": 6,
    }
    assert set(frame["consent_status"]) == {"synthetic"}
    assert set(frame["deidentified"]) == {"true"}
    assert frame["sample_id"].is_unique
    assert frame["reference_text"].str.strip().ne("").all()

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
    assert provenance["sample_count"] == 30
    assert "real manufacturing ASR accuracy" in provenance["prohibited_claims"]
