from pathlib import Path

import pandas as pd
import pytest

from aias_specialist.hf_data import _reuse_manifest, _safe_id


def test_safe_id_removes_path_and_space_characters() -> None:
    assert _safe_id("speaker/001 sample") == "speaker-001-sample"


def test_reuse_manifest_requires_expected_splits_and_audio(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
                "audio_path": "audio.wav",
                "reference_text": "테스트",
                "split": "test",
                "source": "public",
                "consent_status": "public",
            }
        ]
    ).to_csv(manifest, index=False)

    assert _reuse_manifest(manifest, {"test": 1}) is not None
    assert _reuse_manifest(manifest, {"test": 2}) is None


def test_reuse_manifest_returns_none_when_audio_is_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
                "audio_path": "missing.wav",
                "reference_text": "테스트",
                "split": "test",
                "source": "public",
                "consent_status": "public",
            }
        ]
    ).to_csv(manifest, index=False)

    assert _reuse_manifest(manifest, {"test": 1}) is None


def test_reuse_manifest_surfaces_invalid_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"unexpected": "value"}]).to_csv(manifest, index=False)

    with pytest.raises(KeyError):
        _reuse_manifest(manifest, {"test": 1})
