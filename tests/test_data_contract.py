from pathlib import Path

import pandas as pd
import pytest

from aias_specialist.data import prepare_manifest


def test_unapproved_data_is_blocked(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
                "audio_path": "fixture://1",
                "reference_text": "테스트",
                "split": "test",
                "fixture_prediction": "테스트",
                "source": "unknown",
                "consent_status": "unknown",
            }
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="Unapproved"):
        prepare_manifest(manifest, tmp_path / "out.csv", backend="fixture")


def test_invalid_split_is_blocked(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "1",
                "audio_path": "fixture://1",
                "reference_text": "테스트",
                "split": "holdout",
                "fixture_prediction": "테스트",
                "source": "synthetic",
                "consent_status": "synthetic",
            }
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="Unsupported split"):
        prepare_manifest(manifest, tmp_path / "out.csv", backend="fixture")
