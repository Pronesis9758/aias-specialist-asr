from pathlib import Path

import pandas as pd
import pytest
import yaml

from aias_specialist.config import GovernanceConfig
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


def _strict_governance(tmp_path: Path) -> GovernanceConfig:
    approval = tmp_path / "approval.yaml"
    approval.write_text(
        yaml.safe_dump(
            {
                "data_approval": {
                    "approval_id": "approval-001",
                    "data_owner": "owner",
                    "approver": "reviewer",
                    "approved_at": "2026-07-01",
                    "purpose": "ASR evaluation",
                    "retention_until": "2027-07-01",
                    "storage_location": "approved-private-storage",
                    "external_processing_allowed": True,
                    "deidentification_reviewed": True,
                    "prohibited_content_reviewed": True,
                    "deletion_owner": "deletion-owner",
                }
            }
        ),
        encoding="utf-8",
    )
    return GovernanceConfig(
        mode="strict_private",
        approval_file=approval,
        require_speaker_disjoint_splits=True,
        require_label_review=True,
        require_deidentified=True,
        require_external_processing_approval=True,
    )


def _private_rows(audio_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, speaker in [
        ("train", "speaker-01"),
        ("validation", "speaker-02"),
        ("test", "speaker-03"),
    ]:
        rows.append(
            {
                "sample_id": f"{split}-001",
                "audio_path": str(audio_path),
                "reference_text": "프레스 설비를 확인합니다.",
                "split": split,
                "source": "approved-recording",
                "consent_status": "approved",
                "speaker_id": speaker,
                "scenario_id": "scenario-01",
                "noise_condition": "quiet",
                "approval_id": "approval-001",
                "deidentified": "true",
                "label_reviewer": "label-reviewer",
                "label_review_status": "reviewed",
            }
        )
    return rows


def test_strict_private_manifest_accepts_reviewed_speaker_disjoint_data(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(_private_rows(audio)).to_csv(manifest, index=False)

    prepared = prepare_manifest(
        manifest,
        tmp_path / "prepared.csv",
        backend="faster_whisper",
        governance=_strict_governance(tmp_path),
    )

    assert set(prepared["split"]) == {"train", "validation", "test"}


def test_strict_private_manifest_blocks_speaker_leakage(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    rows = _private_rows(audio)
    rows[1]["speaker_id"] = rows[0]["speaker_id"]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="must not cross"):
        prepare_manifest(
            manifest,
            tmp_path / "prepared.csv",
            backend="faster_whisper",
            governance=_strict_governance(tmp_path),
        )
