from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import GovernanceConfig

REQUIRED_COLUMNS = {
    "sample_id",
    "audio_path",
    "reference_text",
    "split",
    "source",
    "consent_status",
}
APPROVED_CONSENT = {"approved", "synthetic", "public"}
ALLOWED_SPLITS = {"train", "validation", "test"}
STRICT_PRIVATE_COLUMNS = {
    "speaker_id",
    "scenario_id",
    "noise_condition",
    "approval_id",
    "deidentified",
    "label_reviewer",
    "label_review_status",
}
APPROVAL_REQUIRED_FIELDS = {
    "approval_id",
    "data_owner",
    "approver",
    "approved_at",
    "purpose",
    "retention_until",
    "storage_location",
    "external_processing_allowed",
    "deidentification_reviewed",
    "prohibited_content_reviewed",
    "deletion_owner",
}
TRUE_VALUES = {"true", "yes", "y", "1"}
REVIEWED_VALUES = {"approved", "reviewed"}


def _is_placeholder(value: object) -> bool:
    normalized = str(value).strip().lower()
    return not normalized or normalized in {
        "to_be_completed",
        "to_be_decided",
        "required",
        "tbd",
        "미정",
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def _load_approval(governance: GovernanceConfig) -> dict[str, Any]:
    if governance.approval_file is None:
        raise ValueError("strict_private governance requires governance.approval_file")
    if not governance.approval_file.exists():
        raise FileNotFoundError(f"Data approval file not found: {governance.approval_file}")
    payload = yaml.safe_load(governance.approval_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Data approval file must contain a YAML mapping")
    approval = payload.get("data_approval", payload)
    if not isinstance(approval, dict):
        raise ValueError("data_approval must be a YAML mapping")
    missing = sorted(
        field
        for field in APPROVAL_REQUIRED_FIELDS
        if field not in approval or _is_placeholder(approval[field])
    )
    if missing:
        raise ValueError(f"Data approval is incomplete: {', '.join(missing)}")
    try:
        approved_at = date.fromisoformat(str(approval["approved_at"]))
        retention_until = date.fromisoformat(str(approval["retention_until"]))
    except ValueError as exc:
        raise ValueError("approved_at and retention_until must use YYYY-MM-DD") from exc
    if retention_until < approved_at:
        raise ValueError("retention_until must not be earlier than approved_at")
    if retention_until < date.today():
        raise ValueError("Data approval retention period has expired")
    if governance.require_external_processing_approval and not _as_bool(
        approval["external_processing_allowed"]
    ):
        raise ValueError("External processing approval is required for this config")
    if governance.require_deidentified and not _as_bool(approval["deidentification_reviewed"]):
        raise ValueError("De-identification review approval is required")
    if not _as_bool(approval["prohibited_content_reviewed"]):
        raise ValueError("Prohibited-content review approval is required")
    return approval


def _validate_strict_private(
    frame: pd.DataFrame,
    governance: GovernanceConfig,
) -> None:
    required = STRICT_PRIVATE_COLUMNS | set(governance.required_manifest_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Strict private manifest is missing governance columns: {', '.join(missing)}"
        )
    for column in required:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Strict private manifest column '{column}' contains blanks")

    approval = _load_approval(governance)
    expected_approval = str(approval["approval_id"]).strip()
    manifest_approvals = set(frame["approval_id"].str.strip())
    if manifest_approvals != {expected_approval}:
        raise ValueError("Manifest approval_id must match the completed data approval file")
    if set(frame["consent_status"].str.strip().str.lower()) != {"approved"}:
        raise ValueError("Strict private samples must use consent_status=approved")
    if governance.require_deidentified:
        invalid = ~frame["deidentified"].str.strip().str.lower().isin(TRUE_VALUES)
        if invalid.any():
            raise ValueError("Every private sample must be marked deidentified=true")
    if governance.require_label_review:
        invalid = ~frame["label_review_status"].str.strip().str.lower().isin(REVIEWED_VALUES)
        if invalid.any():
            raise ValueError("Every transcript must have label_review_status=reviewed")
    if governance.require_speaker_disjoint_splits:
        split_counts = (
            frame.assign(
                speaker_id=frame["speaker_id"].str.strip(),
                split=frame["split"].str.strip().str.lower(),
            )
            .groupby("speaker_id")["split"]
            .nunique()
        )
        overlapping = split_counts.loc[split_counts > 1].index.tolist()
        if overlapping:
            raise ValueError(
                f"speaker_id values must not cross train/validation/test splits: {overlapping[:10]}"
            )


def validate_manifest(
    manifest_path: Path,
    backend: str,
    governance: GovernanceConfig | None = None,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Copy data/sample/manifest.csv to the configured "
            "private path and replace it with reviewed records."
        )

    frame = pd.read_csv(manifest_path, dtype=str).fillna("")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Manifest contains no samples")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values: {duplicates[:10]}")
    if (frame["reference_text"].str.strip() == "").any():
        raise ValueError("reference_text must be non-empty for every sample")

    unapproved = sorted(set(frame["consent_status"].str.lower()) - APPROVED_CONSENT)
    if unapproved:
        raise ValueError(
            "Unapproved consent_status values found: "
            f"{unapproved}. Allowed values: {sorted(APPROVED_CONSENT)}"
        )

    invalid_splits = sorted(set(frame["split"].str.lower()) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(
            f"Unsupported split values found: {invalid_splits}. "
            f"Allowed values: {sorted(ALLOWED_SPLITS)}"
        )

    if backend == "fixture":
        if "fixture_prediction" not in frame.columns:
            raise ValueError("fixture backend requires a fixture_prediction column")
    else:
        missing_audio: list[str] = []
        resolved_audio: list[str] = []
        for raw_path in frame["audio_path"]:
            audio = Path(raw_path).expanduser()
            if not audio.is_absolute():
                audio = (manifest_path.parent / audio).resolve()
            resolved_audio.append(str(audio))
            if not audio.exists():
                missing_audio.append(str(audio))
        if missing_audio:
            preview = "\n".join(missing_audio[:10])
            raise FileNotFoundError(f"Audio files are missing:\n{preview}")
        frame["audio_path"] = resolved_audio

    if governance and governance.mode == "strict_private":
        _validate_strict_private(frame, governance)
    return frame


def prepare_manifest(
    manifest_path: Path,
    output_path: Path,
    backend: str,
    governance: GovernanceConfig | None = None,
) -> pd.DataFrame:
    frame = validate_manifest(manifest_path, backend, governance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def load_domain_terms(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Domain term dictionary not found: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"canonical", "aliases", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Domain term dictionary is missing columns: {', '.join(missing)}")
    if frame["canonical"].str.strip().eq("").any():
        raise ValueError("Domain term canonical values must be non-empty")
    return frame
