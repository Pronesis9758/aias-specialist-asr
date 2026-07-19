from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def prepare_manifest(manifest_path: Path, output_path: Path, backend: str) -> pd.DataFrame:
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
