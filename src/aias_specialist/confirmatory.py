from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_settings
from .data import load_domain_terms
from .evaluation import evaluate_predictions
from .pipeline import run_pipeline
from .quality_targets import evaluate_quality_targets
from .utils import sha256_file, write_json


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def _selection_payload(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("model"), dict):
        raise ValueError(f"Invalid selection file: {path}")
    return selection


def _read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "audio_path", "reference_text", "split", "speaker_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns {sorted(missing)}: {path}")
    return frame


def validate_confirmatory_cohort(
    reference_manifest_path: str | Path,
    confirmatory_manifest_path: str | Path,
    *,
    expected_samples: int = 600,
    minimum_speakers: int = 30,
    minimum_term_occurrences: int = 1000,
    minimum_negative_samples: int = 100,
) -> dict[str, Any]:
    reference_path = Path(reference_manifest_path).expanduser().resolve()
    confirmatory_path = Path(confirmatory_manifest_path).expanduser().resolve()
    reference = _read_manifest(reference_path)
    confirmatory = _read_manifest(confirmatory_path)
    reference_test = reference.loc[reference["split"].str.lower().eq("test")].copy()

    if len(confirmatory) != expected_samples:
        raise ValueError(
            f"Confirmatory cohort must contain {expected_samples} samples: {len(confirmatory)}"
        )
    if set(confirmatory["split"].str.lower()) != {"test"}:
        raise ValueError("Confirmatory manifest may contain only the test split")
    confirmatory_speaker_count = int(confirmatory["speaker_id"].nunique())
    if confirmatory_speaker_count < minimum_speakers:
        raise ValueError(
            f"Confirmatory cohort requires at least {minimum_speakers} speakers: "
            f"{confirmatory_speaker_count}"
        )
    for column in ("sample_id", "audio_path", "reference_text"):
        if confirmatory[column].duplicated().any():
            raise ValueError(f"Confirmatory manifest has duplicate {column} values")

    overlap_checks = {
        "sample_id": set(reference["sample_id"]) & set(confirmatory["sample_id"]),
        "reference_text": set(reference["reference_text"])
        & set(confirmatory["reference_text"]),
        "speaker_id": set(reference["speaker_id"]) & set(confirmatory["speaker_id"]),
    }
    for label, overlap in overlap_checks.items():
        if overlap:
            raise ValueError(
                f"Confirmatory cohort overlaps v1 by {label}: {sorted(overlap)[:3]}"
            )

    audio_hash_overlap: set[str] = set()
    if "audio_sha256" in reference.columns and "audio_sha256" in confirmatory.columns:
        reference_hashes = {value for value in reference["audio_sha256"] if value}
        confirmatory_hashes = {value for value in confirmatory["audio_sha256"] if value}
        if len(confirmatory_hashes) != len(confirmatory):
            raise ValueError("Confirmatory manifest requires a unique audio_sha256 per sample")
        audio_hash_overlap = reference_hashes & confirmatory_hashes
        if audio_hash_overlap:
            raise ValueError("Confirmatory audio content overlaps v1")

    acoustic_overlap: set[tuple[str, str, str]] = set()
    acoustic_columns = ["tts_voice", "tts_rate", "tts_pitch"]
    if all(column in reference.columns for column in acoustic_columns) and all(
        column in confirmatory.columns for column in acoustic_columns
    ):
        reference_profiles = set(
            reference[acoustic_columns].itertuples(index=False, name=None)
        )
        confirmatory_profiles = set(
            confirmatory[acoustic_columns].itertuples(index=False, name=None)
        )
        acoustic_overlap = reference_profiles & confirmatory_profiles
        if acoustic_overlap:
            raise ValueError("Confirmatory acoustic speaker profiles overlap v1")

    negative_samples = 0
    term_occurrences = 0
    if "term_targets" in confirmatory.columns:
        negative_samples = int(confirmatory["term_targets"].eq("").sum())
        term_occurrences = sum(
            len([term for term in value.split("|") if term])
            for value in confirmatory["term_targets"]
        )
    if term_occurrences < minimum_term_occurrences:
        raise ValueError(
            f"Confirmatory cohort requires at least {minimum_term_occurrences} term "
            f"occurrences: {term_occurrences}"
        )
    if negative_samples < minimum_negative_samples:
        raise ValueError(
            f"Confirmatory cohort requires at least {minimum_negative_samples} negative "
            f"samples: {negative_samples}"
        )

    return {
        "reference_test_samples": int(len(reference_test)),
        "confirmatory_test_samples": int(len(confirmatory)),
        "reference_speakers": int(reference["speaker_id"].nunique()),
        "confirmatory_speakers": confirmatory_speaker_count,
        "sample_id_overlap": 0,
        "reference_text_overlap": 0,
        "speaker_id_overlap": 0,
        "audio_hash_overlap": len(audio_hash_overlap),
        "acoustic_profile_overlap": len(acoustic_overlap),
        "negative_samples": negative_samples,
        "term_occurrences": term_occurrences,
        "status": "PASS",
    }


def _safe_cohort_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip()).strip("-._")
    if not cleaned:
        raise ValueError("cohort_id must contain at least one safe character")
    return cleaned


def _metric_row(
    cohort: str,
    metrics: dict[str, Any],
    quality_gate: dict[str, Any],
    speaker_count: int,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "sample_count": int(metrics["sample_count"]),
        "speaker_count": int(speaker_count),
        "domain_term_precision": float(metrics["domain_term_precision"]),
        "domain_term_recall": float(metrics["domain_term_recall"]),
        "domain_term_f1": float(metrics["domain_term_f1"]),
        "cer": float(metrics["cer"]),
        "wer": float(metrics["wer"]),
        "true_positive": int(metrics["domain_term_true_positive"]),
        "false_positive": int(metrics["domain_term_false_positive"]),
        "false_negative": int(metrics["domain_term_false_negative"]),
        "quality_gate_pass": bool(quality_gate["overall_pass"]),
    }


def _write_summary(path: Path, comparison: pd.DataFrame, audit: dict[str, Any]) -> None:
    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            f"| {row.cohort} | {row.sample_count} | {row.speaker_count} | "
            f"{row.domain_term_precision:.2%} | {row.domain_term_recall:.2%} | "
            f"{row.domain_term_f1:.2%} | {row.cer:.2%} | {row.wer:.2%} | "
            f"{'PASS' if row.quality_gate_pass else 'FAIL'} |"
        )
    text = """# Speaker-held-out confirmatory Test

The v1 model, LoRA adapter, decoding, correction, precision, and domain-term snapshot
were frozen before v2 inference. No v2 result is used for model or threshold selection.

| Cohort | Samples | Speakers | Precision | Recall | F1 | CER | WER | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows)
    text += (
        "\n\n## Cohort separation audit\n\n"
        f"- Status: {audit['status']}\n"
        f"- Speaker overlap: {audit['speaker_id_overlap']}\n"
        f"- Sentence overlap: {audit['reference_text_overlap']}\n"
        f"- Audio hash overlap: {audit['audio_hash_overlap']}\n"
        f"- Confirmatory negative samples: {audit['negative_samples']}\n"
        f"- Confirmatory manufacturing-term occurrences: {audit['term_occurrences']}\n"
    )
    path.write_text(text, encoding="utf-8")


def run_confirmatory_evaluation(
    selection_path: str | Path,
    base_config_path: str | Path,
    confirmatory_manifest_path: str | Path,
    *,
    cohort_id: str = "speaker-heldout-v2",
    expected_samples: int = 600,
    minimum_speakers: int = 30,
    minimum_term_occurrences: int = 1000,
    minimum_negative_samples: int = 100,
    reference_result_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_path = Path(selection_path).expanduser().resolve()
    base_path = Path(base_config_path).expanduser().resolve()
    cohort_manifest = Path(confirmatory_manifest_path).expanduser().resolve()
    cohort_name = _safe_cohort_id(cohort_id)
    reference_result = (
        Path(reference_result_path).expanduser().resolve()
        if reference_result_path
        else selected_path.parent / "final_test_result.json"
    )
    if not reference_result.exists():
        raise FileNotFoundError(
            "The immutable v1 final_test_result.json is required before confirmatory v2"
        )

    output_dir = selected_path.parent / "confirmatory" / cohort_name
    result_path = output_dir / "confirmatory_test_result.json"
    protocol_path = output_dir / "protocol_lock.json"
    reference_payload = json.loads(reference_result.read_text(encoding="utf-8"))
    reference_run_dir = Path(str(reference_payload["run_dir"])).expanduser().resolve()
    reference_manifest = reference_run_dir / "prepared_manifest.csv"
    reference_terms = reference_run_dir / "domain_terms.snapshot.csv"
    reference_predictions = reference_run_dir / "predictions_corrected.csv"
    for required_path in (reference_manifest, reference_terms, reference_predictions):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing immutable v1 artifact: {required_path}")

    audit = validate_confirmatory_cohort(
        reference_manifest,
        cohort_manifest,
        expected_samples=expected_samples,
        minimum_speakers=minimum_speakers,
        minimum_term_occurrences=minimum_term_occurrences,
        minimum_negative_samples=minimum_negative_samples,
    )
    lock = {
        "cohort_id": cohort_name,
        "selection_sha256": sha256_file(selected_path),
        "base_config_sha256": sha256_file(base_path),
        "reference_result_sha256": sha256_file(reference_result),
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "reference_domain_terms_sha256": sha256_file(reference_terms),
        "confirmatory_manifest_sha256": sha256_file(cohort_manifest),
        "expected_samples": expected_samples,
        "minimum_speakers": minimum_speakers,
        "minimum_term_occurrences": minimum_term_occurrences,
        "minimum_negative_samples": minimum_negative_samples,
        "frozen_after_v1": True,
        "v2_tuning_permitted": False,
    }
    if result_path.exists():
        if (
            not protocol_path.exists()
            or json.loads(protocol_path.read_text(encoding="utf-8")) != lock
        ):
            raise ValueError("Existing confirmatory result has a different protocol identity")
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        cached["cache_reused"] = True
        print("[confirmatory-test] immutable v2 result reused without inference", flush=True)
        return cached

    output_dir.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        existing_lock = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_lock != lock:
            raise ValueError("Confirmatory cohort ID already has a different protocol lock")
    else:
        write_json(protocol_path, lock)
    write_json(output_dir / "cohort_separation_audit.json", audit)

    selection = _selection_payload(selected_path)
    settings = load_settings(base_path)
    raw = deepcopy(settings.raw)
    raw["paths"] = {
        "manifest": str(cohort_manifest),
        "domain_terms": str(reference_terms),
        "artifacts_dir": str(settings.paths.artifacts_dir),
        "database": str(settings.paths.database),
        "model_lock": str(settings.paths.model_lock),
    }
    raw["model"] = deepcopy(selection["model"])
    raw["evaluation"] = {**raw.get("evaluation", {}), "split": "test"}
    raw["training"] = {
        "enabled": False,
        "reason": "speaker-held-out confirmatory test with frozen v1 selection",
    }
    raw["dataset"] = {
        **raw.get("dataset", {}),
        "repo_id": f"local/{cohort_name}",
        "revision": lock["confirmatory_manifest_sha256"],
        "test_samples": expected_samples,
        "confirmatory_against_run_id": str(reference_payload["run_id"]),
    }
    raw.setdefault("report", {})["title"] = (
        f"화자 완전 분리 확인 Test - {cohort_name}"
    )
    config_path = output_dir / "confirmatory_test_config.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    run_result = run_pipeline(load_settings(config_path))
    confirmatory_predictions = run_result.run_dir / "predictions_corrected.csv"
    v1_frame = pd.read_csv(reference_predictions, dtype=str).fillna("")
    v2_frame = pd.read_csv(confirmatory_predictions, dtype=str).fillna("")
    terms = load_domain_terms(reference_terms)
    v1_metrics = evaluate_predictions(v1_frame, terms)
    v2_metrics = evaluate_predictions(v2_frame, terms)
    combined_metrics = evaluate_predictions(
        pd.concat([v1_frame, v2_frame], ignore_index=True), terms
    )
    v1_gate = evaluate_quality_targets(v1_metrics, settings.quality_targets)
    v2_gate = evaluate_quality_targets(v2_metrics, settings.quality_targets)
    combined_gate = evaluate_quality_targets(combined_metrics, settings.quality_targets)
    reference_speakers = _read_manifest(reference_manifest)
    reference_speakers = reference_speakers.loc[
        reference_speakers["split"].str.lower().eq("test")
    ]
    confirmatory_speakers = _read_manifest(cohort_manifest)
    comparison = pd.DataFrame(
        [
            _metric_row(
                "test_v1",
                v1_metrics,
                v1_gate,
                reference_speakers["speaker_id"].nunique(),
            ),
            _metric_row(
                "test_v2",
                v2_metrics,
                v2_gate,
                confirmatory_speakers["speaker_id"].nunique(),
            ),
            _metric_row(
                "combined_v1_v2",
                combined_metrics,
                combined_gate,
                pd.concat(
                    [reference_speakers["speaker_id"], confirmatory_speakers["speaker_id"]]
                ).nunique(),
            ),
        ]
    )
    comparison_path = output_dir / "v1_v2_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "confirmatory_summary.md"
    _write_summary(summary_path, comparison, audit)

    summary = {
        "cohort_id": cohort_name,
        "selection_path": str(selected_path),
        "protocol_lock_path": str(protocol_path),
        "cohort_audit_path": str(output_dir / "cohort_separation_audit.json"),
        "config_path": str(config_path),
        "reference_result_path": str(reference_result),
        "reference_run_id": str(reference_payload["run_id"]),
        "confirmatory_run_id": run_result.run_id,
        "confirmatory_run_dir": str(run_result.run_dir),
        "confirmatory_report_path": str(run_result.report_path),
        "comparison_path": str(comparison_path),
        "summary_path": str(summary_path),
        "metrics": {
            "test_v1": v1_metrics,
            "test_v2": v2_metrics,
            "combined_v1_v2": combined_metrics,
        },
        "quality_gates": {
            "test_v1": v1_gate,
            "test_v2": v2_gate,
            "combined_v1_v2": combined_gate,
        },
        "selection_frozen": True,
        "v2_result_used_for_tuning": False,
        "human_review_required": True,
    }
    write_json(result_path, summary)
    return summary
