from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_settings
from .data import load_domain_terms
from .evaluation import evaluate_predictions
from .quality_targets import evaluate_quality_targets
from .utils import sha256_file, write_json

METRIC_KEYS = (
    "domain_term_precision",
    "domain_term_recall",
    "domain_term_f1",
    "cer",
    "wer",
)


def _read_result(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_dir_value = payload.get("confirmatory_run_dir") or payload.get("run_dir")
    if not run_dir_value:
        raise ValueError(f"Result does not contain a run directory: {path}")
    run_dir = Path(str(run_dir_value)).expanduser().resolve()
    return payload, run_dir


def _load_cohort(path: Path, label: str) -> tuple[pd.DataFrame, Path]:
    _, run_dir = _read_result(path)
    predictions_path = run_dir / "predictions_corrected.csv"
    manifest_path = run_dir / "prepared_manifest.csv"
    if not predictions_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing immutable predictions or manifest in {run_dir}")
    predictions = pd.read_csv(predictions_path, dtype=str).fillna("")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    if "speaker_id" not in predictions.columns:
        predictions = predictions.merge(
            manifest[["sample_id", "speaker_id"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    if predictions["speaker_id"].eq("").any():
        raise ValueError(f"Every prediction requires speaker_id for cluster bootstrap: {label}")
    predictions["cohort"] = label
    predictions["bootstrap_cluster"] = label + "::" + predictions["speaker_id"]
    return predictions, manifest_path


def _metric_row(
    label: str,
    role: str,
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cohort": label,
        "evidence_role": role,
        "sample_count": int(metrics["sample_count"]),
        "speaker_profile_count": int(frame["bootstrap_cluster"].nunique()),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        "true_positive": int(metrics["domain_term_true_positive"]),
        "false_positive": int(metrics["domain_term_false_positive"]),
        "false_negative": int(metrics["domain_term_false_negative"]),
        "quality_gate_pass": bool(gate["overall_pass"]),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    terms: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    clusters = list(frame["bootstrap_cluster"].drop_duplicates())
    if len(clusters) < 2:
        raise ValueError("Cluster bootstrap requires at least two speaker profiles")
    grouped = {key: value for key, value in frame.groupby("bootstrap_cluster")}
    rng = random.Random(seed)
    observed = evaluate_predictions(frame, terms)
    samples = {key: [] for key in METRIC_KEYS}
    for _ in range(resamples):
        selected = [rng.choice(clusters) for _ in clusters]
        resampled = pd.concat([grouped[key] for key in selected], ignore_index=True)
        metrics = evaluate_predictions(resampled, terms)
        for key in METRIC_KEYS:
            samples[key].append(float(metrics[key]))
    return {
        key: {
            "estimate": float(observed[key]),
            "lower_95": float(pd.Series(values).quantile(0.025)),
            "upper_95": float(pd.Series(values).quantile(0.975)),
        }
        for key, values in samples.items()
    }


def _paired_before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    terms: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    required = {"sample_id", "reference_text", "prediction_text", "speaker_id"}
    for name, frame in (("before", before), ("after", after)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Paired {name} predictions miss columns: {sorted(missing)}")
    before_ids = set(before["sample_id"])
    after_ids = set(after["sample_id"])
    if before_ids != after_ids:
        raise ValueError("Paired v2 comparison requires identical sample IDs")
    joined = before[list(required)].merge(
        after[list(required)],
        on="sample_id",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    if not joined["reference_text_before"].eq(joined["reference_text_after"]).all():
        raise ValueError("Paired v2 comparison requires identical references")
    before_metrics = evaluate_predictions(before, terms)
    after_metrics = evaluate_predictions(after, terms)
    rows = []
    for key in METRIC_KEYS:
        rows.append(
            {
                "metric": key,
                "before": float(before_metrics[key]),
                "after": float(after_metrics[key]),
                "delta_after_minus_before": float(after_metrics[key] - before_metrics[key]),
            }
        )

    clusters = list(after["speaker_id"].drop_duplicates())
    before_grouped = {key: value for key, value in before.groupby("speaker_id")}
    after_grouped = {key: value for key, value in after.groupby("speaker_id")}
    if set(before_grouped) != set(after_grouped):
        raise ValueError("Paired v2 comparison requires identical speaker profiles")
    rng = random.Random(seed)
    deltas = {key: [] for key in METRIC_KEYS}
    for _ in range(resamples):
        selected = [rng.choice(clusters) for _ in clusters]
        sampled_before = pd.concat([before_grouped[key] for key in selected], ignore_index=True)
        sampled_after = pd.concat([after_grouped[key] for key in selected], ignore_index=True)
        before_sample_metrics = evaluate_predictions(sampled_before, terms)
        after_sample_metrics = evaluate_predictions(sampled_after, terms)
        for key in METRIC_KEYS:
            deltas[key].append(
                float(after_sample_metrics[key] - before_sample_metrics[key])
            )
    intervals = {
        key: {
            "delta_estimate": float(after_metrics[key] - before_metrics[key]),
            "delta_lower_95": float(pd.Series(values).quantile(0.025)),
            "delta_upper_95": float(pd.Series(values).quantile(0.975)),
        }
        for key, values in deltas.items()
    }
    return pd.DataFrame(rows), intervals


def build_generalization_summary(
    *,
    v1_result_path: str | Path,
    v2_result_path: str | Path,
    v3_result_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    historical_v2_result_path: str | Path | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 20260827,
) -> Path:
    result_paths = {
        "test_v1": Path(v1_result_path).expanduser().resolve(),
        "test_v2": Path(v2_result_path).expanduser().resolve(),
        "test_v3": Path(v3_result_path).expanduser().resolve(),
    }
    settings = load_settings(config_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, Path] = {}
    for label, path in result_paths.items():
        frames[label], manifests[label] = _load_cohort(path, label)
    _, v1_run_dir = _read_result(result_paths["test_v1"])
    terms_path = v1_run_dir / "domain_terms.snapshot.csv"
    terms = load_domain_terms(terms_path)

    roles = {
        "test_v1": "historical-regression",
        "test_v2": "failure-informed-regression",
        "test_v3": "independent-confirmatory",
    }
    rows: list[dict[str, Any]] = []
    intervals: dict[str, Any] = {}
    for index, (label, frame) in enumerate(frames.items()):
        metrics = evaluate_predictions(frame, terms)
        gate = evaluate_quality_targets(metrics, settings.quality_targets)
        rows.append(_metric_row(label, roles[label], frame, metrics, gate))
        intervals[label] = _cluster_bootstrap(
            frame,
            terms,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index,
        )
    combined = pd.concat(frames.values(), ignore_index=True)
    combined_metrics = evaluate_predictions(combined, terms)
    combined_gate = evaluate_quality_targets(combined_metrics, settings.quality_targets)
    rows.append(
        _metric_row(
            "combined_v1_v2_v3",
            "micro-aggregate",
            combined,
            combined_metrics,
            combined_gate,
        )
    )
    intervals["combined_v1_v2_v3"] = _cluster_bootstrap(
        combined,
        terms,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 3,
    )
    comparison = pd.DataFrame(rows)
    comparison_path = output / "generalization_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    interval_rows = []
    for cohort, cohort_metrics in intervals.items():
        for metric, values in cohort_metrics.items():
            interval_rows.append({"cohort": cohort, "metric": metric, **values})
    pd.DataFrame(interval_rows).to_csv(
        output / "generalization_confidence_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_payload: dict[str, Any] | None = None
    if historical_v2_result_path is not None:
        historical_path = Path(historical_v2_result_path).expanduser().resolve()
        historical_v2, _ = _load_cohort(historical_path, "historical_test_v2")
        paired, paired_intervals = _paired_before_after(
            historical_v2,
            frames["test_v2"],
            terms,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 4,
        )
        paired.to_csv(
            output / "test_v2_before_after.csv", index=False, encoding="utf-8-sig"
        )
        paired_payload = {
            "historical_result": str(historical_path),
            "historical_result_sha256": sha256_file(historical_path),
            "confidence_intervals": paired_intervals,
        }

    payload = {
        "method": "speaker-profile cluster bootstrap and micro aggregation",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "quality_targets": {
            "minimum_domain_term_recall": settings.quality_targets.minimum_domain_term_recall,
            "maximum_cer": settings.quality_targets.maximum_cer,
            "maximum_wer": settings.quality_targets.maximum_wer,
        },
        "result_files": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in result_paths.items()
        },
        "manifest_files": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in manifests.items()
        },
        "comparison_path": str(comparison_path),
        "confidence_intervals": intervals,
        "paired_v2_before_after": paired_payload,
        "claim_boundary": (
            "Synthetic acoustic-profile generalization evidence only; real-factory "
            "production readiness requires approved shadow evaluation and human review."
        ),
    }
    result_path = output / "generalization_summary.json"
    write_json(result_path, payload)
    return result_path
