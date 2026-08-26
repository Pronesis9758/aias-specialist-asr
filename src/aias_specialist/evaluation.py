from __future__ import annotations

import re
from typing import Any

import pandas as pd
from jiwer import cer, wer


def normalize_text(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", str(value))
    return re.sub(r"\s+", " ", value).strip().lower()


def _term_metrics(frame: pd.DataFrame, terms: pd.DataFrame) -> dict[str, Any]:
    canonical_terms = [normalize_text(item) for item in terms["canonical"].astype(str)]
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for row in frame.itertuples(index=False):
        reference = normalize_text(row.reference_text)
        prediction = normalize_text(row.prediction_text)
        for term in canonical_terms:
            if not term:
                continue
            expected = term in reference
            predicted = term in prediction
            true_positive += int(expected and predicted)
            false_positive += int(not expected and predicted)
            false_negative += int(expected and not predicted)
    predicted_count = true_positive + false_positive
    expected_count = true_positive + false_negative
    precision = true_positive / predicted_count if predicted_count else 1.0
    recall = true_positive / expected_count if expected_count else 1.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "domain_term_precision": float(precision),
        "domain_term_precision_applicable": bool(predicted_count > 0),
        "domain_term_recall": float(recall),
        "domain_term_recall_applicable": bool(expected_count > 0),
        "domain_term_f1": float(f1_score),
        "domain_term_true_positive": int(true_positive),
        "domain_term_false_positive": int(false_positive),
        "domain_term_false_negative": int(false_negative),
        "domain_terms_predicted": int(predicted_count),
        "domain_terms_detected": int(true_positive),
        "domain_terms_expected": int(expected_count),
    }


def _numeric_column(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    values = frame[name] if name in frame.columns else pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)


def evaluate_predictions(frame: pd.DataFrame, terms: pd.DataFrame) -> dict[str, Any]:
    references = [normalize_text(item) for item in frame["reference_text"].astype(str)]
    predictions = [normalize_text(item) for item in frame["prediction_text"].astype(str)]
    term_metrics = _term_metrics(frame, terms)
    latencies = _numeric_column(frame, "latency_seconds")
    rtfs = _numeric_column(frame, "real_time_factor")
    durations = _numeric_column(frame, "audio_duration_seconds")
    model_sizes = _numeric_column(frame, "model_size_bytes")
    process_memory = _numeric_column(frame, "process_rss_mb")
    gpu_memory = _numeric_column(frame, "gpu_memory_mb")
    preparation = _numeric_column(frame, "model_preparation_seconds")
    total_duration = float(durations.sum())
    total_latency = float(latencies.sum())

    return {
        "sample_count": int(len(frame)),
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
        **term_metrics,
        "mean_latency_seconds": float(latencies.mean()),
        "p95_latency_seconds": float(latencies.quantile(0.95)),
        "mean_real_time_factor": float(rtfs.mean()),
        "aggregate_real_time_factor": (
            total_latency / total_duration if total_duration > 0 else 0.0
        ),
        "audio_duration_seconds": total_duration,
        "evaluation_runtime_seconds": total_latency,
        "model_preparation_seconds": float(preparation.max()),
        "model_size_bytes": int(model_sizes.max()),
        "peak_process_memory_mb": float(process_memory.max()),
        "peak_gpu_memory_mb": float(gpu_memory.max()),
    }


def per_sample_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["sample_wer"] = [
        wer(normalize_text(ref), normalize_text(hyp))
        for ref, hyp in zip(output["reference_text"], output["prediction_text"], strict=True)
    ]
    output["sample_cer"] = [
        cer(normalize_text(ref), normalize_text(hyp))
        for ref, hyp in zip(output["reference_text"], output["prediction_text"], strict=True)
    ]
    return output


def compare_metrics(baseline: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    term_recall_applicable = bool(
        baseline.get("domain_term_recall_applicable")
        and corrected.get("domain_term_recall_applicable")
    )
    return {
        "wer_absolute_reduction": float(baseline["wer"] - corrected["wer"]),
        "cer_absolute_reduction": float(baseline["cer"] - corrected["cer"]),
        "domain_term_recall_gain": float(
            corrected["domain_term_recall"] - baseline["domain_term_recall"]
        ),
        "domain_term_recall_applicable": term_recall_applicable,
    }
