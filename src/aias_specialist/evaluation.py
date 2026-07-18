from __future__ import annotations

import re
from typing import Any

import pandas as pd
from jiwer import cer, wer


def normalize_text(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", str(value))
    return re.sub(r"\s+", " ", value).strip().lower()


def _term_recall(frame: pd.DataFrame, terms: pd.DataFrame) -> tuple[float, int, int]:
    canonical_terms = [normalize_text(item) for item in terms["canonical"].astype(str)]
    expected = 0
    detected = 0
    for row in frame.itertuples(index=False):
        reference = normalize_text(row.reference_text)
        prediction = normalize_text(row.prediction_text)
        for term in canonical_terms:
            if term and term in reference:
                expected += 1
                if term in prediction:
                    detected += 1
    recall = detected / expected if expected else 1.0
    return recall, detected, expected


def evaluate_predictions(frame: pd.DataFrame, terms: pd.DataFrame) -> dict[str, Any]:
    references = [normalize_text(item) for item in frame["reference_text"].astype(str)]
    predictions = [normalize_text(item) for item in frame["prediction_text"].astype(str)]
    term_recall, detected_terms, expected_terms = _term_recall(frame, terms)
    latencies = pd.to_numeric(frame.get("latency_seconds", 0.0), errors="coerce").fillna(0.0)
    rtfs = pd.to_numeric(frame.get("real_time_factor", 0.0), errors="coerce").fillna(0.0)

    return {
        "sample_count": int(len(frame)),
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
        "domain_term_recall": float(term_recall),
        "domain_term_recall_applicable": bool(expected_terms > 0),
        "domain_terms_detected": int(detected_terms),
        "domain_terms_expected": int(expected_terms),
        "mean_latency_seconds": float(latencies.mean()),
        "p95_latency_seconds": float(latencies.quantile(0.95)),
        "mean_real_time_factor": float(rtfs.mean()),
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
