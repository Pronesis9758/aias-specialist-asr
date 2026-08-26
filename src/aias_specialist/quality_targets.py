from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .config import QualityTargetsConfig


def evaluate_quality_targets(
    metrics: Mapping[str, Any],
    targets: QualityTargetsConfig,
) -> dict[str, Any]:
    """Evaluate ASR metrics against the manufacturing quality contract."""
    recall = float(metrics.get("domain_term_recall", 0.0))
    cer = float(metrics.get("cer", 0.0))
    wer = float(metrics.get("wer", 0.0))
    recall_applicable = bool(metrics.get("domain_term_recall_applicable", True))

    recall_gap = max(targets.minimum_domain_term_recall - recall, 0.0)
    cer_gap = max(cer - targets.maximum_cer, 0.0)
    wer_gap = max(wer - targets.maximum_wer, 0.0)
    recall_pass = recall_applicable and recall_gap == 0.0
    cer_pass = cer_gap == 0.0
    wer_pass = wer_gap == 0.0
    enabled = targets.enabled
    overall_pass = enabled and recall_pass and cer_pass and wer_pass

    return {
        "enabled": enabled,
        "priority": list(targets.priority),
        "targets": asdict(targets),
        "observed": {
            "domain_term_recall": recall,
            "cer": cer,
            "wer": wer,
        },
        "checks": {
            "domain_term_recall": {
                "passed": recall_pass,
                "gap": recall_gap,
                "applicable": recall_applicable,
            },
            "cer": {"passed": cer_pass, "gap": cer_gap},
            "wer": {"passed": wer_pass, "gap": wer_gap},
        },
        "miss_count": int(not recall_pass) + int(not cer_pass) + int(not wer_pass),
        "overall_pass": overall_pass,
        "status": "passed" if overall_pass else "not_configured" if not enabled else "failed",
    }


def quality_target_columns(
    metrics: Mapping[str, Any],
    targets: QualityTargetsConfig,
) -> dict[str, Any]:
    """Flatten the target assessment for experiment comparison CSV files."""
    result = evaluate_quality_targets(metrics, targets)
    checks = result["checks"]
    return {
        "quality_targets_enabled": targets.enabled,
        "target_domain_term_recall": targets.minimum_domain_term_recall,
        "target_maximum_cer": targets.maximum_cer,
        "target_maximum_wer": targets.maximum_wer,
        "domain_term_recall_target_pass": checks["domain_term_recall"]["passed"],
        "cer_target_pass": checks["cer"]["passed"],
        "wer_target_pass": checks["wer"]["passed"],
        "domain_term_recall_gap": checks["domain_term_recall"]["gap"],
        "cer_gap": checks["cer"]["gap"],
        "wer_gap": checks["wer"]["gap"],
        "quality_target_miss_count": result["miss_count"],
        "quality_target_pass": result["overall_pass"],
    }
