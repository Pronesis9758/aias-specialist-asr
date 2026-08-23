from pathlib import Path

import pandas as pd
import pytest

from aias_specialist.config import QualityTargetsConfig
from aias_specialist.experiments import _write_comparison
from aias_specialist.quality_targets import evaluate_quality_targets


def test_quality_targets_require_all_three_manufacturing_metrics() -> None:
    targets = QualityTargetsConfig(enabled=True)

    passed = evaluate_quality_targets(
        {
            "domain_term_recall": 0.85,
            "domain_term_recall_applicable": True,
            "cer": 0.07,
            "wer": 0.15,
        },
        targets,
    )
    failed = evaluate_quality_targets(
        {
            "domain_term_recall": 0.84,
            "domain_term_recall_applicable": True,
            "cer": 0.06,
            "wer": 0.14,
        },
        targets,
    )

    assert passed["overall_pass"] is True
    assert passed["miss_count"] == 0
    assert failed["overall_pass"] is False
    assert failed["miss_count"] == 1
    assert failed["checks"]["domain_term_recall"]["gap"] == pytest.approx(0.01)


def test_comparison_ranking_prioritizes_term_recall_target_gap(tmp_path: Path) -> None:
    common = {
        "status": "completed",
        "quality_targets_enabled": True,
        "quality_target_pass": False,
        "quality_target_miss_count": 3,
        "cer_gap": 0.1,
        "wer_gap": 0.2,
        "aggregate_real_time_factor": 0.1,
    }
    rows = {
        "lower-cer": {
            **common,
            "member_id": "lower-cer",
            "domain_term_recall": 0.50,
            "domain_term_recall_gap": 0.35,
            "cer": 0.10,
            "wer": 0.20,
        },
        "higher-recall": {
            **common,
            "member_id": "higher-recall",
            "domain_term_recall": 0.70,
            "domain_term_recall_gap": 0.15,
            "cer": 0.15,
            "wer": 0.25,
        },
    }

    comparison = _write_comparison(rows, tmp_path / "comparison.csv")

    ranked = comparison.sort_values("rank")
    assert ranked.iloc[0]["member_id"] == "higher-recall"
    assert pd.to_numeric(ranked.iloc[0]["rank"]) == 1
