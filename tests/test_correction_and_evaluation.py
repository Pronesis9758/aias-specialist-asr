from pathlib import Path

import pandas as pd

from aias_specialist.correction import apply_term_correction
from aias_specialist.data import load_domain_terms
from aias_specialist.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]


def test_domain_term_correction_improves_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "reference_text": ["컨베이어 모터 전류를 확인해 주세요."],
            "prediction_text": ["콘베이어 모터 전류를 확인해 주세요."],
            "latency_seconds": [0.1],
            "real_time_factor": [0.05],
        }
    )
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")
    baseline = evaluate_predictions(predictions, terms)
    corrected_frame = apply_term_correction(predictions, terms)
    corrected = evaluate_predictions(corrected_frame, terms)

    assert corrected_frame.loc[0, "prediction_text"].startswith("컨베이어")
    assert corrected["wer"] < baseline["wer"]
    assert corrected["domain_term_recall"] > baseline["domain_term_recall"]


def test_domain_term_recall_is_not_applicable_without_terms() -> None:
    predictions = pd.DataFrame(
        {
            "reference_text": ["오늘 날씨가 맑습니다."],
            "prediction_text": ["오늘 날씨가 맑습니다."],
            "latency_seconds": [0.1],
            "real_time_factor": [0.05],
        }
    )
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    metrics = evaluate_predictions(predictions, terms)

    assert metrics["domain_terms_expected"] == 0
    assert metrics["domain_term_recall_applicable"] is False
