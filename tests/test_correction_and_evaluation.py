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


def test_alias_correction_emits_standard_acronyms_and_model_codes() -> None:
    predictions = pd.DataFrame(
        {
            "prediction_text": [
                "일 호기 피엘씨와 에이치엠아이에서 에이지브이 모델 큐 오백이를 확인합니다."
            ]
        }
    )
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    corrected = apply_term_correction(predictions, terms)

    assert corrected.loc[0, "prediction_text"] == (
        "1호기 PLC와 HMI에서 AGV 모델 Q502를 확인합니다."
    )
    assert corrected.loc[0, "correction_methods"] == "alias"


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


def test_information_retrieval_corrects_unlisted_near_match() -> None:
    predictions = pd.DataFrame({"prediction_text": ["콘베이아 모터를 점검합니다."]})
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    corrected = apply_term_correction(
        predictions,
        terms,
        alias_enabled=False,
        information_retrieval_enabled=True,
        min_ir_score=0.55,
        top_k=1,
        max_ngram_tokens=1,
    )

    assert corrected.loc[0, "prediction_text"].startswith("컨베이어")
    assert corrected.loc[0, "correction_methods"] == "ir"


def test_nearest_neighbor_correction_is_independently_selectable() -> None:
    predictions = pd.DataFrame({"prediction_text": ["프레스 삼오기를 확인합니다."]})
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    corrected = apply_term_correction(
        predictions,
        terms,
        alias_enabled=False,
        nearest_neighbor_enabled=True,
        min_nn_score=0.72,
        nn_backend="char_ngram",
        top_k=1,
        max_ngram_tokens=2,
    )

    assert corrected.loc[0, "prediction_text"].startswith("프레스 3호기")
    assert corrected.loc[0, "correction_methods"] == "nn"


def test_retrieval_features_can_be_disabled_without_changing_prediction() -> None:
    predictions = pd.DataFrame({"prediction_text": ["콘베이아 모터를 점검합니다."]})
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    unchanged = apply_term_correction(
        predictions,
        terms,
        alias_enabled=False,
        information_retrieval_enabled=False,
        nearest_neighbor_enabled=False,
    )

    assert unchanged.loc[0, "prediction_text"] == predictions.loc[0, "prediction_text"]
    assert unchanged.loc[0, "correction_count"] == 0


def test_hybrid_consensus_rejects_single_method_match() -> None:
    predictions = pd.DataFrame({"prediction_text": ["콘베이아 모터를 점검합니다."]})
    terms = load_domain_terms(ROOT / "data/domain_terms/manufacturing_terms.csv")

    corrected = apply_term_correction(
        predictions,
        terms,
        alias_enabled=False,
        information_retrieval_enabled=True,
        nearest_neighbor_enabled=True,
        min_ir_score=0.55,
        min_nn_score=1.0,
        require_consensus=True,
        top_k=1,
        max_ngram_tokens=1,
    )

    assert corrected.loc[0, "prediction_text"] == predictions.loc[0, "prediction_text"]
    assert corrected.loc[0, "correction_count"] == 0
