from pathlib import Path

import pandas as pd

from aias_specialist.error_mining import mine_domain_term_errors


def test_error_mining_finds_observed_term_confusion(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    terms = tmp_path / "terms.csv"
    output = tmp_path / "errors.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "v1",
                "reference_text": "체결 토크 상태를 확인합니다",
                "prediction_text": "체별 토크 상태를 확인합니다",
            }
        ]
    ).to_csv(predictions, index=False)
    pd.DataFrame(
        [{"canonical": "체결 토크", "aliases": "체결토크", "category": "조립"}]
    ).to_csv(terms, index=False)

    mine_domain_term_errors(predictions, terms, output)

    result = pd.read_csv(output)
    assert result.iloc[0]["canonical_term"] == "체결 토크"
    assert result.iloc[0]["observed_form"] == "체별 토크"
    assert result.iloc[0]["occurrence_count"] == 1
