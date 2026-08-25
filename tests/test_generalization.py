import json
from pathlib import Path

import pandas as pd
import yaml

from aias_specialist.generalization import build_generalization_summary

ROOT = Path(__file__).resolve().parents[1]


def _result(tmp_path: Path, label: str, prediction_suffix: str = "") -> Path:
    run_dir = tmp_path / "runs" / label
    run_dir.mkdir(parents=True)
    rows = []
    for speaker_index in range(2):
        for sample_index in range(2):
            reference = "PLC 상태 정상" if sample_index == 0 else "일반 작업 정상"
            prediction = reference + prediction_suffix
            rows.append(
                {
                    "sample_id": f"{label}-{speaker_index}-{sample_index}",
                    "audio_path": f"audio/{label}-{speaker_index}-{sample_index}.wav",
                    "reference_text": reference,
                    "prediction_text": prediction,
                    "split": "test",
                    "speaker_id": f"{label}-speaker-{speaker_index}",
                }
            )
    frame = pd.DataFrame(rows)
    frame.drop(columns=["prediction_text"]).to_csv(
        run_dir / "prepared_manifest.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(
        run_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
    )
    terms = ROOT / "data/domain_terms/manufacturing_terms.csv"
    (run_dir / "domain_terms.snapshot.csv").write_bytes(terms.read_bytes())
    result = tmp_path / f"{label}.json"
    result.write_text(
        json.dumps({"run_id": label, "run_dir": str(run_dir)}), encoding="utf-8"
    )
    return result


def test_generalization_summary_uses_micro_aggregation_and_cluster_ci(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/local_benchmark_base.yaml").read_text(encoding="utf-8")
    )
    config["quality_targets"] = {
        "enabled": True,
        "minimum_domain_term_recall": 0.85,
        "maximum_cer": 0.07,
        "maximum_wer": 0.15,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    v1 = _result(tmp_path, "v1")
    v2 = _result(tmp_path, "v2")
    v3 = _result(tmp_path, "v3")

    result = build_generalization_summary(
        v1_result_path=v1,
        v2_result_path=v2,
        v3_result_path=v3,
        config_path=config_path,
        output_dir=tmp_path / "summary",
        bootstrap_resamples=100,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    comparison = pd.read_csv(payload["comparison_path"])
    assert list(comparison["cohort"]) == [
        "test_v1",
        "test_v2",
        "test_v3",
        "combined_v1_v2_v3",
    ]
    assert comparison.loc[3, "sample_count"] == 12
    assert comparison.loc[2, "evidence_role"] == "independent-confirmatory"
    intervals = pd.read_csv(
        tmp_path / "summary/generalization_confidence_intervals.csv"
    )
    assert set(intervals["metric"]) == {
        "domain_term_precision",
        "domain_term_recall",
        "domain_term_f1",
        "cer",
        "wer",
    }
