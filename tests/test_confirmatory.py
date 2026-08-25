import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from aias_specialist.confirmatory import (
    run_confirmatory_evaluation,
    validate_confirmatory_cohort,
)

ROOT = Path(__file__).resolve().parents[1]


def _manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _row(prefix: str, index: int, reference: str, speaker: str) -> dict[str, str]:
    target = next((term for term in ("PLC", "HMI", "AGV") if term in reference), "")
    return {
        "sample_id": f"{prefix}-{index}",
        "audio_path": f"audio/{prefix}-{index}.wav",
        "audio_sha256": f"{prefix}-hash-{index}",
        "reference_text": reference,
        "split": "test",
        "speaker_id": speaker,
        "tts_voice": f"voice-{prefix}",
        "tts_rate": f"{index:+d}%",
        "tts_pitch": f"{index:+d}Hz",
        "term_targets": target,
    }


def test_confirmatory_cohort_requires_disjoint_speakers_sentences_and_audio(
    tmp_path: Path,
) -> None:
    v1 = _manifest(
        tmp_path / "v1.csv",
        [_row("v1", 1, "PLC 상태를 확인합니다.", "speaker-v1")],
    )
    v2 = _manifest(
        tmp_path / "v2.csv",
        [_row("v2", 1, "HMI 상태를 점검합니다.", "speaker-v2")],
    )

    audit = validate_confirmatory_cohort(
        v1,
        v2,
        expected_samples=1,
        minimum_speakers=1,
        minimum_term_occurrences=1,
        minimum_negative_samples=0,
    )

    assert audit["status"] == "PASS"
    assert audit["speaker_id_overlap"] == 0
    assert audit["reference_text_overlap"] == 0
    assert audit["audio_hash_overlap"] == 0

    overlapped = pd.read_csv(v2, dtype=str)
    overlapped.loc[0, "speaker_id"] = "speaker-v1"
    overlapped.to_csv(v2, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="speaker_id"):
        validate_confirmatory_cohort(
            v1,
            v2,
            expected_samples=1,
            minimum_speakers=1,
            minimum_term_occurrences=1,
            minimum_negative_samples=0,
        )


def test_confirmatory_cohort_checks_additional_reference_manifests(
    tmp_path: Path,
) -> None:
    v1 = _manifest(
        tmp_path / "v1.csv",
        [_row("v1", 1, "PLC 상태를 확인합니다.", "speaker-v1")],
    )
    development = _manifest(
        tmp_path / "development.csv",
        [_row("dev", 1, "HMI 상태를 확인합니다.", "speaker-dev")],
    )
    candidate = _manifest(
        tmp_path / "v3.csv",
        [_row("v3", 1, "HMI 상태를 확인합니다.", "speaker-v3")],
    )

    with pytest.raises(ValueError, match="reference_text"):
        validate_confirmatory_cohort(
            v1,
            candidate,
            additional_reference_manifest_paths=[development],
            expected_samples=1,
            minimum_speakers=1,
            minimum_term_occurrences=1,
            minimum_negative_samples=0,
        )


def test_confirmatory_evaluation_freezes_v1_selection_and_writes_three_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_run = tmp_path / "artifacts/runs/reference-v1"
    reference_run.mkdir(parents=True)
    v1_manifest = _manifest(
        reference_run / "prepared_manifest.csv",
        [
            _row("v1", 1, "PLC 상태를 확인합니다.", "speaker-v1-a"),
            _row("v1", 2, "HMI 상태를 확인합니다.", "speaker-v1-b"),
        ],
    )
    terms_snapshot = reference_run / "domain_terms.snapshot.csv"
    shutil.copyfile(ROOT / "data/domain_terms/manufacturing_terms.csv", terms_snapshot)
    v1_predictions = pd.read_csv(v1_manifest, dtype=str).fillna("")
    v1_predictions["prediction_text"] = v1_predictions["reference_text"]
    v1_predictions.to_csv(
        reference_run / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
    )

    v2_manifest = _manifest(
        tmp_path / "v2.csv",
        [
            _row("v2", 1, "AGV 상태를 확인합니다.", "speaker-v2-a"),
            _row("v2", 2, "일반 작업 내용을 확인합니다.", "speaker-v2-b"),
        ],
    )
    base = yaml.safe_load(
        (ROOT / "configs/local_benchmark_base.yaml").read_text(encoding="utf-8")
    )
    base["paths"] = {
        "manifest": str(v1_manifest),
        "domain_terms": str(ROOT / "data/domain_terms/manufacturing_terms.csv"),
        "artifacts_dir": str(tmp_path / "artifacts/runs"),
        "database": str(tmp_path / "backdata/experiments.sqlite3"),
        "model_lock": str(tmp_path / "models/model-lock.yaml"),
    }
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection_path = selection_dir / "quantization_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump(
            {
                "selection": {
                    "model_id": "frozen-fixture",
                    "variant_id": "float16",
                    "model": {
                        "backend": "fixture",
                        "repo_id": "fixture/frozen-whisper",
                        "revision": "frozen-revision",
                        "local_dir": str(tmp_path / "models/frozen"),
                        "language": "ko",
                        "device": "cpu",
                        "compute_type": "float32",
                        "beam_size": 8,
                    },
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (selection_dir / "final_test_result.json").write_text(
        json.dumps({"run_id": "reference-v1", "run_dir": str(reference_run)}),
        encoding="utf-8",
    )

    calls = 0

    def fake_run_pipeline(settings):
        nonlocal calls
        calls += 1
        run_dir = tmp_path / "artifacts/runs/confirmatory-v2"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(settings.paths.manifest, dtype=str).fillna("")
        predictions["prediction_text"] = predictions["reference_text"]
        predictions.to_csv(
            run_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
        )
        report = run_dir / "evaluation_report.docx"
        report.write_bytes(b"fixture")
        return SimpleNamespace(
            run_id="confirmatory-v2",
            run_dir=run_dir,
            report_path=report,
        )

    monkeypatch.setattr("aias_specialist.confirmatory.run_pipeline", fake_run_pipeline)

    result = run_confirmatory_evaluation(
        selection_path,
        base_config,
        v2_manifest,
        expected_samples=2,
        minimum_speakers=2,
        minimum_term_occurrences=1,
        minimum_negative_samples=1,
    )

    assert result["selection_frozen"] is True
    assert result["v2_result_used_for_tuning"] is False
    comparison = pd.read_csv(result["comparison_path"])
    assert list(comparison["cohort"]) == ["test_v1", "test_v2", "combined_v1_v2"]
    generated_config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert generated_config["model"]["repo_id"] == "fixture/frozen-whisper"
    assert generated_config["model"]["beam_size"] == 8
    assert generated_config["training"]["enabled"] is False
    assert generated_config["paths"]["manifest"] == str(v2_manifest.resolve())
    assert generated_config["paths"]["domain_terms"] == str(terms_snapshot.resolve())

    cached = run_confirmatory_evaluation(
        selection_path,
        base_config,
        v2_manifest,
        expected_samples=2,
        minimum_speakers=2,
        minimum_term_occurrences=1,
        minimum_negative_samples=1,
    )
    assert cached["cache_reused"] is True
    assert calls == 1
