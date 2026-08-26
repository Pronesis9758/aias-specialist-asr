from pathlib import Path

import pandas as pd
import yaml

from aias_specialist.config import load_settings
from aias_specialist.pipeline import run_pipeline
from aias_specialist.store import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_pipeline_creates_auditable_artifacts(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/local_smoke.yaml").read_text(encoding="utf-8"))
    config["paths"]["manifest"] = str(ROOT / "data/sample/manifest.csv")
    config["paths"]["domain_terms"] = str(ROOT / "data/domain_terms/manufacturing_terms.csv")
    config["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    config["paths"]["database"] = str(tmp_path / "backdata/experiments.sqlite3")
    config["paths"]["model_lock"] = str(tmp_path / "models/model-lock.yaml")
    config["quality_targets"] = {
        "enabled": True,
        "minimum_domain_term_recall": 0.85,
        "maximum_cer": 0.07,
        "maximum_wer": 0.15,
        "priority": ["domain_term_recall", "cer", "wer"],
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = run_pipeline(load_settings(config_path))

    required = [
        "config.snapshot.yaml",
        "environment.json",
        "prepared_manifest.csv",
        "predictions_baseline.csv",
        "predictions_corrected.csv",
        "correction_audit.csv",
        "correction_audit.jsonl",
        "correction_audit.md",
        "metrics.json",
        "quality_gate.json",
        "run_summary.md",
        "reports/evaluation_report.docx",
    ]
    for relative in required:
        assert (result.run_dir / relative).exists(), relative
    assert result.metrics["corrected"]["wer"] < result.metrics["baseline"]["wer"]
    assert result.metrics["quality_gate"]["overall_pass"] is True
    correction_audit = pd.read_csv(result.run_dir / "correction_audit.csv")
    assert correction_audit["text_changed"].any()
    assert (correction_audit["outcome"] == "improved").any()
    assert correction_audit["cer_absolute_reduction"].max() > 0
    history = ExperimentStore(tmp_path / "backdata/experiments.sqlite3").history()
    assert history[0]["status"] == "completed"


def test_pipeline_evaluates_only_test_split(tmp_path: Path) -> None:
    source = pd.read_csv(ROOT / "data/sample/manifest.csv", dtype=str)
    extra = source.iloc[[0]].copy()
    extra["sample_id"] = "train-only"
    extra["split"] = "train"
    manifest = tmp_path / "manifest.csv"
    pd.concat([source.iloc[[0]], extra], ignore_index=True).to_csv(manifest, index=False)

    config = yaml.safe_load((ROOT / "configs/local_smoke.yaml").read_text(encoding="utf-8"))
    config["paths"]["manifest"] = str(manifest)
    config["paths"]["domain_terms"] = str(ROOT / "data/domain_terms/manufacturing_terms.csv")
    config["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    config["paths"]["database"] = str(tmp_path / "backdata/experiments.sqlite3")
    config["paths"]["model_lock"] = str(tmp_path / "models/model-lock.yaml")
    config_path = tmp_path / "split-smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = run_pipeline(load_settings(config_path))

    assert result.metrics["baseline"]["sample_count"] == 1
    assert len(pd.read_csv(result.run_dir / "prepared_manifest.csv")) == 2
    assert len(pd.read_csv(result.run_dir / "predictions_baseline.csv")) == 1
