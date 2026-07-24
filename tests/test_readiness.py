from pathlib import Path

import yaml

from aias_specialist.config import load_settings
from aias_specialist.readiness import write_assessment_readiness

ROOT = Path(__file__).resolve().parents[1]


def test_assessment_audit_reports_pre_data_blockers(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/local_smoke.yaml").read_text(encoding="utf-8"))
    config["paths"]["manifest"] = str(tmp_path / "private/manifest.csv")
    config["paths"]["domain_terms"] = str(ROOT / "data/domain_terms/manufacturing_terms.csv")
    config["paths"]["artifacts_dir"] = str(tmp_path / "artifacts/runs")
    config["paths"]["database"] = str(tmp_path / "backdata/experiments.sqlite3")
    config["paths"]["model_lock"] = str(tmp_path / "models/model-lock.yaml")
    config["governance"] = {
        "mode": "strict_private",
        "approval_file": str(tmp_path / "private/data_approval.yaml"),
        "require_speaker_disjoint_splits": True,
        "require_label_review": True,
        "require_deidentified": True,
    }
    config["assessment"] = {
        "model_matrix": str(ROOT / "configs/benchmarks/manufacturing_whisper_models_template.yaml"),
        "benchmark_id": "manufacturing-whisper-model-benchmark-v1",
        "quantization_id": "manufacturing-whisper-quantization-v1",
        "hardware_profile": str(ROOT / "configs/hardware/colab_gpu_validation.yaml"),
        "human_signoff": str(tmp_path / "private/human_review_signoff.yaml"),
    }
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result, json_path, markdown_path = write_assessment_readiness(
        load_settings(config_path),
        tmp_path / "readiness",
    )

    assert result["overall_status"] == "not_ready"
    assert result["blockers_remaining"] > 0
    assert json_path.exists()
    assert markdown_path.exists()
    assert "waiting for approved manufacturing audio" in markdown_path.read_text(encoding="utf-8")
