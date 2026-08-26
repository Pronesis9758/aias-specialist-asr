from pathlib import Path

import pandas as pd
import yaml

from aias_specialist.config import load_settings
from aias_specialist.correction_sweep import (
    run_correction_sweep,
    select_correction_candidate,
)
from aias_specialist.experiments import run_model_benchmark, select_experiment_member

ROOT = Path(__file__).resolve().parents[1]


def _selected_fixture_run(tmp_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(
        (ROOT / "configs/local_benchmark_base.yaml").read_text(encoding="utf-8")
    )
    config["paths"] = {
        "manifest": str(ROOT / "data/sample/benchmark_manifest.csv"),
        "domain_terms": str(ROOT / "data/domain_terms/manufacturing_terms.csv"),
        "artifacts_dir": str(tmp_path / "artifacts/runs"),
        "database": str(tmp_path / "backdata/experiments.sqlite3"),
        "model_lock": str(tmp_path / "models/model-lock.yaml"),
    }
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    load_settings(base_config)

    matrix = yaml.safe_load(
        (ROOT / "configs/benchmarks/local_fixture.yaml").read_text(encoding="utf-8")
    )
    matrix["benchmark"]["id"] = "correction-sweep-source"
    matrix["benchmark"]["base_config"] = str(base_config)
    matrix["benchmark"]["isolated_process"] = False
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(matrix, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    benchmark = run_model_benchmark(matrix_path)
    selection = select_experiment_member(
        benchmark.group_dir,
        "tiny",
        reviewer="test",
        reason="Correction validation fixture",
    )
    return base_config, selection


def _write_spec(
    tmp_path: Path,
    base_config: Path,
    candidates: list[dict[str, object]],
    sweep_id: str,
) -> Path:
    spec = {
        "correction_sweep": {
            "id": sweep_id,
            "name": "Correction sweep fixture",
            "base_config": str(base_config),
            "evaluation_split": "validation",
            "acceptance": {
                "min_cer_absolute_reduction": 1e-9,
                "min_wer_absolute_reduction": 0.0,
                "min_domain_term_recall_gain": 0.0,
                "max_degraded_sample_rate": 0.0,
            },
            "candidates": candidates,
        }
    }
    path = tmp_path / f"{sweep_id}.yaml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return path


def test_correction_sweep_selects_validation_improvement(tmp_path: Path) -> None:
    base_config, model_selection = _selected_fixture_run(tmp_path)
    spec = _write_spec(
        tmp_path,
        base_config,
        [
            {
                "id": "no-correction",
                "fallback": True,
                "enabled": False,
                "alias_enabled": False,
            },
            {"id": "alias-only", "enabled": True, "alias_enabled": True},
        ],
        "safe-correction-sweep",
    )

    result = run_correction_sweep(spec, model_selection)

    comparison = pd.read_csv(result.comparison_path)
    alias = comparison.loc[comparison["candidate_id"] == "alias-only"].iloc[0]
    assert alias["accepted"]
    assert alias["cer_absolute_reduction"] > 0
    assert alias["degraded_count"] == 0
    assert result.recommended_candidate == "alias-only"
    assert result.report_path.exists()
    assert (result.sweep_dir / "candidates/alias-only/correction_audit.csv").exists()

    selection_path = select_correction_candidate(
        result.sweep_dir,
        None,
        reviewer="test",
        reason="Passed validation regression gates",
        human_reviewed=True,
    )
    selected = yaml.safe_load(selection_path.read_text(encoding="utf-8"))["selection"]
    assert selected["candidate_id"] == "alias-only"
    assert selected["correction"]["alias_enabled"] is True


def test_correction_sweep_falls_back_when_no_candidate_improves(tmp_path: Path) -> None:
    base_config, model_selection = _selected_fixture_run(tmp_path)
    spec = _write_spec(
        tmp_path,
        base_config,
        [
            {
                "id": "no-correction",
                "fallback": True,
                "enabled": False,
                "alias_enabled": False,
            },
            {"id": "unchanged", "enabled": True, "alias_enabled": False},
        ],
        "fallback-correction-sweep",
    )

    result = run_correction_sweep(spec, model_selection)

    comparison = pd.read_csv(result.comparison_path)
    unchanged = comparison.loc[comparison["candidate_id"] == "unchanged"].iloc[0]
    assert not unchanged["accepted"]
    assert result.recommended_candidate == "no-correction"
