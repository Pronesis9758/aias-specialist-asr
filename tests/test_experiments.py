from pathlib import Path

import pandas as pd
import pytest
import yaml

from aias_specialist.experiments import (
    _execute_config,
    run_final_evaluation,
    run_model_benchmark,
    run_quantization_sweep,
    select_experiment_member,
)
from aias_specialist.store import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path) -> Path:
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
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _experiment_spec(
    source: Path,
    destination: Path,
    section_name: str,
    experiment_id: str,
    base_config: Path,
) -> Path:
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    spec[section_name]["id"] = experiment_id
    spec[section_name]["base_config"] = str(base_config)
    spec[section_name]["isolated_process"] = False
    destination.write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def test_benchmark_quantization_selection_and_final_test(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path)
    matrix = _experiment_spec(
        ROOT / "configs/benchmarks/local_fixture.yaml",
        tmp_path / "matrix.yaml",
        "benchmark",
        "fixture-models-test",
        base_config,
    )

    benchmark = run_model_benchmark(matrix)

    assert benchmark.status == "completed"
    leaderboard = pd.read_csv(benchmark.comparison_path)
    assert set(leaderboard["member_id"]) == {"tiny", "base"}
    assert set(leaderboard["evaluation_split"]) == {"validation"}
    assert benchmark.report_path.exists()

    model_selection = select_experiment_member(
        benchmark.group_dir,
        "tiny",
        reviewer="test-reviewer",
        reason="Fixture selection for orchestration validation",
    )
    selected = yaml.safe_load(model_selection.read_text(encoding="utf-8"))["selection"]
    assert selected["model_id"] == "tiny"
    assert selected["final_test_required"] is True

    quantization_spec = _experiment_spec(
        ROOT / "configs/quantization/local_fixture.yaml",
        tmp_path / "quantization.yaml",
        "quantization",
        "fixture-quantization-test",
        base_config,
    )
    quantization = run_quantization_sweep(quantization_spec, model_selection)

    quantization_rows = pd.read_csv(quantization.comparison_path)
    assert set(quantization_rows["member_id"]) == {"float16", "int8-float16"}
    assert "cer_delta_vs_reference" in quantization_rows
    assert "rtf_speedup_vs_reference" in quantization_rows
    assert quantization.report_path.exists()

    quantization_selection = select_experiment_member(
        quantization.group_dir,
        "int8-float16",
        reviewer="test-reviewer",
        reason="Fixture quantization selection",
    )
    final = run_final_evaluation(quantization_selection, base_config)

    assert final["metrics"]["baseline"]["sample_count"] == 1
    assert Path(final["report_path"]).exists()
    store = ExperimentStore(tmp_path / "backdata/experiments.sqlite3")
    assert len(store.group_members("fixture-models-test")) == 2
    assert len(store.group_members("fixture-quantization-test")) == 2


def test_benchmark_rejects_test_split_for_model_selection(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path)
    matrix = yaml.safe_load(
        (ROOT / "configs/benchmarks/local_fixture.yaml").read_text(encoding="utf-8")
    )
    matrix["benchmark"]["id"] = "invalid-test-selection"
    matrix["benchmark"]["base_config"] = str(base_config)
    matrix["benchmark"]["evaluation_split"] = "test"
    matrix_path = tmp_path / "invalid.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")

    try:
        run_model_benchmark(matrix_path)
    except ValueError as exc:
        assert "Reserve test" in str(exc)
    else:
        raise AssertionError("Benchmark should reject selection on the held-out test split")


def test_isolated_worker_streams_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_config = _base_config(tmp_path)
    result_path = tmp_path / "worker-result.json"

    result = _execute_config(base_config, result_path, isolated_process=True)

    output = capsys.readouterr().out
    assert "[pipeline] preparing manifest" in output
    assert "[pipeline] completed" in output
    assert result["run_id"]
    assert result_path.exists()
