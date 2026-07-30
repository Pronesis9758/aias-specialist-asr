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
    train_selected_whisper_lora,
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
    assert selected["human_reviewed"] is True
    assert selected["selection_scope"] == "human_review"

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

    repeated_model_selection = select_experiment_member(
        benchmark.group_dir,
        "tiny",
        reviewer="test-reviewer",
        reason="Fixture selection for orchestration validation",
    )
    repeated_selection = yaml.safe_load(
        repeated_model_selection.read_text(encoding="utf-8")
    )["selection"]
    assert repeated_selection["selection_id"] != selected["selection_id"]

    reused_quantization = run_quantization_sweep(
        quantization_spec,
        repeated_model_selection,
    )
    assert reused_quantization.group_dir == quantization.group_dir
    assert reused_quantization.status == "completed"

    changed_model_selection = select_experiment_member(
        benchmark.group_dir,
        "base",
        reviewer="test-reviewer",
        reason="Materially different model selection",
    )
    with pytest.raises(ValueError, match="different specification"):
        run_quantization_sweep(quantization_spec, changed_model_selection)

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


def test_selected_model_drives_lora_training_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = _base_config(tmp_path)
    matrix = _experiment_spec(
        ROOT / "configs/benchmarks/local_fixture.yaml",
        tmp_path / "matrix-selected-training.yaml",
        "benchmark",
        "fixture-selected-training-test",
        base_config,
    )
    benchmark = run_model_benchmark(matrix)
    selection_path = select_experiment_member(
        benchmark.group_dir,
        "tiny",
        reviewer="test-reviewer",
        reason="Selected training configuration test",
    )
    fake_run_dir = tmp_path / "artifacts/runs/train-selected"
    fake_run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "aias_specialist.experiments.train_whisper_lora",
        lambda settings: fake_run_dir,
    )

    result = train_selected_whisper_lora(selection_path, base_config)

    generated = yaml.safe_load(
        (benchmark.group_dir / "selected_training_config.yaml").read_text(encoding="utf-8")
    )
    assert generated["training"]["repo_id"] == "fixture/whisper-tiny"
    assert generated["training"]["enabled"] is True
    assert result["run_dir"] == str(fake_run_dir)


def test_public_proxy_selection_is_not_marked_as_human_reviewed(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path)
    matrix = _experiment_spec(
        ROOT / "configs/benchmarks/local_fixture.yaml",
        tmp_path / "matrix-public-proxy.yaml",
        "benchmark",
        "fixture-public-proxy-selection",
        base_config,
    )
    benchmark = run_model_benchmark(matrix)

    selection_path = select_experiment_member(
        benchmark.group_dir,
        "tiny",
        reviewer="AUTOMATED_PUBLIC_PROXY",
        reason="Pipeline and artifact smoke test only",
        human_reviewed=False,
    )

    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))["selection"]
    assert selection["human_reviewed"] is False
    assert selection["selection_scope"] == "automated_public_proxy"


def test_strict_private_governance_rejects_automated_selection(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path)
    matrix = _experiment_spec(
        ROOT / "configs/benchmarks/local_fixture.yaml",
        tmp_path / "matrix-strict-private.yaml",
        "benchmark",
        "fixture-strict-private-selection",
        base_config,
    )
    benchmark = run_model_benchmark(matrix)
    strict_config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    strict_config["governance"] = {
        "mode": "strict_private",
        "approval_file": str(tmp_path / "data_approval.yaml"),
        "require_speaker_disjoint_splits": True,
        "require_label_review": True,
        "require_deidentified": True,
    }
    base_config.write_text(
        yaml.safe_dump(strict_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Automated proxy selection is not allowed"):
        select_experiment_member(
            benchmark.group_dir,
            "tiny",
            reviewer="AUTOMATED_PUBLIC_PROXY",
            reason="This must not pass strict private governance",
            human_reviewed=False,
        )
