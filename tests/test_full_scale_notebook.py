from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_full_scale_performance.ipynb"
FULL_CONFIG = ROOT / "configs/synthetic_manufacturing_full_scale.yaml"


def test_full_scale_notebook_has_cost_gates_and_test_once_contract() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "GENERATE_SYNTHETIC_AUDIO = False" in source
    assert "RUN_A100_LORA = False" in source
    assert "RUN_FINAL_TEST_ONCE = False" in source
    assert "GENERATE_CONFIRMATORY_V2_AUDIO = False" in source
    assert "RUN_CONFIRMATORY_V2_ONCE = False" in source
    assert "plan-synthetic-dataset" in source
    assert "lora-learning-curve" in source
    assert "merge-selected-lora" in source
    assert "decoding-sweep-selected-lora" in source
    assert "mine-term-errors" in source
    assert "select-deployment-profiles" in source
    assert "finalize-evaluation" in source
    assert "confirmatory-evaluation" in source
    assert "synthetic_manufacturing_test_v2.yaml" in source
    assert "v1_v2_comparison.csv" in source


def test_full_scale_notebook_explains_synthetic_evidence_limit() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "생산 준비 완료를 주장" in source
    assert "실제 배포 전 승인된 shadow test" in source
    assert "Validation" in source
    assert "v2 결과는 설정 선택이나 재튜닝에 사용하지 않습니다" in source


def test_full_scale_a100_config_keeps_effective_batch_and_avoids_duplicate_eval() -> None:
    config = yaml.safe_load(FULL_CONFIG.read_text(encoding="utf-8"))
    training = config["training"]

    assert training["train_batch_size"] * training["gradient_accumulation_steps"] == 8
    assert training["preprocess_num_proc"] == 1
    assert training["evaluate_during_training"] is False
    assert training["repeat_final_evaluation"] is False
