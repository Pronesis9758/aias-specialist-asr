from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_full_scale_performance.ipynb"


def test_full_scale_notebook_has_cost_gates_and_test_once_contract() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "GENERATE_SYNTHETIC_AUDIO = False" in source
    assert "RUN_A100_LORA = False" in source
    assert "RUN_FINAL_TEST_ONCE = False" in source
    assert "plan-synthetic-dataset" in source
    assert "lora-learning-curve" in source
    assert "merge-selected-lora" in source
    assert "decoding-sweep-selected-lora" in source
    assert "mine-term-errors" in source
    assert "select-deployment-profiles" in source
    assert "finalize-evaluation" in source


def test_full_scale_notebook_explains_synthetic_evidence_limit() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "생산 준비 완료를 주장" in source
    assert "실제 배포 전 승인된 shadow test" in source
    assert "Validation" in source
