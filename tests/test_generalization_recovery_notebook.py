from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_generalization_recovery_v3.ipynb"


def _source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_recovery_notebook_has_cost_and_incremental_stage_gates() -> None:
    source = _source()

    assert "GENERATE_GENERALIZATION_AUDIO = False" in source
    assert "RUN_LORA_THROUGH_STAGE = None" in source
    assert "--max-stage" in source
    assert "pilot-5h" in source
    assert "target-10h" in source
    assert "extended-15h" in source


def test_recovery_notebook_does_not_reuse_failed_confirmatory_test_for_selection() -> None:
    source = _source()

    assert "Test는 포함하지 않습니다" in source
    assert "v1/v2 결과는 이 재튜닝에 사용하지 않습니다" in source
    assert "frozen Test v3" in source
    assert "생산 적용 주장은" in source
