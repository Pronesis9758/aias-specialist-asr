from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/colab_generalization_final_validation.ipynb"


def _source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_final_notebook_separates_regression_and_independent_test_roles() -> None:
    source = _source()

    assert "generalization-regression-v2" in source
    assert "independent-heldout-v3" in source
    assert "--cohort-role', 'regression'" in source
    assert "--cohort-role', 'confirmatory'" in source
    assert "--development-influence" in source
    assert "synthetic_manufacturing_test_v3.yaml" in source
    assert "generalization-summary" in source


def test_final_notebook_has_cost_gates_and_no_test_tuning_claim() -> None:
    source = _source()

    assert "GENERATE_TEST_V3_AUDIO = False" in source
    assert "RUN_FINAL_GENERALIZATION = False" in source
    assert "Validation에서만 선택" in source
    assert "실제 작업자·공장" in source
