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


def test_final_notebook_refreshes_editable_install_for_running_colab_kernel() -> None:
    source = _source()

    assert "PROJECT_SRC = str(Path(PROJECT_DIR) / 'src')" in source
    assert "sys.path.insert(0, PROJECT_SRC)" in source
    assert "importlib.invalidate_caches()" in source
    assert "import aias_specialist" in source


def test_final_notebook_keeps_long_running_colab_cell_ids_stable() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell_ids = [cell.id for cell in notebook.cells]

    assert len(cell_ids) == 24
    assert len(set(cell_ids)) == len(cell_ids)
    assert cell_ids[13] == "f71d3408"


def test_final_notebook_resolves_runtime_correction_base_config() -> None:
    source = _source()

    assert (
        "correction_spec['correction_sweep']['base_config'] = "
        "str(Path(GENERALIZATION_CONFIG).resolve())"
    ) in source
