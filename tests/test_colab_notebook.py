from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_manufacturing_assessment.ipynb"


def test_colab_setup_refreshes_editable_package_import_path() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "%pip install -q -e" in cell.source
    ]

    assert len(setup_cells) == 1
    setup_source = setup_cells[0]
    assert 'PROJECT_SRC = os.path.join(PROJECT_DIR, "src")' in setup_source
    assert "sys.path.insert(0, PROJECT_SRC)" in setup_source
    assert 'find_spec("aias_specialist")' in setup_source


def test_colab_code_lines_have_korean_explanatory_comments() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        for index, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            assert index > 0, f"code line has no explanation: {line}"
            comment = lines[index - 1].lstrip()
            assert comment.startswith("# "), f"code line has no explanation: {line}"
            assert any("가" <= character <= "힣" for character in comment), (
                f"explanation is not Korean: {comment}"
            )
