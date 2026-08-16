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


def test_colab_storage_path_comments_explain_persistence() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    mode_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and 'PROJECT_DIR = "/content/AIAS"' in cell.source
    )

    assert "Colab 임시 경로" in mode_cell
    assert "런타임 종료·초기화 시 삭제" in mode_cell
    assert "내 Google Drive 경로" in mode_cell
    assert "런타임 종료 후에도 유지" in mode_cell


def test_colab_mode_comments_explain_each_distinct_purpose() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    mode_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and '"PUBLIC_PROXY": {' in cell.source
    )

    assert "공개 Zeroth 한국어 음성으로 전체 파이프라인만 검증" in mode_cell
    assert "제조 용어가 포함된 합성 TTS 30개로 기능을 검증" in mode_cell
    assert "승인된 실제 제조 녹음과 사람 검수 전사" in mode_cell
    assert "tiny·base·small 후보 목록" in mode_cell
    assert "엄격한 거버넌스 조건" in mode_cell


def test_colab_comments_do_not_use_old_broad_boilerplate() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    comments = {
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if line.lstrip().startswith("#")
    }

    assert "# 현재 실행 모드의 설정 항목을 정의합니다." not in comments
    assert "# 이 단계에 필요한 코드 구문을 실행합니다." not in comments
    assert "# 진행 상태 또는 선택 결과를 실행 로그에 출력합니다." not in comments
