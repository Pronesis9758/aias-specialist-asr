from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import pandas as pd

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
            if re.fullmatch(r"[)\]}]+,?", line.strip()):
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
    assert "제조 용어·난이도·소음 조건을 균형화한 합성 TTS 600개" in mode_cell
    assert "승인된 실제 제조 녹음과 사람 검수 전사" in mode_cell
    assert "tiny부터 large-v3까지 정확도·자원 후보 목록" in mode_cell
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
    assert not any("바로 위 함수·목록·사전 구문" in comment for comment in comments)
    assert not any("이 줄의 연산 결과" in comment for comment in comments)
    assert not any("앞에서 시작한 코드 구문" in comment for comment in comments)
    assert not any("여러 값으로 구성된 자료 구조" in comment for comment in comments)
    assert not any("변수에 이 연구 단계에서 계산하거나 선택" in comment for comment in comments)


def test_colab_drive_mount_is_idempotent_and_explains_popup_failure() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    drive_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "drive.mount(" in cell.source
    )

    assert 'DRIVE_MY_DRIVE = DRIVE_MOUNT_POINT / "MyDrive"' in drive_cell
    assert "if DRIVE_MY_DRIVE.is_dir():" in drive_cell
    assert "Google Drive already mounted" in drive_cell
    assert "except ValueError as exc:" in drive_cell
    assert "accounts.google.com 화면이 검게" in drive_cell
    assert "일반 Chrome" in drive_cell
    assert "if not DRIVE_MY_DRIVE.is_dir():" in drive_cell


def test_colab_experiment_ids_support_reuse_or_one_shared_timestamp() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    mode_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "FORCE_NEW_EXPERIMENT" in cell.source
    )

    assert "FORCE_NEW_EXPERIMENT = False" in mode_cell
    assert 'datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")' in mode_cell
    assert 'f"{design_id}-{EXPERIMENT_SESSION_ID}"' in mode_cell
    assert 'BENCHMARK_ID = experiment_id(mode["benchmark_id"])' in mode_cell
    assert 'QUANTIZATION_ID = experiment_id(mode["quantization_id"])' in mode_cell


def test_colab_runtime_specs_share_the_selected_experiment_ids() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    runtime_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "runtime_correction_sweep_path" in cell.source
    )

    assert 'runtime_config.setdefault("assessment", {})["benchmark_id"] = BENCHMARK_ID' in (
        runtime_cell
    )
    assert 'runtime_config["assessment"]["quantization_id"] = QUANTIZATION_ID' in runtime_cell
    assert 'runtime_matrix["benchmark"]["id"] = BENCHMARK_ID' in runtime_cell
    assert 'runtime_quantization["quantization"]["id"] = QUANTIZATION_ID' in runtime_cell
    assert (
        'Path(PROJECT_DIR) / "configs/correction/aias_runtime_correction_sweep.yaml"'
        in runtime_cell
    )
    assert "correction_sweep_id = experiment_id(" in runtime_cell
    assert "CORRECTION_SWEEP_SPEC = str(runtime_correction_sweep_path)" in runtime_cell


def test_colab_final_test_displays_correction_effect_and_audit_counts() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    final_test_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "final_test_result_path" in cell.source
    )

    assert 'final_test_metrics = final_test_result["metrics"]' in final_test_cell
    assert "build_test_metric_row" in final_test_cell
    assert '"domain_term_recall"' in final_test_cell
    assert 'final_test_result["quality_gate"]' in final_test_cell
    assert '"현업 정확도 목표:"' in final_test_cell
    assert 'Path(final_test_result["run_dir"]) / "correction_audit.csv"' in final_test_cell
    assert 'final_test_audit["outcome"].value_counts()' in final_test_cell
    assert '"improved_count"' in final_test_cell
    assert '"degraded_count"' in final_test_cell
    assert "final_test_changed_mask" in final_test_cell
    assert "final_test_review_columns" in final_test_cell


def test_colab_final_test_summary_code_calculates_expected_values(tmp_path: Path) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    final_test_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "final_test_result_path" in cell.source
    )
    selection_path = tmp_path / "quantization_selection.yaml"
    selection_path.write_text("selection: {}\n", encoding="utf-8")
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    final_result = {
        "run_dir": str(run_dir),
        "metrics": {
            "baseline": {"wer": 0.4, "cer": 0.2, "domain_term_recall": 0.3},
            "corrected": {"wer": 0.3, "cer": 0.15, "domain_term_recall": 0.5},
        },
        "quality_gate": {
            "overall_pass": False,
            "observed": {"wer": 0.3, "cer": 0.15, "domain_term_recall": 0.5},
            "checks": {
                "wer": {"passed": False, "gap": 0.15},
                "cer": {"passed": False, "gap": 0.08},
                "domain_term_recall": {"passed": False, "gap": 0.35},
            },
        },
    }
    (tmp_path / "final_test_result.json").write_text(json.dumps(final_result), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "outcome": "improved",
                "reference_text": "프레스 3호기",
                "recognized_before": "프레스 삼오기",
                "corrected_after": "프레스 3호기",
                "correction_methods": "alias",
                "cer_absolute_reduction": 0.2,
                "text_changed": True,
            },
            {
                "sample_id": "sample-2",
                "outcome": "unchanged",
                "reference_text": "베어링 점검",
                "recognized_before": "베어링 점검",
                "corrected_after": "베어링 점검",
                "correction_methods": "",
                "cer_absolute_reduction": 0.0,
                "text_changed": False,
            },
        ]
    ).to_csv(run_dir / "correction_audit.csv", index=False)
    displayed: list[pd.DataFrame] = []
    namespace = {
        "CONFIG": "unused-test-config.yaml",
        "Path": Path,
        "RUN_QUANTIZATION": True,
        "display": displayed.append,
        "json": json,
        "model_selection": selection_path,
        "pd": pd,
        "quantization_selection": selection_path,
        "run_aias": lambda *args: None,
    }

    exec(compile(final_test_cell, str(NOTEBOOK), "exec"), namespace)

    metric_table = namespace["final_test_metric_table"]
    change_table = namespace["final_test_change_table"]
    assert metric_table["absolute_improvement"].round(3).tolist() == [0.1, 0.05, 0.2]
    assert change_table.iloc[0].to_dict() == {
        "sample_count": 2,
        "changed_count": 1,
        "improved_count": 1,
        "degraded_count": 0,
        "unchanged_count": 1,
    }
    assert len(displayed) == 4
