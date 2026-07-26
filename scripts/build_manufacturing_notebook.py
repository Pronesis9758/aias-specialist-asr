from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_manufacturing_assessment.ipynb"


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": "AI Specialist Manufacturing ASR Assessment",
            "provenance": [],
        },
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 제조 음성 ASR 심사 파이프라인\n\n"
            "하나의 노트북에서 두 가지 데이터 모드를 사용합니다.\n\n"
            "- `PUBLIC_PROXY`: 공개 Zeroth 한국어 음성으로 데이터 준비부터 모델 비교, "
            "자동 프록시 선택, LoRA, 양자화, 고정 Test, 보고서·백데이터까지 전체 동작을 "
            "검증합니다.\n"
            "- `PRIVATE_MANUFACTURING`: 나중에 승인된 제조 녹음과 검수 전사로 같은 코드를 "
            "다시 실행합니다. 이 모드의 모델·양자화 선택은 반드시 사람이 수행합니다.\n\n"
            "> 공개 프록시 결과는 코드와 산출물의 정상 동작 증거입니다. 제조 현장 성능, "
            "배포 적합성 또는 심사 최종 결론의 증거로 사용하면 안 됩니다.\n\n"
            "**보안:** 카메라·마이크·패스키를 사용하지 않습니다. 실제 음성은 GitHub에 "
            "올리지 않고 승인된 비공개 Drive 경로만 사용합니다."
        ),
        nbf.v4.new_markdown_cell("## 0. 실행 모드"),
        nbf.v4.new_code_cell(
            '# "PUBLIC_PROXY" 또는 "PRIVATE_MANUFACTURING"\n'
            'DATA_MODE = "PUBLIC_PROXY"\n\n'
            'GITHUB_REPO_URL = "https://github.com/Pronesis9758/aias-specialist-asr.git"\n'
            'GITHUB_BRANCH = "codex/whisper-benchmark-quantization"  # PR 병합 후 main\n'
            'PROJECT_DIR = "/content/AIAS"\n'
            'DRIVE_ROOT = "/content/drive/MyDrive/AI_Specialist_ASR_Project"\n\n'
            "MODE_SETTINGS = {\n"
            '    "PUBLIC_PROXY": {\n'
            '        "config": "configs/public_proxy_assessment.yaml",\n'
            '        "matrix": "configs/benchmarks/public_proxy_whisper_models.yaml",\n'
            '        "quantization": '
            '"configs/quantization/public_proxy_whisper_quantization.yaml",\n'
            '        "benchmark_id": "public-proxy-whisper-model-benchmark-v1",\n'
            '        "quantization_id": "public-proxy-whisper-quantization-v1",\n'
            "    },\n"
            '    "PRIVATE_MANUFACTURING": {\n'
            '        "config": "configs/manufacturing_private_template.yaml",\n'
            '        "matrix": '
            '"configs/benchmarks/manufacturing_whisper_models_template.yaml",\n'
            '        "quantization": '
            '"configs/quantization/manufacturing_whisper_quantization_template.yaml",\n'
            '        "benchmark_id": "manufacturing-whisper-model-benchmark-v1",\n'
            '        "quantization_id": "manufacturing-whisper-quantization-v1",\n'
            "    },\n"
            "}\n"
            "if DATA_MODE not in MODE_SETTINGS:\n"
            "    raise ValueError(f'지원하지 않는 DATA_MODE: {DATA_MODE}')\n\n"
            "mode = MODE_SETTINGS[DATA_MODE]\n"
            "CONFIG = mode['config']\n"
            "MODEL_MATRIX = mode['matrix']\n"
            "QUANTIZATION_SPEC = mode['quantization']\n"
            "BENCHMARK_ID = mode['benchmark_id']\n"
            "QUANTIZATION_ID = mode['quantization_id']\n"
            'PRIVATE_ROOT = f"{DRIVE_ROOT}/data/private/manufacturing"\n'
            "IS_PUBLIC_PROXY = DATA_MODE == 'PUBLIC_PROXY'\n"
            'print("Mode:", DATA_MODE)\n'
            'print("Config:", CONFIG)'
        ),
        nbf.v4.new_markdown_cell("## 1. GPU와 Google Drive 연결"),
        nbf.v4.new_code_cell(
            '!nvidia-smi\nfrom google.colab import drive\n\ndrive.mount("/content/drive")'
        ),
        nbf.v4.new_markdown_cell("## 2. GitHub 코드 동기화"),
        nbf.v4.new_code_cell(
            "import os\n"
            "import subprocess\n\n"
            "if not os.path.exists(PROJECT_DIR):\n"
            "    subprocess.run(\n"
            "        [\n"
            '            "git", "clone", "--branch", GITHUB_BRANCH, "--single-branch",\n'
            "            GITHUB_REPO_URL, PROJECT_DIR,\n"
            "        ],\n"
            "        check=True,\n"
            "    )\n"
            "else:\n"
            "    subprocess.run(\n"
            '        ["git", "-C", PROJECT_DIR, "fetch", "origin", GITHUB_BRANCH],\n'
            "        check=True,\n"
            "    )\n"
            "    subprocess.run(\n"
            '        ["git", "-C", PROJECT_DIR, "checkout", GITHUB_BRANCH],\n'
            "        check=True,\n"
            "    )\n"
            "    subprocess.run(\n"
            "        [\n"
            '            "git", "-C", PROJECT_DIR, "merge", "--ff-only",\n'
            '            f"origin/{GITHUB_BRANCH}",\n'
            "        ],\n"
            "        check=True,\n"
            "    )\n"
            "os.chdir(PROJECT_DIR)\n"
            'print("Git commit:", subprocess.check_output(\n'
            '    ["git", "rev-parse", "HEAD"], text=True\n'
            ").strip())"
        ),
        nbf.v4.new_markdown_cell("## 3. 의존성 설치와 실행 함수"),
        nbf.v4.new_code_cell(
            "%pip uninstall -y torchao\n"
            '%pip install -q -e ".[train]" "transformers>=4.46,<5" "peft>=0.14,<0.19"\n\n'
            "import sys\n\n"
            "def run_aias(*args):\n"
            "    command = [sys.executable, '-m', 'aias_specialist.cli', *args]\n"
            '    print("\\nRunning:", " ".join(command), flush=True)\n'
            "    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}\n"
            "    subprocess.run(command, check=True, env=environment)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. 데이터 준비\n\n"
            "`PUBLIC_PROXY`에서는 고정 revision의 `kresnik/zeroth_korean` 일부만 스트리밍해 "
            "Drive에 저장합니다. `PRIVATE_MANUFACTURING`에서는 기존 파일을 덮어쓰지 않고 "
            "입력 양식을 준비합니다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import shutil\n"
            "import pandas as pd\n\n"
            "if IS_PUBLIC_PROXY:\n"
            "    run_aias('prepare-hf-dataset', '--config', CONFIG)\n"
            "    public_root = Path(DRIVE_ROOT) / 'data/public/zeroth_korean'\n"
            "    display(pd.read_csv(public_root / 'manifest.csv').groupby('split').size())\n"
            "    provenance_path = public_root / 'dataset_provenance.json'\n"
            "    if provenance_path.exists():\n"
            "        display(json.loads(provenance_path.read_text(encoding='utf-8')))\n"
            "    print('PUBLIC PROXY: 제조 성능 증거가 아닌 전체동작 검증 데이터입니다.')\n"
            "else:\n"
            "    private_root = Path(PRIVATE_ROOT)\n"
            "    (private_root / 'audio').mkdir(parents=True, exist_ok=True)\n"
            "    templates = {\n"
            "        Path('data/templates/manufacturing_manifest_template.csv'):\n"
            "            private_root / 'manifest.csv',\n"
            "        Path('data/templates/data_approval_template.yaml'):\n"
            "            private_root / 'data_approval.yaml',\n"
            "        Path('data/templates/human_review_signoff_template.yaml'):\n"
            "            private_root / 'human_review_signoff.yaml',\n"
            "        Path('configs/assessment/acceptance_criteria_template.yaml'):\n"
            "            private_root / 'acceptance_criteria.yaml',\n"
            "    }\n"
            "    for source, destination in templates.items():\n"
            "        if not destination.exists():\n"
            "            shutil.copy2(source, destination)\n"
            "            print('Created:', destination)\n"
            "        else:\n"
            "            print('Preserved existing:', destination)\n"
            "    run_aias(\n"
            "        'assessment-audit', '--config', CONFIG,\n"
            "        '--output-dir', "
            "f'{DRIVE_ROOT}/reports/assessment_readiness/private_manufacturing',\n"
            "    )"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Whisper 모델 비교\n\n"
            "공개 프록시는 빠른 전체동작 검증을 위해 `tiny`, `base`, `small`을 비교합니다. "
            "실제 제조 모드는 6개 후보를 비교합니다. 모델과 변환본은 Drive 캐시에 재사용됩니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias('model-matrix-lock', '--matrix', MODEL_MATRIX)\n"
            "run_aias('benchmark-models', '--matrix', MODEL_MATRIX)\n"
            "benchmark_dir = Path(DRIVE_ROOT) / 'artifacts/benchmarks' / BENCHMARK_ID\n"
            "benchmark_table = pd.read_csv(benchmark_dir / 'benchmark_comparison.csv')\n"
            "display(benchmark_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. 모델 선택\n\n"
            "공개 프록시는 완료 후보 중 rank 1을 자동 선택하지만 사람 검토로 기록하지 않습니다. "
            "실제 제조 모드에서는 아래 사람 검토 값을 직접 입력해야 다음 단계로 진행됩니다."
        ),
        nbf.v4.new_code_cell(
            "def best_completed_member(frame):\n"
            "    completed = frame.loc[frame['status'].eq('completed')].copy()\n"
            "    if completed.empty:\n"
            "        raise RuntimeError('완료된 후보가 없습니다. 위 오류를 먼저 확인하세요.')\n"
            "    completed['rank'] = pd.to_numeric(completed['rank'], errors='coerce')\n"
            "    return completed.sort_values(\n"
            "        ['rank', 'cer', 'wer', 'aggregate_real_time_factor'],\n"
            "        na_position='last',\n"
            "    ).iloc[0]\n\n"
            "if IS_PUBLIC_PROXY:\n"
            "    selected_row = best_completed_member(benchmark_table)\n"
            "    SELECTED_MODEL = str(selected_row['member_id'])\n"
            "    REVIEWER = 'AUTOMATED_PUBLIC_PROXY'\n"
            "    MODEL_REASON = (\n"
            "        '공개 Zeroth 프록시 rank 1 자동 선택. 코드·산출물 smoke test 전용이며 '\n"
            "        '제조 모델 선정 또는 사람 검토 증거가 아님.'\n"
            "    )\n"
            "    extra_selection_args = ['--automated-proxy']\n"
            "else:\n"
            "    SELECTED_MODEL = 'small'  # 비교표를 보고 수정\n"
            "    REVIEWER = 'TO_BE_COMPLETED'\n"
            "    MODEL_REASON = 'TO_BE_COMPLETED: 정확도·속도·메모리·거버넌스 근거'\n"
            "    if 'TO_BE_COMPLETED' in REVIEWER or 'TO_BE_COMPLETED' in MODEL_REASON:\n"
            "        raise ValueError(\n"
            "            '제조 모드에서는 사람 검토자와 모델 선택 근거를 입력하세요.'\n"
            "        )\n"
            "    extra_selection_args = []\n\n"
            "run_aias(\n"
            "    'select-model', '--benchmark-dir', str(benchmark_dir),\n"
            "    '--model-id', SELECTED_MODEL, '--reviewer', REVIEWER,\n"
            "    '--reason', MODEL_REASON, *extra_selection_args,\n"
            ")\n"
            "model_selection = benchmark_dir / 'model_selection.yaml'\n"
            "print('Selected model:', SELECTED_MODEL)\n"
            "print(model_selection.read_text(encoding='utf-8'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. 선택 모델 LoRA\n\n"
            "공개 프록시는 40개 train 샘플·30 step의 짧은 실행으로 학습 코드, checkpoint, "
            "Base/LoRA 비교 산출물을 검증합니다. 실제 제조 모드는 별도 설정의 500 step을 "
            "사용하며 데이터 규모에 맞춰 조정합니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'train-selected-whisper', '--selection', str(model_selection),\n"
            "    '--config', CONFIG,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell("## 8. 양자화 비교"),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'quantization-sweep', '--spec', QUANTIZATION_SPEC,\n"
            "    '--selection', str(model_selection),\n"
            ")\n"
            "quantization_dir = Path(DRIVE_ROOT) / 'artifacts/quantization' / QUANTIZATION_ID\n"
            "quantization_table = pd.read_csv(\n"
            "    quantization_dir / 'quantization_comparison.csv'\n"
            ")\n"
            "display(quantization_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## 9. 양자화 선택\n\n"
            "공개 프록시는 종합 rank 1을 자동 선택합니다. 실제 제조 모드는 정확도 손실, RTF, "
            "GPU 메모리와 모델 용량을 사람이 함께 검토합니다."
        ),
        nbf.v4.new_code_cell(
            "if IS_PUBLIC_PROXY:\n"
            "    selected_quantization_row = best_completed_member(quantization_table)\n"
            "    SELECTED_VARIANT = str(selected_quantization_row['member_id'])\n"
            "    QUANTIZATION_REASON = (\n"
            "        '공개 Zeroth 프록시 종합 rank 1 자동 선택. 양자화 코드·산출물 smoke test '\n"
            "        '전용이며 실제 배포 결정 또는 사람 검토 증거가 아님.'\n"
            "    )\n"
            "    extra_quantization_args = ['--automated-proxy']\n"
            "else:\n"
            "    SELECTED_VARIANT = 'float16'  # 비교표를 보고 수정\n"
            "    QUANTIZATION_REASON = (\n"
            "        'TO_BE_COMPLETED: 정확도 손실·속도·메모리·모델 용량 근거'\n"
            "    )\n"
            "    if 'TO_BE_COMPLETED' in QUANTIZATION_REASON:\n"
            "        raise ValueError('제조 모드에서는 양자화 선택 근거를 입력하세요.')\n"
            "    extra_quantization_args = []\n\n"
            "run_aias(\n"
            "    'select-quantization', '--quantization-dir', str(quantization_dir),\n"
            "    '--variant-id', SELECTED_VARIANT, '--reviewer', REVIEWER,\n"
            "    '--reason', QUANTIZATION_REASON, *extra_quantization_args,\n"
            ")\n"
            "quantization_selection = quantization_dir / 'quantization_selection.yaml'\n"
            "print('Selected variant:', SELECTED_VARIANT)\n"
            "print(quantization_selection.read_text(encoding='utf-8'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. 고정 Test 최종평가\n\n"
            "모델·양자화 선택이 끝난 뒤에만 그동안 보지 않은 Test split을 한 번 평가합니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'finalize-evaluation', '--selection', str(quantization_selection),\n"
            "    '--config', CONFIG,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## 11. 산출물·심사 준비도 확인\n\n"
            "공개 프록시에서는 `not_ready`가 정상입니다. 공개 데이터, 자동 선택, 미완료 사람 "
            "서명은 제조 심사 증거를 대체하지 못합니다."
        ),
        nbf.v4.new_code_cell(
            "readiness_dir = (\n"
            "    Path(DRIVE_ROOT)\n"
            "    / 'reports/assessment_readiness'\n"
            "    / DATA_MODE.lower()\n"
            ")\n"
            "run_aias(\n"
            "    'assessment-audit', '--config', CONFIG,\n"
            "    '--output-dir', str(readiness_dir),\n"
            ")\n"
            "readiness = json.loads(\n"
            "    (readiness_dir / 'assessment_readiness.json').read_text(encoding='utf-8')\n"
            ")\n"
            "print('Assessment readiness:', readiness['overall_status'])\n"
            "display(pd.DataFrame(readiness['checks'])[\n"
            "    ['criterion', 'check_id', 'status', 'message', 'evidence']\n"
            "])"
        ),
        nbf.v4.new_markdown_cell("## 12. 생성 결과 위치 요약"),
        nbf.v4.new_code_cell(
            "expected_outputs = {\n"
            "    'benchmark_table': benchmark_dir / 'benchmark_comparison.csv',\n"
            "    'benchmark_report': benchmark_dir / 'reports/benchmark_report.docx',\n"
            "    'model_selection': benchmark_dir / 'model_selection.yaml',\n"
            "    'selected_training': benchmark_dir / 'selected_training_result.json',\n"
            "    'quantization_table': quantization_dir / 'quantization_comparison.csv',\n"
            "    'quantization_report': "
            "quantization_dir / 'reports/quantization_report.docx',\n"
            "    'quantization_selection': quantization_dir / 'quantization_selection.yaml',\n"
            "    'final_test': quantization_dir / 'final_test_result.json',\n"
            "    'readiness_json': readiness_dir / 'assessment_readiness.json',\n"
            "    'readiness_markdown': readiness_dir / 'assessment_readiness.md',\n"
            "    'experiment_database': Path(DRIVE_ROOT) / 'backdata/experiments.sqlite3',\n"
            "}\n"
            "display(pd.DataFrame([\n"
            "    {'artifact': name, 'exists': path.exists(), 'path': str(path)}\n"
            "    for name, path in expected_outputs.items()\n"
            "]))\n\n"
            "if IS_PUBLIC_PROXY:\n"
            "    print(\n"
            "        '다음 단계: DATA_MODE을 PRIVATE_MANUFACTURING으로 바꾸고 승인된 제조 '\n"
            "        '녹음·정답 전사를 넣은 뒤 같은 순서를 다시 실행합니다.'\n"
            "    )\n"
            "else:\n"
            "    print(\n"
            "        '최종 보고서와 오류 샘플을 사람이 검수하고 human_review_signoff.yaml을 '\n"
            "        '완료한 뒤 assessment-audit --fail-on-blocker로 최종 확인하세요.'\n"
            "    )"
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(OUTPUT)],
        check=True,
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
