from __future__ import annotations

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
            "# 제조 음성 ASR 심사 실행 노트북\n\n"
            "## Goal\n\n"
            "승인된 제조 음성과 검수 전사가 준비된 뒤 모델 비교, 사람 선택, 선택 모델 LoRA, "
            "양자화, 고정 Test와 심사 증빙 점검을 순서대로 실행합니다.\n\n"
            "**보안:** 카메라·마이크·패스키를 사용하지 않습니다. 실제 음성은 GitHub에 "
            "올리지 않고 승인된 비공개 Drive 경로만 사용합니다."
        ),
        nbf.v4.new_code_cell(
            'GITHUB_REPO_URL = "https://github.com/Pronesis9758/aias-specialist-asr.git"\n'
            'GITHUB_BRANCH = "codex/whisper-benchmark-quantization"  # 병합 후 main\n'
            'PROJECT_DIR = "/content/AIAS"\n'
            'DRIVE_ROOT = "/content/drive/MyDrive/AI_Specialist_ASR_Project"\n'
            'PRIVATE_ROOT = f"{DRIVE_ROOT}/data/private/manufacturing"\n'
            'CONFIG = "configs/manufacturing_private_template.yaml"\n'
            'MODEL_MATRIX = "configs/benchmarks/manufacturing_whisper_models_template.yaml"\n'
            "QUANTIZATION_SPEC = (\n"
            '    "configs/quantization/manufacturing_whisper_quantization_template.yaml"\n'
            ")\n"
            'BENCHMARK_ID = "manufacturing-whisper-model-benchmark-v1"\n'
            'QUANTIZATION_ID = "manufacturing-whisper-quantization-v1"'
        ),
        nbf.v4.new_markdown_cell("## Setup\n\n### 1. GPU와 비공개 Drive 연결"),
        nbf.v4.new_code_cell(
            '!nvidia-smi\nfrom google.colab import drive\n\ndrive.mount("/content/drive")'
        ),
        nbf.v4.new_markdown_cell("### 2. GitHub 코드 동기화"),
        nbf.v4.new_code_cell(
            "import os, subprocess\n\n"
            "if not os.path.exists(PROJECT_DIR):\n"
            "    subprocess.run(\n"
            '        ["git", "clone", "--branch", GITHUB_BRANCH, "--single-branch",\n'
            "         GITHUB_REPO_URL, PROJECT_DIR], check=True,\n"
            "    )\n"
            "else:\n"
            "    subprocess.run(\n"
            '        ["git", "-C", PROJECT_DIR, "fetch", "origin", GITHUB_BRANCH], check=True\n'
            "    )\n"
            "    subprocess.run(\n"
            '        ["git", "-C", PROJECT_DIR, "checkout", GITHUB_BRANCH], check=True\n'
            "    )\n"
            "    subprocess.run(\n"
            '        ["git", "-C", PROJECT_DIR, "merge", "--ff-only",\n'
            '         f"origin/{GITHUB_BRANCH}"], check=True\n'
            "    )\n"
            "os.chdir(PROJECT_DIR)\n"
            'print("Git commit:", subprocess.check_output(\n'
            '    ["git", "rev-parse", "HEAD"], text=True).strip())'
        ),
        nbf.v4.new_markdown_cell("### 3. 의존성 설치"),
        nbf.v4.new_code_cell(
            "%pip uninstall -y torchao\n"
            '%pip install -q -e ".[train]" "transformers>=4.46,<5" "peft>=0.14,<0.19"\n'
            "import sys\n\n"
            "def run_aias(*args):\n"
            "    command = [sys.executable, '-m', 'aias_specialist.cli', *args]\n"
            '    print("Running:", " ".join(command))\n'
            "    subprocess.run(command, check=True)"
        ),
        nbf.v4.new_markdown_cell(
            "### 4. 승인·manifest·사람 검토 양식 준비\n\n"
            "기존 파일은 덮어쓰지 않습니다. 생성된 세 파일을 실제 값으로 채우고 음성을 "
            "`PRIVATE_ROOT/audio/` 아래에 업로드합니다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import shutil\n\n"
            "private_root = Path(PRIVATE_ROOT)\n"
            "(private_root / 'audio').mkdir(parents=True, exist_ok=True)\n"
            "templates = {\n"
            "    Path('data/templates/manufacturing_manifest_template.csv'): "
            "private_root / 'manifest.csv',\n"
            "    Path('data/templates/data_approval_template.yaml'): "
            "private_root / 'data_approval.yaml',\n"
            "    Path('data/templates/human_review_signoff_template.yaml'): "
            "private_root / 'human_review_signoff.yaml',\n"
            "    Path('configs/assessment/acceptance_criteria_template.yaml'): "
            "private_root / 'acceptance_criteria.yaml',\n"
            "}\n"
            "for source, destination in templates.items():\n"
            "    if not destination.exists():\n"
            "        shutil.copy2(source, destination)\n"
            "        print('Created:', destination)\n"
            "    else:\n"
            "        print('Preserved existing:', destination)"
        ),
        nbf.v4.new_markdown_cell(
            "## Checks\n\n### 5. 데이터 전 준비도 점검\n\n"
            "음성과 승인정보가 아직 없으면 `not_ready`가 정상입니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'assessment-audit', '--config', CONFIG,\n"
            "    '--output-dir', f'{DRIVE_ROOT}/reports/assessment_readiness',\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## Steps\n\n아래부터는 승인 문서, manifest, 음성과 전사 검수가 완료된 뒤 실행합니다."
        ),
        nbf.v4.new_markdown_cell("### 6. 모델 비교"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "\n"
            "run_aias('benchmark-models', '--matrix', MODEL_MATRIX)\n"
            "benchmark_dir = Path(DRIVE_ROOT) / 'artifacts/benchmarks' / BENCHMARK_ID\n"
            "display(pd.read_csv(benchmark_dir / 'benchmark_comparison.csv'))"
        ),
        nbf.v4.new_markdown_cell("### 7. 모델 선택"),
        nbf.v4.new_code_cell(
            'SELECTED_MODEL = "small"\n'
            'REVIEWER = "TO_BE_COMPLETED"\n'
            'MODEL_REASON = "TO_BE_COMPLETED: 정확도·속도·메모리·거버넌스 근거"\n\n'
            "run_aias(\n"
            "    'select-model', '--benchmark-dir', str(benchmark_dir),\n"
            "    '--model-id', SELECTED_MODEL, '--reviewer', REVIEWER,\n"
            "    '--reason', MODEL_REASON,\n"
            ")\n"
            "model_selection = benchmark_dir / 'model_selection.yaml'"
        ),
        nbf.v4.new_markdown_cell("### 8. 선택 모델 LoRA"),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'train-selected-whisper', '--selection', str(model_selection),\n"
            "    '--config', CONFIG,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell("### 9. 양자화 비교 및 선택"),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'quantization-sweep', '--spec', QUANTIZATION_SPEC,\n"
            "    '--selection', str(model_selection),\n"
            ")\n"
            "quantization_dir = Path(DRIVE_ROOT) / 'artifacts/quantization' / QUANTIZATION_ID\n"
            "display(pd.read_csv(quantization_dir / 'quantization_comparison.csv'))"
        ),
        nbf.v4.new_code_cell(
            'SELECTED_VARIANT = "float16"\n'
            'QUANTIZATION_REASON = "TO_BE_COMPLETED: 정확도 손실·속도·메모리 근거"\n\n'
            "run_aias(\n"
            "    'select-quantization', '--quantization-dir', str(quantization_dir),\n"
            "    '--variant-id', SELECTED_VARIANT, '--reviewer', REVIEWER,\n"
            "    '--reason', QUANTIZATION_REASON,\n"
            ")\n"
            "quantization_selection = quantization_dir / 'quantization_selection.yaml'"
        ),
        nbf.v4.new_markdown_cell("### 10. 고정 Test 최종평가"),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'finalize-evaluation', '--selection', str(quantization_selection),\n"
            "    '--config', CONFIG,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## Next Steps\n\n"
            "최종 보고서와 오류 샘플을 사람이 검수하고 `human_review_signoff.yaml`을 완료한 뒤 "
            "마지막 점검을 실행합니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'assessment-audit', '--config', CONFIG,\n"
            "    '--output-dir', f'{DRIVE_ROOT}/reports/assessment_readiness',\n"
            "    '--fail-on-blocker',\n"
            ")"
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
