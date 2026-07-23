from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_model_benchmark_quantization.ipynb"


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": "AI Specialist Whisper Model and Quantization Benchmark",
            "provenance": [],
        },
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Whisper 모델·양자화 자동 비교\n\n"
            "고정 Validation 데이터에서 Whisper 모델 크기를 비교하고, 사람이 선택한 모델의 "
            "FP16·INT8-FP16 양자화를 비교한 뒤, 선택된 조합을 고정 Test에서 최종 평가합니다.\n\n"
            "**보안:** 이 노트북은 카메라·마이크·패스키를 요청하지 않습니다. 공개 저장소와 공개 "
            "데이터만 사용하며, 실제 제조 음성은 승인·비식별 여부를 확인한 뒤 사용해야 합니다."
        ),
        nbf.v4.new_code_cell(
            'GITHUB_REPO_URL = "https://github.com/Pronesis9758/aias-specialist-asr.git"\n'
            'GITHUB_BRANCH = "codex/whisper-benchmark-quantization"  # PR 검증 후 main\n'
            'PROJECT_DIR = "/content/AIAS"\n'
            'DRIVE_ROOT = "/content/drive/MyDrive/AI_Specialist_ASR_Project"\n'
            'BASE_CONFIG = "configs/colab_public_sample.yaml"\n'
            'MODEL_MATRIX = "configs/benchmarks/whisper_models_colab.yaml"\n'
            'QUANTIZATION_SPEC = "configs/quantization/whisper_quantization_colab.yaml"\n'
            'BENCHMARK_ID = "public-whisper-model-benchmark-v1"\n'
            'QUANTIZATION_ID = "public-whisper-quantization-v1"'
        ),
        nbf.v4.new_markdown_cell("## 1. GPU 확인과 Google Drive 연결"),
        nbf.v4.new_code_cell(
            '!nvidia-smi\nfrom google.colab import drive\n\ndrive.mount("/content/drive")'
        ),
        nbf.v4.new_markdown_cell(
            "## 2. GitHub 코드 동기화\n\n현재 저장소는 공개이므로 GitHub 토큰이 필요하지 않습니다."
        ),
        nbf.v4.new_code_cell(
            "import os, subprocess\n\n"
            "if not os.path.exists(PROJECT_DIR):\n"
            "    subprocess.run(\n"
            '        ["git", "clone", "--branch", GITHUB_BRANCH, "--single-branch",\n'
            "         GITHUB_REPO_URL, PROJECT_DIR],\n"
            "        check=True,\n"
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
            'print("Git commit:", subprocess.check_output(["git", "rev-parse", "HEAD"],\n'
            "      text=True).strip())"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. 의존성 설치\n\n"
            "Whisper 원본 checkpoint를 CTranslate2 FP16·INT8 형식으로 변환하기 위해 "
            "Transformers 학습 extra를 함께 설치합니다."
        ),
        nbf.v4.new_code_cell(
            "%pip uninstall -y torchao\n"
            '%pip install -q -e ".[train]" "transformers>=4.46,<5" "peft>=0.14,<0.19"\n'
            "import sys, torch, ctranslate2\n\n"
            'print({"python": sys.executable, "cuda": torch.cuda.is_available(),\n'
            '       "gpu_compute_types": sorted(ctranslate2.get_supported_compute_types("cuda"))})'
        ),
        nbf.v4.new_markdown_cell(
            "## 4. 공개 한국어 데이터 준비\n\n"
            "공개 Zeroth-Korean 샘플을 Drive에 준비합니다. 실제 녹음으로 바꿀 때도 동일한 "
            "Manifest의 Validation/Test 분리를 유지해야 합니다."
        ),
        nbf.v4.new_code_cell(
            "def run_aias(*args):\n"
            "    command = [sys.executable, '-m', 'aias_specialist.cli', *args]\n"
            '    print("Running:", " ".join(command))\n'
            "    subprocess.run(command, check=True)\n\n"
            "run_aias('prepare-hf-dataset', '--config', BASE_CONFIG)\n"
            "run_aias('doctor', '--config', BASE_CONFIG)"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Whisper 모델 일괄 비교\n\n"
            "Tiny, Base, Small, Medium, Large-v3, Turbo를 같은 Validation 데이터·FP16·beam "
            "size 5로 실행합니다. 완료된 모델은 Drive에 기록되어 재실행 시 건너뜁니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias('model-matrix-lock', '--matrix', MODEL_MATRIX)\n"
            "run_aias('benchmark-models', '--matrix', MODEL_MATRIX)"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            'benchmark_dir = Path(DRIVE_ROOT) / "artifacts/benchmarks" / BENCHMARK_ID\n'
            'leaderboard = pd.read_csv(benchmark_dir / "benchmark_comparison.csv")\n'
            "display(leaderboard.sort_values(['status', 'rank']))\n"
            'print("Report:", benchmark_dir / "reports/benchmark_report.docx")'
        ),
        nbf.v4.new_markdown_cell(
            "## 6. 모델 선택 — 사람이 확인하는 단계\n\n"
            "`SELECTED_MODEL`을 비교표의 `member_id` 중 하나로 바꾸고 선택 이유를 작성합니다."
        ),
        nbf.v4.new_code_cell(
            'SELECTED_MODEL = "small"\n'
            'REVIEWER = "Pronesis9758"\n'
            "MODEL_SELECTION_REASON = (\n"
            '    "Validation CER, 제조 용어 재현율, RTF와 GPU 메모리의 균형을 검토하여 선택"\n'
            ")\n\n"
            "run_aias(\n"
            "    'select-model', '--benchmark-dir', str(benchmark_dir),\n"
            "    '--model-id', SELECTED_MODEL, '--reviewer', REVIEWER,\n"
            "    '--reason', MODEL_SELECTION_REASON,\n"
            ")\n"
            'model_selection = benchmark_dir / "model_selection.yaml"\n'
            'print("Selection:", model_selection)'
        ),
        nbf.v4.new_markdown_cell(
            "## 7. 선택 모델 양자화 비교\n\n"
            "기본 실행은 FP16과 INT8-FP16입니다. FP32와 CPU INT8은 양자화 YAML에서 "
            "`enabled: true`로 바꾸면 추가할 수 있습니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'quantization-sweep', '--spec', QUANTIZATION_SPEC,\n"
            "    '--selection', str(model_selection),\n"
            ")"
        ),
        nbf.v4.new_code_cell(
            'quantization_dir = Path(DRIVE_ROOT) / "artifacts/quantization" / QUANTIZATION_ID\n'
            'quantization_table = pd.read_csv(quantization_dir / "quantization_comparison.csv")\n'
            "display(quantization_table.sort_values(['status', 'rank']))\n"
            'print("Report:", quantization_dir / "reports/quantization_report.docx")'
        ),
        nbf.v4.new_markdown_cell(
            "## 8. 양자화 단계 선택 — 사람이 확인하는 단계\n\n"
            "FP16 대비 정확도 저하와 속도·메모리 이득을 확인한 뒤 선택합니다."
        ),
        nbf.v4.new_code_cell(
            'SELECTED_VARIANT = "int8-float16"\n'
            "QUANTIZATION_SELECTION_REASON = (\n"
            '    "FP16 대비 CER와 제조 용어 재현율 손실이 허용 범위이며 메모리 효율이 좋아 선택"\n'
            ")\n\n"
            "run_aias(\n"
            "    'select-quantization', '--quantization-dir', str(quantization_dir),\n"
            "    '--variant-id', SELECTED_VARIANT, '--reviewer', REVIEWER,\n"
            "    '--reason', QUANTIZATION_SELECTION_REASON,\n"
            ")\n"
            'quantization_selection = quantization_dir / "quantization_selection.yaml"'
        ),
        nbf.v4.new_markdown_cell(
            "## 9. 고정 Test 최종평가\n\n"
            "모델과 양자화 선택이 끝난 이후에만 Test split을 실행합니다."
        ),
        nbf.v4.new_code_cell(
            "run_aias(\n"
            "    'finalize-evaluation', '--selection', str(quantization_selection),\n"
            "    '--config', BASE_CONFIG,\n"
            ")\n"
            'print("Final result:", quantization_dir / "final_test_result.json")'
        ),
        nbf.v4.new_markdown_cell(
            "## 산출물\n\n"
            "- 모델 비교 CSV·그래프·Word 보고서\n"
            "- 모델 선택자와 선택 이유\n"
            "- 양자화 비교 CSV·그래프·Word 보고서\n"
            "- 양자화 선택자와 선택 이유\n"
            "- 고정 Test 최종 예측·지표·Word 보고서\n"
            "- 모든 독립 run과 SQLite 실험 이력\n\n"
            "최종 보고서 결론과 실제 제조 데이터의 정답·개인정보 승인은 반드시 사람이 검수합니다."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
