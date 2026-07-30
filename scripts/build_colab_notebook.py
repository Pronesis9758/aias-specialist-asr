from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_runner.ipynb"


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": "AI Specialist ASR Colab Runner", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# AI Specialist ASR - Colab Runner\n\n"
            "## Goal\n"
            "GitHub의 동일 코드베이스를 사용해 공개 한국어 음성 샘플을 자동 준비하고 "
            "Baseline과 Whisper LoRA 학습을 실행합니다. 데이터, checkpoint, 실행 이력, "
            "Word 보고서는 Google Drive에 보존합니다."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "런타임 유형을 **T4 GPU** 이상으로 바꾼 뒤 위에서부터 순서대로 실행합니다. "
            "공개 Zeroth-Korean 샘플만 사용하므로 별도 음성 파일은 필요하지 않습니다. "
            "실제 사내 민감 데이터는 승인 없이 Colab에 업로드하지 마세요."
        ),
        nbf.v4.new_code_cell(
            'GITHUB_REPO_URL = "https://github.com/Pronesis9758/aias-specialist-asr.git"\n'
            'GITHUB_BRANCH = "codex/initial-asr-automation"  # PR 병합 후 main으로 변경\n'
            'PROJECT_DIR = "/content/AIAS"\n'
            'DRIVE_ROOT = "/content/drive/MyDrive/AI_Specialist_ASR_Project"\n'
            'CONFIG = "configs/colab_public_sample.yaml"'
        ),
        nbf.v4.new_markdown_cell("## Steps\n\n### 1. GPU와 Drive 연결"),
        nbf.v4.new_code_cell(
            '!nvidia-smi\nfrom google.colab import drive\n\ndrive.mount("/content/drive")'
        ),
        nbf.v4.new_markdown_cell(
            "### 2. GitHub 코드 동기화\n\n"
            "공개 저장소는 그대로 실행됩니다. 비공개 저장소라면 Colab 왼쪽 열쇠 아이콘의 "
            "Secrets에 `GITHUB_TOKEN`을 추가하고 이 노트북의 액세스를 켜세요. 토큰에는 이 "
            "저장소를 읽을 수 있는 최소 권한만 부여합니다."
        ),
        nbf.v4.new_code_cell(
            "import os, stat, subprocess\n"
            "from pathlib import Path\n\n"
            "clone_env = os.environ.copy()\n"
            'askpass = Path("/content/git-askpass.sh")\n'
            "try:\n"
            "    from google.colab import userdata\n\n"
            '    token = userdata.get("GITHUB_TOKEN")\n'
            "except Exception:\n"
            "    token = None\n"
            "if token:\n"
            "    askpass.write_text(\n"
            '        "#!/bin/sh\\n"\n'
            "        'case \"$1\" in\\n'\n"
            "        '  *Username*) echo \"x-access-token\";;\\n'\n"
            "        '  *) echo \"$GITHUB_TOKEN\";;\\n'\n"
            '        "esac\\n"\n'
            "    )\n"
            "    askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)\n"
            "    clone_env.update(\n"
            "        {\n"
            '            "GIT_ASKPASS": str(askpass),\n'
            '            "GIT_TERMINAL_PROMPT": "0",\n'
            '            "GITHUB_TOKEN": token,\n'
            "        }\n"
            "    )\n"
            "try:\n"
            "    if not os.path.exists(PROJECT_DIR):\n"
            "        subprocess.run(\n"
            '            ["git", "clone", "--branch", GITHUB_BRANCH, "--single-branch",\n'
            "             GITHUB_REPO_URL, PROJECT_DIR],\n"
            "            env=clone_env,\n"
            "            check=True,\n"
            "        )\n"
            "    else:\n"
            "        subprocess.run(\n"
            '            ["git", "-C", PROJECT_DIR, "fetch", "origin", GITHUB_BRANCH],\n'
            "            env=clone_env,\n"
            "            check=True,\n"
            "        )\n"
            "        subprocess.run(\n"
            '            ["git", "-C", PROJECT_DIR, "checkout", GITHUB_BRANCH],\n'
            "            env=clone_env,\n"
            "            check=True,\n"
            "        )\n"
            "        subprocess.run(\n"
            "            [\n"
            '                "git", "-C", PROJECT_DIR, "merge", "--ff-only",\n'
            '                f"origin/{GITHUB_BRANCH}",\n'
            "            ],\n"
            "            env=clone_env,\n"
            "            check=True,\n"
            "        )\n"
            "finally:\n"
            "    if askpass.exists():\n"
            "        askpass.unlink()\n"
            "os.chdir(PROJECT_DIR)\n"
            "result = subprocess.run("
            '["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)\n'
            'print("Git commit:", result.stdout.strip())'
        ),
        nbf.v4.new_markdown_cell("### 3. 의존성 설치"),
        nbf.v4.new_code_cell(
            "# Colab 이미지를 기준으로 설치합니다. 기본 CUDA PyTorch는 유지하고,\n"
            "# Colab에 사전 설치된 torchao와 Gradio는\n"
            "# Whisper/PEFT 의존성과 충돌할 수 있어 제거합니다.\n"
            "%pip uninstall -y torchao gradio gradio-client\n"
            '%pip install -q -e ".[train]" "transformers>=4.46,<5" "peft>=0.14,<0.19"\n'
            "import sys, torch, transformers, peft\n"
            'print({"python": sys.executable, "torch": torch.__version__,\n'
            '       "cuda": torch.cuda.is_available(),\n'
            '       "transformers": transformers.__version__,\n'
            '       "peft": peft.__version__})'
        ),
        nbf.v4.new_markdown_cell(
            "### 4. 공개 한국어 음성 샘플 준비\n\n"
            "CC BY 4.0 Zeroth-Korean의 고정된 리비전에서 학습 40·검증 8·테스트 16개를 "
            "스트리밍합니다. 두 번째 실행부터는 Drive의 완성된 manifest와 오디오를 재사용합니다."
        ),
        nbf.v4.new_code_cell(
            '!{sys.executable} -m aias_specialist.cli prepare-hf-dataset --config "{CONFIG}"'
        ),
        nbf.v4.new_markdown_cell("### 5. 모델 버전 고정과 Baseline 실행"),
        nbf.v4.new_code_cell(
            '!{sys.executable} -m aias_specialist.cli doctor --config "{CONFIG}"\n'
            '!{sys.executable} -m aias_specialist.cli model-lock --config "{CONFIG}"\n'
            '!{sys.executable} -m aias_specialist.cli download-model --config "{CONFIG}"\n'
            '!{sys.executable} -m aias_specialist.cli run --config "{CONFIG}"'
        ),
        nbf.v4.new_markdown_cell("### 6. Whisper LoRA 학습"),
        nbf.v4.new_code_cell(
            '!{sys.executable} -m aias_specialist.cli train-whisper --config "{CONFIG}"'
        ),
        nbf.v4.new_markdown_cell("## Checks\n\nDrive에 결과와 checkpoint가 남았는지 확인합니다."),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "for required in [\n"
            '    Path(DRIVE_ROOT) / "artifacts/runs",\n'
            '    Path(DRIVE_ROOT) / "backdata/experiments.sqlite3",\n'
            '    Path(DRIVE_ROOT) / "checkpoints",\n'
            "]:\n"
            '    print(required, "OK" if required.exists() else "MISSING")\n\n'
            'training_runs = sorted((Path(DRIVE_ROOT) / "artifacts/runs").glob("train-*"))\n'
            "if training_runs:\n"
            "    latest = training_runs[-1]\n"
            '    result = json.loads((latest / "metrics.json").read_text(encoding="utf-8"))\n'
            '    print("Latest training run:", latest.name)\n'
            '    print("Base test WER:", result["baseline"]["wer"])\n'
            '    print("LoRA test WER:", result["lora"]["wer"])\n'
            '    reduction = result["lora_improvement"]["wer_absolute_reduction"]\n'
            '    print("WER absolute reduction:", reduction)\n'
            '    print("Report:", latest / "reports/evaluation_report.docx")'
        ),
        nbf.v4.new_markdown_cell(
            "## Next Steps\n\n"
            "이 노트북은 데이터 준비, Baseline 실행, LoRA 학습, 동일 test split의 "
            "Base Whisper·Best LoRA 비교, SQLite 등록, Word 보고서 생성까지 자동화한 "
            "소규모 기술 검증입니다. 공개 데이터가 일반 한국어이므로 제조 현장 성능 근거로 "
            "사용하지 않습니다. 실제 현장 음성·정답 문장을 승인된 저장소에 준비한 후 동일한 "
            "manifest 스키마로 교체하고, 최종 모델 선택과 보고서 결론은 현업 담당자가 검토합니다."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
