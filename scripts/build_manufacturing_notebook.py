from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_manufacturing_assessment.ipynb"


def _line_comment(
    line: str,
    next_code: str = "",
    future_code: tuple[str, ...] = (),
) -> str:
    """Return a concise, value-aware Korean explanation for one notebook code line."""
    stripped = line.strip()
    exact_comments = {
        'DATA_MODE = "SYNTHETIC_MANUFACTURING"': (
            "현재 실행을 합성 제조 TTS 데이터 모드로 선택합니다. 실제 데이터 사용 시 값을 바꿉니다."
        ),
        "RUN_LORA = True": "선택된 Whisper의 LoRA 학습 단계를 실행하도록 켭니다.",
        "RUN_QUANTIZATION = True": "선택 모델의 정밀도별 양자화 비교 단계를 실행하도록 켭니다.",
        "ENABLE_INFORMATION_RETRIEVAL = True": (
            "BM25 기반 제조 용어 정보검색 보정을 실행 설정에서 활성화합니다."
        ),
        "ENABLE_NEAREST_NEIGHBOR = True": (
            "벡터 최근접 이웃 검색 기반 제조 용어 보정을 실행 설정에서 활성화합니다."
        ),
        "RUN_DISTILLATION = False  # 추가 GPU 비용이 큰 선택 실험": (
            "Teacher·Student 지식 증류는 비용이 커 기본 비활성화하며 필요할 때 True로 바꿉니다."
        ),
        'GITHUB_REPO_URL = "https://github.com/Pronesis9758/aias-specialist-asr.git"': (
            "Colab에서 복제할 ASR 프로젝트 GitHub 저장소 주소를 지정합니다."
        ),
        'GITHUB_BRANCH = "codex/whisper-benchmark-quantization"  # PR 병합 후 main': (
            "벤치마크·양자화 기능이 있는 작업 브랜치를 선택합니다. PR 병합 후 main으로 바꿉니다."
        ),
        'PROJECT_DIR = "/content/AIAS"': (
            "GitHub 코드를 복제할 Colab 임시 경로입니다. 런타임 종료·초기화 시 삭제됩니다."
        ),
        'DRIVE_ROOT = "/content/drive/MyDrive/AI_Specialist_ASR_Project"': (
            "모델·결과·보고서를 보존할 내 Google Drive 경로입니다. 런타임 종료 후에도 유지됩니다."
        ),
        'DRIVE_MOUNT_POINT = Path("/content/drive")': (
            "Colab 런타임에서 Google Drive를 연결할 기준 폴더를 지정합니다."
        ),
        'DRIVE_MY_DRIVE = DRIVE_MOUNT_POINT / "MyDrive"': (
            "Drive 연결 성공 여부를 판별할 내 드라이브 폴더 경로를 지정합니다."
        ),
        "MODE_SETTINGS = {": (
            "데이터 출처별 설정·모델 후보·양자화 명세·실험 ID를 하나의 표로 정의합니다."
        ),
        '"PUBLIC_PROXY": {': (
            "공개 Zeroth 한국어 음성으로 전체 파이프라인만 검증하는 프록시 모드를 정의합니다."
        ),
        '"SYNTHETIC_MANUFACTURING": {': (
            "제조 용어·난이도·소음 조건을 균형화한 합성 TTS 600개를 사용합니다.\n"
            "    # 실제 제조 데이터 투입 전 전체 연구 기능을 검증하는 기본 실행 모드입니다."
        ),
        '"PRIVATE_MANUFACTURING": {': (
            "승인된 실제 제조 녹음과 사람 검수 전사를 사용하는 최종 연구 모드를 정의합니다."
        ),
        '"config": "configs/public_proxy_assessment.yaml",': (
            "공개 Zeroth 데이터·평가·Drive 산출물 경로가 담긴 설정 파일을 연결합니다."
        ),
        '"config": "configs/synthetic_manufacturing_sample.yaml",': (
            "합성 제조 데이터 분할·학습·평가 조건이 담긴 설정 파일을 연결합니다."
        ),
        '"config": "configs/manufacturing_private_template.yaml",': (
            "실제 제조 데이터 경로와 엄격한 거버넌스 조건을 입력할 템플릿을 연결합니다."
        ),
        '"matrix": "configs/benchmarks/public_proxy_whisper_models.yaml",': (
            "공개 프록시에서 비교할 Whisper 후보와 실행 조건 목록을 지정합니다."
        ),
        '"matrix": "configs/benchmarks/synthetic_manufacturing_whisper_models.yaml",': (
            "합성 제조 데이터에서 비교할 tiny·base·small 후보 목록을 지정합니다."
        ),
        '"matrix": "configs/benchmarks/manufacturing_whisper_models_template.yaml",': (
            "실제 제조 데이터에서 사람이 검토할 확장 Whisper 후보 목록을 지정합니다."
        ),
        '"quantization": "configs/quantization/public_proxy_whisper_quantization.yaml",': (
            "공개 프록시 모델에 적용할 float16·int8 양자화 비교 조건을 지정합니다."
        ),
        (
            '"quantization": "configs/quantization/'
            'synthetic_manufacturing_whisper_quantization.yaml",'
        ): ("합성 제조 선택 모델의 float16·int8-float16 비교 조건을 지정합니다."),
        (
            '"quantization": "configs/quantization/'
            'manufacturing_whisper_quantization_template.yaml",'
        ): ("실제 제조 선택 모델에 적용할 양자화 후보와 허용 손실 조건을 지정합니다."),
        '"benchmark_id": "public-proxy-whisper-model-benchmark-v1",': (
            "공개 프록시 모델 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        '"benchmark_id": "synthetic-manufacturing-whisper-model-benchmark-v1",': (
            "합성 제조 모델 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        '"benchmark_id": "manufacturing-whisper-model-benchmark-v1",': (
            "실제 제조 모델 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        '"quantization_id": "public-proxy-whisper-quantization-v1",': (
            "공개 프록시 양자화 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        '"quantization_id": "synthetic-manufacturing-whisper-quantization-v1",': (
            "합성 제조 양자화 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        '"quantization_id": "manufacturing-whisper-quantization-v1",': (
            "실제 제조 양자화 비교 산출물을 모을 고유 실험 ID를 지정합니다."
        ),
        "mode = MODE_SETTINGS[DATA_MODE]": "선택한 데이터 모드의 설정 묶음만 꺼냅니다.",
        'CONFIG = mode["config"]': "선택 모드의 데이터·학습·평가 설정 파일 경로를 사용합니다.",
        'MODEL_MATRIX = mode["matrix"]': (
            "선택 모드에서 비교할 Whisper 후보 목록 경로를 사용합니다."
        ),
        'QUANTIZATION_SPEC = mode["quantization"]': (
            "선택 모델에 적용할 양자화 후보와 평가 조건 파일 경로를 사용합니다."
        ),
        'BENCHMARK_ID = mode["benchmark_id"]': "모델 비교 결과를 저장할 실험 그룹 ID를 사용합니다.",
        'QUANTIZATION_ID = mode["quantization_id"]': (
            "양자화 비교 결과를 저장할 실험 그룹 ID를 사용합니다."
        ),
        'PRIVATE_ROOT = f"{DRIVE_ROOT}/data/private/manufacturing"': (
            "실제 제조 음성·정답·승인 문서를 둘 비공개 Drive 폴더를 지정합니다."
        ),
        'IS_PUBLIC_PROXY = DATA_MODE == "PUBLIC_PROXY"': (
            "현재 실행이 공개 데이터 기능 검증 모드인지 표시합니다."
        ),
        'IS_SYNTHETIC_MANUFACTURING = DATA_MODE == "SYNTHETIC_MANUFACTURING"': (
            "현재 실행이 합성 제조 데이터 기능 검증 모드인지 표시합니다."
        ),
        "IS_AUTOMATED_PROXY = IS_PUBLIC_PROXY or IS_SYNTHETIC_MANUFACTURING": (
            "공개·합성 모드에서는 사람 결정 대신 자동 선택을 허용하도록 표시합니다."
        ),
        'PROJECT_SRC = os.path.join(PROJECT_DIR, "src")': (
            "editable 설치 직후 프로젝트 패키지를 찾을 src 절대 경로를 계산합니다."
        ),
        'command = [sys.executable, "-m", "aias_specialist.cli", *args]': (
            "현재 Python으로 AIAS CLI와 전달받은 세부 명령을 실행할 명령 배열을 만듭니다."
        ),
        'environment = {**os.environ, "PYTHONUNBUFFERED": "1"}': (
            "학습·평가 진행 로그가 지연 없이 Colab에 표시되도록 실행 환경을 만듭니다."
        ),
        "source_config_path = Path(CONFIG)": (
            "선택 데이터 모드의 원본 YAML을 읽기 위한 Path 객체를 만듭니다."
        ),
        'runtime_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))': (
            "GitHub 원본을 보존하면서 기능 옵션을 적용할 실행용 설정 사본을 읽습니다."
        ),
        "runtime_config = yaml.safe_load(source_config_path.read_text(encoding='utf-8'))": (
            "GitHub 원본을 보존하면서 기능 옵션을 적용할 실행용 설정 사본을 읽습니다."
        ),
        "runtime_config_path = Path(\"/content/aias_runtime_config.yaml\")": (
            "기능 토글이 반영된 임시 설정 YAML의 Colab 경로를 지정합니다."
        ),
        "runtime_config_path = Path('/content/aias_runtime_config.yaml')": (
            "기능 토글이 반영된 임시 설정 YAML의 Colab 경로를 지정합니다."
        ),
        'runtime_matrix = yaml.safe_load(Path(MODEL_MATRIX).read_text(encoding="utf-8"))': (
            "모델 비교가 실행용 설정을 사용하도록 원본 모델 행렬 YAML을 읽습니다."
        ),
        "runtime_matrix = yaml.safe_load(Path(MODEL_MATRIX).read_text(encoding='utf-8'))": (
            "모델 비교가 실행용 설정을 사용하도록 원본 모델 행렬 YAML을 읽습니다."
        ),
        "runtime_matrix_path = Path(\"/content/aias_runtime_model_matrix.yaml\")": (
            "실행용 base config가 연결된 임시 모델 행렬 경로를 지정합니다."
        ),
        "runtime_matrix_path = Path('/content/aias_runtime_model_matrix.yaml')": (
            "실행용 base config가 연결된 임시 모델 행렬 경로를 지정합니다."
        ),
        "runtime_quantization = yaml.safe_load(": (
            "양자화 사용 여부와 실행용 base config를 반영할 원본 명세를 읽기 시작합니다."
        ),
        (
            'runtime_quantization = yaml.safe_load(Path(QUANTIZATION_SPEC).read_text('
            'encoding="utf-8"))'
        ): ("양자화 사용 여부와 실행용 base config를 반영할 원본 명세를 읽습니다."),
        "runtime_quantization_path = Path(\"/content/aias_runtime_quantization.yaml\")": (
            "양자화 토글이 반영된 임시 양자화 명세 경로를 지정합니다."
        ),
        "runtime_quantization_path = Path('/content/aias_runtime_quantization.yaml')": (
            "양자화 토글이 반영된 임시 양자화 명세 경로를 지정합니다."
        ),
        "CONFIG = str(runtime_config_path)": (
            "이후 AIAS 명령이 기능 토글 반영 설정 사본을 사용하도록 경로를 교체합니다."
        ),
        "MODEL_MATRIX = str(runtime_matrix_path)": (
            "이후 모델 비교가 실행용 설정을 연결한 모델 행렬을 사용하도록 교체합니다."
        ),
        "QUANTIZATION_SPEC = str(runtime_quantization_path)": (
            "이후 양자화 단계가 활성화 여부를 반영한 임시 명세를 사용하도록 교체합니다."
        ),
        'public_root = Path(DRIVE_ROOT) / "data/public/zeroth_korean"': (
            "다운로드한 Zeroth 음성과 manifest를 보존할 Drive 폴더를 지정합니다."
        ),
        "synthetic_settings = load_settings(CONFIG)": (
            "합성 제조 manifest와 실험 경로를 YAML 설정에서 읽습니다."
        ),
        "private_root = Path(PRIVATE_ROOT)": (
            "실제 제조 데이터의 비공개 Drive 문자열 경로를 Path 객체로 변환합니다."
        ),
        'benchmark_dir = Path(DRIVE_ROOT) / "artifacts/benchmarks" / BENCHMARK_ID': (
            "모델별 예측·성능표·보고서가 저장된 Drive 벤치마크 폴더를 지정합니다."
        ),
        'benchmark_table = pd.read_csv(benchmark_dir / "benchmark_comparison.csv")': (
            "Whisper 후보별 정확도·속도·메모리·용량 비교표를 읽습니다."
        ),
        'completed = frame.loc[frame["status"].eq("completed")].copy()': (
            "실패 후보를 제외하고 평가가 완료된 모델 또는 양자화 후보만 남깁니다."
        ),
        "selected_row = best_completed_member(benchmark_table)": (
            "완료된 Whisper 후보 중 종합 순위가 가장 높은 행을 선택합니다."
        ),
        'SELECTED_MODEL = str(selected_row["member_id"])': (
            "자동 선택된 Whisper 후보의 member_id를 후속 학습 모델로 사용합니다."
        ),
        'REVIEWER = "AUTOMATED_PUBLIC_PROXY"': (
            "공개 프록시의 자동 선택임을 기록해 사람 검토와 구분합니다."
        ),
        'REVIEWER = "AUTOMATED_SYNTHETIC_FIXTURE"': (
            "합성 데이터의 자동 선택임을 기록해 실제 제조 검토와 구분합니다."
        ),
        'SELECTED_MODEL = "small"  # 비교표를 보고 수정': (
            "실제 제조 모드에서 비교표 검토 후 선택할 모델의 초기 입력값입니다."
        ),
        'REVIEWER = "TO_BE_COMPLETED"': (
            "실제 제조 모델을 검토한 담당자 이름을 반드시 입력해야 하는 자리입니다."
        ),
        'extra_selection_args = ["--automated-proxy"]': (
            "자동 선택 결과를 사람 검토로 오인하지 않도록 프록시 표시 인자를 추가합니다."
        ),
        "extra_selection_args = []": (
            "실제 제조 모드에서는 자동 프록시 표시 없이 사람 검토 기록을 사용합니다."
        ),
        'model_selection = benchmark_dir / "model_selection.yaml"': (
            "선택 모델·revision·성능·선택 사유가 기록된 YAML 경로를 지정합니다."
        ),
        'quantization_dir = Path(DRIVE_ROOT) / "artifacts/quantization" / QUANTIZATION_ID': (
            "양자화별 예측·성능표·보고서가 저장된 Drive 폴더를 지정합니다."
        ),
        'quantization_table = pd.read_csv(quantization_dir / "quantization_comparison.csv")': (
            "양자화 후보별 정확도 변화·속도·메모리·용량 비교표를 읽습니다."
        ),
        "selected_quantization_row = best_completed_member(quantization_table)": (
            "완료된 양자화 후보 중 종합 순위가 가장 높은 행을 선택합니다."
        ),
        'SELECTED_VARIANT = str(selected_quantization_row["member_id"])': (
            "자동 선택된 양자화 variant ID를 최종 Test 평가에 사용합니다."
        ),
        'SELECTED_VARIANT = "float16"  # 비교표를 보고 수정': (
            "실제 제조 모드에서 손실·속도·메모리를 검토한 뒤 선택할 초기 양자화 값입니다."
        ),
        'extra_quantization_args = ["--automated-proxy"]': (
            "양자화 자동 선택을 사람 배포 결정으로 오인하지 않도록 프록시 표시를 추가합니다."
        ),
        "extra_quantization_args = []": (
            "실제 제조 모드에서는 자동 프록시 표시 없이 사람의 양자화 결정을 기록합니다."
        ),
        'quantization_selection = quantization_dir / "quantization_selection.yaml"': (
            "최종 variant·성능·선택 사유가 기록된 YAML 경로를 지정합니다."
        ),
        "distillation_run_dir = None": (
            "지식 증류를 실행하지 않은 상태를 나타내도록 결과 폴더를 비워 둡니다."
        ),
        "before_distillation = set(": (
            "새 증류 실행을 식별하기 위해 실행 전 distill run 폴더 목록을 수집합니다."
        ),
        'before_distillation = set(Path(DRIVE_ROOT, "artifacts/runs").glob("distill-*"))': (
            "새 증류 실행을 식별하기 위해 실행 전 distill run 폴더 목록을 수집합니다."
        ),
        "after_distillation = set(": (
            "증류 명령 완료 후 distill run 폴더 목록을 다시 수집합니다."
        ),
        'after_distillation = set(Path(DRIVE_ROOT, "artifacts/runs").glob("distill-*"))': (
            "증류 명령 완료 후 distill run 폴더 목록을 다시 수집합니다."
        ),
        "new_distillation_runs = sorted(": (
            "실행 전후 차이로 새 증류 결과만 찾아 수정 시각 순으로 정렬합니다."
        ),
        "distillation_run_dir = new_distillation_runs[-1]": (
            "가장 최근에 생성된 지식 증류 run을 결과 확인 대상으로 선택합니다."
        ),
        "quantization_table = None": (
            "양자화를 끈 경우에도 후속 조건 분기가 가능하도록 비교표를 비워 둡니다."
        ),
        "quantization_selection = None": (
            "양자화를 끈 경우 모델 선택을 최종 선택으로 사용할 수 있게 초기화합니다."
        ),
        "final_selection = (": (
            "양자화 실행 여부에 따라 최종 Test에 사용할 선택 YAML을 결정합니다."
        ),
        "final_selection = quantization_selection if RUN_QUANTIZATION else model_selection": (
            "양자화 실행 여부에 따라 최종 Test에 사용할 선택 YAML을 결정합니다."
        ),
        'readiness_dir = Path(DRIVE_ROOT) / "reports/assessment_readiness" / DATA_MODE.lower()': (
            "현재 데이터 모드의 심사 준비도 JSON·Markdown을 저장할 Drive 폴더를 지정합니다."
        ),
        (
            'readiness = json.loads((readiness_dir / "assessment_readiness.json")'
            '.read_text(encoding="utf-8"))'
        ): ("심사 항목별 통과·대기·실패 상태가 담긴 JSON 결과를 읽습니다."),
        "expected_outputs = {": "최종 확인할 필수 표·보고서·선택 기록·백데이터 목록을 정의합니다.",
    }
    if stripped in exact_comments:
        return exact_comments[stripped]

    semantic_comments = {
        '"git",': "운영체제에 설치된 Git 실행 파일을 호출합니다.",
        '"clone",': "원격 저장소를 Colab 임시 디스크로 복제하는 clone 명령을 선택합니다.",
        '"--branch",': "전체 브랜치 중 실행할 특정 브랜치를 지정하는 옵션입니다.",
        "GITHUB_BRANCH,": "앞에서 지정한 벤치마크·양자화 작업 브랜치 이름을 전달합니다.",
        '"--single-branch",': "선택 브랜치 이력만 받아 다운로드 시간과 디스크 사용량을 줄입니다.",
        "GITHUB_REPO_URL,": "복제 원본인 AIAS GitHub 저장소 주소를 전달합니다.",
        "PROJECT_DIR,": "저장소를 복제하거나 Git 명령을 실행할 Colab 로컬 경로를 전달합니다.",
        '"-C",': "현재 셸 위치를 바꾸지 않고 지정 저장소에서 Git을 실행하는 옵션입니다.",
        '"merge",': "원격 브랜치의 최신 커밋을 현재 체크아웃에 반영합니다.",
        '"--ff-only",': "로컬 변경과 충돌하면 새 merge commit을 만들지 않고 안전하게 실패시킵니다.",
        'f"origin/{GITHUB_BRANCH}",': "동기화할 origin 원격 브랜치 이름을 구성합니다.",
        '["git", "-C", PROJECT_DIR, "fetch", "origin", GITHUB_BRANCH],': (
            "프로젝트 폴더의 origin에서 선택 브랜치 최신 이력만 가져오는 Git 명령을 구성합니다."
        ),
        '["git", "-C", PROJECT_DIR, "checkout", GITHUB_BRANCH],': (
            "프로젝트 폴더에서 지정 작업 브랜치로 전환하는 Git 명령을 구성합니다."
        ),
        "check=True,": "Git·CLI 명령이 실패하면 다음 셀로 진행하지 않고 즉시 예외를 발생시킵니다.",
        'encoding="utf-8",': "한글이 손상되지 않도록 실행용 YAML을 UTF-8로 저장합니다.",
        "encoding='utf-8',": "한글이 손상되지 않도록 실행용 YAML을 UTF-8로 저장합니다.",
        "key=lambda path: path.stat().st_mtime,": (
            "새 증류 결과 폴더를 파일 수정 시각 기준으로 정렬하는 함수를 전달합니다."
        ),
        "sys.path.insert(0, PROJECT_SRC)": (
            "방금 설치한 저장소의 src를 Python 검색 경로 최우선 순위에 추가합니다."
        ),
        'print("aias_specialist import OK:", PROJECT_SRC)': (
            "실제로 import되는 프로젝트 소스 경로를 확인 로그로 출력합니다."
        ),
        'print("\\nRunning:", " ".join(command), flush=True)': (
            "실행할 AIAS 명령 전체를 즉시 출력해 장시간 작업의 시작점을 보여줍니다."
        ),
        "subprocess.run(command, check=True, env=environment)": (
            "로그 즉시 출력 환경으로 AIAS 명령을 실행하고 실패 시 파이프라인을 중단합니다."
        ),
        'print("Google Drive already mounted at /content/drive")': (
            "이 런타임은 이미 Drive에 연결됐으므로 추가 승인 없이 재사용한다고 알립니다."
        ),
        'print("Google Drive ready:", DRIVE_MY_DRIVE)': (
            "검증을 통과한 내 드라이브 경로를 출력해 저장 준비 완료를 확인합니다."
        ),
        'run_aias("prepare-hf-dataset", "--config", CONFIG)': (
            "공개 Zeroth 일부를 설정에 맞게 내려받고 manifest·출처 정보를 생성합니다."
        ),
        'print("PUBLIC PROXY: 제조 성능 증거가 아닌 전체동작 검증 데이터입니다.")': (
            "공개 음성 결과를 제조 현장 성능으로 오해하지 않도록 사용 한계를 출력합니다."
        ),
        'synthetic_settings.paths.manifest, backend="faster_whisper"': (
            "설정의 합성 manifest를 Faster-Whisper 입력 규칙으로 검증하도록 전달합니다."
        ),
        'print("SYNTHETIC: 실제 제조 성능이나 사람 검수 증거가 아닙니다.")': (
            "합성 TTS 결과가 실제 제조·사람 검수 증거가 아님을 출력합니다."
        ),
        '"Google Drive 연결에 실패했습니다. accounts.google.com 화면이 검게 "': (
            "인증 팝업 렌더링 실패의 대표 증상을 오류 안내 첫 문장에 포함합니다."
        ),
        '"멈추면 Codex 내장 브라우저 대신 일반 Chrome에서 이 노트북을 "': (
            "문제가 반복될 때 사용할 정상 브라우저와 대상 노트북을 안내합니다."
        ),
        '"열고 팝업·리디렉션을 허용한 뒤 이 셀을 다시 실행하세요."': (
            "Chrome에서 허용할 권한과 재실행할 단계를 안내합니다."
        ),
        ") from exc": "사용자 안내 오류에 Colab의 원래 마운트 실패 원인을 연결해 보존합니다.",
        '(private_root / "audio").mkdir(parents=True, exist_ok=True)': (
            "실제 녹음 WAV를 넣을 비공개 Drive 폴더를 상위 폴더와 함께 생성합니다."
        ),
        (
            'Path("data/templates/manufacturing_manifest_template.csv"): '
            'private_root / "manifest.csv",'
        ): ("음성 경로·정답·split·출처를 작성할 manifest CSV 양식을 비공개 폴더에 배치합니다."),
        (
            'Path("data/templates/data_approval_template.yaml"): '
            'private_root / "data_approval.yaml",'
        ): ("데이터 사용 승인·동의·비식별화 상태를 기록할 YAML 양식을 배치합니다."),
        'Path("data/templates/human_review_signoff_template.yaml"): private_root': (
            "모델·용어·결과의 사람 검토 서명 양식 원본과 비공개 대상 폴더를 연결합니다."
        ),
        '/ "human_review_signoff.yaml",': (
            "사람 검토 완료 여부를 기록할 대상 파일명을 human_review_signoff.yaml로 지정합니다."
        ),
        'Path("configs/assessment/acceptance_criteria_template.yaml"): private_root': (
            "정확도·속도·메모리 합격기준 양식 원본과 비공개 대상 폴더를 연결합니다."
        ),
        '/ "acceptance_criteria.yaml",': (
            "과제 합격기준을 작성할 대상 파일명을 acceptance_criteria.yaml로 지정합니다."
        ),
        "shutil.copy2(source, destination)": (
            "원본 양식의 메타데이터를 보존해 비공개 Drive 대상 경로로 복사합니다."
        ),
        'print("Created:", destination)': "이번 실행에서 새로 만든 입력 양식 경로를 출력합니다.",
        'print("Preserved existing:", destination)': (
            "기존 작성 내용을 덮어쓰지 않고 유지한 파일 경로를 출력합니다."
        ),
        'f"{DRIVE_ROOT}/reports/assessment_readiness/private_manufacturing",': (
            "실제 제조 입력 준비 상태 점검표를 저장할 Drive 경로를 전달합니다."
        ),
        'run_aias("model-matrix-lock", "--matrix", MODEL_MATRIX)': (
            "모든 Whisper 후보의 Hugging Face revision을 해시로 고정해 재현성을 확보합니다."
        ),
        'run_aias("benchmark-models", "--matrix", MODEL_MATRIX)': (
            "고정된 후보들을 같은 validation split에서 정확도·속도·메모리로 비교합니다."
        ),
        'completed["rank"] = pd.to_numeric(completed["rank"], errors="coerce")': (
            "문자열이나 결측값이 섞인 rank 열을 안전한 숫자형으로 변환합니다."
        ),
        '["rank", "cer", "wer", "aggregate_real_time_factor"],': (
            "종합순위, 글자·단어 오류율, 실시간 처리속도 순으로 후보를 정렬합니다."
        ),
        ").iloc[0]": "정렬 결과의 첫 행, 즉 최우선 완료 후보 하나를 반환합니다.",
        '"select-model",': "검토한 Whisper 모델을 공식 선택 기록으로 저장하는 명령입니다.",
        '"--benchmark-dir",': "후보 비교표와 실행 근거가 있는 벤치마크 폴더 옵션입니다.",
        "str(benchmark_dir),": "현재 모드의 벤치마크 산출물 폴더 경로를 전달합니다.",
        '"--model-id",': "최종 선택할 Whisper 후보 ID를 지정하는 옵션입니다.",
        "SELECTED_MODEL,": "자동 순위 또는 사람 검토로 선택한 모델 ID를 전달합니다.",
        '"--reviewer",': "선택을 수행한 자동 절차 또는 사람 검토자를 기록하는 옵션입니다.",
        "REVIEWER,": "현재 선택의 검토자 식별값을 감사 기록에 전달합니다.",
        '"--reason",': "정확도·속도·메모리와 데이터 한계를 선택 근거로 기록하는 옵션입니다.",
        "MODEL_REASON,": "Whisper 모델을 선택한 구체적 판단 근거를 전달합니다.",
        "*extra_selection_args,": (
            "공개·합성 자동 결과인 경우 사람 검토가 아니라는 표시 인자를 펼쳐 전달합니다."
        ),
        'print(model_selection.read_text(encoding="utf-8"))': (
            "저장된 모델 ID·revision·성능·검토자·선택 사유를 화면에서 확인합니다."
        ),
        '"train-selected-whisper",': "선택된 Whisper에 LoRA를 적용하는 학습 명령입니다.",
        '"--selection",': "앞 단계의 선택 YAML을 입력으로 받는 옵션입니다.",
        "str(model_selection),": (
            "모델·revision·실행 조건이 고정된 model_selection.yaml을 전달합니다."
        ),
        '"--config",': "데이터 경로·split·학습·평가 조건 YAML을 지정하는 옵션입니다.",
        "CONFIG,": "현재 데이터 모드에 대응하는 핵심 설정 YAML 경로를 전달합니다.",
        '"quantization-sweep",': "선택 모델의 여러 정밀도를 비교 평가하는 양자화 명령입니다.",
        '"--spec",': "양자화 후보·캐시·평가 기준 YAML을 지정하는 옵션입니다.",
        "QUANTIZATION_SPEC,": "현재 모드의 float16·int8 비교 명세 경로를 전달합니다.",
        '"select-quantization",': "검토한 양자화 variant를 공식 선택 기록으로 저장하는 명령입니다.",
        '"--quantization-dir",': "양자화 비교표와 실행 근거가 있는 폴더 옵션입니다.",
        "str(quantization_dir),": "현재 모드의 양자화 비교 산출물 폴더를 전달합니다.",
        '"--variant-id",': "최종 선택할 정밀도 variant ID를 지정하는 옵션입니다.",
        "SELECTED_VARIANT,": "자동 순위 또는 사람 검토로 선택한 양자화 ID를 전달합니다.",
        "QUANTIZATION_REASON,": "정확도 손실·속도·메모리·용량을 고려한 선택 근거를 전달합니다.",
        "*extra_quantization_args,": (
            "공개·합성 자동 결과인 경우 배포 결정이 아니라는 표시 인자를 펼쳐 전달합니다."
        ),
        'print(quantization_selection.read_text(encoding="utf-8"))': (
            "저장된 variant·성능·검토자·선택 사유를 화면에서 확인합니다."
        ),
        '"finalize-evaluation",': "선택 완료 후 미사용 Test split을 평가하는 최종 명령입니다.",
        "str(quantization_selection),": (
            "선택 모델과 정밀도가 고정된 quantization_selection.yaml을 전달합니다."
        ),
        '"assessment-audit",': "심사 증거의 완성도와 차단 항목을 검사하는 감사 명령입니다.",
        '"--output-dir",': "감사 결과 JSON과 Markdown을 저장할 폴더 옵션입니다.",
        "str(readiness_dir),": "현재 모드의 심사 준비도 보고서 폴더를 전달합니다.",
        (
            'pd.DataFrame(readiness["checks"])[['
            '"criterion", "check_id", "status", "message", "evidence"]]'
        ): ("심사기준·검사항목·상태·설명·증거 경로 열만 선택해 검토표를 만듭니다."),
        "pd.DataFrame(": "산출물 존재 여부 목록을 Colab에서 읽기 쉬운 표로 변환합니다.",
        '{"artifact": name, "exists": path.exists(), "path": str(path)}': (
            "각 산출물의 이름·존재 여부·실제 저장 경로를 한 행으로 구성합니다."
        ),
        '"다음 단계: DATA_MODE을 PRIVATE_MANUFACTURING으로 바꾸고 승인된 제조 "': (
            "기능 검증 다음에는 실행 모드를 실제 제조 데이터로 바꾸라고 안내합니다."
        ),
        '"녹음·정답 전사를 넣은 뒤 같은 순서를 다시 실행합니다."': (
            "승인된 녹음·정답으로 동일 파이프라인을 재실행하라는 절차를 완성합니다."
        ),
        '"최종 보고서와 오류 샘플을 사람이 검수하고 human_review_signoff.yaml을 "': (
            "실제 제조 결과의 보고서·오류 샘플·사람 검토 서명을 요구합니다."
        ),
        '"완료한 뒤 assessment-audit --fail-on-blocker로 최종 확인하세요."': (
            "서명 후 차단 항목이 남으면 실패하는 최종 감사 명령을 안내합니다."
        ),
        "display(benchmark_table)": (
            "Whisper 후보별 CER·WER·RTF·메모리·용량·종합순위를 표로 표시합니다."
        ),
        "display(quantization_table)": (
            "정밀도별 정확도 변화·속도·메모리·용량·종합순위를 표로 표시합니다."
        ),
        'display(synthetic_manifest.groupby(["split", "noise_condition"]).size())': (
            "train·validation·test와 소음 조건별 합성 음성 개수를 표로 확인합니다."
        ),
        'display(json.loads(provenance_path.read_text(encoding="utf-8")))': (
            "데이터 출처·생성기·개인정보 여부·용도 제한 정보를 화면에 표시합니다."
        ),
    }
    if stripped in semantic_comments:
        return semantic_comments[stripped]

    path_key_comments = {
        '"benchmark_table"': "Whisper 모델 후보 비교 CSV 경로를 등록합니다.",
        '"benchmark_report"': "모델 비교 결과 Word 보고서 경로를 등록합니다.",
        '"model_selection"': "선택 모델과 근거를 기록한 YAML 경로를 등록합니다.",
        '"selected_training"': "선택 모델 LoRA 학습·비교 결과 JSON 경로를 등록합니다.",
        '"quantization_table"': "양자화 후보 비교 CSV 경로를 등록합니다.",
        '"quantization_report"': "양자화 비교 결과 Word 보고서 경로를 등록합니다.",
        '"quantization_selection"': "선택 양자화와 근거를 기록한 YAML 경로를 등록합니다.",
        '"final_test"': "선택 완료 후 고정 Test 결과 JSON 경로를 등록합니다.",
        '"readiness_json"': "심사 준비도 기계 판독용 JSON 경로를 등록합니다.",
        '"readiness_markdown"': "심사 준비도 사람이 읽을 Markdown 경로를 등록합니다.",
        '"experiment_database"': "모든 실험 이력을 누적한 SQLite 백데이터 경로를 등록합니다.",
    }
    for key, comment in path_key_comments.items():
        if stripped.startswith(f"{key}:"):
            return comment

    if stripped == "run_aias(":
        command_comments = {
            '"train-selected-whisper",': (
                "선택된 Whisper에 LoRA를 학습하고 Base 대비 결과를 생성합니다."
            ),
            '"quantization-sweep",': "선택 모델의 각 양자화 variant를 동일 조건에서 평가합니다.",
            '"select-model",': "검토한 모델과 선택자·근거를 변경 불가능한 선택 기록으로 남깁니다.",
            '"select-quantization",': "검토한 양자화와 선택자·근거를 선택 기록으로 남깁니다.",
            '"finalize-evaluation",': (
                "선택이 끝난 모델을 미사용 고정 Test split에서 한 번 평가합니다."
            ),
            '"assessment-audit",': "문서·데이터·실험·거버넌스·사람 검토 증거를 종합 점검합니다.",
        }
        return command_comments.get(next_code, "프로젝트 CLI의 지정된 연구 단계를 실행합니다.")

    if stripped == "subprocess.run(":
        nearby = " ".join(future_code[:14])
        if '"clone",' in nearby:
            return "지정 브랜치의 GitHub 저장소를 Colab 임시 경로에 처음 복제합니다."
        if '"fetch"' in nearby:
            return "이미 복제된 저장소에서 원격 브랜치의 최신 커밋 정보를 가져옵니다."
        if '"checkout"' in nearby:
            return "기존 저장소의 작업 브랜치를 노트북이 요구하는 브랜치로 전환합니다."
        if '"merge",' in nearby:
            return "원격 최신 커밋을 fast-forward 방식으로 로컬 코드에 반영합니다."
        return "외부 명령을 실행하고 성공 여부를 확인합니다."

    if stripped.startswith("%pip uninstall"):
        return "Colab 기본 패키지 중 충돌 가능성이 있는 항목을 제거합니다."
    if stripped.startswith("%pip install"):
        return "프로젝트와 학습용 의존성을 현재 Colab 런타임에 설치합니다."
    if stripped == "!nvidia-smi":
        return "할당된 GPU 종류와 메모리 상태를 확인합니다."
    if stripped.startswith("from ") and " import " in stripped:
        module = stripped.split(" import ", 1)[0].removeprefix("from ")
        return f"{module} 모듈에서 필요한 기능을 불러옵니다."
    if stripped.startswith("import "):
        module = stripped.removeprefix("import ").split(" as ", 1)[0]
        return f"{module} 모듈을 불러옵니다."
    if stripped.startswith("def "):
        function_name = stripped.split("def ", 1)[1].split("(", 1)[0]
        return f"{function_name} 재사용 함수를 정의합니다."
    condition_comments = {
        "if DATA_MODE not in MODE_SETTINGS:": "지원 목록에 없는 데이터 모드를 조기에 차단합니다.",
        "if PROJECT_SRC not in sys.path:": (
            "프로젝트 src 경로가 Python 검색 경로에 없는지 확인합니다."
        ),
        'if find_spec("aias_specialist") is None:': (
            "AIAS 패키지를 실제로 import할 수 있는지 확인합니다."
        ),
        "if IS_PUBLIC_PROXY:": "공개 Zeroth 프록시 모드일 때의 데이터·선택 절차를 실행합니다.",
        "elif IS_SYNTHETIC_MANUFACTURING:": "합성 제조 TTS 모드일 때 manifest와 출처를 검증합니다.",
        "if IS_AUTOMATED_PROXY:": "공개·합성 기능 검증 모드이면 순위 기반 자동 선택을 사용합니다.",
        'if "TO_BE_COMPLETED" in REVIEWER or "TO_BE_COMPLETED" in MODEL_REASON:': (
            "실제 제조 모델의 검토자 또는 선택 근거가 미입력 상태인지 확인합니다."
        ),
        'if "TO_BE_COMPLETED" in QUANTIZATION_REASON:': (
            "실제 제조 양자화 선택 근거가 미입력 상태인지 확인합니다."
        ),
        "if completed.empty:": "정상 완료된 비교 후보가 하나도 없는지 확인합니다.",
        "if provenance_path.exists():": "데이터 출처·라이선스 JSON이 생성됐는지 확인합니다.",
        "if not destination.exists():": "기존 비공개 입력·검토 문서를 덮어쓰지 않도록 확인합니다.",
        "if not os.path.exists(PROJECT_DIR):": (
            "Colab 임시 디스크에 저장소가 아직 없는지 확인합니다."
        ),
        "if DRIVE_MY_DRIVE.is_dir():": (
            "내 드라이브 폴더가 이미 보이면 추가 승인 없이 기존 연결을 재사용합니다."
        ),
        "if not DRIVE_MY_DRIVE.is_dir():": (
            "승인 이후에도 내 드라이브 폴더가 없으면 연결 실패로 처리합니다."
        ),
    }
    if stripped in condition_comments:
        return condition_comments[stripped]
    if stripped.startswith("if "):
        return "이 줄에 명시된 보호 조건을 검사해 다음 처리 경로를 결정합니다."
    if stripped.startswith("elif "):
        return "앞 모드가 아닐 때 이 줄의 다음 데이터 모드 조건을 검사합니다."
    if stripped == "else:":
        nearby = " ".join(future_code[:12])
        if "drive.mount" in nearby:
            return "기존 Drive 연결이 없으므로 새 인증과 마운트를 시도합니다."
        if "private_root = Path(PRIVATE_ROOT)" in nearby:
            return "공개·합성 모드가 아니므로 실제 제조 데이터 입력 양식을 준비합니다."
        if 'SELECTED_MODEL = "small"' in nearby:
            return "실제 제조 모드이므로 자동 순위 대신 사람이 모델을 검토하고 입력합니다."
        if 'SELECTED_VARIANT = "float16"' in nearby:
            return "실제 제조 모드이므로 사람이 양자화 손실과 자원 절감을 검토합니다."
        if '"fetch"' in nearby:
            return "저장소가 이미 있으므로 재복제하지 않고 원격 최신 코드와 동기화합니다."
        if "Preserved existing" in nearby:
            return "대상 양식이 이미 있으므로 사용자가 작성한 기존 파일을 그대로 보존합니다."
        if "AUTOMATED_SYNTHETIC_FIXTURE" in nearby:
            return "공개 프록시가 아니므로 합성 제조 자동 선택임을 별도로 기록합니다."
        if "human_review_signoff.yaml" in nearby:
            return "실제 제조 모드의 최종 사람 검수와 서명 완료 절차를 안내합니다."
        return "앞 조건이 거짓이므로 이 코드 블록에 정의된 대체 절차를 실행합니다."
    if stripped.startswith("for "):
        return "각 항목을 순회하며 같은 처리를 반복합니다."
    if stripped.startswith("raise "):
        return "필수 조건을 만족하지 않으면 명확한 오류로 실행을 중단합니다."
    if stripped.startswith("return "):
        return "계산하거나 선택한 결과를 호출한 곳에 반환합니다."
    if stripped.startswith("print("):
        if "Selected model" in stripped:
            return "자동 또는 사람 검토로 선택된 Whisper 모델 ID를 출력합니다."
        if "Selected variant" in stripped:
            return "자동 또는 사람 검토로 선택된 양자화 variant ID를 출력합니다."
        if "Assessment readiness" in stripped:
            return "현재 모드의 최종 심사 준비 상태를 출력합니다."
        if "Git commit" in stripped:
            return "재현성을 위해 실제 실행 중인 Git 커밋 해시를 출력합니다."
        if "Mode:" in stripped:
            return "사용자가 확인할 수 있도록 현재 데이터 모드를 출력합니다."
        if "Config:" in stripped:
            return "현재 모드가 사용하는 핵심 YAML 설정 경로를 출력합니다."
        if stripped == "print(":
            if "PRIVATE_MANUFACTURING" in next_code:
                return "기능 검증 완료 후 실제 제조 데이터 모드로 전환하는 다음 단계를 출력합니다."
            if "human_review_signoff.yaml" in " ".join(future_code[:5]):
                return "실제 제조 결과의 사람 검수·서명·최종 감사 절차를 출력합니다."
        return "해당 단계의 상태·선택 근거·안내 문구를 실행 로그에 출력합니다."
    if stripped.startswith("display("):
        return "결과를 Colab 표 형태로 표시합니다."
    if stripped.startswith("os.chdir("):
        return "이후 상대 경로가 저장소를 기준으로 동작하도록 작업 폴더를 바꿉니다."
    if stripped.startswith("drive.mount("):
        return "새 런타임에 모델·산출물을 보존할 Google Drive 접근 승인을 요청합니다."
    if stripped == "try:":
        return "Google Drive 인증 실패를 사용자가 이해할 수 있는 안내로 변환합니다."
    if stripped.startswith("except ValueError as exc:"):
        return "Colab의 Drive 마운트 실패 예외를 잡아 브라우저 해결 방법을 안내합니다."
    if stripped.startswith((")", "]", "}")):
        return ""
    if stripped == "[":
        if '"clone",' in future_code[:8]:
            return "Git clone 실행 파일·하위 명령·옵션·원본·대상 순서로 명령 목록을 구성합니다."
        nearby = " ".join(future_code[:10])
        if '"merge",' in nearby:
            return "Git 저장소 위치·merge 방식·원격 브랜치를 순서대로 명령 목록에 담습니다."
        if '"artifact"' in nearby:
            return "필수 산출물마다 존재 여부와 경로를 계산할 행 목록을 시작합니다."
        return "순서가 중요한 입력값들을 묶는 목록을 시작합니다."
    if stripped.startswith("#"):
        return ""
    assignment_comments = {
        "provenance_path": (
            "현재 데이터셋의 출처·생성 방식·라이선스 정보를 담은 JSON 경로를 지정합니다."
        ),
        "synthetic_manifest": "합성 WAV 존재 여부·정답·split·해시를 검증한 manifest를 저장합니다.",
        "templates": "실제 제조 모드에서 필요한 manifest·승인·검토·합격기준 양식을 연결합니다.",
        "MODEL_REASON": "모델 선택의 데이터 범위·정확도·속도·검토 한계를 근거로 기록합니다.",
        "QUANTIZATION_REASON": "양자화 선택의 정확도 손실·속도·메모리·용량 근거를 기록합니다.",
        "na_position": "순위 정렬 시 결측 성능값을 가장 뒤로 보내도록 지정합니다.",
    }
    assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
    if assignment:
        variable = assignment.group(1)
        return assignment_comments.get(
            variable,
            f"{variable} 변수에 이 연구 단계에서 계산하거나 선택한 값을 저장합니다.",
        )
    if re.match(r"[\"'][^\"']+[\"']\s*:", stripped):
        return "이 키가 나타내는 세부 설정값을 현재 실행 모드에 연결합니다."
    if "TO_BE_COMPLETED" in stripped:
        return "실제 제조 모드에서 사람이 검토 후 반드시 교체해야 하는 입력값입니다."
    if "자동 선택" in stripped or "사람 검토" in stripped:
        return "자동 프록시 결과의 범위와 사람 검토가 아님을 선택 근거에 명시합니다."
    if stripped.startswith("(") or stripped.endswith(","):
        return "이 줄의 값을 바로 위에서 설명한 연구 단계의 입력으로 사용합니다."
    return "이 연산으로 현재 데이터 준비·평가·선택 단계의 상태를 갱신합니다."


def _annotate_code(source: str) -> str:
    """Add a Korean explanation immediately before every non-comment code line."""
    annotated: list[str] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            annotated.append(line)
            continue
        next_code = next(
            (
                candidate.strip()
                for candidate in lines[index + 1 :]
                if candidate.strip() and not candidate.lstrip().startswith("#")
            ),
            "",
        )
        future_code = tuple(
            candidate.strip()
            for candidate in lines[index + 1 :]
            if candidate.strip() and not candidate.lstrip().startswith("#")
        )
        comment = _line_comment(line, next_code, future_code)
        if comment:
            indentation = line[: len(line) - len(line.lstrip())]
            annotated.append(f"{indentation}# {comment}")
        annotated.append(line)
    return "\n".join(annotated)


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
            "하나의 노트북에서 세 가지 데이터 모드를 사용합니다.\n\n"
            "- `PUBLIC_PROXY`: 공개 Zeroth 한국어 음성으로 데이터 준비부터 모델 비교, "
            "자동 프록시 선택, LoRA, 양자화, 고정 Test, 보고서·백데이터까지 전체 동작을 "
            "검증합니다.\n"
            "- `SYNTHETIC_MANUFACTURING`: 저장소의 한국어 TTS 제조 문장 600개로 실제 제조 "
            "데이터를 넣기 전 동일한 코드 경로를 검증합니다.\n"
            "- `PRIVATE_MANUFACTURING`: 나중에 승인된 제조 녹음과 검수 전사로 같은 코드를 "
            "다시 실행합니다. 이 모드의 모델·양자화 선택은 반드시 사람이 수행합니다.\n\n"
            "> 공개·합성 결과는 코드와 산출물의 정상 동작 증거입니다. 제조 현장 성능, "
            "배포 적합성 또는 심사 최종 결론의 증거로 사용하면 안 됩니다.\n\n"
            "양자화, BM25 정보검색, 벡터 최근접 이웃 검색, 지식 증류는 실행 모드와 "
            "별도로 각각 켜고 끌 수 있습니다. 지식 증류는 추가 GPU 시간과 모델 다운로드가 "
            "필요하므로 기본값은 꺼짐입니다.\n\n"
            "**보안:** 카메라·마이크·패스키를 사용하지 않습니다. 실제 음성은 GitHub에 "
            "올리지 않고 승인된 비공개 Drive 경로만 사용합니다."
        ),
        nbf.v4.new_markdown_cell("## 0. 실행 모드"),
        nbf.v4.new_code_cell(
            '# "PUBLIC_PROXY", "SYNTHETIC_MANUFACTURING", "PRIVATE_MANUFACTURING"\n'
            'DATA_MODE = "SYNTHETIC_MANUFACTURING"\n\n'
            "RUN_LORA = True\n"
            "RUN_QUANTIZATION = True\n"
            "ENABLE_INFORMATION_RETRIEVAL = True\n"
            "ENABLE_NEAREST_NEIGHBOR = True\n"
            "RUN_DISTILLATION = False  # 추가 GPU 비용이 큰 선택 실험\n\n"
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
            '    "SYNTHETIC_MANUFACTURING": {\n'
            '        "config": "configs/synthetic_manufacturing_sample.yaml",\n'
            '        "matrix": '
            '"configs/benchmarks/synthetic_manufacturing_whisper_models.yaml",\n'
            '        "quantization": '
            '"configs/quantization/synthetic_manufacturing_whisper_quantization.yaml",\n'
            '        "benchmark_id": "synthetic-manufacturing-whisper-model-benchmark-v1",\n'
            '        "quantization_id": "synthetic-manufacturing-whisper-quantization-v1",\n'
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
            "IS_SYNTHETIC_MANUFACTURING = DATA_MODE == 'SYNTHETIC_MANUFACTURING'\n"
            "IS_AUTOMATED_PROXY = IS_PUBLIC_PROXY or IS_SYNTHETIC_MANUFACTURING\n"
            'print("Mode:", DATA_MODE)\n'
            'print("Config:", CONFIG)'
        ),
        nbf.v4.new_markdown_cell("## 1. GPU와 Google Drive 연결"),
        nbf.v4.new_code_cell(
            "!nvidia-smi\n"
            "from pathlib import Path\n"
            "from google.colab import drive\n\n"
            "DRIVE_MOUNT_POINT = Path('/content/drive')\n"
            "DRIVE_MY_DRIVE = DRIVE_MOUNT_POINT / 'MyDrive'\n\n"
            "if DRIVE_MY_DRIVE.is_dir():\n"
            "    print('Google Drive already mounted at /content/drive')\n"
            "else:\n"
            "    try:\n"
            "        drive.mount(str(DRIVE_MOUNT_POINT))\n"
            "    except ValueError as exc:\n"
            "        raise RuntimeError(\n"
            "            'Google Drive 연결에 실패했습니다. accounts.google.com 화면이 검게 '\n"
            "            '멈추면 Codex 내장 브라우저 대신 일반 Chrome에서 이 노트북을 '\n"
            "            '열고 팝업·리디렉션을 허용한 뒤 이 셀을 다시 실행하세요.'\n"
            "        ) from exc\n\n"
            "if not DRIVE_MY_DRIVE.is_dir():\n"
            "    raise RuntimeError(\n"
            "        'Google Drive 승인이 완료됐지만 /content/drive/MyDrive를 찾을 수 없습니다.'\n"
            "    )\n"
            "print('Google Drive ready:', DRIVE_MY_DRIVE)"
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
            "%pip uninstall -y torchao gradio gradio-client\n"
            '%pip install -q -e ".[train]" "transformers>=4.46,<5" "peft>=0.14,<0.19"\n\n'
            "from importlib.util import find_spec\n"
            "import sys\n\n"
            "PROJECT_SRC = os.path.join(PROJECT_DIR, 'src')\n"
            "if PROJECT_SRC not in sys.path:\n"
            "    sys.path.insert(0, PROJECT_SRC)\n"
            "if find_spec('aias_specialist') is None:\n"
            "    raise ModuleNotFoundError('aias_specialist import path refresh failed')\n"
            "print('aias_specialist import OK:', PROJECT_SRC)\n\n"
            "def run_aias(*args):\n"
            "    command = [sys.executable, '-m', 'aias_specialist.cli', *args]\n"
            '    print("\\nRunning:", " ".join(command), flush=True)\n'
            "    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}\n"
            "    subprocess.run(command, check=True, env=environment)"
        ),
        nbf.v4.new_markdown_cell(
            "### 3-1. 선택 기능을 실행용 설정에 반영\n\n"
            "GitHub 원본 YAML은 수정하지 않고 `/content`에 실행용 사본을 만듭니다. "
            "따라서 위 Boolean 값만 바꿔 양자화·IR·NN Search·Distillation을 독립적으로 "
            "실행할 수 있습니다. `char_ngram` NN은 다운로드 없는 재현용 벡터 검색이며, "
            "실제 임베딩 모델을 쓰려면 YAML의 backend를 `transformers`로 변경합니다."
        ),
        nbf.v4.new_code_cell(
            "import yaml\n\n"
            "source_config_path = Path(CONFIG)\n"
            "runtime_config = yaml.safe_load(source_config_path.read_text(encoding='utf-8'))\n"
            "runtime_config.setdefault('correction', {})\n"
            "runtime_config['correction'].setdefault('information_retrieval', {})[\n"
            "    'enabled'\n"
            "] = ENABLE_INFORMATION_RETRIEVAL\n"
            "runtime_config['correction'].setdefault('nearest_neighbor', {})[\n"
            "    'enabled'\n"
            "] = ENABLE_NEAREST_NEIGHBOR\n"
            "runtime_config.setdefault('distillation', {})['enabled'] = RUN_DISTILLATION\n"
            "runtime_config_path = Path('/content/aias_runtime_config.yaml')\n"
            "runtime_config_path.write_text(\n"
            "    yaml.safe_dump(runtime_config, allow_unicode=True, sort_keys=False),\n"
            "    encoding='utf-8',\n"
            ")\n\n"
            "runtime_matrix = yaml.safe_load(Path(MODEL_MATRIX).read_text(encoding='utf-8'))\n"
            "runtime_matrix['benchmark']['base_config'] = str(runtime_config_path)\n"
            "runtime_matrix_path = Path('/content/aias_runtime_model_matrix.yaml')\n"
            "runtime_matrix_path.write_text(\n"
            "    yaml.safe_dump(runtime_matrix, allow_unicode=True, sort_keys=False),\n"
            "    encoding='utf-8',\n"
            ")\n\n"
            "runtime_quantization = yaml.safe_load(\n"
            "    Path(QUANTIZATION_SPEC).read_text(encoding='utf-8')\n"
            ")\n"
            "runtime_quantization['quantization']['enabled'] = RUN_QUANTIZATION\n"
            "runtime_quantization['quantization']['base_config'] = str(runtime_config_path)\n"
            "runtime_quantization_path = Path('/content/aias_runtime_quantization.yaml')\n"
            "runtime_quantization_path.write_text(\n"
            "    yaml.safe_dump(runtime_quantization, allow_unicode=True, sort_keys=False),\n"
            "    encoding='utf-8',\n"
            ")\n\n"
            "CONFIG = str(runtime_config_path)\n"
            "MODEL_MATRIX = str(runtime_matrix_path)\n"
            "QUANTIZATION_SPEC = str(runtime_quantization_path)\n"
            "print({\n"
            "    'lora': RUN_LORA,\n"
            "    'quantization': RUN_QUANTIZATION,\n"
            "    'information_retrieval': ENABLE_INFORMATION_RETRIEVAL,\n"
            "    'nearest_neighbor': ENABLE_NEAREST_NEIGHBOR,\n"
            "    'distillation': RUN_DISTILLATION,\n"
            "})"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. 데이터 준비\n\n"
            "`PUBLIC_PROXY`에서는 고정 revision의 `kresnik/zeroth_korean` 일부만 스트리밍해 "
            "Drive에 저장합니다. `SYNTHETIC_MANUFACTURING`에서는 저장소에 포함된 TTS WAV와 "
            "정답 manifest를 검증합니다. `PRIVATE_MANUFACTURING`에서는 기존 파일을 덮어쓰지 "
            "않고 입력 양식을 준비합니다."
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
            "elif IS_SYNTHETIC_MANUFACTURING:\n"
            "    from aias_specialist.config import load_settings\n"
            "    from aias_specialist.data import validate_manifest\n\n"
            "    synthetic_settings = load_settings(CONFIG)\n"
            "    synthetic_manifest = validate_manifest(\n"
            "        synthetic_settings.paths.manifest, backend='faster_whisper'\n"
            "    )\n"
            "    display(synthetic_manifest.groupby(['split', 'noise_condition']).size())\n"
            "    provenance_path = (\n"
            "        synthetic_settings.paths.manifest.parent / 'dataset_provenance.json'\n"
            "    )\n"
            "    display(json.loads(provenance_path.read_text(encoding='utf-8')))\n"
            "    print('SYNTHETIC: 실제 제조 성능이나 사람 검수 증거가 아닙니다.')\n"
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
            "공개·합성 모드는 빠른 전체동작 검증을 위해 `tiny`, `base`, `small`을 비교합니다. "
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
            "공개·합성 모드는 완료 후보 중 rank 1을 자동 선택하지만 사람 검토로 기록하지 "
            "않습니다. "
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
            "if IS_AUTOMATED_PROXY:\n"
            "    selected_row = best_completed_member(benchmark_table)\n"
            "    SELECTED_MODEL = str(selected_row['member_id'])\n"
            "    if IS_PUBLIC_PROXY:\n"
            "        REVIEWER = 'AUTOMATED_PUBLIC_PROXY'\n"
            "        MODEL_REASON = (\n"
            "            '공개 Zeroth 프록시 rank 1 자동 선택. 코드·산출물 smoke test '\n"
            "            '전용이며 제조 모델 선정 또는 사람 검토 증거가 아님.'\n"
            "        )\n"
            "    else:\n"
            "        REVIEWER = 'AUTOMATED_SYNTHETIC_FIXTURE'\n"
            "        MODEL_REASON = (\n"
            "            '합성 제조 TTS rank 1 자동 선택. 기능 검증 전용이며 실제 제조 '\n"
            "            '모델 선정 또는 사람 검토 증거가 아님.'\n"
            "        )\n"
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
            "공개·합성 모드는 작은 train split과 30 step의 짧은 실행으로 학습 코드, "
            "checkpoint, Base/LoRA 비교 산출물을 검증합니다. 실제 제조 모드는 500 step을 "
            "사용하며 데이터 규모에 맞춰 조정합니다."
        ),
        nbf.v4.new_code_cell(
            "if RUN_LORA:\n"
            "    run_aias(\n"
            "        'train-selected-whisper', '--selection', str(model_selection),\n"
            "        '--config', CONFIG,\n"
            "    )\n"
            "else:\n"
            "    print('LoRA skipped by RUN_LORA=False')"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Knowledge Distillation (선택)\n\n"
            "고정된 Teacher Whisper의 token 분포와 정답 label을 결합해 더 작은 Student를 "
            "학습합니다. GPU 시간과 모델 다운로드 비용이 크므로 `RUN_DISTILLATION=True`인 "
            "경우에만 실행합니다."
        ),
        nbf.v4.new_code_cell(
            "distillation_run_dir = None\n"
            "if RUN_DISTILLATION:\n"
            "    run_aias('model-lock', '--config', CONFIG)\n"
            "    before_distillation = set(\n"
            "        Path(DRIVE_ROOT, 'artifacts/runs').glob('distill-*')\n"
            "    )\n"
            "    run_aias('train-whisper-distillation', '--config', CONFIG)\n"
            "    after_distillation = set(\n"
            "        Path(DRIVE_ROOT, 'artifacts/runs').glob('distill-*')\n"
            "    )\n"
            "    new_distillation_runs = sorted(\n"
            "        after_distillation - before_distillation,\n"
            "        key=lambda path: path.stat().st_mtime,\n"
            "    )\n"
            "    if not new_distillation_runs:\n"
            "        raise RuntimeError('새 지식 증류 run 디렉터리를 찾지 못했습니다.')\n"
            "    distillation_run_dir = new_distillation_runs[-1]\n"
            "    display(json.loads(\n"
            "        (distillation_run_dir / 'metrics.json').read_text(encoding='utf-8')\n"
            "    ))\n"
            "else:\n"
            "    print('Knowledge distillation skipped by RUN_DISTILLATION=False')"
        ),
        nbf.v4.new_markdown_cell("## 9. 양자화 비교"),
        nbf.v4.new_code_cell(
            "quantization_dir = Path(DRIVE_ROOT) / 'artifacts/quantization' / QUANTIZATION_ID\n"
            "quantization_table = None\n"
            "if RUN_QUANTIZATION:\n"
            "    run_aias(\n"
            "        'quantization-sweep', '--spec', QUANTIZATION_SPEC,\n"
            "        '--selection', str(model_selection),\n"
            "    )\n"
            "    quantization_table = pd.read_csv(\n"
            "        quantization_dir / 'quantization_comparison.csv'\n"
            "    )\n"
            "    display(quantization_table)\n"
            "else:\n"
            "    print('Quantization skipped by RUN_QUANTIZATION=False')"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. 양자화 선택\n\n"
            "공개·합성 모드는 종합 rank 1을 자동 선택합니다. 실제 제조 모드는 정확도 손실, "
            "RTF, GPU 메모리와 모델 용량을 사람이 함께 검토합니다."
        ),
        nbf.v4.new_code_cell(
            "quantization_selection = None\n"
            "if not RUN_QUANTIZATION:\n"
            "    print('Quantization selection skipped because quantization is disabled.')\n"
            "elif IS_AUTOMATED_PROXY:\n"
            "    selected_quantization_row = best_completed_member(quantization_table)\n"
            "    SELECTED_VARIANT = str(selected_quantization_row['member_id'])\n"
            "    if IS_PUBLIC_PROXY:\n"
            "        QUANTIZATION_REASON = (\n"
            "            '공개 Zeroth 프록시 종합 rank 1 자동 선택. 양자화 코드·산출물 '\n"
            "            'smoke test 전용이며 실제 배포 결정 또는 사람 검토 증거가 아님.'\n"
            "        )\n"
            "    else:\n"
            "        QUANTIZATION_REASON = (\n"
            "            '합성 제조 TTS 종합 rank 1 자동 선택. 기능 검증 전용이며 실제 '\n"
            "            '배포 결정 또는 사람 검토 증거가 아님.'\n"
            "        )\n"
            "    extra_quantization_args = ['--automated-proxy']\n"
            "elif RUN_QUANTIZATION:\n"
            "    SELECTED_VARIANT = 'float16'  # 비교표를 보고 수정\n"
            "    QUANTIZATION_REASON = (\n"
            "        'TO_BE_COMPLETED: 정확도 손실·속도·메모리·모델 용량 근거'\n"
            "    )\n"
            "    if 'TO_BE_COMPLETED' in QUANTIZATION_REASON:\n"
            "        raise ValueError('제조 모드에서는 양자화 선택 근거를 입력하세요.')\n"
            "    extra_quantization_args = []\n\n"
            "if RUN_QUANTIZATION:\n"
            "    run_aias(\n"
            "        'select-quantization', '--quantization-dir', str(quantization_dir),\n"
            "        '--variant-id', SELECTED_VARIANT, '--reviewer', REVIEWER,\n"
            "        '--reason', QUANTIZATION_REASON, *extra_quantization_args,\n"
            "    )\n"
            "    quantization_selection = quantization_dir / 'quantization_selection.yaml'\n"
            "    print('Selected variant:', SELECTED_VARIANT)\n"
            "    print(quantization_selection.read_text(encoding='utf-8'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 11. 고정 Test 최종평가\n\n"
            "모델·양자화 선택이 끝난 뒤에만 그동안 보지 않은 Test split을 한 번 평가합니다."
        ),
        nbf.v4.new_code_cell(
            "final_selection = (\n"
            "    quantization_selection if RUN_QUANTIZATION else model_selection\n"
            ")\n"
            "run_aias(\n"
            "    'finalize-evaluation', '--selection', str(final_selection),\n"
            "    '--config', CONFIG,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## 12. 산출물·심사 준비도 확인\n\n"
            "공개·합성 모드에서는 `not_ready`가 정상입니다. 공개·합성 데이터, 자동 선택, "
            "미완료 사람 서명은 제조 심사 증거를 대체하지 못합니다."
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
        nbf.v4.new_markdown_cell("## 13. 생성 결과 위치 요약"),
        nbf.v4.new_code_cell(
            "expected_outputs = {\n"
            "    'benchmark_table': benchmark_dir / 'benchmark_comparison.csv',\n"
            "    'benchmark_report': benchmark_dir / 'reports/benchmark_report.docx',\n"
            "    'model_selection': benchmark_dir / 'model_selection.yaml',\n"
            "    'final_test': final_selection.parent / 'final_test_result.json',\n"
            "    'readiness_json': readiness_dir / 'assessment_readiness.json',\n"
            "    'readiness_markdown': readiness_dir / 'assessment_readiness.md',\n"
            "    'experiment_database': Path(DRIVE_ROOT) / 'backdata/experiments.sqlite3',\n"
            "}\n"
            "if RUN_LORA:\n"
            "    expected_outputs['selected_training'] = (\n"
            "        benchmark_dir / 'selected_training_result.json'\n"
            "    )\n"
            "if RUN_QUANTIZATION:\n"
            "    expected_outputs.update({\n"
            "        'quantization_table': quantization_dir / 'quantization_comparison.csv',\n"
            "        'quantization_report': (\n"
            "            quantization_dir / 'reports/quantization_report.docx'\n"
            "        ),\n"
            "        'quantization_selection': quantization_selection,\n"
            "    })\n"
            "if distillation_run_dir is not None:\n"
            "    expected_outputs.update({\n"
            "        'distillation_metrics': distillation_run_dir / 'metrics.json',\n"
            "        'distillation_report': (\n"
            "            distillation_run_dir / 'reports/evaluation_report.docx'\n"
            "        ),\n"
            "    })\n"
            "display(pd.DataFrame([\n"
            "    {'artifact': name, 'exists': path.exists(), 'path': str(path)}\n"
            "    for name, path in expected_outputs.items()\n"
            "]))\n\n"
            "if IS_AUTOMATED_PROXY:\n"
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
    if OUTPUT.exists():
        existing_notebook = nbf.read(OUTPUT, as_version=4)
        if len(existing_notebook.cells) == len(notebook.cells):
            for cell, existing_cell in zip(notebook.cells, existing_notebook.cells, strict=True):
                if "id" in existing_cell:
                    cell["id"] = existing_cell["id"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(OUTPUT)],
        check=True,
    )
    formatted_notebook = nbf.read(OUTPUT, as_version=4)
    for cell in formatted_notebook.cells:
        if cell.cell_type == "code":
            cell.source = _annotate_code(cell.source)
    nbf.write(formatted_notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
