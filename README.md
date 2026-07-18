# AI Specialist 현업과제 - 제조 음성 ASR 자동화

제조 현장의 한국어 음성을 대상으로 데이터 검증, ASR 추론, 도메인 용어 보정,
WER/CER/용어 정확도 평가, 실행 이력 누적, Word 보고서 생성을 하나의 재현 가능한
파이프라인으로 구성한 프로젝트입니다.

## 현재 자동화 범위

- 입력 manifest 및 도메인 용어 사전 검증
- 샘플 fixture 또는 Hugging Face `faster-whisper` 모델 기반 ASR 추론
- 제조 용어 alias 보정
- WER, CER, Domain Term Recall, Latency 측정
- 실행별 설정/환경/예측/지표/보고서 저장
- SQLite 실험 레지스트리에 백데이터 누적
- 로컬과 Google Colab에서 동일한 설정 파일 사용

Fine-tuning과 실제 현장 데이터 사용은 데이터·보안·GPU 확인 후 활성화합니다.

## 음성 파일 없이 Colab 데모 실행

공개 [Zeroth-Korean](https://huggingface.co/datasets/kresnik/zeroth_korean) 데이터셋의
고정된 커밋에서 학습 40개, 검증 8개, 테스트 16개를 스트리밍해 Google Drive에
저장합니다. 전체 2.88GB를 한 번에 내려받지 않으며, 준비가 끝난 샘플은 다음 실행에서
재사용합니다. 데이터셋은 CC BY 4.0이므로 결과물에 출처를 유지해야 합니다.

1. GitHub에서 `notebooks/colab_runner.ipynb`를 Colab으로 엽니다.
2. 런타임 유형을 T4 GPU 이상으로 설정합니다.
3. 저장소가 비공개라면 Colab Secrets에 읽기 전용 `GITHUB_TOKEN`을 추가합니다.
4. 모든 셀을 위에서부터 실행합니다.

노트북은 공개 데이터 준비, 모델 리비전 고정, Baseline 추론·평가·Word 보고서,
Whisper-tiny LoRA 데모 학습, checkpoint 및 SQLite 백데이터 Drive 보존을 실행합니다.
공개 데이터는 일반 한국어이므로 제조 현장 성능을 입증하지 않습니다.

GPU 없이 공개 데이터 준비와 4개 샘플의 CPU Baseline만 점검하려면:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" sync --extra data --extra dev
& "$env:USERPROFILE\.local\bin\uv.exe" run aias prepare-hf-dataset --config configs/local_public_sample.yaml
& "$env:USERPROFILE\.local\bin\uv.exe" run aias run --config configs/local_public_sample.yaml
```

## 빠른 시작

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" sync --extra dev
& "$env:USERPROFILE\.local\bin\uv.exe" run aias doctor
& "$env:USERPROFILE\.local\bin\uv.exe" run aias run --config configs/local_smoke.yaml
```

위 smoke 실행은 모델 다운로드 없이 샘플 예측으로 전체 산출물 흐름을 검증합니다.

저장소에 포함된 무음 WAV로 모델 로딩·오디오 디코딩·추론 경로만 검증하려면:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run aias run --config configs/local_model_smoke.yaml
```

이 실행 결과는 정확도 근거로 사용하지 않습니다. 실제 성능 평가는 검수된 음성으로 실행합니다.

실제 Hugging Face 모델 실행:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run aias model-lock --config configs/local_baseline.yaml
& "$env:USERPROFILE\.local\bin\uv.exe" run aias download-model --config configs/local_baseline.yaml
& "$env:USERPROFILE\.local\bin\uv.exe" run aias run --config configs/local_baseline.yaml
```

## 실제 데이터 준비

`data/sample/manifest.csv`와 같은 열을 가진 CSV를 준비합니다.

| 열 | 설명 |
|---|---|
| `sample_id` | 중복되지 않는 음성 ID |
| `audio_path` | 저장소 루트 기준 wav/mp3 경로 |
| `reference_text` | 사람이 검수한 정답 문장 |
| `split` | train/validation/test |
| `fixture_prediction` | smoke 모드 전용 예측 문장 |
| `source` | 데이터 출처 |
| `consent_status` | 사용 승인/동의 상태 |

실제 음성은 Git에 올리지 말고 `data/private/` 또는 승인된 사내 저장소에 둡니다.

## 주요 명령

```text
aias doctor                         환경 및 입력 준비 상태 확인
aias model-lock --config ...        Hugging Face 모델 revision을 commit SHA로 고정
aias download-model --config ...    고정된 모델 파일 다운로드
aias prepare-hf-dataset --config ... 공개 음성 샘플과 manifest 준비
aias run --config ...               전체 파이프라인 실행
aias history                        누적 실행 이력 조회
```

## 디렉터리

```text
configs/                 로컬/Colab 실행 설정
data/                    샘플 및 사용자 입력 스키마
src/aias_specialist/     파이프라인 구현
notebooks/               Colab 실행 전용 노트북
artifacts/runs/          실행별 불변 산출물
backdata/                SQLite 실험 레지스트리
docs/                    계획, 거버넌스, 연동 설명
tests/                   단위/통합 테스트
```

## 사람이 결정해야 하는 항목

- 실제 사내 음성의 외부 반출/Colab 사용 가능 여부
- 정답 transcript와 제조 용어 사전의 현업 검수
- GitHub 저장소 공개/비공개 여부와 저장소 이름
- Colab 요금제 및 GPU 사용 승인
- 정확도·속도·메모리·보안 간 최종 모델 선택
