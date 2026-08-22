# AI Specialist 현업과제 - 제조 음성 ASR 자동화

제조 현장의 한국어 음성을 대상으로 데이터 검증, ASR 추론, 도메인 용어 보정,
WER/CER/용어 정확도 평가, 실행 이력 누적, Word 보고서 생성을 하나의 재현 가능한
파이프라인으로 구성한 프로젝트입니다.

## 현재 자동화 범위

- 입력 manifest 및 도메인 용어 사전 검증
- 샘플 fixture 또는 Hugging Face `faster-whisper` 모델 기반 ASR 추론
- 제조 용어 alias·BM25 Information Retrieval·벡터 NN Search 선택형 보정
- WER, CER, Domain Term Recall, Latency 측정
- Whisper Tiny/Base/Small/Medium/Large-v3/Turbo Validation 비교
- 선택 모델의 FP16·INT8-FP16 양자화 및 자원 사용량 비교
- Teacher Whisper에서 경량 Student Whisper로 선택형 Knowledge Distillation
- 사람의 모델·양자화 선택 기록 후 고정 Test 최종평가
- 실행별 설정/환경/예측/지표/보고서 저장
- SQLite 실험 레지스트리에 백데이터 누적
- 로컬과 Google Colab에서 동일한 설정 파일 사용
- 승인·비식별·전사 검수와 화자 독립 split을 강제하는 제조 데이터 모드
- 네 가지 심사기준의 제출 증빙 누락을 자동 확인하는 준비도 감사

Fine-tuning과 실제 현장 데이터 사용은 데이터·보안·GPU 확인 후 활성화합니다.

## 실제 제조 데이터 전 준비

실제 음성과 정답 전사가 없어도 다음 항목은 준비되어 있습니다.

- Solution 후보 조사와 Whisper Baseline 선정 근거
- strict private manifest·데이터 승인·사람 최종 검토 양식
- 화자가 Train/Validation/Test에 중복되지 않는지 자동 검증
- 모델 선택 결과를 받아 해당 모델을 LoRA 학습하는 명령
- warm-up 제외 및 3회 반복 중앙값 기반 HW 성능 측정 설정
- 모델 비교·LoRA·양자화·최종 Test·사람 승인의 증빙 완성도 자동 점검

통합 실행 진입점은 `notebooks/colab_manufacturing_assessment.ipynb`입니다. 노트북의
`DATA_MODE`에 따라 다음 설정을 사용합니다.

- `PUBLIC_PROXY`: 고정 revision의 공개 `kresnik/zeroth_korean` 샘플로 데이터 준비,
  3개 모델 비교, 자동 프록시 선택, LoRA, 양자화, 고정 Test, 보고서·SQLite 등록까지
  전체 동작을 검증합니다.
- `SYNTHETIC_MANUFACTURING`: 저장소에 포함된 한국어 TTS 제조 문장 30개와 정답 전사로
  실제 데이터 투입 전 동일한 모델 비교·LoRA·양자화·보고서 경로를 검증합니다.
- `PRIVATE_MANUFACTURING`: `configs/manufacturing_private_template.yaml`과 승인된 제조
  녹음·검수 전사를 사용하며, 모델과 양자화 선택을 사람 검토로 강제합니다.

공개·합성 모드의 자동 선택에는 `human_reviewed: false`와
`selection_scope: automated_public_proxy`가 기록됩니다. 따라서 정상 동작 확인에는 쓸 수
있지만 제조 성능이나 심사 최종 결론의 근거로는 인정하지 않습니다.

노트북 상단의 `RUN_QUANTIZATION`, `ENABLE_INFORMATION_RETRIEVAL`,
`ENABLE_NEAREST_NEIGHBOR`, `RUN_DISTILLATION` 값을 각각 변경하면 네 기능을 독립적으로
실행할 수 있습니다. 지식 증류는 GPU 비용이 크므로 기본값은 `False`입니다. 세부 YAML
옵션과 산출물은 `docs/OPTIONAL_MODELING_FEATURES.md`에 정리되어 있습니다.

합성 제조 데이터는 다음 명령으로 Windows에서 재생성할 수 있습니다.

```powershell
uv run python scripts/generate_synthetic_manufacturing_dataset.py --overwrite
```

현재 제조 데이터 준비 상태를 확인하려면:

```powershell
uv run aias assessment-audit `
  --config configs/manufacturing_private_template.yaml `
  --output-dir reports/generated/assessment_readiness
```

음성이 준비되기 전에는 데이터·실험 항목이 `waiting`으로 표시되는 것이 정상입니다.

## 모델·양자화 비교 Colab 실행

`notebooks/colab_model_benchmark_quantization.ipynb`는 다음 순서로 실행됩니다.

1. 공개 한국어 데이터의 Validation/Test 분리 확인
2. 모든 Whisper 후보의 Hugging Face revision을 commit SHA로 고정
3. 동일 Validation 데이터에서 FP16 모델 크기 비교
4. 모델 담당자와 선택 이유 기록
5. 선택 모델의 FP16·INT8-FP16 비교
6. 양자화 담당자와 선택 이유 기록
7. 선택된 모델·양자화 조합의 고정 Test 최종평가

모델·양자화 후보는 각각 독립 `run_id`를 가지며, 비교 CSV·그래프·Word 보고서와
SQLite 실험 그룹 관계가 Google Drive에 보존됩니다. Colab 연결이 끊긴 뒤 같은 실험 ID로
다시 실행하면 완료된 후보는 건너뜁니다. 실험 설정이나 데이터가 바뀌면 YAML의 `id`를 새
버전으로 변경해야 합니다.

모델 다운로드·변환과 샘플별 추론 진행률은 셀 출력에 실시간 표시됩니다. 변환된
CTranslate2 모델은 Drive의 `models/ct2/`에 계속 보존되어 다음 Colab 런타임에서
재사용됩니다. 변환 전 Hugging Face 원본 모델까지 Drive에 보존하려면 노트북의
`PERSIST_HF_SOURCE_CACHE=True`를 사용합니다. 이 옵션은 약 15GB 이상의 추가 공간을
사용할 수 있으므로 기본값은 `False`입니다.

로컬 fixture로 오케스트레이션만 확인하려면:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run aias benchmark-models `
  --matrix configs/benchmarks/local_fixture.yaml
```

실제 Colab 모델 비교 설정은 `configs/benchmarks/whisper_models_colab.yaml`, 양자화 설정은
`configs/quantization/whisper_quantization_colab.yaml`에 있습니다.

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
aias model-matrix-lock --matrix ... 모델 행렬 전체 revision 고정
aias benchmark-models --matrix ...  Validation 모델 비교
aias select-model ...               사람의 모델 선택과 이유 기록
aias train-selected-whisper ...     선택된 모델의 LoRA 학습과 Base/Test 비교
aias train-whisper-distillation ... Teacher 지식을 경량 Student에 증류
aias quantization-sweep ...         선택 모델의 양자화 비교
aias select-quantization ...        사람의 양자화 선택과 이유 기록
aias finalize-evaluation ...        고정 Test 최종평가
aias history                        누적 실행 이력 조회
aias assessment-audit ...           심사기준별 제출 증빙 준비 상태 점검
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
