# 선택형 ASR 모델링 기능

이 문서는 제조 음성 ASR 핵심 흐름에 추가된 네 기능을 독립적으로 실행하는 방법을
설명합니다. 모든 실제 성능 주장은 승인되고 사람이 검수한 고정 test split에서 다시
검증해야 합니다.

## 1. Quantization

양자화 명세의 최상위 스위치입니다.

```yaml
quantization:
  enabled: true
  reference_variant: float16
  variants:
    - id: float16
      enabled: true
    - id: int8-float16
      enabled: true
```

```powershell
uv run aias quantization-sweep `
  --spec configs/quantization/synthetic_manufacturing_whisper_quantization.yaml `
  --selection <model_selection.yaml>
```

정확도, RTF, 메모리, 모델 크기와 기준 variant 대비 변화가 비교 CSV와 Word 보고서에
저장됩니다.

## 2. Information Retrieval

제조 용어 canonical·alias를 문자 2/3-gram BM25 색인으로 검색합니다. 기존 alias에 정확히
일치하지 않는 유사 오인식도 후보로 찾으며, `min_score` 미만은 변경하지 않습니다.

```yaml
correction:
  enabled: true
  alias_enabled: true
  information_retrieval:
    enabled: true
    min_score: 0.62
    weight: 0.5
```

각 적용 내역은 `predictions_corrected.csv`의 `correction_details`,
`correction_methods`, `correction_count`에 기록됩니다.

실행별 `correction_audit.csv`, `correction_audit.jsonl`, `correction_audit.md`에는 샘플별
정답 문장, 보정 전·후 전사, 적용 방법과 점수, CER/WER 보정 전·후 값, 절대 감소량 및
`improved`·`unchanged`·`degraded` 판정이 저장됩니다. 실제 전사 내용은 콘솔에 출력하지 않고
실행 폴더 안에만 보존하며 변경되거나 악화된 문장은 사람이 검토해야 합니다.

## 3. Nearest Neighbor Search

기본 `char_ngram` backend는 별도 모델 다운로드 없이 문자 2/3/4-gram 벡터의 cosine 최근접
이웃을 검색합니다. 기능·재현성 검증에 적합합니다.

```yaml
correction:
  nearest_neighbor:
    enabled: true
    backend: char_ngram
    min_score: 0.78
    weight: 0.5
```

문맥 임베딩 모델을 사용하려면 다음처럼 설정합니다. 이 경우 모델 revision은
`aias model-lock`이 commit SHA로 고정합니다.

```yaml
correction:
  nearest_neighbor:
    enabled: true
    backend: transformers
    model_repo_id: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    model_revision: main
    device: auto
    min_score: 0.78
```

IR과 NN을 동시에 켜면 두 점수의 가중 평균을 사용하고 적용 방법을 `hybrid`로 기록합니다.

### Validation 보정 후보 탐색과 안전한 선택

IR·NN 보정은 모든 문장을 무조건 개선하지 않습니다. 따라서 test split을 보면서 임계값을
고르지 않고, 모델 선정이 끝난 뒤 고정된 validation 예측에 대해서만 보정 후보를 비교합니다.

```powershell
uv run aias correction-sweep `
  --spec configs/correction/synthetic_manufacturing_correction_sweep.yaml `
  --selection <model_selection.yaml>

uv run aias select-correction `
  --sweep-dir <correction_sweep_dir> `
  --reviewer <reviewer> `
  --reason "validation acceptance gates passed"
```

탐색 후보에는 무보정 fallback, 정확한 alias 치환, IR, NN, IR·NN 합의형 hybrid가 포함됩니다.
후보가 채택되려면 CER이 개선되고 WER·제조 용어 recall이 악화되지 않으며, 악화된 문장 비율
제한도 통과해야 합니다. 어느 후보도 통과하지 못하면 `no-correction`을 선택하여 기존 인식
성능을 유지합니다. 선택 후에만 고정 test split을 한 번 평가하며, test 결과로 임계값을 다시
조정하지 않습니다.

주요 산출물은 다음과 같습니다.

- `correction_sweep_comparison.csv`: 후보별 정확도 변화, 악화 표본 수, 게이트 통과 여부와 순위
- `candidates/<candidate_id>/correction_audit.*`: 문장별 보정 전·후 및 개선·악화 근거
- `reports/correction_sweep_report.docx`: 후보 비교와 권고 후보 보고서
- `correction_selection.yaml`: 선택된 보정 정책, validation 지표, 검토자와 선택 사유

## 4. Knowledge Distillation

Teacher와 Student는 모두 model lock에 고정합니다. Student의 정답 token 교차엔트로피와
temperature-scaled Teacher KL divergence를 결합합니다.

```yaml
distillation:
  enabled: true
  teacher_repo_id: openai/whisper-small
  teacher_revision: main
  student_repo_id: openai/whisper-tiny
  student_revision: main
  output_dir: /content/drive/MyDrive/AI_Specialist_ASR_Project/checkpoints/distilled
  hard_label_weight: 0.5
  temperature: 2.0
  max_steps: 30
```

```powershell
uv run aias model-lock --config configs/synthetic_manufacturing_sample.yaml
uv run aias train-whisper-distillation --config configs/synthetic_manufacturing_sample.yaml
```

각 실행은 다음을 생성합니다.

- `predictions_baseline.csv`: 증류 전 Student
- `predictions_distilled.csv`: 증류 후 Student
- `predictions_corrected.csv`: 증류 후 Student에 선택형 IR/NN 보정 적용
- `metrics.json`, `training_metrics.json`
- `run_summary.md`, `reports/evaluation_report.docx`
- `checkpoints/distilled_student/`

## Colab 권장 기본값

```python
RUN_LORA = True
RUN_QUANTIZATION = True
RUN_CORRECTION_SWEEP = True
ENABLE_INFORMATION_RETRIEVAL = False
ENABLE_NEAREST_NEIGHBOR = False
RUN_DISTILLATION = False
```

IR·NN은 위의 직접 적용 스위치로 바로 켜기보다 `RUN_CORRECTION_SWEEP`에서 후보로 검증한 뒤
선택 결과를 적용하는 것이 기본 흐름입니다. 먼저 보정 탐색·양자화까지 전체 실행을 확인한 뒤
별도 Colab GPU 세션에서 지식 증류만 켜는 것을 권장합니다. 실제 제조 데이터에서는 자동 치환
표본과 보정·Teacher/Student 선택 근거를 현업 담당자가 검토해야 합니다.
