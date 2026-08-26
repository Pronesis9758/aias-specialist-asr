# AI Solution 조사 및 모델 선정 근거

## 문제 정의

제조 현장의 한국어 발화를 텍스트로 변환하고, 설비명·부품명·이상 상태와 같은 제조 용어의
인식 오류를 측정·개선한다. 모델 선정은 정확도만이 아니라 지연시간, Real-time Factor,
모델 크기, CPU/GPU 메모리, 데이터 보안과 재현성을 함께 고려한다.

최종 성공 기준의 수치는 실제 사용 시나리오와 목표 장비가 정해진 뒤 과제 수행자가 승인한다.

- 최대 허용 CER: `TO_BE_DECIDED`
- 최소 제조 용어 재현율: `TO_BE_DECIDED`
- 최대 aggregate RTF: `TO_BE_DECIDED`
- 최대 P95 지연시간: `TO_BE_DECIDED`
- 최대 모델 크기·메모리: `TO_BE_DECIDED`

## 후보 Solution 검토

| 후보 | 장점 | 제약 | 본 과제 판단 |
|---|---|---|---|
| OpenAI Whisper | 다국어·한국어 zero-shot, 모델 크기 선택, 공개 가중치, LoRA 가능 | 제조 용어와 현장 소음에 별도 검증 필요 | Baseline 및 주 모델군 |
| Wav2Vec2/XLS-R 한국어 모델 | CTC 기반 구조, 특정 한국어 데이터로 fine-tuning 가능 | tokenizer·학습 데이터 의존, 배포 경로를 별도 구축해야 함 | 대안 모델 조사 근거로 유지 |
| Conformer 계열 | 스트리밍·저지연 설계 가능 | 학습·배포 복잡도와 데이터 요구량이 큼 | 실시간 요구가 강할 때 2차 후보 |
| 외부 Speech API | 빠른 초기 구축, 운영형 API | 제조 음성 외부 전송, 비용, revision 통제와 재현성 제약 | 승인된 비민감 데이터에서만 비교 가능 |

## Whisper를 Baseline으로 선택한 이유

1. 공개 가중치로 동일 revision을 고정하고 재현할 수 있다.
2. Tiny부터 Large-v3·Turbo까지 동일한 디코딩 조건에서 용량·정확도 trade-off를 비교할 수 있다.
3. PEFT LoRA로 제조 용어와 음향 조건에 적응시킬 수 있다.
4. Faster-Whisper/CTranslate2를 이용해 FP16·INT8 변환과 CPU/GPU 배포 실험이 가능하다.
5. 코드와 모델 가중치의 공식 라이선스가 MIT로 명시되어 있다.

참고:

- OpenAI Whisper: <https://github.com/openai/whisper>
- Faster-Whisper: <https://github.com/SYSTRAN/faster-whisper>
- Hugging Face PEFT: <https://github.com/huggingface/peft>
- NVIDIA NeMo ASR/Conformer: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html>

## 선정 절차

1. 검수된 Validation split에서 Tiny/Base/Small/Medium/Large-v3/Turbo를 동일 조건으로 평가한다.
2. WER, CER, 제조 용어 재현율, RTF, 지연시간, 모델 크기와 메모리를 비교한다.
3. 담당자가 모델과 이유를 `model_selection.yaml` 및 SQLite에 기록한다.
4. 선택 모델을 실제 제조 Train/Validation 데이터로 LoRA fine-tuning한다.
5. FP16·INT8-FP16을 동일 Validation에서 비교하고 사람이 양자화 단계를 선택한다.
6. 모든 선택이 끝난 뒤 고정 Test를 한 번 실행한다.

공개 Zeroth-Korean 결과는 파이프라인과 Tool 활용을 검증하는 예비 실험이며, 제조 현장 성능
근거로 사용하지 않는다.
