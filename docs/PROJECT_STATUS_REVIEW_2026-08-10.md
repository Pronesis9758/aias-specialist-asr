# AI Specialist 제조 음성 ASR 프로젝트 진행 리뷰

- 기준일: 2026-08-10
- 작업 브랜치: `codex/whisper-benchmark-quantization`
- 기준 커밋: `373033b` 이후 작업

## 1. 현재 단계

프로젝트는 **공개·합성 데이터로 전체 자동화 검증을 수행하고, 실제 제조 데이터 투입을 준비하는
단계**다. 코드 골격과 Colab GPU 파이프라인은 구현됐으며, 심사 제출을 위해 남은 핵심은 실제
제조 음성, 데이터 승인, 사람 검토와 실제 대상 하드웨어 검증이다.

## 2. 과제 목표

제조 현장의 한국어 음성을 텍스트로 변환하고 다음 절차를 재현 가능하게 자동화한다.

1. 음성·정답 전사 준비와 데이터 계약 검증
2. Train/Validation/Test 분리와 화자 누수 확인
3. 여러 Whisper 모델의 정확도·속도·메모리 비교
4. 모델 선택 근거 기록
5. 선택 모델 LoRA 파인튜닝
6. FP16·INT8-FP16 양자화 비교와 선택 근거 기록
7. 고정 Test 최종평가
8. Word 보고서, 불변 run 디렉터리, SQLite 백데이터 생성
9. 데이터 승인·라벨 검수·사람 최종 서명 감사

## 3. 구현된 기능

| 영역 | 구현 내용 |
|---|---|
| 데이터 | manifest 검증, split 검증, 화자 독립 검증, 출처·승인 필드 |
| ASR | Faster-Whisper 추론과 제조 용어 보정 |
| 모델 비교 | Tiny, Base, Small 및 제조용 전체 후보 비교 설정 |
| 모델 고정 | Hugging Face revision을 commit SHA로 기록 |
| 파인튜닝 | Transformers Whisper + PEFT LoRA |
| 양자화 | CTranslate2 FP16과 INT8-FP16 비교 |
| 평가 | WER, CER, Domain Term Recall, RTF, 지연시간, 메모리 |
| 산출물 | CSV, JSON, Markdown, Word 보고서, 그래프 |
| 백데이터 | `backdata/experiments.sqlite3` 실험 레지스트리 |
| 실행환경 | Python 3.12, `uv`, pytest, Ruff, GitHub, Colab GPU |
| 거버넌스 | 비식별화, 외부 처리 승인, 라벨 검수, 사람 서명 게이트 |

## 4. 공개 프록시 전체 실행 결과

공개 `kresnik/zeroth_korean` 고정 revision의 64개 샘플을 사용했다.

- Train 40개
- Validation 8개
- Test 16개
- 라이선스: CC BY 4.0

Colab T4 GPU에서 전체 노트북을 처음부터 끝까지 실행했다.

- Tiny, Base, Small 모델 비교 완료
- 공개 Validation 기준 `Whisper-small(float16)` 1위
- Small WER 약 0.3364, CER 약 0.1143
- Small RTF 약 0.0626, GPU 메모리 약 724 MB
- 30 step LoRA smoke 학습 완료
- `int8-float16` 양자화가 공개 프록시 종합 1위
- 모델 크기 약 487 MB에서 248 MB로 감소
- GPU 메모리 약 724 MB에서 436 MB로 감소
- 고정 Test 평가와 최종 산출물 11종 생성 확인

이 결과는 코드와 산출물 계약의 정상 동작 증거이며 실제 제조 성능 증거가 아니다.

## 5. 심사기준별 상태

### AI 기본·모델링

Whisper 후보 조사, baseline 선정, 동일 Validation 비교, 모델 revision 고정과 선택 기록 기능은
준비됐다. 실제 제조 데이터 기반 비교와 사람이 검토한 최종 선택 근거는 남아 있다.

### 프로그래밍·Tool 활용

Python CLI, `uv`, pytest, Ruff, Hugging Face, Colab GPU, PEFT LoRA, CTranslate2, Word 보고서와
SQLite 등록을 구현했다.

### AI Application·On-device

수집 양식, 추론, 파인튜닝, 양자화와 보고서 생성은 구현했다. 현재 하드웨어 프로파일은 Colab
개발 검증용이므로 실제 배포 대상 장비를 정하고 같은 지표를 다시 측정해야 한다.

### AI Governance

실제 음성의 GitHub 업로드 금지, Drive 비공개 경로, 비식별화, 외부 처리 승인, 라벨 검수,
화자 독립 split과 사람 최종 서명 게이트가 준비됐다. 실제 승인서와 서명은 아직 없다.

## 6. 심사 준비도

2026-07-24 로컬 감사 기준 17개 점검 중 8개 통과, 9개 대기이며 종합 상태는
`not_ready`다. 대기 항목은 다음과 같다.

1. 실제 합격 기준
2. 승인된 제조 manifest
3. 제조 데이터 모델 비교
4. 사람 모델 선택
5. 선택 모델 LoRA 증거
6. 제조 데이터 양자화 비교
7. 사람 양자화 선택
8. 고정 Test 최종평가
9. 사람 최종 서명

이는 코드 실패가 아니라 실제 제조 데이터와 사람 검토 증거가 없기 때문에 발생한 정상적인
게이트다.

## 7. 합성 제조 음성의 역할

실제 녹음 전 기능 검증을 위해 `data/sample/manufacturing_synthetic/`에 사람 음성이 아닌 한국어
TTS 데이터셋을 둔다. 이 데이터는 제조 문장, 정답 전사, 잡음 조건과 split을 포함하므로 동일한
코드를 실행할 수 있지만 다음을 증명하지 못한다.

- 실제 제조 현장 정확도
- 실제 화자와 억양에 대한 일반화
- 개인정보·노무·외부 처리 승인 완료
- 실제 라벨 사람 검수
- 생산 배포 준비 완료

## 8. 다음 실행 순서

1. 합성 제조 음성 모드로 전체 코드와 산출물 생성 재검증
2. 실제 제조 녹음과 검수 전사 준비
3. 데이터 승인서와 합격 기준 작성
4. 실제 제조 데이터로 Whisper 후보 비교
5. 사람이 모델을 선택하고 근거 기록
6. LoRA 파인튜닝
7. 양자화 비교 후 사람이 최종 방식 선택
8. 고정 Test 평가
9. 실제 배포 대상 하드웨어 검증
10. 보고서 검수와 `human_review_signoff.yaml` 작성
11. `assessment-audit --fail-on-blocker` 최종 통과

## 9. 주요 진입점

- 통합 Colab: `notebooks/colab_manufacturing_assessment.ipynb`
- 합성 데이터 설정: `configs/synthetic_manufacturing_sample.yaml`
- 실제 제조 템플릿: `configs/manufacturing_private_template.yaml`
- 모델 고정: `models/model-lock.yaml`
- 심사 증거 계획: `docs/ASSESSMENT_EVIDENCE_PLAN.md`
- 거버넌스 점검: `docs/GOVERNANCE_CHECKLIST.md`
- 하드웨어 검증 계획: `docs/HARDWARE_VALIDATION_PLAN.md`

## 결론

자동화 시스템과 공개 데이터 전체 실행은 완료됐다. 합성 제조 데이터는 실제 녹음 전 코드의
기능 검증 범위를 넓히기 위한 중간 단계다. 최종 심사 증거는 승인된 실제 제조 음성과 사람 검토를
통해 별도로 만들어야 한다.
