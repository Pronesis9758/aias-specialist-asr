# AI Specialist 제조 음성 ASR 현업과제 실행계획

## 목표

제조 음성 데이터의 준비부터 ASR 추론, 용어 보정, 평가, 모델 개선, 경량화, 결과 축적,
보고서 생성을 재현 가능한 단일 파이프라인으로 구축한다. Codex는 코드 작성·테스트·실행·문서화를
담당하고, 과제 수행자는 문제 정의·데이터 승인·라벨 검수·최종 판단을 담당한다.

## 완료 기준

1. 동일 config와 모델 commit으로 동일한 평가 결과를 재생성할 수 있다.
2. 각 실행의 입력 스냅샷, 환경, 예측, 지표, 보고서가 고유 run ID 아래 보존된다.
3. Baseline과 보정/학습/양자화 모델을 고정 test split에서 비교한다.
4. WER, CER, Domain Term Recall, Latency, Real-time Factor를 자동 산출한다.
5. 실제 사내 데이터는 승인된 환경에서만 처리한다.

## WBS와 게이트

| 단계 | 자동화 작업 | 사람 검토 게이트 | 산출물 |
|---|---|---|---|
| 0. 기반 구축 | 저장소, 환경, config, CI | 저장소 공개 범위 | 재현 가능한 개발환경 |
| 1. 데이터 준비 | manifest 검증, split, 스냅샷 | 동의/보안/전사 품질 | prepared_manifest.csv |
| 2. Baseline | Whisper 추론, 성능/속도 측정 | 오류 샘플 검수 | baseline predictions/metrics |
| 3. 용어 보정 | alias 치환, 재평가 | 잘못된 치환 검수 | corrected predictions/metrics |
| 4. 모델 개선 | Colab LoRA 학습, checkpoint resume | 학습 데이터 및 GPU 승인 | adapter/checkpoints |
| 5. 경량화 | FP16/INT8 비교 | 현장 장비 기준 선택 | benchmark 결과 |
| 6. 통합 평가 | 고정 test split 비교 | 최종 모델 선택 | 비교표/오류 분석 |
| 7. 보고서 | Word 보고서와 실행 근거 연결 | 결론·성과 문구 승인 | evaluation_report.docx |

## 현재 상태

- 단계 0: 로컬 프로젝트·CI·Colab 노트북과 GitHub 저장소 자동화 완료
- 단계 1: CC BY 4.0 Zeroth-Korean 고정 리비전에서 64개 공개 샘플 자동 준비 가능
- 단계 2~3: 실제 공개 음성의 CPU Baseline End-to-End와 Word 보고서 생성 검증 완료
- 단계 4: 선택 모델 LoRA, checkpoint 재개와 Base/LoRA 고정 Test 비교 코드 완료
- 단계 5: Whisper 6종 Validation 비교와 FP16·INT8-FP16 양자화 GPU 실행 완료
- 단계 6: 선택 기록·고정 Test 자동화 완료, 실제 제조 데이터 결과 대기
- 단계 7: 실행별 Word 보고서·SQLite 백데이터 자동 생성 및 구조 감사 완료
- 심사 대응: Solution 조사, strict governance, HW 측정 계획과 준비도 감사 CLI 완료

공개 샘플은 파이프라인 기술 검증 전용이다. 제조 도메인 일반화 성능, 제조 용어 재현율,
현장 배포 가능성은 검수된 제조 음성으로 별도 평가하기 전까지 주장하지 않는다.
