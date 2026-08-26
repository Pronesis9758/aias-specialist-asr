# AI Specialist 심사 증빙 계획

## 기준별 증빙

| 심사기준 | 필수 증빙 |
|---|---|
| AI 기본·모델링 | 문제 정의, Solution 비교, Baseline, 전체 모델 비교표, 선택 이유, 고정 Test 결과 |
| 프로그래밍·Tool | Git 이력, Python package, uv lock, Colab, HF, CI, 테스트 결과, 자동 보고서 |
| Application·HW | manifest 검증, LoRA 학습 로그, checkpoint 선택, 양자화표, 환경·메모리·RTF |
| AI Governance | 데이터 승인, provenance, 라이선스 notice, 비식별·전사 검수, 사람 최종 승인 |

## 제출 산출물

- `benchmark_comparison.csv`, `benchmark_report.docx`, `model_selection.yaml`
- `training_metrics.json`, Base/LoRA Test 예측과 `evaluation_report.docx`
- `quantization_comparison.csv`, `quantization_report.docx`, `quantization_selection.yaml`
- `final_test_result.json`이 가리키는 불변 run 디렉터리
- `config.snapshot.yaml`, `environment.json`, `model-lock.snapshot.yaml`
- `experiments.sqlite3` 또는 개인정보를 제거한 registry export
- `assessment_readiness.json`과 `assessment_readiness.md`
- 완료된 데이터 승인 및 사람 검토 문서

## 자동 점검

```bash
aias assessment-audit \
  --config configs/manufacturing_private_template.yaml \
  --output-dir reports/generated/assessment_readiness
```

실제 데이터가 없을 때는 관련 항목을 `waiting`으로 표시한다. 모든 실험과 사람 검토가 완료된
뒤 `--fail-on-blocker`를 사용하면 누락된 필수 증빙이 있을 때 비정상 종료한다.

## 공개 프록시와 제조 제출 증거의 분리

`configs/public_proxy_assessment.yaml`은 공개 Zeroth 한국어 음성의 작은 고정 표본으로
파이프라인과 산출물 계약을 빠르게 검증한다. 이 실행은 다음 항목을 확인하는 용도다.

- Hugging Face 데이터 revision·라이선스·provenance 기록
- 모델 비교표와 Word 보고서 생성
- 선택 모델 LoRA 실행과 checkpoint·평가 산출물 생성
- FP16·INT8-FP16 양자화 비교
- 고정 Test 평가와 `experiments.sqlite3` 백데이터 등록

공개 프록시 자동 선택은 `human_reviewed: false`로 기록되며 준비도 감사에서 사람 선택으로
통과하지 않는다. `strict_private` 제조 설정은 자동 프록시 선택을 거부한다. 따라서 최종
제출 전에는 반드시 승인된 제조 데이터로 다시 실행하고 모델·양자화 trade-off와 보고서
결론을 사람이 검토해야 한다.

민감한 음성, 전사, 예측과 checkpoint는 공개 GitHub나 공개 제출물에 포함하지 않는다.
필요한 경우 승인된 심사 채널에만 제공하거나 비식별 요약 지표로 대체한다.
