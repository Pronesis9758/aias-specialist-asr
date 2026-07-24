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

민감한 음성, 전사, 예측과 checkpoint는 공개 GitHub나 공개 제출물에 포함하지 않는다.
필요한 경우 승인된 심사 채널에만 제공하거나 비식별 요약 지표로 대체한다.
