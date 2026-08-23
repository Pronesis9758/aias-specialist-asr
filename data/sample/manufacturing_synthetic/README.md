# 합성 제조 음성 테스트 세트

이 디렉터리는 실제 임직원·고객 음성이 아닌 Windows 한국어 TTS로 생성한 기능 검증용
데이터입니다.

## 구성

- Train 360개, Validation 120개, Test 120개(총 600개 고유 문장)
- 16 kHz, mono, PCM 16-bit WAV
- 일반 제조 문장, 용어 밀집 문장, 모델 번호가 포함된 오류 유발 문장을 각 200개 구성
- 조용한 음성, 팬 잡음, 설비 저주파 잡음, 8 dB 고난도 공장 잡음 조건
- split마다 겹치지 않는 TTS 속도 변형을 synthetic speaker ID로 기록
- 각 WAV의 SHA-256과 생성 출처를 `manifest.csv`에 기록
- `spoken_text`에는 TTS가 실제로 읽은 한글 발음을, `reference_text`에는 제조 문서에서
  요구하는 영문 약어·모델 코드 표준 표기를 기록

표기 예시:

| spoken_text(음성 발음) | reference_text(평가 정답) |
| --- | --- |
| 피엘씨 입력 신호를 확인합니다. | PLC 입력 신호를 확인합니다. |
| 에이지브이 모델 큐 오백이 | AGV 모델 Q502 |
| 일 호기 씨엔씨 선반 | 1호기 CNC 선반 |

`PLC`, `HMI`, `CNC`, `AGV`, `AOI`와 영숫자 모델 코드는 영문 표기를 정답으로 사용하고,
설비 번호는 `1호기`, `2호기`처럼 아라비아 숫자로 기록하며,
센서·컨베이어·베어링처럼 현장에서 통상 한글로 기록하는 일반 용어는 한글을 유지합니다.

재생성 명령:

```powershell
uv run python scripts/generate_synthetic_manufacturing_dataset.py --overwrite
```

기존 WAV를 다시 합성하지 않고 정답 표기와 출처 정보만 갱신하는 명령:

```powershell
uv run python scripts/generate_synthetic_manufacturing_dataset.py --overwrite --reuse-existing-audio
```

생성에는 Windows에 설치된 `Microsoft Heami Desktop` 음성이 필요합니다. 생성된 WAV는 Colab에서
다시 합성할 필요 없이 저장소에서 바로 사용합니다.

## 제한사항

- 실제 사람이나 실제 제조 현장의 음성이 아닙니다.
- TTS 속도 변형은 실제 화자 다양성을 의미하지 않습니다.
- 합성 잡음은 실제 공장 음향을 대표하지 않습니다.
- 모델 정확도, 현장 배포 적합성, 사람 라벨 검수 또는 개인정보 승인 증거로 사용할 수 없습니다.
- 프로젝트 외부로 재배포하기 전에는 사용한 Windows 음성에 적용되는 Microsoft 조건을 확인해야
  합니다.

용도는 데이터 계약, ASR 추론, 모델 비교, LoRA, 양자화, 보고서 및 백데이터 생성 코드의 기능
검증으로 제한합니다.

IR·NN 후보와 임계값은 Validation 120개에서만 탐색하고 Test 120개는 최종 평가에만 사용합니다.
합성 데이터 확대가 후처리 성능 향상을 보장하지 않으며, 실제 개선 여부는 생성된 비교표로 판단합니다.
