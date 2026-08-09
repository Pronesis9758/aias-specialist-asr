# 합성 제조 음성 테스트 세트

이 디렉터리는 실제 임직원·고객 음성이 아닌 Windows 한국어 TTS로 생성한 기능 검증용
데이터입니다.

## 구성

- Train 18개, Validation 6개, Test 6개
- 16 kHz, mono, PCM 16-bit WAV
- 제조 설비·안전·검사 문장 30개와 정답 전사
- 조용한 음성, 팬 잡음, 설비 저주파 잡음 조건
- split마다 서로 다른 TTS 속도 변형을 synthetic speaker ID로 기록
- 각 WAV의 SHA-256과 생성 출처를 `manifest.csv`에 기록

재생성 명령:

```powershell
uv run python scripts/generate_synthetic_manufacturing_dataset.py --overwrite
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
