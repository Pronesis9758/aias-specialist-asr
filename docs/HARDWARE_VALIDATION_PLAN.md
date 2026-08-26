# 개발 인프라 및 HW 검증 계획

## 현재 범위

Google Colab GPU는 모델 조사, LoRA 학습과 양자화 비교를 위한 개발·검증 인프라다. 실행마다
GPU 이름, 메모리, 드라이버, Python·패키지 버전과 Git SHA를 `environment.json`에 기록한다.
Colab 결과만으로 생산 On-device 준비 완료를 주장하지 않는다.

## 측정 방법

- 동일 모델 revision, 데이터, beam size와 전처리를 사용한다.
- 모델 로딩 후 최소 1개 샘플을 warm-up으로 실행한다.
- 각 샘플을 최소 3회 실행하고 중앙 지연시간을 사용한다.
- WER, CER, 제조 용어 재현율, 평균/P95 지연시간, aggregate RTF를 기록한다.
- 모델 파일 크기, peak process memory와 peak GPU memory를 기록한다.
- GPU 종류가 바뀐 Colab 실행 결과를 하나의 성능 수치로 합치지 않는다.

`evaluation.warmup_samples`와 `evaluation.timing_repetitions`로 측정 조건을 설정한다.

## On-device로 확장할 경우

`configs/hardware/target_device_template.yaml`을 복사해 CPU/GPU/RAM/OS와 허용 기준을
채운다. 실제 목표 장비에서 FP16·INT8·CPU INT8을 실행하고 정확도 손실과 자원 절감이
허용 기준을 만족하는지 판단한다.

최종 모델은 가장 작은 모델이 아니라, 승인된 정확도·지연시간·메모리 기준을 모두 만족하는
후보 중에서 선택한다.
