# Whisper 모델·양자화 비교 운영 절차

## 평가 원칙

- 모델과 양자화 선택에는 `validation`만 사용한다.
- 최종 성능 확인에는 선택 완료 후 `test`를 한 번 사용한다.
- 모델 비교 단계는 동일한 FP16, 디코딩, VAD, 언어 설정을 사용한다.
- 비교 순위는 후처리 이전의 원본 ASR 예측 지표로 작성한다.
- CER, WER, 제조 용어 재현율, RTF, 지연시간, 모델 크기, CPU/GPU 메모리를 함께 본다.
- 최저 CER 후보를 자동 확정하지 않고 담당자와 선택 이유를 남긴다.

## Colab 명령

```bash
aias model-matrix-lock \
  --matrix configs/benchmarks/whisper_models_colab.yaml

aias benchmark-models \
  --matrix configs/benchmarks/whisper_models_colab.yaml

aias select-model \
  --benchmark-dir <Drive benchmark directory> \
  --model-id small \
  --reviewer Pronesis9758 \
  --reason "정확도·속도·메모리 균형"

aias quantization-sweep \
  --spec configs/quantization/whisper_quantization_colab.yaml \
  --selection <benchmark directory>/model_selection.yaml

aias select-quantization \
  --quantization-dir <Drive quantization directory> \
  --variant-id int8-float16 \
  --reviewer Pronesis9758 \
  --reason "허용 가능한 CER 손실과 메모리 절감"

aias finalize-evaluation \
  --selection <quantization directory>/quantization_selection.yaml \
  --config configs/colab_public_sample.yaml
```

## 재시작

비교 그룹 ID와 snapshot이 같으면 완료된 멤버는 다시 실행하지 않는다. 실행 중 끊겼거나
실패한 멤버만 다시 시도한다. 동일 ID의 snapshot과 현재 YAML이 다르면 실행을 차단한다.
데이터, 모델 목록, 양자화 단계, 디코딩 설정을 바꿀 때는 그룹 ID의 버전을 올린다.

실행 중에는 revision 고정, 모델 다운로드·변환, 샘플별 추론 진행률과 완료 지표를 즉시
출력한다. `benchmark-models`가 revision 고정까지 수행하므로 바로 앞에서
`model-matrix-lock`을 별도로 실행할 필요는 없다.

## Colab 모델 캐시

- 변환된 CTranslate2 모델은 항상 Google Drive의
  `AI_Specialist_ASR_Project/models/ct2/` 아래에 저장하며 다음 런타임에서 재사용한다.
- 같은 Colab 런타임에서는 Hugging Face 원본 모델도 `/content/cache/huggingface`에서
  자동 재사용한다.
- 런타임을 자주 재시작하면서 모델을 다시 변환해야 한다면 노트북 설정의
  `PERSIST_HF_SOURCE_CACHE=True`로 원본까지 Drive에 보존할 수 있다.
- 원본 캐시는 약 15GB 이상의 추가 공간을 사용할 수 있고 Drive I/O가 로컬 `/content`보다
  느릴 수 있으므로 기본값은 `False`다. 일반적인 반복 평가에는 변환 모델 Drive 캐시만으로
  충분하다.

## 실제 녹음으로 전환

공개 데이터용 base config를 복사하고 `paths.manifest`를 승인된 실제 데이터 manifest로
바꾼다. 실제 음성은 `data/private/` 또는 승인된 저장소에 보관하며 Git에 커밋하지 않는다.
Manifest에는 출처와 동의·승인 상태가 있어야 하며, 정답 문장과 split은 사람이 검수한다.

## 보고서 해석

Validation 보고서는 선택 근거이고 최종 성능 보고서가 아니다. Test 보고서도 실제 제조
분포, 화자, 소음, 장비, 용어를 충분히 대표하는지 사람이 확인하기 전에는 운영 준비 완료로
표현하지 않는다.
