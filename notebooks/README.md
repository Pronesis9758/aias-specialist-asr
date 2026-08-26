# Colab notebook

- `colab_runner.ipynb`: 공개 데이터 Baseline과 Whisper LoRA 데모
- `colab_model_benchmark_quantization.ipynb`: Whisper 모델 크기 비교, 모델 선택,
  FP16·INT8-FP16 양자화 비교, 양자화 선택, 고정 Test 최종평가
- `colab_manufacturing_assessment.ipynb`: 공개 프록시, 합성 제조 TTS 또는 승인된 실제
  제조 데이터를 선택해 거버넌스 검증, 모델 비교, 선택 모델 LoRA, 양자화, 고정 Test와
  심사 준비도 점검

모델 비교 노트북은 변환된 CTranslate2 모델을 Google Drive에 캐시합니다. 변환 전
Hugging Face 원본 모델도 런타임 간 재사용하려면 첫 설정 셀의
`PERSIST_HF_SOURCE_CACHE=True`를 사용합니다.

각 노트북은 대응하는 `scripts/build_*_notebook.py`로 생성합니다. 두 노트북 모두
카메라·마이크·패스키 권한을 사용하지 않습니다.

로컬에서는 Colab 전용 Drive/GPU 셀을 실행하지 않습니다. 구조 검증은 다음과 같이 수행합니다.

```powershell
uv run python scripts/build_colab_notebook.py
uv run python scripts/build_benchmark_notebook.py
uv run python scripts/build_manufacturing_notebook.py
uv run python -m json.tool notebooks/colab_runner.ipynb > $null
uv run python -m json.tool notebooks/colab_model_benchmark_quantization.ipynb > $null
uv run python -m json.tool notebooks/colab_manufacturing_assessment.ipynb > $null
```

실제 실행 검증은 GPU 런타임의 Colab에서 위에서부터 순서대로 수행해야 합니다.
