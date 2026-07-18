# Colab notebook

`colab_runner.ipynb`는 `scripts/build_colab_notebook.py`로 생성합니다.

로컬에서는 Colab 전용 Drive/GPU 셀을 실행하지 않습니다. 구조 검증은 다음과 같이 수행합니다.

```powershell
uv run python scripts/build_colab_notebook.py
uv run python -m json.tool notebooks/colab_runner.ipynb > $null
```

실제 실행 검증은 GPU 런타임의 Colab에서 위에서부터 순서대로 수행해야 합니다.
