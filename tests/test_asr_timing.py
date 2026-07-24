from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from aias_specialist import asr
from aias_specialist.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_inference_warmup_and_repetitions_are_recorded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"model")
    config = yaml.safe_load((ROOT / "configs/local_model_smoke.yaml").read_text(encoding="utf-8"))
    config["paths"]["model_lock"] = str(tmp_path / "model-lock.yaml")
    config["model"]["local_dir"] = str(model_dir)
    config["evaluation"] = {
        "split": "test",
        "warmup_samples": 1,
        "timing_repetitions": 3,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = load_settings(config_path)

    calls: list[str] = []

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, audio_path: str, **kwargs):
            calls.append(audio_path)
            return iter([SimpleNamespace(text=" 정상 예측 ")]), SimpleNamespace(duration=2.0)

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(
        asr,
        "download_baseline_model",
        lambda settings: (model_dir, "a" * 40),
    )
    monkeypatch.setattr(asr, "_gpu_memory_used_mb", lambda: 0.0)
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "audio_path": str(audio),
                "reference_text": "정상 예측",
                "split": "test",
                "source": "synthetic",
                "consent_status": "synthetic",
            }
        ]
    )

    result, revision = asr.run_inference(frame, settings)

    assert revision == "a" * 40
    assert len(calls) == 4
    assert result.loc[0, "prediction_text"] == "정상 예측"
    assert result.loc[0, "warmup_samples"] == 1
    assert result.loc[0, "timing_repetitions"] == 3
