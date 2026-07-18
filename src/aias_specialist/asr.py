from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import Settings
from .models import download_baseline_model


def _runtime_device(settings: Settings) -> tuple[str, str]:
    if settings.model.device != "auto":
        device = settings.model.device
    else:
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if settings.model.compute_type != "auto":
        compute_type = settings.model.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def run_inference(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, str]:
    if settings.model.backend == "fixture":
        output = frame.copy()
        output["prediction_text"] = output["fixture_prediction"].astype(str)
        output["latency_seconds"] = 0.0
        output["audio_duration_seconds"] = 0.0
        output["real_time_factor"] = 0.0
        output["backend"] = "fixture"
        output["model_repo"] = settings.model.repo_id
        output["model_revision"] = "fixture"
        return output, "fixture"

    model_path, revision = download_baseline_model(settings)
    device, compute_type = _runtime_device(settings)

    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        audio_path = Path(str(record["audio_path"]))
        started = time.perf_counter()
        segments, info = model.transcribe(
            str(audio_path),
            language=settings.model.language,
            beam_size=settings.model.beam_size,
            vad_filter=True,
        )
        prediction = " ".join(segment.text.strip() for segment in segments).strip()
        latency = time.perf_counter() - started
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        rows.append(
            {
                **record,
                "prediction_text": prediction,
                "latency_seconds": latency,
                "audio_duration_seconds": duration,
                "real_time_factor": latency / duration if duration > 0 else 0.0,
                "backend": "faster_whisper",
                "model_repo": settings.model.repo_id,
                "model_revision": revision,
            }
        )
    return pd.DataFrame(rows), revision
