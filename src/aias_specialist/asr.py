from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil

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


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _gpu_memory_used_mb() -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [
            float(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip().replace(".", "", 1).isdigit()
        ]
        return max(values, default=0.0)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return 0.0


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
        output["runtime_device"] = settings.model.device
        output["compute_type"] = settings.model.compute_type
        output["storage_quantization"] = settings.model.conversion_quantization or "fixture"
        output["model_preparation_seconds"] = 0.0
        output["model_size_bytes"] = 0
        output["process_rss_mb"] = 0.0
        output["gpu_memory_mb"] = 0.0
        return output, "fixture"

    preparation_started = time.perf_counter()
    model_path, revision = download_baseline_model(settings)
    preparation_seconds = time.perf_counter() - preparation_started
    model_size_bytes = _directory_size_bytes(model_path)
    device, compute_type = _runtime_device(settings)

    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    peak_gpu_memory_mb = _gpu_memory_used_mb()
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
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 * 1024))
        peak_gpu_memory_mb = max(peak_gpu_memory_mb, _gpu_memory_used_mb())
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
                "runtime_device": device,
                "compute_type": compute_type,
                "storage_quantization": (
                    settings.model.conversion_quantization or "source-default"
                ),
                "model_preparation_seconds": preparation_seconds,
                "model_size_bytes": model_size_bytes,
                "process_rss_mb": peak_rss_mb,
                "gpu_memory_mb": peak_gpu_memory_mb,
            }
        )
    return pd.DataFrame(rows), revision
