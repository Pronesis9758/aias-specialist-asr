from __future__ import annotations

import subprocess
import time
from pathlib import Path
from statistics import median

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
        output["warmup_samples"] = settings.evaluation.warmup_samples
        output["timing_repetitions"] = settings.evaluation.timing_repetitions
        return output, "fixture"

    preparation_started = time.perf_counter()
    model_path, revision = download_baseline_model(settings)
    preparation_seconds = time.perf_counter() - preparation_started
    model_size_bytes = _directory_size_bytes(model_path)
    device, compute_type = _runtime_device(settings)
    print(
        f"[inference] loading model device={device} compute_type={compute_type} "
        f"size_mb={model_size_bytes / (1024 * 1024):.1f}",
        flush=True,
    )

    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    peak_gpu_memory_mb = _gpu_memory_used_mb()
    rows: list[dict[str, object]] = []
    records = frame.to_dict(orient="records")
    warmup_count = min(settings.evaluation.warmup_samples, len(records))
    for warmup_index, record in enumerate(records[:warmup_count], start=1):
        print(f"[inference] warmup {warmup_index}/{warmup_count}", flush=True)
        segments, _ = model.transcribe(
            str(Path(str(record["audio_path"]))),
            language=settings.model.language,
            beam_size=settings.model.beam_size,
            vad_filter=True,
        )
        list(segments)

    for index, record in enumerate(records, start=1):
        print(f"[inference] sample {index}/{len(records)}", flush=True)
        audio_path = Path(str(record["audio_path"]))
        latencies: list[float] = []
        predictions: list[str] = []
        durations: list[float] = []
        for repetition in range(1, settings.evaluation.timing_repetitions + 1):
            if settings.evaluation.timing_repetitions > 1:
                print(
                    f"[inference] sample {index}/{len(records)} "
                    f"repetition {repetition}/{settings.evaluation.timing_repetitions}",
                    flush=True,
                )
            started = time.perf_counter()
            segments, info = model.transcribe(
                str(audio_path),
                language=settings.model.language,
                beam_size=settings.model.beam_size,
                vad_filter=True,
            )
            predictions.append(" ".join(segment.text.strip() for segment in segments).strip())
            latencies.append(time.perf_counter() - started)
            durations.append(float(getattr(info, "duration", 0.0) or 0.0))
            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 * 1024))
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, _gpu_memory_used_mb())
        prediction = predictions[0]
        latency = median(latencies)
        duration = median(durations)
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
                "warmup_samples": warmup_count,
                "timing_repetitions": settings.evaluation.timing_repetitions,
            }
        )
        print(
            f"[inference] sample {index}/{len(records)} completed "
            f"latency={latency:.2f}s rtf="
            f"{latency / duration if duration > 0 else 0.0:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows), revision
