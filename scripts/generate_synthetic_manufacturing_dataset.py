from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import wave
from array import array
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "sample" / "manufacturing_synthetic"
VOICE_NAME = "Microsoft Heami Desktop"
SAMPLE_RATE = 16_000
APPROVAL_ID = "SYNTHETIC-FIXTURE-NO-HUMAN-DATA"

SPLIT_RATES = {
    "train": -2,
    "validation": 0,
    "test": 2,
}

UTTERANCES = [
    ("train", "press", "프레스 3호기 안전 센서를 점검합니다."),
    ("train", "press", "프레스 3호기 유압 압력을 확인하세요."),
    ("train", "bearing", "베어링 온도가 기준값을 초과했습니다."),
    ("train", "bearing", "베어링 교체 작업을 시작합니다."),
    ("train", "conveyor", "컨베이어 운전 속도를 낮춰 주세요."),
    ("train", "conveyor", "컨베이어 비상 정지 버튼을 확인합니다."),
    ("train", "vision", "비전 검사기 조명을 조정합니다."),
    ("train", "vision", "비전 검사기에서 불량품을 검출했습니다."),
    ("train", "equipment_anomaly", "설비 이상 경보가 발생했습니다."),
    ("train", "equipment_anomaly", "설비 이상 원인을 점검해 주세요."),
    ("train", "safety", "작업자는 보호 장갑을 착용하세요."),
    ("train", "safety", "생산 라인 전원을 차단합니다."),
    ("train", "motor", "모터 진동 수치를 기록해 주세요."),
    ("train", "assembly", "조립 공정의 체결 토크를 확인합니다."),
    ("train", "material", "자재 투입 수량을 다시 확인하세요."),
    ("train", "cooling", "냉각수 누수 여부를 점검합니다."),
    ("train", "robot", "로봇 팔의 동작 범위를 확인하세요."),
    ("train", "quality", "품질 검사 결과를 작업 일지에 기록합니다."),
    ("validation", "press", "프레스 3호기의 금형 위치를 조정합니다."),
    ("validation", "bearing", "베어링 소음이 평소보다 크게 들립니다."),
    ("validation", "conveyor", "컨베이어 센서에 이물질이 감지되었습니다."),
    ("validation", "vision", "비전 검사기 카메라 초점을 맞춰 주세요."),
    ("validation", "equipment_anomaly", "설비 이상 발생 시 즉시 관리자에게 보고하세요."),
    ("validation", "packaging", "포장 공정의 라벨 부착 상태를 확인합니다."),
    ("test", "press", "프레스 3호기 가동을 일시 중지합니다."),
    ("test", "bearing", "베어링 윤활유 상태를 확인해 주세요."),
    ("test", "conveyor", "컨베이어 위의 제품 간격을 조정합니다."),
    ("test", "vision", "비전 검사기 판정 결과를 저장합니다."),
    ("test", "equipment_anomaly", "설비 이상 경보를 해제하기 전에 원인을 확인하세요."),
    ("test", "shipping", "출하 전 최종 품질 검사를 실시합니다."),
]

NOISE_PROFILES = (
    ("quiet_tts", None, 0.0),
    ("fan_noise_snr24db", 24.0, 120.0),
    ("equipment_hum_snr18db", 18.0, 60.0),
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("Windows PowerShell is required to generate the TTS fixture")
    return executable


def _synthesize(text: str, output_path: Path, rate: int, voice: str) -> None:
    escaped_text = text.replace("'", "''")
    escaped_voice = voice.replace("'", "''")
    escaped_path = str(output_path.resolve()).replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    {SAMPLE_RATE},
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
try {{
    $synth.SelectVoice('{escaped_voice}')
    $synth.Rate = {rate}
    $synth.SetOutputToWaveFile('{escaped_path}', $format)
    $synth.Speak('{escaped_text}')
}} finally {{
    $synth.Dispose()
}}
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        [_powershell(), "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"TTS synthesis failed for {output_path.name}: {details}")


def _add_noise(path: Path, snr_db: float | None, hum_hz: float, seed: int) -> None:
    if snr_db is None:
        return
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != SAMPLE_RATE:
        raise ValueError(f"Unexpected WAV format: {path}")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    signal_rms = math.sqrt(sum(value * value for value in samples) / max(len(samples), 1))
    noise_rms = max(signal_rms, 1.0) / (10 ** (snr_db / 20.0))
    random_source = random.Random(seed)
    hum_amplitude = noise_rms * 0.45
    gaussian_scale = noise_rms * 0.9
    for index, value in enumerate(samples):
        hum = hum_amplitude * math.sin(2 * math.pi * hum_hz * index / SAMPLE_RATE)
        noisy = round(value + hum + random_source.gauss(0.0, gaussian_scale))
        samples[index] = max(-32_768, min(32_767, noisy))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())


def _wav_metadata(path: Path) -> tuple[float, str]:
    with wave.open(str(path), "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return duration, digest


def generate_dataset(output_dir: Path, voice: str, overwrite: bool) -> Path:
    if platform.system() != "Windows":
        raise RuntimeError("This generator uses the Windows System.Speech TTS runtime")
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {manifest_path}. Use --overwrite to rebuild."
        )
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    split_sequence: Counter[str] = Counter()
    for global_index, (split, scenario, text) in enumerate(UTTERANCES, start=1):
        split_sequence[split] += 1
        sample_id = f"syn_{split}_{split_sequence[split]:03d}"
        audio_path = audio_dir / f"{sample_id}.wav"
        rate = SPLIT_RATES[split]
        noise_name, snr_db, hum_hz = NOISE_PROFILES[(global_index - 1) % len(NOISE_PROFILES)]
        _synthesize(text, audio_path, rate, voice)
        _add_noise(audio_path, snr_db, hum_hz, seed=20_260_810 + global_index)
        duration, digest = _wav_metadata(audio_path)
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": f"audio/{audio_path.name}",
                "reference_text": text,
                "split": split,
                "source": "synthetic_windows_system_speech",
                "consent_status": "synthetic",
                "speaker_id": f"synthetic_heami_rate_{rate:+d}",
                "scenario_id": scenario,
                "noise_condition": noise_name,
                "approval_id": APPROVAL_ID,
                "deidentified": "true",
                "label_reviewer": "not_applicable_synthetic_source_text",
                "label_review_status": "synthetic_generated",
                "tts_voice": voice,
                "tts_rate": str(rate),
                "audio_duration_seconds": f"{duration:.3f}",
                "audio_sha256": digest,
            }
        )

    fieldnames = list(rows[0])
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "dataset_id": "synthetic-manufacturing-korean-asr-v1",
        "generated_on": "2026-08-10",
        "generator": "scripts/generate_synthetic_manufacturing_dataset.py",
        "human_voice_data": False,
        "contains_personal_information": False,
        "tts_engine": "Windows System.Speech.Synthesis.SpeechSynthesizer",
        "tts_voice": voice,
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "speaker_definition": (
            "Synthetic rate variants of one installed Korean TTS voice; not real speakers"
        ),
        "noise_profiles": [profile[0] for profile in NOISE_PROFILES],
        "intended_use": "ASR pipeline, reporting, and artifact-contract functional testing",
        "prohibited_claims": [
            "real manufacturing ASR accuracy",
            "human speaker generalization",
            "production readiness",
            "human transcript review completion",
        ],
        "licensing_note": (
            "Generated locally from project-authored phrases with an installed Microsoft Windows "
            "voice. Confirm applicable Microsoft terms before redistribution outside this project."
        ),
        "documentation": (
            "https://learn.microsoft.com/dotnet/api/system.speech.synthesis.speechsynthesizer"
        ),
        "integrity": "SHA-256 per file is recorded in manifest.csv",
    }
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Korean synthetic manufacturing ASR set"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voice", default=VOICE_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(args.output_dir.resolve(), args.voice, args.overwrite)
    print(manifest)


if __name__ == "__main__":
    main()
