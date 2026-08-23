from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "sample" / "manufacturing_synthetic"
VOICE_NAME = "Microsoft Heami Desktop"
SAMPLE_RATE = 16_000
APPROVAL_ID = "SYNTHETIC-FIXTURE-NO-HUMAN-DATA"
DATASET_ID = "synthetic-manufacturing-korean-asr-v3"
GENERATED_ON = "2026-08-23"

SPLIT_RATES = {
    "train": (-4, -3, -2),
    "validation": (-1, 0),
    "test": (1, 2, 3),
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    canonical_term: str
    related_term: str
    subjects: tuple[str, ...]
    components: tuple[str, ...]


SCENARIOS = (
    Scenario(
        "press",
        "프레스 3호기",
        "피엘씨",
        (
            "프레스 3호기 금형부",
            "프레스 3호기 유압부",
            "프레스 3호기 투입부",
            "프레스 3호기 배출부",
            "프레스 3호기 안전 구역",
        ),
        ("안전 센서", "유압 압력", "금형 위치", "비상 정지 회로"),
    ),
    Scenario(
        "bearing",
        "베어링",
        "설비 이상",
        ("주축 베어링", "모터 베어링", "감속기 베어링", "펌프 베어링", "팬 베어링"),
        ("온도", "진동", "윤활유", "회전 소음"),
    ),
    Scenario(
        "conveyor",
        "컨베이어",
        "에이지브이",
        (
            "투입 컨베이어",
            "검사 컨베이어",
            "조립 컨베이어",
            "포장 컨베이어",
            "출하 컨베이어",
        ),
        ("운전 속도", "제품 간격", "정렬 센서", "비상 정지 버튼"),
    ),
    Scenario(
        "vision",
        "비전 검사기",
        "에이오아이 검사기",
        (
            "전면 비전 검사기",
            "후면 비전 검사기",
            "조립 비전 검사기",
            "포장 비전 검사기",
            "출하 비전 검사기",
        ),
        ("카메라 초점", "조명 밝기", "검출 임계값", "불량 판정 결과"),
    ),
    Scenario(
        "equipment_anomaly",
        "설비 이상",
        "에이치엠아이",
        (
            "설비 이상 감시 화면",
            "설비 이상 경보 장치",
            "설비 이상 분석 모듈",
            "설비 이상 이력 화면",
            "설비 이상 진단 서버",
        ),
        ("경보 코드", "발생 시각", "진동 추세", "복구 상태"),
    ),
    Scenario(
        "plc",
        "피엘씨",
        "에이치엠아이",
        ("프레스 피엘씨", "조립 피엘씨", "포장 피엘씨", "검사 피엘씨", "출하 피엘씨"),
        ("입력 신호", "출력 신호", "인터록", "통신 상태"),
    ),
    Scenario(
        "cnc",
        "씨엔씨 선반",
        "체결 토크",
        (
            "일 호기 씨엔씨 선반",
            "이 호기 씨엔씨 선반",
            "삼 호기 씨엔씨 선반",
            "사 호기 씨엔씨 선반",
            "오 호기 씨엔씨 선반",
        ),
        ("주축 회전수", "공구 보정값", "절삭유 유량", "가공 원점"),
    ),
    Scenario(
        "agv",
        "에이지브이",
        "컨베이어",
        (
            "자재 운반 에이지브이",
            "공정 이동 에이지브이",
            "검사 이동 에이지브이",
            "포장 이동 에이지브이",
            "출하 이동 에이지브이",
        ),
        ("배터리 잔량", "주행 경로", "정지 센서", "도킹 위치"),
    ),
    Scenario(
        "aoi",
        "에이오아이 검사기",
        "비전 검사기",
        (
            "전공정 에이오아이 검사기",
            "후공정 에이오아이 검사기",
            "표면 에이오아이 검사기",
            "납땜 에이오아이 검사기",
            "출하 에이오아이 검사기",
        ),
        ("검사 해상도", "조명 각도", "불량 좌표", "판정 임계값"),
    ),
    Scenario(
        "welding_robot",
        "용접 로봇",
        "체결 토크",
        (
            "차체 용접 로봇",
            "프레임 용접 로봇",
            "브래킷 용접 로봇",
            "패널 용접 로봇",
            "보강재 용접 로봇",
        ),
        ("용접 전류", "용접 속도", "토치 위치", "보호 가스 압력"),
    ),
)

DIFFICULTIES = ("normal", "term_dense", "error_prone")
DIFFICULTY_SPLIT_COUNTS = {"train": 12, "validation": 4, "test": 4}
MODEL_CODE_PAIRS = (
    ("엑스 백일", "X101"),
    ("엑스 백이", "X102"),
    ("케이 이백일", "K201"),
    ("케이 이백이", "K202"),
    ("에이 삼백일", "A301"),
    ("에이 삼백이", "A302"),
    ("엠 사백일", "M401"),
    ("엠 사백이", "M402"),
    ("큐 오백일", "Q501"),
    ("큐 오백이", "Q502"),
    ("티 육백일", "T601"),
    ("티 육백이", "T602"),
    ("브이 칠백일", "V701"),
    ("브이 칠백이", "V702"),
    ("알 팔백일", "R801"),
    ("알 팔백이", "R802"),
    ("에스 구백일", "S901"),
    ("에스 구백이", "S902"),
    ("지 천일", "G1001"),
    ("지 천이", "G1002"),
)
MODEL_CODES = tuple(spoken for spoken, _ in MODEL_CODE_PAIRS)
EQUIPMENT_NUMBER_PAIRS = (
    ("일 호기", "1호기"),
    ("이 호기", "2호기"),
    ("삼 호기", "3호기"),
    ("사 호기", "4호기"),
    ("오 호기", "5호기"),
)
STANDARD_TRANSCRIPT_PAIRS = (
    ("에이오아이 검사기", "AOI 검사기"),
    ("에이치엠아이", "HMI"),
    ("씨엔씨 선반", "CNC 선반"),
    ("에이지브이", "AGV"),
    ("피엘씨", "PLC"),
    *EQUIPMENT_NUMBER_PAIRS,
    *MODEL_CODE_PAIRS,
)
ACTION_STEMS = ("점검", "확인", "기록", "검증")

NOISE_PROFILES = (
    ("quiet_tts", None, 0.0),
    ("fan_noise_snr20db", 20.0, 120.0),
    ("equipment_hum_snr14db", 14.0, 60.0),
    ("hard_factory_noise_snr8db", 8.0, 90.0),
)


def _sentence(scenario: Scenario, subject: str, component: str, difficulty: str) -> str:
    action = ACTION_STEMS[scenario.components.index(component)]
    combination_index = (
        scenario.subjects.index(subject) * len(scenario.components)
        + scenario.components.index(component)
    )
    if difficulty == "normal":
        return f"{subject}의 {component} 상태를 {action}합니다."
    if difficulty == "term_dense":
        return (
            f"{subject}의 {component} 값을 {action}하고 "
            f"{scenario.related_term} 기록과 비교합니다."
        )
    model_code = MODEL_CODES[combination_index]
    return (
        f"{subject} 모델 {model_code}의 {component} 항목을 {action}한 뒤 "
        f"{scenario.related_term} 상태를 확인합니다."
    )


def _to_reference_text(spoken_text: str) -> str:
    """Convert a Korean pronunciation script into the required factory transcript style."""
    reference_text = spoken_text
    for spoken, standard in STANDARD_TRANSCRIPT_PAIRS:
        reference_text = reference_text.replace(spoken, standard)
    return reference_text


def build_utterances(seed: int = 20_260_823) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_order = ("train", "validation", "test")
    for scenario_index, scenario in enumerate(SCENARIOS):
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            combinations = [
                (subject, component)
                for subject in scenario.subjects
                for component in scenario.components
            ]
            random.Random(seed + scenario_index * 100 + difficulty_index).shuffle(combinations)
            start = 0
            for split in split_order:
                count = DIFFICULTY_SPLIT_COUNTS[split]
                for subject, component in combinations[start : start + count]:
                    spoken_text = _sentence(scenario, subject, component, difficulty)
                    target_terms = [_to_reference_text(scenario.canonical_term)]
                    if difficulty != "normal":
                        target_terms.append(_to_reference_text(scenario.related_term))
                    if difficulty == "error_prone":
                        target_terms.extend(
                            standard
                            for spoken, standard in MODEL_CODE_PAIRS
                            if spoken in spoken_text
                        )
                    target_terms.extend(
                        standard
                        for spoken, standard in EQUIPMENT_NUMBER_PAIRS
                        if spoken in spoken_text
                    )
                    rows.append(
                        {
                            "split": split,
                            "scenario_id": scenario.scenario_id,
                            "difficulty_group": difficulty,
                            "term_targets": "|".join(target_terms),
                            "spoken_text": spoken_text,
                            "reference_text": _to_reference_text(spoken_text),
                        }
                    )
                start += count
    split_rank = {split: index for index, split in enumerate(split_order)}
    rows.sort(
        key=lambda row: (
            split_rank[row["split"]],
            row["scenario_id"],
            row["difficulty_group"],
            row["spoken_text"],
        )
    )
    expected_counts = {"train": 360, "validation": 120, "test": 120}
    if Counter(row["split"] for row in rows) != Counter(expected_counts):
        raise AssertionError("Synthetic split construction did not produce 360/120/120")
    if len({row["reference_text"] for row in rows}) != len(rows):
        raise AssertionError("Synthetic reference texts must be unique across all splits")
    if len({row["spoken_text"] for row in rows}) != len(rows):
        raise AssertionError("Synthetic spoken texts must be unique across all splits")
    return rows


UTTERANCES = build_utterances()


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("Windows PowerShell is required to generate the TTS fixture")
    return executable


def _synthesize_batch(items: list[dict[str, str | int]], voice: str) -> None:
    script = f"""
param(
    [Parameter(Mandatory=$true)][string]$PayloadPath,
    [Parameter(Mandatory=$true)][string]$VoiceName
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    {SAMPLE_RATE},
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
try {{
    $synth.SelectVoice($VoiceName)
    $items = Get-Content -Raw -Encoding UTF8 -LiteralPath $PayloadPath | ConvertFrom-Json
    $index = 0
    foreach ($item in $items) {{
        $synth.Rate = [int]$item.rate
        $synth.SetOutputToWaveFile([string]$item.output_path, $format)
        $synth.Speak([string]$item.text)
        $synth.SetOutputToNull()
        $index += 1
        if (($index % 50) -eq 0) {{
            Write-Output ("Synthesized {{0}}/{{1}}" -f $index, $items.Count)
        }}
    }}
}} finally {{
    $synth.Dispose()
}}
"""
    with tempfile.TemporaryDirectory(prefix="aias-synthetic-tts-") as temporary:
        temporary_dir = Path(temporary)
        payload_path = temporary_dir / "tts_payload.json"
        script_path = temporary_dir / "synthesize.ps1"
        payload_path.write_text(
            json.dumps(items, ensure_ascii=False), encoding="utf-8"
        )
        script_path.write_text(script, encoding="utf-8-sig")
        result = subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-PayloadPath",
                str(payload_path),
                "-VoiceName",
                voice,
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"TTS batch synthesis failed: {details}")
    if result.stdout.strip():
        print(result.stdout.strip())


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


def generate_dataset(
    output_dir: Path,
    voice: str,
    overwrite: bool,
    reuse_existing_audio: bool = False,
) -> Path:
    if platform.system() != "Windows":
        raise RuntimeError("This generator uses the Windows System.Speech TTS runtime")
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {manifest_path}. Use --overwrite to rebuild."
        )
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    synthesis_items: list[dict[str, str | int]] = []
    planned_rows: list[dict[str, str | int | Path]] = []
    split_sequence: Counter[str] = Counter()
    for global_index, utterance in enumerate(UTTERANCES, start=1):
        split = utterance["split"]
        split_sequence[split] += 1
        sample_id = f"syn_{split}_{split_sequence[split]:03d}"
        audio_path = audio_dir / f"{sample_id}.wav"
        split_index = split_sequence[split] - 1
        rate = SPLIT_RATES[split][split_index % len(SPLIT_RATES[split])]
        noise_name, snr_db, hum_hz = NOISE_PROFILES[split_index % len(NOISE_PROFILES)]
        synthesis_items.append(
            {
                "text": utterance["spoken_text"],
                "output_path": str(audio_path),
                "rate": rate,
            }
        )
        planned_rows.append(
            {
                **utterance,
                "global_index": global_index,
                "sample_id": sample_id,
                "audio_path_object": audio_path,
                "rate": rate,
                "noise_name": noise_name,
                "snr_db": snr_db,
                "hum_hz": hum_hz,
            }
        )

    if reuse_existing_audio:
        missing_audio = [
            str(item["output_path"])
            for item in synthesis_items
            if not Path(str(item["output_path"])).exists()
        ]
        if missing_audio:
            preview = "\n".join(missing_audio[:10])
            raise FileNotFoundError(f"Existing synthetic WAV files are missing:\n{preview}")
        print(f"Reusing {len(synthesis_items)} existing WAV files; refreshing labels only...")
    else:
        print(f"Synthesizing {len(synthesis_items)} unique Korean manufacturing utterances...")
        _synthesize_batch(synthesis_items, voice)

    rows: list[dict[str, str]] = []
    for planned in planned_rows:
        audio_path = Path(planned["audio_path_object"])
        global_index = int(planned["global_index"])
        snr_value = planned["snr_db"]
        snr_db = None if snr_value is None else float(snr_value)
        hum_hz = float(planned["hum_hz"])
        if not reuse_existing_audio:
            _add_noise(audio_path, snr_db, hum_hz, seed=20_260_810 + global_index)
        duration, digest = _wav_metadata(audio_path)
        rows.append(
            {
                "sample_id": str(planned["sample_id"]),
                "audio_path": f"audio/{audio_path.name}",
                "reference_text": str(planned["reference_text"]),
                "spoken_text": str(planned["spoken_text"]),
                "split": str(planned["split"]),
                "source": "synthetic_windows_system_speech",
                "consent_status": "synthetic",
                "speaker_id": f"synthetic_heami_rate_{int(planned['rate']):+d}",
                "scenario_id": str(planned["scenario_id"]),
                "difficulty_group": str(planned["difficulty_group"]),
                "term_targets": str(planned["term_targets"]),
                "noise_condition": str(planned["noise_name"]),
                "approval_id": APPROVAL_ID,
                "deidentified": "true",
                "label_reviewer": "not_applicable_synthetic_source_text",
                "label_review_status": "synthetic_generated",
                "tts_voice": voice,
                "tts_rate": str(planned["rate"]),
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
        "dataset_id": DATASET_ID,
        "generated_on": GENERATED_ON,
        "generator": "scripts/generate_synthetic_manufacturing_dataset.py",
        "human_voice_data": False,
        "contains_personal_information": False,
        "tts_engine": "Windows System.Speech.Synthesis.SpeechSynthesizer",
        "tts_voice": voice,
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": len(rows),
        "unique_reference_count": len({row["reference_text"] for row in rows}),
        "unique_spoken_text_count": len({row["spoken_text"] for row in rows}),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "difficulty_counts": dict(Counter(row["difficulty_group"] for row in rows)),
        "noise_condition_counts": dict(Counter(row["noise_condition"] for row in rows)),
        "scenario_counts": dict(Counter(row["scenario_id"] for row in rows)),
        "speaker_definition": (
            "Seven split-disjoint rate variants of one installed Korean TTS voice; "
            "not real speakers"
        ),
        "noise_profiles": [profile[0] for profile in NOISE_PROFILES],
        "audio_generation_mode": (
            "reused_existing_wav" if reuse_existing_audio else "generated_from_spoken_text"
        ),
        "transcription_policy": {
            "spoken_text": "Korean pronunciation script passed to TTS",
            "reference_text": (
                "Required final transcript with official Latin acronyms and model codes"
            ),
            "standardized_terms": [
                "PLC",
                "HMI",
                "CNC",
                "AGV",
                "AOI",
                "alphanumeric model codes",
                "Arabic-number equipment identifiers",
            ],
        },
        "intended_use": (
            "ASR pipeline, domain-term correction tuning on validation, held-out test "
            "evaluation, reporting, and artifact-contract functional testing"
        ),
        "postprocessing_design": {
            "train": "model training only",
            "validation": "IR and nearest-neighbor thresholds and candidate selection",
            "test": "held-out final evaluation only; never threshold tuning",
            "guaranteed_improvement": False,
        },
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
    parser.add_argument(
        "--reuse-existing-audio",
        action="store_true",
        help="Keep existing WAV files and rebuild only manifest labels and provenance",
    )
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_dir.resolve(),
        args.voice,
        args.overwrite,
        reuse_existing_audio=args.reuse_existing_audio,
    )
    print(manifest)


if __name__ == "__main__":
    main()
