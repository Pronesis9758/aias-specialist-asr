from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import shutil
import subprocess
import wave
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import utc_now

SPLITS = ("train", "validation", "test")
DEFAULT_VOICES = (
    "ko-KR-SunHiNeural",
    "ko-KR-InJoonNeural",
    "ko-KR-HyunsuNeural",
)
PROSODY = (
    ("slow_low", "-12%", "-8Hz"),
    ("slow_high", "-8%", "+8Hz"),
    ("neutral_low", "+0%", "-5Hz"),
    ("neutral_high", "+0%", "+5Hz"),
    ("fast_low", "+10%", "-6Hz"),
    ("fast_high", "+12%", "+7Hz"),
    ("careful", "-5%", "+0Hz"),
    ("urgent", "+15%", "+2Hz"),
)
NOISE_PROFILES = (
    ("clean", None, 0.0),
    ("fan_snr20", 20.0, 120.0),
    ("equipment_hum_snr14", 14.0, 60.0),
    ("hard_factory_snr8", 8.0, 90.0),
)
TARGET_DURATIONS = (5.4, 5.8, 6.2, 6.6)
UNITS = ("뉴턴미터", "밀리미터", "바", "도", "퍼센트", "알피엠")
AREAS = ("가공", "조립", "검사", "물류", "포장", "출하")


def load_synthetic_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    section = payload.get("synthetic_dataset") if isinstance(payload, dict) else None
    if not isinstance(section, dict):
        raise ValueError("Synthetic data spec requires a synthetic_dataset mapping")
    return spec_path, section


def _project_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _speaker_profiles(section: dict[str, Any]) -> list[dict[str, str]]:
    voices = tuple(str(value) for value in section.get("voices", DEFAULT_VOICES))
    configured_prosody = section.get("prosody_profiles")
    if configured_prosody is None:
        prosody = PROSODY
    else:
        if not isinstance(configured_prosody, list) or not configured_prosody:
            raise ValueError("synthetic_dataset.prosody_profiles must be a non-empty list")
        prosody = tuple(
            (
                str(item["id"]),
                str(item["rate"]),
                str(item["pitch"]),
            )
            for item in configured_prosody
        )
    counts = section.get("speaker_profiles", {"train": 16, "validation": 4, "test": 4})
    if not isinstance(counts, dict):
        raise ValueError("synthetic_dataset.speaker_profiles must be a mapping")
    assignment = str(section.get("profile_assignment", "aligned_cycle"))
    if assignment not in {"aligned_cycle", "cartesian"}:
        raise ValueError(
            "synthetic_dataset.profile_assignment must be aligned_cycle or cartesian"
        )
    acoustic_capacity = len(voices) * len(prosody)
    requested_profiles = sum(int(counts.get(split, 0)) for split in SPLITS)
    require_unique = bool(section.get("require_unique_acoustic_profiles", False))
    if assignment == "cartesian" and require_unique and requested_profiles > acoustic_capacity:
        raise ValueError(
            "Requested speaker profiles exceed the unique voice x prosody capacity: "
            f"{requested_profiles} > {acoustic_capacity}"
        )
    speaker_prefix = str(section.get("speaker_id_prefix", "synthetic"))
    profiles: list[dict[str, str]] = []
    offset = 0
    for split in SPLITS:
        count = int(counts.get(split, 0))
        for local_index in range(count):
            profile_index = offset + local_index
            if assignment == "cartesian":
                # 음성별로 모든 속도·피치 조합을 순회해 화자 ID만 늘고 실제 음향 조합은
                # 반복되는 문제를 방지합니다. split offset을 공유하므로 train/validation의
                # 음향 조합도 요청 용량 내에서는 겹치지 않습니다.
                voice = voices[profile_index % len(voices)]
                style, rate, pitch = prosody[
                    (profile_index // len(voices)) % len(prosody)
                ]
            else:
                # 기존 v1/v2의 불변 데이터 계획과 재현성을 보존합니다.
                voice = voices[profile_index % len(voices)]
                style, rate, pitch = prosody[profile_index % len(prosody)]
            profiles.append(
                {
                    "speaker_id": f"{speaker_prefix}_{split}_{local_index + 1:02d}_{style}",
                    "split": split,
                    "voice": voice,
                    "rate": rate,
                    "pitch": pitch,
                }
            )
        offset += count
    return profiles


def _spoken_form(canonical: str, aliases: str) -> str:
    candidates = [item.strip() for item in str(aliases).split("|") if item.strip()]
    if any(character.isascii() and character.isalnum() for character in canonical):
        return candidates[0] if candidates else canonical
    if canonical[:1].isdigit():
        return candidates[0] if candidates else canonical
    return canonical


def _sentence(
    *,
    split: str,
    index: int,
    primary_spoken: str,
    primary_reference: str,
    secondary_spoken: str | None,
    secondary_reference: str | None,
) -> tuple[str, str, str]:
    area = AREAS[index % len(AREAS)]
    unit = UNITS[index % len(UNITS)]
    value = 10 + (index * 17) % 890
    order = f"{split[:1].upper()}{index + 1:04d}"
    if secondary_spoken and secondary_reference:
        spoken = (
            f"{area} 공정 작업 번호 {order}입니다. {primary_spoken} 상태와 "
            f"{secondary_spoken} 상태를 함께 확인하고 측정값 {value} {unit}를 기록합니다."
        )
        reference = (
            f"{area} 공정 작업 번호 {order}입니다. {primary_reference} 상태와 "
            f"{secondary_reference} 상태를 함께 확인하고 측정값 {value} {unit}를 기록합니다."
        )
        return spoken, reference, f"{primary_reference}|{secondary_reference}"
    spoken = (
        f"{area} 공정 작업 번호 {order}입니다. {primary_spoken} 상태를 확인하고 "
        f"측정값 {value} {unit}를 작업 표준서에 기록합니다."
    )
    reference = (
        f"{area} 공정 작업 번호 {order}입니다. {primary_reference} 상태를 확인하고 "
        f"측정값 {value} {unit}를 작업 표준서에 기록합니다."
    )
    return spoken, reference, primary_reference


def _confirmatory_sentence(
    *,
    index: int,
    spoken_terms: list[str],
    reference_terms: list[str],
) -> tuple[str, str, str]:
    area = AREAS[(index + 3) % len(AREAS)]
    unit = UNITS[(index + 2) % len(UNITS)]
    value = 15 + (index * 29) % 880
    order = f"V2-{index + 1:04d}"
    spoken_joined = ", ".join(spoken_terms)
    reference_joined = ", ".join(reference_terms)
    spoken = (
        f"{area} 구역 교대 점검 {order} 결과를 보고합니다. {spoken_joined}의 표시 상태를 "
        f"차례로 확인했습니다. 기준 수치 {value} {unit}는 허용 범위입니다."
    )
    reference = (
        f"{area} 구역 교대 점검 {order} 결과를 보고합니다. {reference_joined}의 표시 상태를 "
        f"차례로 확인했습니다. 기준 수치 {value} {unit}는 허용 범위입니다."
    )
    return spoken, reference, "|".join(reference_terms)


def _confirmatory_negative_sentence(index: int) -> tuple[str, str, str]:
    order = f"V2-N{index + 1:04d}"
    sentence = (
        f"오늘 교대 작업 인수인계 {order} 내용을 기록합니다. 담당 구역과 작업 순서를 "
        "다시 확인하고 완료 시간을 문서에 남깁니다."
    )
    return sentence, sentence, ""


def _primary_term_indexes(
    *,
    split: str,
    count: int,
    term_records: list[dict[str, str]],
    section: dict[str, Any],
    seed: int,
) -> list[int]:
    if split != "train":
        indexes = [index % len(term_records) for index in range(count)]
        random.Random(seed).shuffle(indexes)
        return indexes
    priority = {str(term) for term in section.get("priority_terms", [])}
    priority_count = int(section.get("priority_train_occurrences_per_term", 200))
    counts = {
        index: priority_count if term["canonical"] in priority else 0
        for index, term in enumerate(term_records)
    }
    remaining = count - sum(counts.values())
    non_priority = [index for index, value in counts.items() if value == 0]
    if remaining < 0 or not non_priority:
        raise ValueError("Priority-term allocation exceeds the Train sample count")
    for offset in range(remaining):
        counts[non_priority[offset % len(non_priority)]] += 1
    indexes = [index for index, occurrence in counts.items() for _ in range(occurrence)]
    random.Random(seed).shuffle(indexes)
    return indexes


def build_dataset_plan(spec_path: str | Path) -> pd.DataFrame:
    path, section = load_synthetic_spec(spec_path)
    root = _project_root(path)
    term_path = _resolve(root, str(section["domain_terms"]))
    terms = pd.read_csv(term_path, dtype=str).fillna("")
    if not {"canonical", "aliases"}.issubset(terms.columns) or terms.empty:
        raise ValueError("Domain term dictionary must contain canonical and aliases")
    term_records = [
        {
            "canonical": str(row.canonical).strip(),
            "spoken": _spoken_form(str(row.canonical).strip(), str(row.aliases)),
        }
        for row in terms.itertuples(index=False)
    ]
    profiles = _speaker_profiles(section)
    seed = int(section.get("seed", 20_260_823))
    split_counts = section.get(
        "split_counts", {"train": 6000, "validation": 600, "test": 600}
    )
    rows: list[dict[str, Any]] = []
    dataset_role = str(section.get("dataset_role", "development"))
    sample_prefix = str(section.get("sample_id_prefix", "syn7200"))
    for split_index, split in enumerate(SPLITS):
        count = int(split_counts.get(split, 0))
        if count <= 0:
            continue
        split_profiles = [profile for profile in profiles if profile["split"] == split]
        if not split_profiles:
            raise ValueError(f"No synthetic speaker profiles configured for {split}")
        negative_count = (
            int(section.get("negative_test_samples", 0)) if split == "test" else 0
        )
        if negative_count < 0 or negative_count >= count:
            raise ValueError("negative_test_samples must be between 0 and test count - 1")
        positive_count = count - negative_count
        primary_indexes = _primary_term_indexes(
            split=split,
            count=positive_count,
            term_records=term_records,
            section=section,
            seed=seed + split_index,
        )
        negative_positions: set[int] = set()
        if negative_count:
            positions = list(range(count))
            random.Random(seed + 900 + split_index).shuffle(positions)
            negative_positions = set(positions[:negative_count])
        triple_count = (
            int(section.get("triple_term_test_samples", 0)) if split == "test" else 0
        )
        if triple_count < 0 or triple_count > positive_count:
            raise ValueError("triple_term_test_samples exceeds positive Test samples")
        secondary_order = list(range(len(term_records)))
        random.Random(seed + 100 + split_index).shuffle(secondary_order)
        positive_index = 0
        for index in range(count):
            if index in negative_positions:
                spoken, reference, targets = _confirmatory_negative_sentence(index)
                primary = None
                secondary = None
            else:
                primary = term_records[primary_indexes[positive_index]]
                secondary = None
                tertiary = None
                if split == "test":
                    secondary = term_records[
                        secondary_order[(positive_index * 7 + 11) % len(secondary_order)]
                    ]
                    if secondary["canonical"] == primary["canonical"]:
                        secondary = term_records[
                            secondary_order[(positive_index * 7 + 12) % len(secondary_order)]
                        ]
                if split == "test" and positive_index < triple_count:
                    tertiary = term_records[
                        secondary_order[(positive_index * 11 + 19) % len(secondary_order)]
                    ]
                    if tertiary["canonical"] in {
                        primary["canonical"],
                        secondary["canonical"] if secondary else "",
                    }:
                        tertiary = term_records[
                            secondary_order[(positive_index * 11 + 20) % len(secondary_order)]
                        ]
                if dataset_role == "confirmatory_test":
                    selected_terms = [primary, *([secondary] if secondary else [])]
                    if tertiary:
                        selected_terms.append(tertiary)
                    spoken, reference, targets = _confirmatory_sentence(
                        index=index,
                        spoken_terms=[term["spoken"] for term in selected_terms],
                        reference_terms=[term["canonical"] for term in selected_terms],
                    )
                else:
                    spoken, reference, targets = _sentence(
                        split=split,
                        index=index,
                        primary_spoken=primary["spoken"],
                        primary_reference=primary["canonical"],
                        secondary_spoken=secondary["spoken"] if secondary else None,
                        secondary_reference=secondary["canonical"] if secondary else None,
                    )
                positive_index += 1
            profile = split_profiles[index % len(split_profiles)]
            noise_name, snr_db, hum_hz = NOISE_PROFILES[index % len(NOISE_PROFILES)]
            sample_id = f"{sample_prefix}_{split}_{index + 1:04d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": f"audio/{sample_id}.wav",
                    "reference_text": reference,
                    "spoken_text": spoken,
                    "split": split,
                    "source": "synthetic_edge_tts",
                    "consent_status": "synthetic",
                    "speaker_id": profile["speaker_id"],
                    "tts_voice": profile["voice"],
                    "tts_rate": profile["rate"],
                    "tts_pitch": profile["pitch"],
                    "scenario_id": (
                        f"{split}_sentence_family_{index % 12:02d}"
                        if dataset_role == "development"
                        else f"{dataset_role}_{split}_family_{index % 18:02d}"
                    ),
                    "difficulty_group": (
                        "domain_negative"
                        if not targets
                        else "term_dense" if split == "test" else "normal"
                    ),
                    "term_targets": targets,
                    "noise_condition": noise_name,
                    "noise_snr_db": "" if snr_db is None else str(snr_db),
                    "noise_hum_hz": str(hum_hz),
                    "target_duration_seconds": f"{TARGET_DURATIONS[index % 4]:.1f}",
                    "approval_id": "SYNTHETIC-NO-HUMAN-DATA",
                    "deidentified": "true",
                    "label_reviewer": "synthetic_source_text",
                    "label_review_status": "synthetic_generated",
                }
            )
    plan = pd.DataFrame(rows)
    validate_dataset_plan(plan, section)
    return plan


def _term_counts(frame: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for targets in frame["term_targets"].astype(str):
        counts.update(term for term in targets.split("|") if term)
    return counts


def validate_dataset_plan(frame: pd.DataFrame, section: dict[str, Any]) -> dict[str, Any]:
    expected = {
        key: int(value)
        for key, value in section["split_counts"].items()
        if int(value) > 0
    }
    observed = frame.groupby("split").size().to_dict()
    if observed != expected:
        raise ValueError(f"Split counts do not match: expected={expected}, observed={observed}")
    if frame["sample_id"].duplicated().any() or frame["reference_text"].duplicated().any():
        raise ValueError("Synthetic sample IDs and reference texts must be globally unique")
    speaker_overlap = frame.groupby("speaker_id")["split"].nunique()
    if int(speaker_overlap.max()) != 1:
        raise ValueError("Synthetic speaker profiles must be split-disjoint")
    minimum_speakers = int(section.get("minimum_speaker_profiles", 20))
    if frame["speaker_id"].nunique() < minimum_speakers:
        raise ValueError(f"At least {minimum_speakers} synthetic speaker profiles are required")
    train_counts = _term_counts(frame.loc[frame["split"].eq("train")])
    if expected.get("train", 0):
        minimum_train = int(section.get("minimum_train_occurrences_per_term", 100))
        maximum_train = int(section.get("maximum_train_occurrences_per_term", 200))
        outside = {
            term: count
            for term, count in train_counts.items()
            if count < minimum_train or count > maximum_train
        }
        if outside:
            raise ValueError(f"Train term coverage is outside the configured range: {outside}")
    test_occurrences = sum(_term_counts(frame.loc[frame["split"].eq("test")]).values())
    minimum_test = int(section.get("minimum_test_term_occurrences", 1000))
    if expected.get("test", 0) and test_occurrences < minimum_test:
        raise ValueError(
            f"Test term occurrences must be at least {minimum_test}: {test_occurrences}"
        )
    negative_count = int(
        frame.loc[frame["split"].eq("test"), "term_targets"].astype(str).eq("").sum()
    )
    minimum_negatives = int(section.get("minimum_negative_test_samples", 0))
    if negative_count < minimum_negatives:
        raise ValueError(
            f"Negative Test samples must be at least {minimum_negatives}: {negative_count}"
        )
    split_texts = {
        split: set(frame.loc[frame["split"].eq(split), "reference_text"])
        for split in expected
    }
    split_names = list(split_texts)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if not split_texts[left].isdisjoint(split_texts[right]):
                raise ValueError("Reference sentences must not cross data splits")
    return {
        "split_counts": observed,
        "speaker_profile_count": int(frame["speaker_id"].nunique()),
        "train_term_occurrence_min": min(train_counts.values()) if train_counts else 0,
        "train_term_occurrence_max": max(train_counts.values()) if train_counts else 0,
        "test_term_occurrences": test_occurrences,
        "negative_test_samples": negative_count,
    }


def write_dataset_plan(spec_path: str | Path) -> Path:
    path, section = load_synthetic_spec(spec_path)
    root = _project_root(path)
    output_dir = _resolve(root, str(section["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_dataset_plan(path)
    plan_path = output_dir / "manifest.plan.csv"
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    coverage = validate_dataset_plan(plan, section)
    (output_dir / "plan_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan_path


def _add_noise(path: Path, snr_db: float | None, hum_hz: float, seed: int) -> None:
    if snr_db is None:
        return
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    signal_rms = math.sqrt(sum(value * value for value in samples) / max(len(samples), 1))
    noise_rms = max(signal_rms, 1.0) / (10 ** (snr_db / 20.0))
    source = random.Random(seed)
    for index, value in enumerate(samples):
        hum = noise_rms * 0.45 * math.sin(2 * math.pi * hum_hz * index / params.framerate)
        noisy = round(value + hum + source.gauss(0.0, noise_rms * 0.9))
        samples[index] = max(-32_768, min(32_767, noisy))
    with wave.open(str(path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())


async def _synthesize_one(
    row: dict[str, str],
    output_dir: Path,
    retries: int = 3,
    request_timeout_seconds: float = 30.0,
    retry_base_seconds: float = 2.0,
    retry_max_seconds: float = 30.0,
) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Install synthetic TTS support with: pip install edge-tts") from exc
    audio_path = output_dir / row["audio_path"]
    if audio_path.exists() and audio_path.stat().st_size > 44:
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert synthetic TTS audio to 16 kHz WAV")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = audio_path.with_suffix(".partial.mp3")
    raw_path = audio_path.with_suffix(".raw.wav")
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(
                row["spoken_text"],
                row["tts_voice"],
                rate=row["tts_rate"],
                pitch=row["tts_pitch"],
            )
            # Edge TTS occasionally leaves a websocket open without returning an error.  Bound
            # each request so one stalled sample cannot block all 7,200 synthesis tasks forever.
            await asyncio.wait_for(
                communicate.save(str(mp3_path)),
                timeout=request_timeout_seconds,
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(mp3_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(raw_path),
                ],
                check=True,
            )
            with wave.open(str(raw_path), "rb") as reader:
                raw_duration = reader.getnframes() / reader.getframerate()
            target_duration = float(row["target_duration_seconds"])
            tempo = raw_duration / target_duration
            tempo_parts: list[float] = []
            while tempo > 2.0:
                tempo_parts.append(2.0)
                tempo /= 2.0
            while tempo < 0.5:
                tempo_parts.append(0.5)
                tempo /= 0.5
            tempo_parts.append(tempo)
            audio_filter = ",".join(
                [*(f"atempo={value:.6f}" for value in tempo_parts), "apad"]
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw_path),
                    "-af",
                    audio_filter,
                    "-t",
                    f"{target_duration:.3f}",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_path),
                ],
                check=True,
            )
            mp3_path.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
            snr = row.get("noise_snr_db", "")
            _add_noise(
                audio_path,
                None if not snr else float(snr),
                float(row["noise_hum_hz"]),
                int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8], 16),
            )
            return
        except Exception as exc:
            mp3_path.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            if attempt == retries:
                print(
                    "[synthetic-tts] failed "
                    f"sample_id={row['sample_id']} voice={row['tts_voice']} "
                    f"rate={row['tts_rate']} pitch={row['tts_pitch']} "
                    f"attempts={retries} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            delay = min(retry_max_seconds, retry_base_seconds * (2 ** (attempt - 1)))
            print(
                "[synthetic-tts] retry "
                f"sample_id={row['sample_id']} attempt={attempt}/{retries} "
                f"delay_seconds={delay:g} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            await asyncio.sleep(delay)


async def _synthesize_all(
    plan: pd.DataFrame,
    output_dir: Path,
    concurrency: int,
    request_timeout_seconds: float,
    progress_every: int,
    retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def guarded(row: dict[str, str]) -> None:
        nonlocal completed
        async with semaphore:
            await _synthesize_one(
                row,
                output_dir,
                retries=retries,
                request_timeout_seconds=request_timeout_seconds,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
        async with lock:
            completed += 1
            if completed % progress_every == 0 or completed == len(plan):
                print(f"[synthetic-tts] completed={completed}/{len(plan)}", flush=True)

    await asyncio.gather(*(guarded(row) for row in plan.astype(str).to_dict("records")))


def synthesize_dataset(spec_path: str | Path, concurrency: int = 4) -> Path:
    path, section = load_synthetic_spec(spec_path)
    root = _project_root(path)
    output_dir = _resolve(root, str(section["output_dir"]))
    plan_path = output_dir / "manifest.plan.csv"
    if not plan_path.exists():
        plan_path = write_dataset_plan(path)
    plan = pd.read_csv(plan_path, dtype=str).fillna("")
    validate_dataset_plan(plan, section)
    request_timeout_seconds = float(section.get("request_timeout_seconds", 30.0))
    progress_every = max(1, int(section.get("progress_every", 10)))
    retries = max(1, int(section.get("tts_retries", 3)))
    retry_base_seconds = max(0.0, float(section.get("tts_retry_base_seconds", 2.0)))
    retry_max_seconds = max(
        retry_base_seconds,
        float(section.get("tts_retry_max_seconds", 30.0)),
    )
    print(
        "[synthetic-tts] start "
        f"samples={len(plan)} concurrency={max(1, concurrency)} "
        f"request_timeout_seconds={request_timeout_seconds:g} retries={retries}",
        flush=True,
    )
    asyncio.run(
        _synthesize_all(
            plan,
            output_dir,
            max(1, concurrency),
            request_timeout_seconds,
            progress_every,
            retries,
            retry_base_seconds,
            retry_max_seconds,
        )
    )

    durations: list[float] = []
    digests: list[str] = []
    for relative in plan["audio_path"]:
        audio_path = output_dir / relative
        with wave.open(str(audio_path), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getframerate() != 16_000:
                raise ValueError(f"Unexpected synthetic WAV format: {audio_path}")
            durations.append(reader.getnframes() / reader.getframerate())
        digests.append(hashlib.sha256(audio_path.read_bytes()).hexdigest())
    manifest = plan.copy()
    manifest["audio_duration_seconds"] = [f"{value:.3f}" for value in durations]
    manifest["audio_sha256"] = digests
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    duration_hours = (
        manifest.assign(duration=pd.to_numeric(manifest["audio_duration_seconds"]))
        .groupby("split")["duration"]
        .sum()
        .div(3600)
        .to_dict()
    )
    target_duration_hours = (
        plan.assign(target=pd.to_numeric(plan["target_duration_seconds"]))
        .groupby("split")["target"]
        .sum()
        .div(3600)
        .to_dict()
    )
    for split, target in target_duration_hours.items():
        if abs(float(duration_hours[split]) - float(target)) > 0.02:
            raise ValueError(
                f"Synthetic duration contract failed for {split}: "
                f"target={target:.3f} h actual={duration_hours[split]:.3f} h"
            )
    provenance = {
        "dataset_id": str(section["id"]),
        "dataset_role": str(section.get("dataset_role", "development")),
        "generated_at": utc_now().isoformat(),
        "human_voice_data": False,
        "contains_personal_information": False,
        "generator": "edge-tts with project-authored manufacturing sentences",
        "synthetic_speaker_profiles_are_not_humans": True,
        "sample_count": len(manifest),
        "split_counts": manifest.groupby("split").size().to_dict(),
        "duration_hours": duration_hours,
        "target_duration_hours": target_duration_hours,
        "speaker_profile_count": int(manifest["speaker_id"].nunique()),
        "selection_policy": (
            "frozen-model confirmatory evaluation; no tuning after cohort results"
            if str(section.get("dataset_role", "development")) == "confirmatory_test"
            else "validation-only tuning; held-out test once after all choices"
        ),
        "limitations": [
            "Synthetic voices do not prove human-speaker or factory-noise generalization.",
            "Production readiness requires a separately approved real-world shadow evaluation.",
        ],
    }
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path
