from types import SimpleNamespace

import pandas as pd

from aias_specialist.training import (
    _extract_input_features,
    _gradient_checkpointing_enabled,
    _lora_config_kwargs,
    _prediction_frame,
)


def test_whisper_lora_config_uses_generic_peft_wrapper() -> None:
    config = _lora_config_kwargs({})

    assert "task_type" not in config
    assert config["target_modules"] == ["q_proj", "v_proj"]


def test_whisper_lora_config_applies_overrides() -> None:
    config = _lora_config_kwargs(
        {
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        }
    )

    assert config["r"] == 8
    assert config["lora_alpha"] == 16
    assert config["lora_dropout"] == 0.1


def test_whisper_lora_disables_gradient_checkpointing_by_default() -> None:
    assert _gradient_checkpointing_enabled({}) is False
    assert _gradient_checkpointing_enabled({"gradient_checkpointing": False}) is False
    assert _gradient_checkpointing_enabled({"gradient_checkpointing": True}) is True


def test_whisper_features_request_and_preserve_attention_mask() -> None:
    calls = []

    class FeatureExtractor:
        def __call__(self, audio, **kwargs):  # noqa: ANN001, ANN202
            calls.append((audio, kwargs))
            return SimpleNamespace(input_features=[[1.0, 2.0]], attention_mask=[[1, 0]])

    processor = SimpleNamespace(feature_extractor=FeatureExtractor())
    result = _extract_input_features(processor, [0.1, 0.2], 16_000)

    assert calls == [
        ([0.1, 0.2], {"sampling_rate": 16_000, "return_attention_mask": True})
    ]
    assert result == {"input_features": [1.0, 2.0], "attention_mask": [1, 0]}


def test_prediction_frame_uses_aggregate_batch_timing() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "reference_text": ["안녕", "세계"],
            "audio_duration_seconds": [2.0, 3.0],
        }
    )

    result = _prediction_frame(frame, ["안녕", "세계"], 1.0, "best_lora")

    assert result["model_label"].tolist() == ["best_lora", "best_lora"]
    assert result["latency_seconds"].tolist() == [0.5, 0.5]
    assert result["real_time_factor"].tolist() == [0.2, 0.2]
    assert result["sample_wer"].tolist() == [0.0, 0.0]
