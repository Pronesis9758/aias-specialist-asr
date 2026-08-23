from types import SimpleNamespace

import pandas as pd
import pytest

from aias_specialist.distillation import distillation_loss
from aias_specialist.training import (
    _extract_input_features,
    _gradient_checkpointing_enabled,
    _lora_config_kwargs,
    _match_input_features_dtype,
    _model_load_kwargs,
    _prediction_frame,
    _preprocess_num_proc,
)


def test_whisper_collator_matches_input_features_to_model_dtype() -> None:
    class TensorStub:
        def __init__(self) -> None:
            self.requested_dtype = None

        def to(self, *, dtype):  # noqa: ANN001, ANN202
            self.requested_dtype = dtype
            return self

    features = TensorStub()
    batch = {"input_features": features, "attention_mask": "unchanged"}

    result = _match_input_features_dtype(batch, "float16")

    assert result is batch
    assert result["input_features"] is features
    assert features.requested_dtype == "float16"
    assert result["attention_mask"] == "unchanged"


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


def test_whisper_preprocessing_worker_count_is_positive() -> None:
    assert _preprocess_num_proc({}) == 1
    assert _preprocess_num_proc({"preprocess_num_proc": 4}) == 4

    with pytest.raises(ValueError, match="preprocess_num_proc"):
        _preprocess_num_proc({"preprocess_num_proc": 0})


def test_whisper_lora_uses_low_memory_float16_loading_by_default() -> None:
    torch_stub = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32")

    assert _model_load_kwargs({}, torch_stub) == {
        "low_cpu_mem_usage": True,
        "torch_dtype": "fp16",
    }
    assert _model_load_kwargs(
        {"low_cpu_mem_usage": False, "load_dtype": "auto"}, torch_stub
    ) == {"low_cpu_mem_usage": False}


def test_whisper_lora_rejects_unknown_load_dtype() -> None:
    torch_stub = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32")

    with pytest.raises(ValueError, match="Unsupported training.load_dtype"):
        _model_load_kwargs({"load_dtype": "int8"}, torch_stub)


def test_whisper_features_request_and_preserve_attention_mask() -> None:
    calls = []

    class FeatureExtractor:
        def __call__(self, audio, **kwargs):  # noqa: ANN001, ANN202
            calls.append((audio, kwargs))
            return SimpleNamespace(input_features=[[1.0, 2.0]], attention_mask=[[1, 0]])

    processor = SimpleNamespace(feature_extractor=FeatureExtractor())
    result = _extract_input_features(processor, [0.1, 0.2], 16_000)

    assert calls == [([0.1, 0.2], {"sampling_rate": 16_000, "return_attention_mask": True})]
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


def test_distillation_loss_supports_independent_hard_and_soft_weights() -> None:
    torch = pytest.importorskip("torch")
    student_logits = torch.tensor([[[2.0, 0.1], [0.2, 1.2]]], requires_grad=True)
    teacher_logits = torch.tensor([[[3.0, 0.1], [0.1, 2.0]]])
    labels = torch.tensor([[0, 1]])

    hard_only = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        hard_label_weight=1.0,
        temperature=2.0,
    )
    blended = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        hard_label_weight=0.5,
        temperature=2.0,
    )

    assert hard_only.item() > 0.0
    assert blended.item() > 0.0
    blended.backward()
    assert student_logits.grad is not None


def test_distillation_loss_validates_options() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, 2))
    labels = torch.zeros((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="hard_label_weight"):
        distillation_loss(
            logits,
            logits,
            labels,
            hard_label_weight=1.5,
            temperature=2.0,
        )
