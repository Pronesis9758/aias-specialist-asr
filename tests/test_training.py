from aias_specialist.training import _lora_config_kwargs


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
