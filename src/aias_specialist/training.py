from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .data import prepare_manifest
from .models import load_model_lock
from .utils import new_run_id, write_json


@dataclass
class SpeechSeq2SeqCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        feature_batch = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(feature_batch, return_tensors="pt")
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _checkpoint(output_dir: Path, requested: str | None) -> str | bool | None:
    if not requested or requested.lower() in {"false", "none"}:
        return None
    if requested != "auto":
        return requested
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda item: int(item.name.rsplit("-", 1)[-1]),
    )
    return str(candidates[-1]) if candidates else None


def train_whisper_lora(settings: Settings) -> Path:
    if not settings.training.enabled:
        raise ValueError("training.enabled is false in the selected config")
    values = settings.training.values or {}
    required = {"repo_id", "output_dir"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Training config is missing: {', '.join(missing)}")

    try:
        import librosa
        import torch
        from datasets import Dataset
        from jiwer import wer
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: uv sync --extra train") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Whisper LoRA training requires a CUDA GPU. Run this command in the Colab GPU runtime."
        )

    run_id = f"train-{new_run_id()}"
    run_dir = settings.paths.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prepared = prepare_manifest(
        settings.paths.manifest,
        run_dir / "prepared_manifest.csv",
        backend="faster_whisper",
    )
    splits = set(prepared["split"].str.lower())
    if not {"train", "validation"}.issubset(splits):
        raise ValueError("Training manifest must contain both train and validation splits")

    lock = load_model_lock(settings)
    train_repo_id = str(values["repo_id"])
    train_lock = lock.get("models", {}).get(train_repo_id)
    if not train_lock:
        raise ValueError("Training model is not locked. Run 'aias model-lock' first.")
    repo_id = str(train_lock["repo_id"])
    revision = str(train_lock["resolved_revision"])
    output_dir = _path(settings.project_root, str(values["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(
        repo_id,
        revision=revision,
        language=settings.model.language,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(repo_id, revision=revision)
    model.generation_config.language = settings.model.language
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora = LoraConfig(
        r=int(values.get("lora_rank", 16)),
        lora_alpha=int(values.get("lora_alpha", 32)),
        lora_dropout=float(values.get("lora_dropout", 0.05)),
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset_frame = prepared[["audio_path", "reference_text", "split"]].rename(
        columns={"audio_path": "audio", "reference_text": "sentence"}
    )
    dataset = Dataset.from_pandas(dataset_frame, preserve_index=False)

    def preprocess(record: dict[str, Any]) -> dict[str, Any]:
        audio, sampling_rate = librosa.load(record["audio"], sr=16_000, mono=True)
        record["input_features"] = processor.feature_extractor(
            audio, sampling_rate=sampling_rate
        ).input_features[0]
        record["labels"] = processor.tokenizer(record["sentence"]).input_ids
        return record

    dataset = dataset.map(preprocess, remove_columns=["audio", "sentence"])
    train_dataset = dataset.filter(lambda row: row["split"].lower() == "train").remove_columns(
        ["split"]
    )
    eval_dataset = dataset.filter(lambda row: row["split"].lower() == "validation").remove_columns(
        ["split"]
    )

    collator = SpeechSeq2SeqCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        prediction_text = processor.batch_decode(prediction_ids, skip_special_tokens=True)
        reference_text = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": float(wer(reference_text, prediction_text))}

    argument_values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(values.get("train_batch_size", 4)),
        "per_device_eval_batch_size": int(values.get("eval_batch_size", 4)),
        "gradient_accumulation_steps": int(values.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(values.get("learning_rate", 1e-4)),
        "max_steps": int(values.get("max_steps", 500)),
        "warmup_steps": int(values.get("warmup_steps", 50)),
        "gradient_checkpointing": True,
        "fp16": True,
        "predict_with_generate": True,
        "generation_max_length": 225,
        "save_steps": int(values.get("save_steps", 50)),
        "eval_steps": int(values.get("eval_steps", 50)),
        "logging_steps": int(values.get("logging_steps", 10)),
        "save_total_limit": int(values.get("save_total_limit", 3)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    )
    argument_values[evaluation_key] = "steps"
    argument_values["save_strategy"] = "steps"
    arguments = Seq2SeqTrainingArguments(**argument_values)

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    resume = _checkpoint(output_dir, str(values.get("resume_from_checkpoint", "auto")))
    train_result = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = trainer.evaluate()
    adapter_dir = output_dir / "best_adapter"
    trainer.save_model(adapter_dir)
    processor.save_pretrained(adapter_dir)

    payload = {
        "run_id": run_id,
        "model_repo": repo_id,
        "model_revision": revision,
        "adapter_dir": str(adapter_dir),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "train_samples": len(train_dataset),
        "validation_samples": len(eval_dataset),
    }
    write_json(run_dir / "training_metrics.json", payload)
    pd.DataFrame([payload]).to_json(
        run_dir / "training_summary.jsonl", orient="records", lines=True, force_ascii=False
    )
    return run_dir
