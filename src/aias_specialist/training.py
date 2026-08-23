from __future__ import annotations

import inspect
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Settings
from .correction import apply_term_correction, resolved_correction_options
from .correction_audit import write_correction_audit
from .data import load_domain_terms, prepare_manifest
from .environment import collect_environment
from .evaluation import compare_metrics, evaluate_predictions, per_sample_metrics
from .models import load_model_lock
from .reporting import build_report, create_metrics_chart
from .store import ExperimentStore, register_run_artifacts
from .utils import git_sha, new_run_id, utc_now, write_json


@dataclass
class SpeechSeq2SeqCollator:
    processor: Any
    decoder_start_token_id: int
    input_features_dtype: Any | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        feature_batch = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(feature_batch, return_tensors="pt")
        # Whisper's first convolution requires its input features and parameters to
        # share a dtype.  A low-memory float16 model load does not automatically cast
        # the float32 log-Mel features on every Transformers/Accelerate combination
        # used by Colab, so make that contract explicit before Trainer moves the batch
        # to the GPU.
        batch = _match_input_features_dtype(batch, self.input_features_dtype)
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _match_input_features_dtype(
    batch: dict[str, Any], input_features_dtype: Any | None
) -> dict[str, Any]:
    """Cast Whisper log-Mel features to the loaded model's parameter dtype."""
    if input_features_dtype is not None:
        batch["input_features"] = batch["input_features"].to(dtype=input_features_dtype)
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


def _lora_config_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    """Build a Whisper-compatible LoRA configuration.

    Whisper consumes ``input_features`` instead of the text-model ``input_ids``
    expected by PEFT's task-specific sequence-to-sequence wrapper. Leaving
    ``task_type`` unset selects the generic PEFT wrapper and preserves Whisper's
    forward signature.
    """
    return {
        "r": int(values.get("lora_rank", 16)),
        "lora_alpha": int(values.get("lora_alpha", 32)),
        "lora_dropout": float(values.get("lora_dropout", 0.05)),
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }


def _gradient_checkpointing_enabled(values: dict[str, Any]) -> bool:
    """Return the explicit checkpointing choice, disabled by default for Whisper LoRA.

    PEFT freezes the base Whisper feature path. With checkpointing enabled, PyTorch can
    receive no gradient-bearing checkpoint inputs and drop the graph before LoRA's backward
    pass. The small Colab validation models fit on a T4 without this memory optimization.
    """
    return bool(values.get("gradient_checkpointing", False))


def _model_load_kwargs(values: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    """Build memory-conscious Transformers loading options for Colab GPU training.

    Loading a large Whisper checkpoint in float32 can temporarily consume more than the
    standard Colab system-RAM allowance before the Trainer moves it to the GPU.  LoRA does
    not need float32 base weights, so the default is a single-copy, low-memory float16 load.
    The options remain explicit in the run snapshot and can be overridden for other hardware.
    """
    options: dict[str, Any] = {
        "low_cpu_mem_usage": bool(values.get("low_cpu_mem_usage", True)),
    }
    dtype_name = str(values.get("load_dtype", "float16")).strip().lower()
    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if dtype_name == "auto":
        return options
    if dtype_name not in dtype_map:
        supported = "auto, float16, bfloat16, float32"
        raise ValueError(f"Unsupported training.load_dtype={dtype_name!r}; use one of: {supported}")
    options["torch_dtype"] = dtype_map[dtype_name]
    return options


def _decode_prediction_text(prediction_output: Any, processor: Any) -> list[str]:
    prediction_ids = prediction_output.predictions
    if isinstance(prediction_ids, tuple):
        prediction_ids = prediction_ids[0]
    return list(processor.batch_decode(prediction_ids, skip_special_tokens=True))


def _extract_input_features(processor: Any, audio: Any, sampling_rate: int) -> dict[str, Any]:
    features = processor.feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
    )
    return {
        "input_features": features.input_features[0],
        "attention_mask": features.attention_mask[0],
    }


def _prediction_frame(
    test_frame: pd.DataFrame,
    prediction_texts: list[str],
    runtime_seconds: float,
    model_label: str,
) -> pd.DataFrame:
    if len(test_frame) != len(prediction_texts):
        raise ValueError(
            "Prediction count does not match the fixed test split: "
            f"{len(prediction_texts)} != {len(test_frame)}"
        )
    output = test_frame.copy().reset_index(drop=True)
    total_audio_seconds = float(output["audio_duration_seconds"].sum())
    sample_count = max(len(output), 1)
    output["prediction_text"] = prediction_texts
    output["model_label"] = model_label
    output["latency_seconds"] = runtime_seconds / sample_count
    output["real_time_factor"] = (
        runtime_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0
    )
    return per_sample_metrics(output)


def _evaluation_metrics(frame: pd.DataFrame, terms: pd.DataFrame) -> dict[str, Any]:
    metrics = evaluate_predictions(frame, terms)
    runtime_seconds = float(frame["latency_seconds"].sum())
    audio_seconds = float(frame["audio_duration_seconds"].sum())
    metrics.update(
        {
            "evaluation_runtime_seconds": runtime_seconds,
            "audio_duration_seconds": audio_seconds,
            "samples_per_second": len(frame) / runtime_seconds if runtime_seconds > 0 else 0.0,
            "aggregate_real_time_factor": (
                runtime_seconds / audio_seconds if audio_seconds > 0 else 0.0
            ),
        }
    )
    return metrics


def _write_training_summary(
    path: Path, settings: Settings, run_id: str, metrics: dict[str, Any]
) -> None:
    baseline = metrics["baseline"]
    lora = metrics["lora"]
    improvement = metrics["lora_improvement"]
    text = f"""# Training run summary: {run_id}

- Project: {settings.project.name}
- Training backend: Transformers Whisper + PEFT LoRA
- Model: {metrics["training"]["model_repo"]}@{metrics["training"]["model_revision"]}
- Train samples: {metrics["training"]["train_samples"]}
- Validation samples: {metrics["training"]["validation_samples"]}
- Test samples: {baseline["sample_count"]}
- Base Whisper test WER: {baseline["wer"]:.4f}
- Best LoRA test WER: {lora["wer"]:.4f}
- LoRA WER absolute reduction: {improvement["wer_absolute_reduction"]:.4f}
- Base Whisper test CER: {baseline["cer"]:.4f}
- Best LoRA test CER: {lora["cer"]:.4f}
- Evidence confidence: low (small public general-Korean test split; single seed)
- Human review required: transcript labels, privacy approval, domain terms, model trade-offs,
  and final report conclusions
"""
    path.write_text(text, encoding="utf-8")


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
        from peft import LoraConfig, get_peft_model
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
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = run_dir / "checkpoints" / "best_adapter"
    store = ExperimentStore(settings.paths.database)
    revision: str | None = None
    train_repo_id = str(values["repo_id"])

    store.start_run(
        {
            "run_id": run_id,
            "started_at": utc_now().isoformat(),
            "project_name": settings.project.name,
            "config_path": str(settings.config_path),
            "backend": "transformers_whisper_lora",
            "model_repo": train_repo_id,
            "git_sha": git_sha(settings.project_root),
            "run_dir": str(run_dir),
        }
    )

    try:
        store.event(run_id, "snapshot", "started")
        (run_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(settings.raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        write_json(run_dir / "environment.json", collect_environment(settings.project_root))
        store.event(run_id, "snapshot", "completed")

        store.event(run_id, "prepare", "started")
        prepared = prepare_manifest(
            settings.paths.manifest,
            run_dir / "prepared_manifest.csv",
            backend="faster_whisper",
            governance=settings.governance,
        )
        splits = set(prepared["split"].str.lower())
        if not {"train", "validation", "test"}.issubset(splits):
            raise ValueError(
                "Training manifest must contain train, validation, and test splits for comparison"
            )
        terms = load_domain_terms(settings.paths.domain_terms)
        terms.to_csv(run_dir / "domain_terms.snapshot.csv", index=False, encoding="utf-8-sig")
        store.event(run_id, "prepare", "completed", f"samples={len(prepared)}")

        lock = load_model_lock(settings)
        train_lock = lock.get("models", {}).get(train_repo_id)
        if not train_lock:
            raise ValueError("Training model is not locked. Run 'aias model-lock' first.")
        repo_id = str(train_lock["repo_id"])
        revision = str(train_lock["resolved_revision"])
        checkpoint_root = _path(settings.project_root, str(values["output_dir"]))
        output_dir = checkpoint_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)

        processor = WhisperProcessor.from_pretrained(
            repo_id,
            revision=revision,
            language=settings.model.language,
            task="transcribe",
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            repo_id,
            revision=revision,
            **_model_load_kwargs(values, torch),
        )
        model.generation_config.language = settings.model.language
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        model.config.use_cache = False

        lora = LoraConfig(**_lora_config_kwargs(values))
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        dataset_frame = prepared[
            ["sample_id", "audio_path", "reference_text", "split", "source", "consent_status"]
        ].rename(columns={"audio_path": "audio", "reference_text": "sentence"})
        dataset_frame["audio_duration_seconds"] = [
            float(librosa.get_duration(path=path)) for path in dataset_frame["audio"]
        ]
        test_frame = dataset_frame.loc[dataset_frame["split"].str.lower() == "test"].rename(
            columns={"audio": "audio_path", "sentence": "reference_text"}
        )
        dataset = Dataset.from_pandas(dataset_frame, preserve_index=False)

        def preprocess(record: dict[str, Any]) -> dict[str, Any]:
            audio, sampling_rate = librosa.load(record["audio"], sr=16_000, mono=True)
            record.update(_extract_input_features(processor, audio, sampling_rate))
            record["labels"] = processor.tokenizer(record["sentence"]).input_ids
            return record

        dataset = dataset.map(preprocess, remove_columns=["audio", "sentence"])
        train_dataset = dataset.filter(lambda row: row["split"].lower() == "train").remove_columns(
            ["split"]
        )
        eval_dataset = dataset.filter(
            lambda row: row["split"].lower() == "validation"
        ).remove_columns(["split"])
        test_dataset = dataset.filter(lambda row: row["split"].lower() == "test").remove_columns(
            ["split"]
        )

        collator = SpeechSeq2SeqCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
            input_features_dtype=next(model.parameters()).dtype,
        )

        def compute_metrics(prediction: Any) -> dict[str, float]:
            prediction_ids = prediction.predictions
            if isinstance(prediction_ids, tuple):
                prediction_ids = prediction_ids[0]
            label_ids = prediction.label_ids.copy()
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
            "gradient_checkpointing": _gradient_checkpointing_enabled(values),
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
            "seed": settings.project.seed,
            "data_seed": settings.project.seed,
        }
        signature = inspect.signature(Seq2SeqTrainingArguments)
        evaluation_key = (
            "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
        )
        argument_values[evaluation_key] = "steps"
        argument_values["save_strategy"] = "steps"
        arguments = Seq2SeqTrainingArguments(**argument_values)

        trainer_values: dict[str, Any] = {
            "model": model,
            "args": arguments,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": collator,
            "compute_metrics": compute_metrics,
        }
        # ``processing_class`` replaced the deprecated ``tokenizer`` keyword in
        # newer Transformers releases.  Keep the runner compatible with the
        # older 4.x builds commonly preinstalled in Colab runtimes.
        trainer_key = (
            "processing_class"
            if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
            else "tokenizer"
        )
        trainer_values[trainer_key] = processor
        trainer = Seq2SeqTrainer(**trainer_values)
        resume = _checkpoint(output_dir, str(values.get("resume_from_checkpoint", "auto")))
        store.event(run_id, "training", "started", f"resume={resume or 'none'}")
        train_result = trainer.train(resume_from_checkpoint=resume)
        eval_metrics = trainer.evaluate()
        store.event(
            run_id,
            "training",
            "completed",
            f"best_checkpoint={trainer.state.best_model_checkpoint}",
        )

        trainer.save_model(adapter_dir)
        processor.save_pretrained(adapter_dir)
        trainer.model.config.use_cache = True
        trainer.model.eval()
        torch.cuda.empty_cache()

        store.event(run_id, "test_comparison", "started", f"test_samples={len(test_dataset)}")
        baseline_started = time.perf_counter()
        with trainer.model.disable_adapter():
            baseline_output = trainer.predict(test_dataset, metric_key_prefix="base_test")
        baseline_runtime = time.perf_counter() - baseline_started
        baseline_predictions = _prediction_frame(
            test_frame,
            _decode_prediction_text(baseline_output, processor),
            baseline_runtime,
            "base_whisper",
        )
        baseline_predictions.to_csv(
            run_dir / "predictions_baseline.csv", index=False, encoding="utf-8-sig"
        )
        baseline_metrics = _evaluation_metrics(baseline_predictions, terms)

        lora_started = time.perf_counter()
        lora_output = trainer.predict(test_dataset, metric_key_prefix="lora_test")
        lora_runtime = time.perf_counter() - lora_started
        lora_predictions = _prediction_frame(
            test_frame,
            _decode_prediction_text(lora_output, processor),
            lora_runtime,
            "best_lora",
        )
        lora_predictions.to_csv(run_dir / "predictions_lora.csv", index=False, encoding="utf-8-sig")
        lora_metrics = _evaluation_metrics(lora_predictions, terms)

        corrected_predictions = apply_term_correction(
            baseline_predictions,
            terms,
            **resolved_correction_options(settings),
        )
        corrected_predictions = per_sample_metrics(corrected_predictions)
        corrected_predictions.to_csv(
            run_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
        )
        write_correction_audit(run_dir, baseline_predictions, corrected_predictions)
        corrected_metrics = _evaluation_metrics(corrected_predictions, terms)
        store.event(run_id, "test_comparison", "completed")

        dataset_config = settings.raw.get("dataset", {})
        training_metrics = {
            "model_repo": repo_id,
            "model_revision": revision,
            "adapter_dir": str(adapter_dir),
            "trainer_checkpoint_dir": str(output_dir),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_wer": float(eval_metrics.get("eval_wer", 0.0)),
            "train_samples": len(train_dataset),
            "validation_samples": len(eval_dataset),
            "test_samples": len(test_dataset),
            "max_steps": int(values.get("max_steps", 500)),
            "seed": settings.project.seed,
            **{f"train_{key}": value for key, value in train_result.metrics.items()},
            **{key: value for key, value in eval_metrics.items()},
        }
        metrics = {
            "training": training_metrics,
            "baseline": baseline_metrics,
            "corrected": corrected_metrics,
            "improvement": compare_metrics(baseline_metrics, corrected_metrics),
            "lora": lora_metrics,
            "lora_improvement": compare_metrics(baseline_metrics, lora_metrics),
            "evidence": {
                "dataset": f"{dataset_config.get('repo_id', 'unknown')}@"
                f"{dataset_config.get('revision', 'unknown')}",
                "comparison_population": "identical fixed test split",
                "confidence": "low",
                "reason": "small public general-Korean sample and one training seed",
            },
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "training_metrics.json", training_metrics)
        pd.DataFrame([training_metrics]).to_json(
            run_dir / "training_summary.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        metric_scopes = [
            "training",
            "baseline",
            "corrected",
            "improvement",
            "lora",
            "lora_improvement",
        ]
        for scope in metric_scopes:
            store.add_metrics(run_id, scope, metrics[scope])

        store.event(run_id, "report", "started")
        chart_path = None
        if settings.report.include_charts:
            chart_path = create_metrics_chart(metrics, reports_dir / "metrics_comparison.png")
        build_report(
            reports_dir / "evaluation_report.docx",
            settings.report.title,
            run_id,
            settings.project.owner,
            "transformers_whisper_lora",
            repo_id,
            revision,
            metrics,
            chart_path,
        )
        _write_training_summary(run_dir / "run_summary.md", settings, run_id, metrics)
        store.event(run_id, "report", "completed")

        register_run_artifacts(store, run_id, run_dir)
        store.finish_run(run_id, "completed", model_revision=revision)
        return run_dir
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        store.event(run_id, "training_pipeline", "failed", str(exc))
        register_run_artifacts(store, run_id, run_dir)
        store.finish_run(run_id, "failed", model_revision=revision, error=str(exc))
        raise
