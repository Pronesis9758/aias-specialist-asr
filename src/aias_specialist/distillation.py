from __future__ import annotations

import inspect
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from .config import Settings
from .correction import apply_term_correction, resolved_correction_options
from .data import load_domain_terms, prepare_manifest
from .environment import collect_environment
from .evaluation import compare_metrics, per_sample_metrics
from .models import load_model_lock
from .reporting import build_report, create_metrics_chart
from .store import ExperimentStore, register_run_artifacts
from .training import (
    SpeechSeq2SeqCollator,
    _checkpoint,
    _decode_prediction_text,
    _evaluation_metrics,
    _extract_input_features,
    _path,
    _prediction_frame,
)
from .utils import git_sha, new_run_id, utc_now, write_json


def distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    hard_label_weight: float,
    temperature: float,
    hard_loss: Any | None = None,
) -> Any:
    """Blend supervised token loss with temperature-scaled teacher KL divergence."""
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Knowledge distillation requires PyTorch") from exc

    if not 0.0 <= hard_label_weight <= 1.0:
        raise ValueError("hard_label_weight must be between zero and one")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")

    sequence_length = min(student_logits.shape[1], teacher_logits.shape[1], labels.shape[1])
    vocabulary_size = min(student_logits.shape[2], teacher_logits.shape[2])
    student = student_logits[:, :sequence_length, :vocabulary_size]
    teacher = teacher_logits[:, :sequence_length, :vocabulary_size]
    aligned_labels = labels[:, :sequence_length]

    if hard_loss is None:
        hard_loss = functional.cross_entropy(
            student.reshape(-1, vocabulary_size),
            aligned_labels.reshape(-1),
            ignore_index=-100,
        )

    valid = aligned_labels.ne(-100)
    if valid.any():
        student_distribution = functional.log_softmax(student[valid] / temperature, dim=-1)
        teacher_distribution = functional.softmax(teacher[valid] / temperature, dim=-1)
        soft_loss = functional.kl_div(
            student_distribution,
            teacher_distribution,
            reduction="batchmean",
        ) * (temperature**2)
    else:
        soft_loss = hard_loss.new_zeros(())
    return hard_label_weight * hard_loss + (1.0 - hard_label_weight) * soft_loss


def _write_summary(
    path: Path,
    settings: Settings,
    run_id: str,
    metrics: dict[str, Any],
) -> None:
    baseline = metrics["baseline"]
    distilled = metrics["distilled"]
    corrected = metrics["corrected"]
    text = f"""# Knowledge-distillation run: {run_id}

- Project: {settings.project.name}
- Teacher: {metrics['training']['teacher_repo']}@{metrics['training']['teacher_revision']}
- Student: {metrics['training']['student_repo']}@{metrics['training']['student_revision']}
- Test samples: {baseline['sample_count']}
- Student baseline WER: {baseline['wer']:.4f}
- Distilled student WER: {distilled['wer']:.4f}
- Distilled + optional retrieval correction WER: {corrected['wer']:.4f}
- Distillation WER absolute reduction:
  {metrics['distilled_improvement']['wer_absolute_reduction']:.4f}
- Combined WER absolute reduction: {metrics['improvement']['wer_absolute_reduction']:.4f}
- Human review required: transcript labels, teacher/student trade-off, retrieval substitutions,
  privacy approval, and final deployment conclusion
"""
    path.write_text(text, encoding="utf-8")


def train_whisper_distillation(settings: Settings) -> Path:
    values = settings.distillation.values or {}
    if not settings.distillation.enabled:
        raise ValueError("distillation.enabled is false in the selected config")
    required = {"teacher_repo_id", "student_repo_id", "output_dir"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Distillation config is missing: {', '.join(missing)}")

    try:
        import librosa
        import torch
        from datasets import Dataset
        from jiwer import wer
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: uv sync --extra train") from exc

    if not torch.cuda.is_available() and not bool(values.get("allow_cpu", False)):
        raise RuntimeError(
            "Whisper distillation requires a CUDA GPU by default. Use Colab GPU or explicitly "
            "set distillation.allow_cpu=true for a small diagnostic run."
        )

    hard_label_weight = float(values.get("hard_label_weight", 0.5))
    temperature = float(values.get("temperature", 2.0))
    run_id = f"distill-{new_run_id()}"
    run_dir = settings.paths.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    distilled_dir = run_dir / "checkpoints" / "distilled_student"
    store = ExperimentStore(settings.paths.database)
    teacher_revision: str | None = None
    student_revision: str | None = None
    teacher_repo = str(values["teacher_repo_id"])
    student_repo = str(values["student_repo_id"])

    store.start_run(
        {
            "run_id": run_id,
            "started_at": utc_now().isoformat(),
            "project_name": settings.project.name,
            "config_path": str(settings.config_path),
            "backend": "transformers_whisper_distillation",
            "model_repo": student_repo,
            "git_sha": git_sha(settings.project_root),
            "run_dir": str(run_dir),
        }
    )

    try:
        (run_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(settings.raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        write_json(run_dir / "environment.json", collect_environment(settings.project_root))
        prepared = prepare_manifest(
            settings.paths.manifest,
            run_dir / "prepared_manifest.csv",
            backend="faster_whisper",
            governance=settings.governance,
        )
        required_splits = {"train", "validation", "test"}
        if not required_splits.issubset(set(prepared["split"].str.lower())):
            raise ValueError("Distillation requires train, validation, and test manifest splits")
        terms = load_domain_terms(settings.paths.domain_terms)
        terms.to_csv(run_dir / "domain_terms.snapshot.csv", index=False, encoding="utf-8-sig")

        lock = load_model_lock(settings)
        locked_models = lock.get("models", {})
        teacher_lock = locked_models.get(teacher_repo)
        student_lock = locked_models.get(student_repo)
        if not isinstance(teacher_lock, dict) or not isinstance(student_lock, dict):
            raise ValueError("Teacher and student must be pinned. Run 'aias model-lock' first.")
        teacher_revision = str(teacher_lock["resolved_revision"])
        student_revision = str(student_lock["resolved_revision"])

        processor = WhisperProcessor.from_pretrained(
            student_repo,
            revision=student_revision,
            language=settings.model.language,
            task="transcribe",
        )
        teacher = WhisperForConditionalGeneration.from_pretrained(
            teacher_repo, revision=teacher_revision
        )
        student = WhisperForConditionalGeneration.from_pretrained(
            student_repo, revision=student_revision
        )
        for model in [teacher, student]:
            model.generation_config.language = settings.model.language
            model.generation_config.task = "transcribe"
            model.generation_config.forced_decoder_ids = None
            model.config.forced_decoder_ids = None
            model.config.use_cache = False
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

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
            decoder_start_token_id=student.config.decoder_start_token_id,
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

        output_root = _path(settings.project_root, str(values["output_dir"]))
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        argument_values: dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(values.get("train_batch_size", 2)),
            "per_device_eval_batch_size": int(values.get("eval_batch_size", 2)),
            "gradient_accumulation_steps": int(values.get("gradient_accumulation_steps", 4)),
            "learning_rate": float(values.get("learning_rate", 5e-5)),
            "max_steps": int(values.get("max_steps", 100)),
            "warmup_steps": int(values.get("warmup_steps", 10)),
            "fp16": torch.cuda.is_available(),
            "predict_with_generate": True,
            "generation_max_length": 225,
            "save_steps": int(values.get("save_steps", 25)),
            "eval_steps": int(values.get("eval_steps", 25)),
            "logging_steps": int(values.get("logging_steps", 5)),
            "save_total_limit": int(values.get("save_total_limit", 2)),
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "report_to": [],
            "remove_unused_columns": False,
            "label_names": ["labels"],
            "seed": settings.project.seed,
            "data_seed": settings.project.seed,
        }
        argument_signature = inspect.signature(Seq2SeqTrainingArguments)
        evaluation_key = (
            "eval_strategy"
            if "eval_strategy" in argument_signature.parameters
            else "evaluation_strategy"
        )
        argument_values[evaluation_key] = "steps"
        argument_values["save_strategy"] = "steps"
        arguments = Seq2SeqTrainingArguments(**argument_values)

        class WhisperDistillationTrainer(Seq2SeqTrainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any | None = None,
            ) -> Any:
                del num_items_in_batch
                outputs = model(**inputs)
                teacher_inputs = {
                    key: value
                    for key, value in inputs.items()
                    if key in {"input_features", "attention_mask", "labels"}
                }
                with torch.no_grad():
                    teacher_outputs = teacher(**teacher_inputs)
                loss = distillation_loss(
                    outputs.logits,
                    teacher_outputs.logits,
                    inputs["labels"],
                    hard_label_weight=hard_label_weight,
                    temperature=temperature,
                    hard_loss=outputs.loss,
                )
                return (loss, outputs) if return_outputs else loss

        trainer_values: dict[str, Any] = {
            "model": student,
            "args": arguments,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": collator,
            "compute_metrics": compute_metrics,
        }
        trainer_key = (
            "processing_class"
            if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
            else "tokenizer"
        )
        trainer_values[trainer_key] = processor
        trainer = WhisperDistillationTrainer(**trainer_values)
        teacher.to(trainer.args.device)

        baseline_started = time.perf_counter()
        baseline_output = trainer.predict(test_dataset, metric_key_prefix="student_base_test")
        baseline_runtime = time.perf_counter() - baseline_started
        baseline_predictions = _prediction_frame(
            test_frame,
            _decode_prediction_text(baseline_output, processor),
            baseline_runtime,
            "student_before_distillation",
        )
        baseline_predictions.to_csv(
            run_dir / "predictions_baseline.csv", index=False, encoding="utf-8-sig"
        )
        baseline_metrics = _evaluation_metrics(baseline_predictions, terms)

        resume = _checkpoint(output_dir, str(values.get("resume_from_checkpoint", "auto")))
        store.event(run_id, "distillation", "started", f"resume={resume or 'none'}")
        train_result = trainer.train(resume_from_checkpoint=resume)
        eval_metrics = trainer.evaluate()
        trainer.save_model(distilled_dir)
        processor.save_pretrained(distilled_dir)
        trainer.model.config.use_cache = True
        trainer.model.eval()

        distilled_started = time.perf_counter()
        distilled_output = trainer.predict(test_dataset, metric_key_prefix="distilled_test")
        distilled_runtime = time.perf_counter() - distilled_started
        distilled_predictions = _prediction_frame(
            test_frame,
            _decode_prediction_text(distilled_output, processor),
            distilled_runtime,
            "distilled_student",
        )
        distilled_predictions.to_csv(
            run_dir / "predictions_distilled.csv", index=False, encoding="utf-8-sig"
        )
        distilled_metrics = _evaluation_metrics(distilled_predictions, terms)

        corrected_predictions = apply_term_correction(
            distilled_predictions,
            terms,
            **resolved_correction_options(settings),
        )
        corrected_predictions = per_sample_metrics(corrected_predictions)
        corrected_predictions.to_csv(
            run_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
        )
        corrected_metrics = _evaluation_metrics(corrected_predictions, terms)

        training_metrics = {
            "teacher_repo": teacher_repo,
            "teacher_revision": teacher_revision,
            "student_repo": student_repo,
            "student_revision": student_revision,
            "distilled_student_dir": str(distilled_dir),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "hard_label_weight": hard_label_weight,
            "temperature": temperature,
            "train_samples": len(train_dataset),
            "validation_samples": len(eval_dataset),
            "test_samples": len(test_dataset),
            "max_steps": int(values.get("max_steps", 100)),
            "best_validation_wer": float(eval_metrics.get("eval_wer", 0.0)),
            "seed": settings.project.seed,
            **{f"train_{key}": value for key, value in train_result.metrics.items()},
            **{key: value for key, value in eval_metrics.items()},
        }
        metrics = {
            "training": training_metrics,
            "baseline": baseline_metrics,
            "distilled": distilled_metrics,
            "distilled_improvement": compare_metrics(baseline_metrics, distilled_metrics),
            "corrected": corrected_metrics,
            "correction_improvement": compare_metrics(distilled_metrics, corrected_metrics),
            "improvement": compare_metrics(baseline_metrics, corrected_metrics),
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "training_metrics.json", training_metrics)
        for scope, values_for_scope in metrics.items():
            store.add_metrics(run_id, scope, values_for_scope)

        chart_path = None
        if settings.report.include_charts:
            chart_path = create_metrics_chart(metrics, reports_dir / "metrics_comparison.png")
        build_report(
            reports_dir / "evaluation_report.docx",
            settings.report.title,
            run_id,
            settings.project.owner,
            "transformers_whisper_distillation",
            student_repo,
            student_revision,
            metrics,
            chart_path,
        )
        _write_summary(run_dir / "run_summary.md", settings, run_id, metrics)
        store.event(run_id, "distillation", "completed")
        register_run_artifacts(store, run_id, run_dir)
        store.finish_run(run_id, "completed", model_revision=student_revision)
        return run_dir
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        store.event(run_id, "distillation", "failed", str(exc))
        register_run_artifacts(store, run_id, run_dir)
        store.finish_run(run_id, "failed", model_revision=student_revision, error=str(exc))
        raise
