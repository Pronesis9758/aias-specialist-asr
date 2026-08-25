from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_settings
from .confirmatory import run_confirmatory_evaluation
from .correction_sweep import run_correction_sweep, select_correction_candidate
from .decoding_experiments import run_selected_lora_decoding_sweep
from .distillation import train_whisper_distillation
from .environment import doctor as doctor_check
from .error_mining import mine_domain_term_errors
from .experiments import (
    lock_model_matrix,
    run_final_evaluation,
    run_model_benchmark,
    run_quantization_sweep,
    select_experiment_member,
    train_selected_whisper_lora,
)
from .hf_data import prepare_hf_dataset
from .lora_experiments import run_lora_learning_curve
from .lora_merge import merge_selected_lora
from .models import download_baseline_model, resolve_model_lock
from .ondevice import select_deployment_profiles
from .pipeline import run_pipeline
from .readiness import write_assessment_readiness
from .store import ExperimentStore
from .synthetic_program import synthesize_dataset, write_dataset_plan
from .training import train_whisper_lora

app = typer.Typer(
    no_args_is_help=True,
    help="AI Specialist manufacturing ASR experiment automation",
)


@app.command()
def doctor(
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True, dir_okay=False),
) -> None:
    """Inspect runtime, data, Git, Hugging Face, and GPU readiness."""
    settings = load_settings(config)
    typer.echo(json.dumps(doctor_check(settings), ensure_ascii=False, indent=2))


@app.command("model-lock")
def model_lock(
    config: Path = typer.Option(Path("configs/local_baseline.yaml"), exists=True, dir_okay=False),
) -> None:
    """Resolve mutable Hugging Face revisions to immutable commit SHAs."""
    settings = load_settings(config)
    payload = resolve_model_lock(settings)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    typer.echo(f"Saved: {settings.paths.model_lock}")


@app.command("download-model")
def download_model(
    config: Path = typer.Option(Path("configs/local_baseline.yaml"), exists=True, dir_okay=False),
) -> None:
    """Download the locked baseline model from Hugging Face."""
    settings = load_settings(config)
    local_dir, revision = download_baseline_model(settings)
    typer.echo(f"Downloaded {settings.model.repo_id}@{revision} to {local_dir}")


@app.command("prepare-hf-dataset")
def prepare_hf(
    config: Path = typer.Option(
        Path("configs/colab_public_sample.yaml"), exists=True, dir_okay=False
    ),
    force: bool = typer.Option(False, help="Rebuild an existing public sample manifest."),
) -> None:
    """Stream a pinned public Hugging Face speech sample into the configured manifest."""
    settings = load_settings(config)
    manifest, details = prepare_hf_dataset(settings, force=force)
    typer.echo(json.dumps(details, ensure_ascii=False, indent=2))
    typer.echo(f"Manifest: {manifest}")


@app.command("plan-synthetic-dataset")
def plan_synthetic_dataset(
    spec: Path = typer.Option(
        Path("configs/data/synthetic_manufacturing_7200.yaml"),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Create and validate the deterministic 7,200-item synthetic data plan."""
    path = write_dataset_plan(spec)
    typer.echo(f"Synthetic plan: {path}")


@app.command("synthesize-dataset")
def synthesize_synthetic_dataset(
    spec: Path = typer.Option(
        Path("configs/data/synthetic_manufacturing_7200.yaml"),
        exists=True,
        dir_okay=False,
    ),
    concurrency: int = typer.Option(4, min=1, max=8),
) -> None:
    """Generate resumable Edge TTS WAV files and a verified final manifest."""
    path = synthesize_dataset(spec, concurrency=concurrency)
    typer.echo(f"Synthetic manifest: {path}")


@app.command("mine-term-errors")
def mine_term_errors(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    terms: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
) -> None:
    """Mine missing manufacturing terms and likely recognized forms from validation."""
    path = mine_domain_term_errors(predictions, terms, output)
    typer.echo(f"Term error candidates: {path}")


@app.command("run")
def run(
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True, dir_okay=False),
) -> None:
    """Run preparation, inference, correction, evaluation, and reporting."""
    settings = load_settings(config)
    result = run_pipeline(settings)
    typer.echo(f"Run completed: {result.run_id}")
    typer.echo(f"Artifacts: {result.run_dir}")
    typer.echo(f"Report: {result.report_path}")


@app.command("train-whisper")
def train_whisper(
    config: Path = typer.Option(
        Path("configs/colab_whisper_lora.yaml"), exists=True, dir_okay=False
    ),
) -> None:
    """Run CUDA-only Whisper LoRA fine-tuning and save checkpoints to the configured path."""
    settings = load_settings(config)
    run_dir = train_whisper_lora(settings)
    typer.echo(f"Training completed: {run_dir}")


@app.command("train-whisper-distillation")
def train_distillation(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Distill a locked teacher Whisper model into a smaller locked student model."""
    settings = load_settings(config)
    run_dir = train_whisper_distillation(settings)
    typer.echo(f"Distillation completed: {run_dir}")


@app.command("train-selected-whisper")
def train_selected_whisper(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """LoRA fine-tune the human-selected benchmark model."""
    result = train_selected_whisper_lora(selection, config)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("lora-learning-curve")
def lora_learning_curve(
    spec: Path = typer.Option(
        Path("configs/training/synthetic_manufacturing_lora_a100.yaml"),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Compare Medium, Turbo, and Large-v3 LoRA stages on validation only."""
    comparison = run_lora_learning_curve(spec)
    typer.echo(f"LoRA learning curve: {comparison}")


@app.command("merge-selected-lora")
def merge_lora(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Merge the selected PEFT adapter into a deployable Transformers checkpoint."""
    merged = merge_selected_lora(selection, config)
    typer.echo(f"Merged LoRA model: {merged}")


@app.command("decoding-sweep-selected-lora")
def decoding_sweep_selected_lora(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Tune beam, prompt, hotwords, and VAD on the fixed validation split."""
    selected = run_selected_lora_decoding_sweep(selection, config)
    typer.echo(f"Decoding selection: {selected}")


@app.command()
def history(
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True, dir_okay=False),
    limit: int = typer.Option(20, min=1, max=200),
) -> None:
    """Show the durable experiment history."""
    settings = load_settings(config)
    records = ExperimentStore(settings.paths.database).history(limit=limit)
    typer.echo(json.dumps(records, ensure_ascii=False, indent=2))


@app.command("model-matrix-lock")
def model_matrix_lock(
    matrix: Path = typer.Option(
        Path("configs/benchmarks/whisper_models_colab.yaml"),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Pin every enabled model in a benchmark matrix to an immutable Hub commit."""
    payload = lock_model_matrix(matrix)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("benchmark-models")
def benchmark_models(
    matrix: Path = typer.Option(
        Path("configs/benchmarks/whisper_models_colab.yaml"),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Compare multiple Whisper models on one fixed validation split."""
    result = run_model_benchmark(matrix)
    typer.echo(f"Benchmark {result.status}: {result.group_id}")
    typer.echo(f"Comparison: {result.comparison_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command("select-model")
def select_model(
    benchmark_dir: Path = typer.Option(..., exists=True, file_okay=False),
    model_id: str = typer.Option(..., help="Completed benchmark model ID."),
    reviewer: str = typer.Option(..., help="Human reviewer making the trade-off decision."),
    reason: str = typer.Option(..., help="Accuracy, speed, memory, and governance rationale."),
    automated_proxy: bool = typer.Option(
        False,
        help=(
            "Mark this as an automated public-proxy smoke selection, not a human "
            "manufacturing decision."
        ),
    ),
) -> None:
    """Record a human model decision or a clearly labeled public-proxy smoke selection."""
    path = select_experiment_member(
        benchmark_dir,
        model_id,
        reviewer=reviewer,
        reason=reason,
        human_reviewed=not automated_proxy,
    )
    typer.echo(f"Selection: {path}")


@app.command("correction-sweep")
def correction_sweep(
    spec: Path = typer.Option(..., exists=True, dir_okay=False),
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Tune safe domain-term correction only on a selected model's validation predictions."""
    result = run_correction_sweep(spec, selection)
    typer.echo(f"Correction sweep completed: {result.sweep_id}")
    typer.echo(f"Comparison: {result.comparison_path}")
    typer.echo(f"Recommended: {result.recommended_candidate}")
    typer.echo(f"Report: {result.report_path}")


@app.command("select-correction")
def select_correction(
    sweep_dir: Path = typer.Option(..., exists=True, file_okay=False),
    candidate_id: str | None = typer.Option(
        None,
        help="Accepted correction candidate ID. Omit to select the highest-ranked candidate.",
    ),
    reviewer: str = typer.Option(..., help="Reviewer or labeled automated proxy selector."),
    reason: str = typer.Option(..., help="Validation metrics and regression-gate rationale."),
    automated_proxy: bool = typer.Option(
        False,
        help="Mark selection as synthetic/public automation rather than human approval.",
    ),
) -> None:
    """Record an accepted validation correction policy with a safe fallback."""
    path = select_correction_candidate(
        sweep_dir,
        candidate_id,
        reviewer=reviewer,
        reason=reason,
        human_reviewed=not automated_proxy,
    )
    typer.echo(f"Correction selection: {path}")


@app.command("quantization-sweep")
def quantization_sweep(
    spec: Path = typer.Option(
        Path("configs/quantization/whisper_quantization_colab.yaml"),
        exists=True,
        dir_okay=False,
    ),
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Evaluate storage and runtime quantization variants for a selected model."""
    result = run_quantization_sweep(spec, selection)
    typer.echo(f"Quantization sweep {result.status}: {result.group_id}")
    typer.echo(f"Comparison: {result.comparison_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command("select-quantization")
def select_quantization(
    quantization_dir: Path = typer.Option(..., exists=True, file_okay=False),
    variant_id: str = typer.Option(..., help="Completed quantization variant ID."),
    reviewer: str = typer.Option(..., help="Human reviewer making the deployment decision."),
    reason: str = typer.Option(..., help="Accuracy-loss and efficiency trade-off rationale."),
    automated_proxy: bool = typer.Option(
        False,
        help=(
            "Mark this as an automated public-proxy smoke selection, not a human "
            "manufacturing decision."
        ),
    ),
) -> None:
    """Record a human precision decision or a labeled public-proxy smoke selection."""
    path = select_experiment_member(
        quantization_dir,
        variant_id,
        reviewer=reviewer,
        reason=reason,
        human_reviewed=not automated_proxy,
    )
    typer.echo(f"Selection: {path}")


@app.command("finalize-evaluation")
def finalize_evaluation(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Run one final held-out test evaluation after model and precision selection."""
    result = run_final_evaluation(selection, config)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("confirmatory-evaluation")
def confirmatory_evaluation(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    cohort_id: str = typer.Option("speaker-heldout-v2"),
    expected_samples: int = typer.Option(600, min=1),
    minimum_speakers: int = typer.Option(30, min=1),
    minimum_term_occurrences: int = typer.Option(1000, min=0),
    minimum_negative_samples: int = typer.Option(100, min=0),
    reference_result: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Evaluate a new speaker-held-out cohort without changing the frozen v1 selection."""
    result = run_confirmatory_evaluation(
        selection,
        config,
        manifest,
        cohort_id=cohort_id,
        expected_samples=expected_samples,
        minimum_speakers=minimum_speakers,
        minimum_term_occurrences=minimum_term_occurrences,
        minimum_negative_samples=minimum_negative_samples,
        reference_result_path=reference_result,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("select-deployment-profiles")
def deployment_profiles(
    comparison: Path = typer.Option(..., exists=True, dir_okay=False),
    profiles: Path = typer.Option(
        Path("configs/deployment/ondevice_profiles.yaml"), exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(..., file_okay=False),
) -> None:
    """Select accuracy-first and feasible on-device candidates without relaxing gates."""
    result = select_deployment_profiles(comparison, profiles, output_dir)
    typer.echo(f"Deployment selections: {result}")


@app.command("assessment-audit")
def assessment_audit(
    config: Path = typer.Option(
        Path("configs/manufacturing_private_template.yaml"),
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("reports/generated/assessment_readiness"),
        file_okay=False,
    ),
    fail_on_blocker: bool = typer.Option(
        False,
        help="Return a non-zero exit code while required evidence is incomplete.",
    ),
) -> None:
    """Generate a four-criterion assessment evidence readiness report."""
    settings = load_settings(config)
    result, json_path, markdown_path = write_assessment_readiness(
        settings,
        output_dir.expanduser().resolve(),
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(f"JSON: {json_path}")
    typer.echo(f"Markdown: {markdown_path}")
    if fail_on_blocker and result["overall_status"] != "ready":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
