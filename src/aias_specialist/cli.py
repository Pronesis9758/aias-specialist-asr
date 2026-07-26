from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_settings
from .environment import doctor as doctor_check
from .experiments import (
    lock_model_matrix,
    run_final_evaluation,
    run_model_benchmark,
    run_quantization_sweep,
    select_experiment_member,
    train_selected_whisper_lora,
)
from .hf_data import prepare_hf_dataset
from .models import download_baseline_model, resolve_model_lock
from .pipeline import run_pipeline
from .readiness import write_assessment_readiness
from .store import ExperimentStore
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


@app.command("train-selected-whisper")
def train_selected_whisper(
    selection: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """LoRA fine-tune the human-selected benchmark model."""
    result = train_selected_whisper_lora(selection, config)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


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
