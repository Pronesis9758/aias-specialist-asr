from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_settings
from .environment import doctor as doctor_check
from .hf_data import prepare_hf_dataset
from .models import download_baseline_model, resolve_model_lock
from .pipeline import run_pipeline
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


@app.command()
def history(
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True, dir_okay=False),
    limit: int = typer.Option(20, min=1, max=200),
) -> None:
    """Show the durable experiment history."""
    settings = load_settings(config)
    records = ExperimentStore(settings.paths.database).history(limit=limit)
    typer.echo(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
