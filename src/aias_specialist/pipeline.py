from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .asr import run_inference
from .config import Settings
from .correction import apply_term_correction
from .data import load_domain_terms, prepare_manifest
from .environment import collect_environment
from .evaluation import compare_metrics, evaluate_predictions, per_sample_metrics
from .reporting import build_report, create_metrics_chart
from .store import ExperimentStore, register_run_artifacts
from .utils import git_sha, new_run_id, utc_now, write_json


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    report_path: Path


def _write_summary(path: Path, settings: Settings, run_id: str, metrics: dict[str, Any]) -> None:
    baseline = metrics["baseline"]
    corrected = metrics["corrected"]
    improvement = metrics["improvement"]
    term_recall_applicable = bool(baseline.get("domain_term_recall_applicable"))
    baseline_term_recall = (
        f"{baseline['domain_term_recall']:.4f}" if term_recall_applicable else "N/A"
    )
    corrected_term_recall = (
        f"{corrected['domain_term_recall']:.4f}" if term_recall_applicable else "N/A"
    )
    text = f"""# Run summary: {run_id}

- Project: {settings.project.name}
- Backend: {settings.model.backend}
- Model: {settings.model.repo_id}
- Evaluation split: {settings.evaluation.split}
- Samples: {baseline["sample_count"]}
- Baseline WER: {baseline["wer"]:.4f}
- Corrected WER: {corrected["wer"]:.4f}
- WER absolute reduction: {improvement["wer_absolute_reduction"]:.4f}
- Baseline domain term recall: {baseline_term_recall}
- Corrected domain term recall: {corrected_term_recall}
- Human review required: transcript labels, privacy approval, domain-term substitutions,
  final model choice
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(settings: Settings) -> RunResult:
    run_id = new_run_id()
    run_dir = settings.paths.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(settings.paths.database)
    revision: str | None = None
    print(
        f"[pipeline] run={run_id} model={settings.model.repo_id} split={settings.evaluation.split}",
        flush=True,
    )

    store.start_run(
        {
            "run_id": run_id,
            "started_at": utc_now().isoformat(),
            "project_name": settings.project.name,
            "config_path": str(settings.config_path),
            "backend": settings.model.backend,
            "model_repo": settings.model.repo_id,
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
        print("[pipeline] preparing manifest", flush=True)
        prepared = prepare_manifest(
            settings.paths.manifest,
            run_dir / "prepared_manifest.csv",
            settings.model.backend,
            settings.governance,
        )
        terms = load_domain_terms(settings.paths.domain_terms)
        terms.to_csv(run_dir / "domain_terms.snapshot.csv", index=False, encoding="utf-8-sig")
        store.event(run_id, "prepare", "completed", f"samples={len(prepared)}")

        evaluation_frame = prepared.loc[
            prepared["split"].str.lower() == settings.evaluation.split
        ].copy()
        if evaluation_frame.empty:
            raise ValueError(
                f"Pipeline evaluation requires at least one {settings.evaluation.split} sample"
            )
        store.event(
            run_id,
            "evaluation_split",
            "completed",
            f"split={settings.evaluation.split}; samples={len(evaluation_frame)}",
        )

        store.event(run_id, "baseline", "started")
        print("[pipeline] running baseline inference", flush=True)
        baseline_predictions, revision = run_inference(evaluation_frame, settings)
        baseline_predictions = per_sample_metrics(baseline_predictions)
        baseline_predictions.to_csv(
            run_dir / "predictions_baseline.csv", index=False, encoding="utf-8-sig"
        )
        baseline_metrics = evaluate_predictions(baseline_predictions, terms)
        store.event(run_id, "baseline", "completed", f"revision={revision}")
        if settings.paths.model_lock.exists():
            (run_dir / "model-lock.snapshot.yaml").write_text(
                settings.paths.model_lock.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        store.event(run_id, "correction", "started")
        print("[pipeline] applying domain-term correction", flush=True)
        corrected_predictions = apply_term_correction(
            baseline_predictions,
            terms,
            enabled=settings.correction.enabled,
            case_sensitive=settings.correction.case_sensitive,
        )
        corrected_predictions = per_sample_metrics(corrected_predictions)
        corrected_predictions.to_csv(
            run_dir / "predictions_corrected.csv", index=False, encoding="utf-8-sig"
        )
        corrected_metrics = evaluate_predictions(corrected_predictions, terms)
        store.event(run_id, "correction", "completed")

        metrics = {
            "baseline": baseline_metrics,
            "corrected": corrected_metrics,
            "improvement": compare_metrics(baseline_metrics, corrected_metrics),
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(
            run_dir / "resource_metrics.json",
            {
                key: baseline_metrics[key]
                for key in [
                    "model_preparation_seconds",
                    "model_size_bytes",
                    "peak_process_memory_mb",
                    "peak_gpu_memory_mb",
                    "evaluation_runtime_seconds",
                    "aggregate_real_time_factor",
                ]
            },
        )
        store.add_metrics(run_id, "baseline", baseline_metrics)
        store.add_metrics(run_id, "corrected", corrected_metrics)
        store.add_metrics(run_id, "improvement", metrics["improvement"])

        store.event(run_id, "report", "started")
        print("[pipeline] generating report", flush=True)
        chart_path = None
        if settings.report.include_charts:
            chart_path = create_metrics_chart(metrics, reports_dir / "metrics_comparison.png")
        report_path = build_report(
            reports_dir / "evaluation_report.docx",
            settings.report.title,
            run_id,
            settings.project.owner,
            settings.model.backend,
            settings.model.repo_id,
            revision or settings.model.revision,
            metrics,
            chart_path,
        )
        _write_summary(run_dir / "run_summary.md", settings, run_id, metrics)
        store.event(run_id, "report", "completed")

        if settings.training.enabled:
            store.event(
                run_id,
                "training",
                "blocked",
                "Training is configured but must be run with the dedicated Colab training command.",
            )

        register_run_artifacts(store, run_id, run_dir)
        store.finish_run(run_id, "completed", model_revision=revision)
        print(f"[pipeline] completed run={run_id} report={report_path}", flush=True)
        return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics, report_path=report_path)
    except Exception as exc:
        error_path = run_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        store.event(run_id, "pipeline", "failed", str(exc))
        store.finish_run(run_id, "failed", model_revision=revision, error=str(exc))
        print(f"[pipeline] failed run={run_id}: {exc}", flush=True)
        raise
