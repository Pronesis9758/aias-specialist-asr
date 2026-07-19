from pathlib import Path

from docx import Document

from aias_specialist.reporting import build_report, create_metrics_chart


def _comparison_metrics() -> dict:
    base = {
        "sample_count": 16,
        "wer": 0.50,
        "cer": 0.24,
        "domain_term_recall": 1.0,
        "domain_term_recall_applicable": False,
        "mean_latency_seconds": 0.10,
        "mean_real_time_factor": 0.05,
        "evaluation_runtime_seconds": 1.60,
        "aggregate_real_time_factor": 0.05,
    }
    lora = {
        **base,
        "wer": 0.44,
        "cer": 0.20,
        "mean_latency_seconds": 0.12,
        "mean_real_time_factor": 0.06,
        "evaluation_runtime_seconds": 1.92,
        "aggregate_real_time_factor": 0.06,
    }
    return {
        "training": {
            "model_repo": "openai/whisper-tiny",
            "model_revision": "abc123",
            "train_samples": 40,
            "validation_samples": 8,
            "max_steps": 30,
            "best_validation_wer": 0.47,
            "seed": 42,
        },
        "baseline": base,
        "corrected": base,
        "improvement": {
            "wer_absolute_reduction": 0.0,
            "cer_absolute_reduction": 0.0,
            "domain_term_recall_gain": 0.0,
            "domain_term_recall_applicable": False,
        },
        "lora": lora,
        "lora_improvement": {
            "wer_absolute_reduction": 0.06,
            "cer_absolute_reduction": 0.04,
            "domain_term_recall_gain": 0.0,
            "domain_term_recall_applicable": False,
        },
        "evidence": {
            "dataset": "kresnik/zeroth_korean@fixed",
            "comparison_population": "identical fixed test split",
            "confidence": "low",
        },
    }


def test_lora_comparison_report_contains_required_sections(tmp_path: Path) -> None:
    metrics = _comparison_metrics()
    chart_path = create_metrics_chart(metrics, tmp_path / "comparison.png")
    report_path = build_report(
        tmp_path / "evaluation_report.docx",
        "Whisper LoRA 기술 평가",
        "train-example",
        "Owner",
        "transformers_whisper_lora",
        "openai/whisper-tiny",
        "abc123",
        metrics,
        chart_path,
    )

    assert chart_path.exists() and chart_path.stat().st_size > 0
    assert report_path.exists() and report_path.stat().st_size > 0
    text = "\n".join(paragraph.text for paragraph in Document(report_path).paragraphs)
    for heading in [
        "Technical Summary",
        "Key Findings with Visual Evidence",
        "Scope, Data, and Metric Definitions",
        "Methodology",
        "Limitations, Uncertainty, and Robustness",
        "Recommended Next Steps",
        "Further Questions",
    ]:
        assert heading in text
    assert "동일한 테스트 16개" in text
