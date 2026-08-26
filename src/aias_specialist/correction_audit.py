from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import per_sample_metrics

AUDIT_COLUMNS = [
    "sample_id",
    "reference_text",
    "recognized_before",
    "corrected_after",
    "text_changed",
    "outcome",
    "correction_methods",
    "correction_count",
    "correction_details",
    "wer_before",
    "wer_after",
    "wer_absolute_reduction",
    "cer_before",
    "cer_after",
    "cer_absolute_reduction",
]


def _outcome(cer_reduction: float, wer_reduction: float, tolerance: float = 1e-12) -> str:
    if cer_reduction > tolerance or (
        abs(cer_reduction) <= tolerance and wer_reduction > tolerance
    ):
        return "improved"
    if cer_reduction < -tolerance or (
        abs(cer_reduction) <= tolerance and wer_reduction < -tolerance
    ):
        return "degraded"
    return "unchanged"


def build_correction_audit(
    predictions_before: pd.DataFrame,
    predictions_after: pd.DataFrame,
) -> pd.DataFrame:
    """Build one traceable row per sample for domain-term correction review."""
    if len(predictions_before) != len(predictions_after):
        raise ValueError("Correction audit requires the same samples before and after correction")

    before = per_sample_metrics(predictions_before).reset_index(drop=True)
    after = per_sample_metrics(predictions_after).reset_index(drop=True)
    if "sample_id" in before and "sample_id" in after:
        before_ids = before["sample_id"].astype(str).tolist()
        after_ids = after["sample_id"].astype(str).tolist()
        if before_ids != after_ids:
            raise ValueError("Correction audit sample order or sample IDs do not match")

    rows: list[dict[str, Any]] = []
    for index in range(len(before)):
        before_row = before.iloc[index]
        after_row = after.iloc[index]
        wer_before = float(before_row["sample_wer"])
        wer_after = float(after_row["sample_wer"])
        cer_before = float(before_row["sample_cer"])
        cer_after = float(after_row["sample_cer"])
        wer_reduction = wer_before - wer_after
        cer_reduction = cer_before - cer_after
        recognized_before = str(before_row["prediction_text"])
        corrected_after = str(after_row["prediction_text"])
        details = str(after_row.get("correction_details", "[]"))
        try:
            correction_count = len(json.loads(details))
        except (TypeError, json.JSONDecodeError):
            correction_count = int(after_row.get("correction_count", 0))

        rows.append(
            {
                "sample_id": str(before_row.get("sample_id", index)),
                "reference_text": str(before_row["reference_text"]),
                "recognized_before": recognized_before,
                "corrected_after": corrected_after,
                "text_changed": recognized_before != corrected_after,
                "outcome": _outcome(cer_reduction, wer_reduction),
                "correction_methods": str(after_row.get("correction_methods", "")),
                "correction_count": correction_count,
                "correction_details": details,
                "wer_before": wer_before,
                "wer_after": wer_after,
                "wer_absolute_reduction": wer_reduction,
                "cer_before": cer_before,
                "cer_after": cer_after,
                "cer_absolute_reduction": cer_reduction,
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _write_markdown(path: Path, audit: pd.DataFrame) -> None:
    changed = audit.loc[audit["text_changed"]].copy()
    counts = audit["outcome"].value_counts()
    mean_wer_before = float(audit["wer_before"].mean()) if len(audit) else 0.0
    mean_wer_after = float(audit["wer_after"].mean()) if len(audit) else 0.0
    mean_cer_before = float(audit["cer_before"].mean()) if len(audit) else 0.0
    mean_cer_after = float(audit["cer_after"].mean()) if len(audit) else 0.0
    lines = [
        "# Correction audit",
        "",
        f"- Samples: {len(audit)}",
        f"- Text changed: {len(changed)}",
        f"- Improved: {int(counts.get('improved', 0))}",
        f"- Degraded: {int(counts.get('degraded', 0))}",
        f"- Unchanged: {int(counts.get('unchanged', 0))}",
        f"- Mean sample WER: {mean_wer_before:.4f} -> {mean_wer_after:.4f}",
        f"- Mean sample CER: {mean_cer_before:.4f} -> {mean_cer_after:.4f}",
        "",
        "> Positive reduction values mean the correction improved the score. "
        "Human review is required for every changed or degraded transcript.",
        "",
        "## Changed samples",
        "",
    ]
    if changed.empty:
        lines.append("No transcript text was changed.")
    else:
        lines.extend(
            [
                "| Sample | Outcome | Method | CER before | CER after | CER reduction | "
                "Recognized before | Corrected after | Reference |",
                "|---|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for row in changed.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(row.sample_id),
                        _markdown_cell(row.outcome),
                        _markdown_cell(row.correction_methods),
                        f"{row.cer_before:.4f}",
                        f"{row.cer_after:.4f}",
                        f"{row.cer_absolute_reduction:.4f}",
                        _markdown_cell(row.recognized_before),
                        _markdown_cell(row.corrected_after),
                        _markdown_cell(row.reference_text),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_correction_audit(
    run_dir: Path,
    predictions_before: pd.DataFrame,
    predictions_after: pd.DataFrame,
) -> pd.DataFrame:
    """Persist CSV, JSONL, and reviewer-friendly Markdown correction evidence."""
    audit = build_correction_audit(predictions_before, predictions_after)
    audit.to_csv(run_dir / "correction_audit.csv", index=False, encoding="utf-8-sig")
    jsonl = audit.to_json(orient="records", lines=True, force_ascii=False)
    (run_dir / "correction_audit.jsonl").write_text(jsonl + "\n", encoding="utf-8")
    _write_markdown(run_dir / "correction_audit.md", audit)
    return audit
