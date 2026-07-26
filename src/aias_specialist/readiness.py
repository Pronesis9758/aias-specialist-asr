from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Settings
from .data import validate_manifest
from .utils import utc_now, write_json

RUN_ARTIFACT_CONTRACT = {
    "config.snapshot.yaml",
    "environment.json",
    "prepared_manifest.csv",
    "predictions_baseline.csv",
    "predictions_corrected.csv",
    "metrics.json",
    "run_summary.md",
    "reports/evaluation_report.docx",
}
REQUIRED_DOCUMENTS = {
    "solution_review": "docs/SOLUTION_SELECTION_REVIEW.md",
    "governance_checklist": "docs/GOVERNANCE_CHECKLIST.md",
    "hardware_plan": "docs/HARDWARE_VALIDATION_PLAN.md",
    "assessment_evidence": "docs/ASSESSMENT_EVIDENCE_PLAN.md",
    "third_party_notices": "THIRD_PARTY_NOTICES.md",
}
PLACEHOLDERS = {
    "",
    "to_be_completed",
    "to_be_decided",
    "required",
    "tbd",
    "미정",
}


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    criterion: str
    status: str
    blocker: bool
    message: str
    evidence: str = ""


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _placeholder(value: object) -> bool:
    return str(value).strip().lower() in PLACEHOLDERS


def _add(
    checks: list[ReadinessCheck],
    check_id: str,
    criterion: str,
    status: str,
    blocker: bool,
    message: str,
    evidence: str | Path = "",
) -> None:
    checks.append(
        ReadinessCheck(
            check_id=check_id,
            criterion=criterion,
            status=status,
            blocker=blocker,
            message=message,
            evidence=str(evidence),
        )
    )


def _completed_table(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "comparison table has not been generated"
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty:
        return False, "comparison table is empty"
    if "status" not in frame.columns:
        return False, "comparison table has no status column"
    incomplete = frame.loc[frame["status"].astype(str) != "completed"]
    if not incomplete.empty:
        return False, f"{len(incomplete)} candidate(s) are incomplete"
    return True, f"{len(frame)} candidate(s) completed"


def _check_human_selection(path: Path, label: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"human {label} selection is pending"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
    if selection.get("human_reviewed") is not True:
        scope = str(selection.get("selection_scope", "unknown"))
        return (
            False,
            f"{label} selection is an automated proxy result ({scope}); "
            "human manufacturing review is still required",
        )
    reviewer = str(selection.get("reviewer", "")).strip()
    reason = str(selection.get("reason", "")).strip()
    if _placeholder(reviewer) or _placeholder(reason):
        return False, f"human {label} selection reviewer or rationale is incomplete"
    return True, f"human {label} selection is recorded by {reviewer}"


def _model_repositories(settings: Settings, assessment: dict[str, Any]) -> set[str]:
    repositories = {settings.model.repo_id}
    training = settings.training.values or {}
    if training.get("enabled") and training.get("repo_id"):
        repositories.add(str(training["repo_id"]))
    matrix_value = assessment.get("model_matrix")
    if matrix_value:
        matrix_path = _resolve(settings.project_root, matrix_value)
        if matrix_path.exists():
            payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
            section = payload.get("benchmark", {}) if isinstance(payload, dict) else {}
            for model in section.get("models", []):
                if isinstance(model, dict) and model.get("repo_id"):
                    repositories.add(str(model["repo_id"]))
    return repositories


def _check_model_lock(
    settings: Settings,
    assessment: dict[str, Any],
) -> tuple[bool, str]:
    if not settings.paths.model_lock.exists():
        return False, "model lock does not exist"
    payload = yaml.safe_load(settings.paths.model_lock.read_text(encoding="utf-8"))
    models = payload.get("models", {}) if isinstance(payload, dict) else {}
    missing: list[str] = []
    mutable: list[str] = []
    for repo_id in sorted(_model_repositories(settings, assessment)):
        record = models.get(repo_id)
        if not isinstance(record, dict):
            missing.append(repo_id)
            continue
        revision = str(record.get("resolved_revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            mutable.append(repo_id)
    if missing:
        return False, f"unlocked repositories: {', '.join(missing)}"
    if mutable:
        return False, f"repositories without immutable SHA: {', '.join(mutable)}"
    return True, f"{len(_model_repositories(settings, assessment))} repository revision(s) pinned"


def _check_human_signoff(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "human review sign-off has not been created"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    review = payload.get("human_review", {}) if isinstance(payload, dict) else {}
    required_text = {"reviewer", "reviewed_at", "decision", "conclusion"}
    required_gates = {
        "transcript_labels_verified",
        "privacy_approval_verified",
        "domain_terms_verified",
        "model_tradeoff_verified",
        "report_conclusions_verified",
    }
    incomplete = [
        field for field in required_text if field not in review or _placeholder(review[field])
    ]
    incomplete.extend(field for field in required_gates if review.get(field) is not True)
    if incomplete:
        return False, f"incomplete human review fields: {', '.join(sorted(incomplete))}"
    return True, f"review completed by {review['reviewer']}"


def _check_deployment_profile(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "deployment hardware profile does not exist"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = payload.get("deployment", {}) if isinstance(payload, dict) else {}
    required = {
        "target_name",
        "deployment_mode",
        "on_device_claim",
        "warmup_samples",
        "timing_repetitions",
    }
    incomplete = [
        field for field in required if field not in profile or _placeholder(profile[field])
    ]
    if profile.get("on_device_claim") is True:
        for field in {
            "max_model_size_mb",
            "max_peak_memory_mb",
            "max_aggregate_rtf",
            "max_p95_latency_seconds",
        }:
            if field not in profile or _placeholder(profile[field]):
                incomplete.append(field)
    if incomplete:
        return False, f"incomplete deployment fields: {', '.join(sorted(incomplete))}"
    if int(profile["timing_repetitions"]) < 3:
        return False, "timing_repetitions must be at least 3 for submission evidence"
    return True, f"target={profile['target_name']}; mode={profile['deployment_mode']}"


def _check_acceptance_criteria(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "acceptance criteria file does not exist"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    criteria = payload.get("acceptance_criteria", {}) if isinstance(payload, dict) else {}
    required = {
        "problem_statement_owner",
        "approved_at",
        "minimum_train_samples",
        "minimum_validation_samples",
        "minimum_test_samples",
        "maximum_cer",
        "minimum_domain_term_recall",
        "maximum_aggregate_rtf",
        "maximum_p95_latency_seconds",
        "selection_rule",
    }
    incomplete = [
        field for field in required if field not in criteria or _placeholder(criteria[field])
    ]
    if incomplete:
        return False, f"incomplete acceptance criteria: {', '.join(sorted(incomplete))}"
    return True, f"criteria approved by {criteria['problem_statement_owner']}"


def audit_assessment_readiness(settings: Settings) -> dict[str, Any]:
    root = settings.project_root
    assessment = settings.raw.get("assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    checks: list[ReadinessCheck] = []

    for check_id, relative in REQUIRED_DOCUMENTS.items():
        path = root / relative
        _add(
            checks,
            f"document.{check_id}",
            "1/2/3/4",
            "passed" if path.exists() else "failed",
            True,
            "required assessment document is present"
            if path.exists()
            else "required assessment document is missing",
            path,
        )

    strict = settings.governance.mode == "strict_private"
    _add(
        checks,
        "governance.strict_private",
        "4",
        "passed" if strict else "failed",
        True,
        "strict private-data governance is enabled"
        if strict
        else "manufacturing submission config must use governance.mode=strict_private",
        settings.config_path,
    )

    acceptance_value = assessment.get(
        "acceptance_criteria",
        "configs/assessment/acceptance_criteria_template.yaml",
    )
    acceptance_path = _resolve(root, acceptance_value)
    acceptance_ok, acceptance_message = _check_acceptance_criteria(acceptance_path)
    _add(
        checks,
        "problem.acceptance_criteria",
        "1/3",
        "passed" if acceptance_ok else "waiting",
        True,
        acceptance_message,
        acceptance_path,
    )

    if settings.paths.manifest.exists():
        try:
            frame = validate_manifest(
                settings.paths.manifest,
                settings.model.backend,
                settings.governance,
            )
            split_counts = frame["split"].str.lower().value_counts().to_dict()
            required_splits = {"train", "validation", "test"}
            valid_splits = required_splits.issubset(split_counts)
            _add(
                checks,
                "data.manifest",
                "3/4",
                "passed" if valid_splits else "failed",
                True,
                f"validated manifest with split counts {split_counts}"
                if valid_splits
                else f"manifest must contain train/validation/test; found {split_counts}",
                settings.paths.manifest,
            )
        except Exception as exc:
            _add(
                checks,
                "data.manifest",
                "3/4",
                "failed",
                True,
                f"manifest validation failed: {exc}",
                settings.paths.manifest,
            )
    else:
        _add(
            checks,
            "data.manifest",
            "3/4",
            "waiting",
            True,
            "waiting for approved manufacturing audio and transcripts",
            settings.paths.manifest,
        )

    lock_ok, lock_message = _check_model_lock(settings, assessment)
    _add(
        checks,
        "model.revisions",
        "1",
        "passed" if lock_ok else "failed",
        True,
        lock_message,
        settings.paths.model_lock,
    )

    artifact_root = settings.paths.artifacts_dir.parent
    benchmark_id = str(assessment.get("benchmark_id", "")).strip()
    quantization_id = str(assessment.get("quantization_id", "")).strip()
    benchmark_dir = artifact_root / "benchmarks" / benchmark_id
    quantization_dir = artifact_root / "quantization" / quantization_id

    benchmark_ok, benchmark_message = _completed_table(benchmark_dir / "benchmark_comparison.csv")
    _add(
        checks,
        "experiment.model_benchmark",
        "1/3",
        "passed" if benchmark_ok else "waiting",
        True,
        benchmark_message,
        benchmark_dir,
    )
    model_selection = benchmark_dir / "model_selection.yaml"
    model_selection_ok, model_selection_message = _check_human_selection(
        model_selection,
        "model",
    )
    _add(
        checks,
        "experiment.model_selection",
        "1",
        "passed" if model_selection_ok else "waiting",
        True,
        model_selection_message,
        model_selection,
    )
    selected_training_result = benchmark_dir / "selected_training_result.json"
    training_ok = False
    training_evidence: Path = selected_training_result
    if selected_training_result.exists():
        training_payload = json.loads(selected_training_result.read_text(encoding="utf-8"))
        training_run_dir = Path(str(training_payload.get("run_dir", "")))
        training_evidence = training_run_dir
        training_ok = training_run_dir.exists() and all(
            (training_run_dir / relative).exists()
            for relative in {
                *RUN_ARTIFACT_CONTRACT,
                "training_metrics.json",
                "predictions_lora.csv",
            }
        )
    _add(
        checks,
        "experiment.selected_model_lora",
        "1/3",
        "passed" if training_ok else "waiting",
        True,
        "selected model LoRA run and Base/LoRA comparison are complete"
        if training_ok
        else "selected model LoRA fine-tuning evidence is pending",
        training_evidence,
    )

    quantization_ok, quantization_message = _completed_table(
        quantization_dir / "quantization_comparison.csv"
    )
    _add(
        checks,
        "experiment.quantization",
        "3",
        "passed" if quantization_ok else "waiting",
        True,
        quantization_message,
        quantization_dir,
    )
    quantization_selection = quantization_dir / "quantization_selection.yaml"
    quantization_selection_ok, quantization_selection_message = _check_human_selection(
        quantization_selection,
        "quantization",
    )
    _add(
        checks,
        "experiment.quantization_selection",
        "3",
        "passed" if quantization_selection_ok else "waiting",
        True,
        quantization_selection_message,
        quantization_selection,
    )

    final_result_path = quantization_dir / "final_test_result.json"
    final_run_dir: Path | None = None
    if final_result_path.exists():
        payload = json.loads(final_result_path.read_text(encoding="utf-8"))
        final_run_dir = Path(str(payload.get("run_dir", "")))
        final_ok = final_run_dir.exists() and all(
            (final_run_dir / relative).exists() for relative in RUN_ARTIFACT_CONTRACT
        )
        message = (
            "held-out test result and artifact contract are complete"
            if final_ok
            else "final result exists but the immutable run artifact contract is incomplete"
        )
    else:
        final_ok = False
        message = "held-out test evaluation is pending"
    _add(
        checks,
        "experiment.final_test",
        "1/3",
        "passed" if final_ok else "waiting",
        True,
        message,
        final_run_dir or final_result_path,
    )

    hardware_value = assessment.get(
        "hardware_profile", "configs/hardware/target_device_template.yaml"
    )
    hardware_path = _resolve(root, hardware_value)
    hardware_ok, hardware_message = _check_deployment_profile(hardware_path)
    _add(
        checks,
        "deployment.hardware_profile",
        "3",
        "passed" if hardware_ok else "failed",
        True,
        hardware_message,
        hardware_path,
    )

    signoff_value = assessment.get("human_signoff", "data/private/human_review_signoff.yaml")
    signoff_path = _resolve(root, signoff_value)
    signoff_ok, signoff_message = _check_human_signoff(signoff_path)
    _add(
        checks,
        "review.human_signoff",
        "1/3/4",
        "passed" if signoff_ok else "waiting",
        True,
        signoff_message,
        signoff_path,
    )

    blocker_failures = [
        check for check in checks if check.blocker and check.status in {"failed", "waiting"}
    ]
    return {
        "generated_at": utc_now().isoformat(),
        "config": str(settings.config_path),
        "overall_status": "ready" if not blocker_failures else "not_ready",
        "passed": sum(check.status == "passed" for check in checks),
        "total": len(checks),
        "blockers_remaining": len(blocker_failures),
        "checks": [asdict(check) for check in checks],
    }


def write_assessment_readiness(
    settings: Settings,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    result = audit_assessment_readiness(settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "assessment_readiness.json"
    markdown_path = output_dir / "assessment_readiness.md"
    write_json(json_path, result)

    lines = [
        "# AI Specialist assessment readiness",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Config: `{result['config']}`",
        f"- Overall: **{result['overall_status']}**",
        f"- Passed: {result['passed']}/{result['total']}",
        f"- Blockers remaining: {result['blockers_remaining']}",
        "",
        "| Criterion | Check | Status | Blocker | Message | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for check in result["checks"]:
        message = str(check["message"]).replace("|", "/")
        evidence = str(check["evidence"]).replace("|", "/")
        lines.append(
            f"| {check['criterion']} | {check['check_id']} | {check['status']} | "
            f"{'yes' if check['blocker'] else 'no'} | {message} | `{evidence}` |"
        )
    lines.extend(
        [
            "",
            "This report checks evidence completeness. Human reviewers remain responsible for "
            "transcript correctness, privacy approval, model trade-offs, and final conclusions.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, json_path, markdown_path
