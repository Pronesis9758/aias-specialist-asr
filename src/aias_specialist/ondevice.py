from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def select_deployment_profiles(
    comparison_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
) -> Path:
    comparison = pd.read_csv(comparison_path, keep_default_na=False)
    payload = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    section = payload.get("deployment_selection") if isinstance(payload, dict) else None
    if not isinstance(section, dict) or not isinstance(section.get("profiles"), list):
        raise ValueError("Deployment profile spec requires deployment_selection.profiles")
    completed = comparison.loc[comparison["status"].eq("completed")].copy()
    numeric = [
        "domain_term_recall",
        "cer",
        "wer",
        "aggregate_real_time_factor",
        "model_size_bytes",
    ]
    for column in numeric:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for profile in section["profiles"]:
        profile_id = str(profile["id"])
        feasible = completed.loc[
            completed["domain_term_recall"].ge(
                float(profile["minimum_domain_term_recall"])
            )
            & completed["cer"].le(float(profile["maximum_cer"]))
            & completed["wer"].le(float(profile["maximum_wer"]))
            & completed["aggregate_real_time_factor"].le(float(profile["maximum_rtf"]))
            & completed["model_size_bytes"].le(
                float(profile["maximum_model_size_mb"]) * 1024 * 1024
            )
        ].copy()
        if feasible.empty:
            selections[profile_id] = {
                "status": "blocked",
                "reason": "No candidate satisfies every accuracy and resource constraint.",
            }
            continue
        if str(profile.get("sort_by", "accuracy")) == "efficiency":
            feasible = feasible.sort_values(
                [
                    "model_size_bytes",
                    "aggregate_real_time_factor",
                    "domain_term_recall",
                    "cer",
                ],
                ascending=[True, True, False, True],
            )
        else:
            feasible = feasible.sort_values(
                [
                    "domain_term_recall",
                    "cer",
                    "wer",
                    "aggregate_real_time_factor",
                ],
                ascending=[False, True, True, True],
            )
        selected = feasible.iloc[0]
        selections[profile_id] = {
            "status": "selected",
            "member_id": str(selected["member_id"]),
            "run_id": str(selected["run_id"]),
            "domain_term_recall": float(selected["domain_term_recall"]),
            "cer": float(selected["cer"]),
            "wer": float(selected["wer"]),
            "aggregate_real_time_factor": float(selected["aggregate_real_time_factor"]),
            "model_size_bytes": int(selected["model_size_bytes"]),
        }
        rows.append({"profile_id": profile_id, **selections[profile_id]})
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "deployment_selection_id": str(section["id"]),
        "source_comparison": str(Path(comparison_path).expanduser().resolve()),
        "test_evaluated": False,
        "profiles": selections,
        "optimization_policy": section.get("optimization_policy", {}),
        "human_hardware_validation_required": True,
    }
    result_path = output / "deployment_selections.yaml"
    result_path.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        output / "deployment_feasible_candidates.csv", index=False, encoding="utf-8-sig"
    )
    return result_path
