from pathlib import Path

import pandas as pd
import yaml

from aias_specialist.ondevice import select_deployment_profiles


def test_ondevice_selection_never_relaxes_profile_constraints(tmp_path: Path) -> None:
    comparison = tmp_path / "quantization.csv"
    profiles = tmp_path / "profiles.yaml"
    pd.DataFrame(
        [
            {
                "member_id": "float16",
                "run_id": "r1",
                "status": "completed",
                "domain_term_recall": 0.90,
                "cer": 0.05,
                "wer": 0.10,
                "aggregate_real_time_factor": 0.30,
                "model_size_bytes": 2_000_000_000,
            },
            {
                "member_id": "int8",
                "run_id": "r2",
                "status": "completed",
                "domain_term_recall": 0.86,
                "cer": 0.06,
                "wer": 0.12,
                "aggregate_real_time_factor": 0.20,
                "model_size_bytes": 800_000_000,
            },
        ]
    ).to_csv(comparison, index=False)
    profiles.write_text(
        """deployment_selection:
  id: test
  profiles:
    - id: accuracy
      minimum_domain_term_recall: 0.85
      maximum_cer: 0.07
      maximum_wer: 0.15
      maximum_rtf: 1.0
      maximum_model_size_mb: 3000
      sort_by: accuracy
    - id: edge
      minimum_domain_term_recall: 0.85
      maximum_cer: 0.07
      maximum_wer: 0.15
      maximum_rtf: 0.5
      maximum_model_size_mb: 1000
      sort_by: efficiency
""",
        encoding="utf-8",
    )

    result = select_deployment_profiles(comparison, profiles, tmp_path / "output")
    payload = yaml.safe_load(result.read_text(encoding="utf-8"))

    assert payload["profiles"]["accuracy"]["member_id"] == "float16"
    assert payload["profiles"]["edge"]["member_id"] == "int8"
    assert payload["test_evaluated"] is False
