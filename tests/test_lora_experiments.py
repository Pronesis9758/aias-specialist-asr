from typing import Any

import pytest

from aias_specialist.lora_experiments import _stages_through


def _stages() -> list[dict[str, Any]]:
    return [
        {"id": "pilot-5h"},
        {"id": "target-10h"},
        {"id": "extended-15h"},
    ]


def test_stages_through_supports_incremental_a100_learning_curve() -> None:
    assert [stage["id"] for stage in _stages_through(_stages(), "target-10h")] == [
        "pilot-5h",
        "target-10h",
    ]


def test_stages_through_rejects_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="Unknown max_stage"):
        _stages_through(_stages(), "unknown")
