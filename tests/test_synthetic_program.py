from pathlib import Path

from aias_specialist.synthetic_program import build_dataset_plan, load_synthetic_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/data/synthetic_manufacturing_7200.yaml"


def test_full_scale_synthetic_plan_meets_research_contract() -> None:
    frame = build_dataset_plan(SPEC)

    assert frame.groupby("split").size().to_dict() == {
        "train": 6000,
        "validation": 600,
        "test": 600,
    }
    assert frame["speaker_id"].nunique() == 24
    assert frame.groupby("speaker_id")["split"].nunique().max() == 1
    assert frame["reference_text"].is_unique
    assert frame["sample_id"].is_unique
    assert set(frame["consent_status"]) == {"synthetic"}


def test_full_scale_term_coverage_is_balanced_and_test_is_dense() -> None:
    frame = build_dataset_plan(SPEC)
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    train_counts: dict[str, int] = {}
    for value in train["term_targets"]:
        for term in value.split("|"):
            train_counts[term] = train_counts.get(term, 0) + 1

    assert min(train_counts.values()) >= 100
    assert max(train_counts.values()) <= 200
    assert sum(len(value.split("|")) for value in test["term_targets"]) >= 1000
    assert test["term_targets"].str.contains(r"\|").all()


def test_full_scale_spec_records_expected_duration_targets() -> None:
    _, section = load_synthetic_spec(SPEC)
    frame = build_dataset_plan(SPEC)

    assert section["split_counts"] == {"train": 6000, "validation": 600, "test": 600}
    hours = (
        frame.assign(seconds=frame["target_duration_seconds"].astype(float))
        .groupby("split")["seconds"]
        .sum()
        .div(3600)
        .to_dict()
    )
    assert hours == {"train": 10.0, "validation": 1.0, "test": 1.0}
