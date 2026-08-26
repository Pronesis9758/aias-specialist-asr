from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .evaluation import normalize_text


def _ngrams(text: str, maximum: int) -> list[str]:
    tokens = normalize_text(text).split()
    return [
        " ".join(tokens[start : start + width])
        for width in range(1, min(maximum, len(tokens)) + 1)
        for start in range(len(tokens) - width + 1)
    ]


def mine_domain_term_errors(
    predictions_path: str | Path,
    terms_path: str | Path,
    output_path: str | Path,
) -> Path:
    predictions = pd.read_csv(predictions_path, keep_default_na=False)
    terms = pd.read_csv(terms_path, keep_default_na=False)
    if not {"sample_id", "reference_text", "prediction_text"}.issubset(predictions.columns):
        raise ValueError("Predictions require sample_id, reference_text, and prediction_text")
    observations: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in predictions.itertuples(index=False):
        reference = normalize_text(row.reference_text)
        hypothesis = normalize_text(row.prediction_text)
        for term_row in terms.itertuples(index=False):
            canonical = str(term_row.canonical)
            normalized = normalize_text(canonical)
            if not normalized or normalized not in reference or normalized in hypothesis:
                continue
            aliases = [
                normalize_text(value)
                for value in str(term_row.aliases).split("|")
                if value.strip()
            ]
            targets = [normalized, *aliases]
            candidates = _ngrams(hypothesis, max(1, len(normalized.split()) + 1))
            if not candidates:
                continue
            scored = [
                (
                    max(SequenceMatcher(None, candidate, target).ratio() for target in targets),
                    candidate,
                )
                for candidate in candidates
            ]
            score, observed = max(scored)
            observations[(canonical, observed)].append((score, str(row.sample_id)))
    rows = [
        {
            "canonical_term": canonical,
            "observed_form": observed,
            "occurrence_count": len(values),
            "mean_similarity": sum(score for score, _ in values) / len(values),
            "sample_ids": "|".join(sample_id for _, sample_id in values[:20]),
            "recommended_action": (
                "review_then_add_alias_and_generate_20_to_50_varied_examples"
            ),
        }
        for (canonical, observed), values in observations.items()
    ]
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            ["occurrence_count", "mean_similarity"], ascending=[False, False]
        )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
