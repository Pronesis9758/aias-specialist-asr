from __future__ import annotations

import json

import pandas as pd


def _alias_pairs(terms: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for record in terms.to_dict(orient="records"):
        canonical = str(record["canonical"]).strip()
        aliases = [item.strip() for item in str(record["aliases"]).split("|") if item.strip()]
        for alias in aliases:
            if alias != canonical:
                pairs.append((alias, canonical))
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def apply_term_correction(
    predictions: pd.DataFrame,
    terms: pd.DataFrame,
    enabled: bool = True,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    output = predictions.copy()
    pairs = _alias_pairs(terms)
    corrected_texts: list[str] = []
    details: list[str] = []

    for prediction in output["prediction_text"].astype(str):
        corrected = prediction
        applied: list[dict[str, str]] = []
        for alias, canonical in pairs:
            if not enabled:
                break
            if case_sensitive:
                if alias in corrected:
                    corrected = corrected.replace(alias, canonical)
                    applied.append({"from": alias, "to": canonical})
            else:
                lower_text = corrected.lower()
                lower_alias = alias.lower()
                if lower_alias in lower_text:
                    start = lower_text.index(lower_alias)
                    end = start + len(alias)
                    corrected = f"{corrected[:start]}{canonical}{corrected[end:]}"
                    applied.append({"from": alias, "to": canonical})
        corrected_texts.append(corrected)
        details.append(json.dumps(applied, ensure_ascii=False))

    output["baseline_prediction_text"] = output["prediction_text"]
    output["prediction_text"] = corrected_texts
    output["correction_details"] = details
    output["correction_count"] = [len(json.loads(item)) for item in details]
    return output
