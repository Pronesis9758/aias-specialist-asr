from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .retrieval import retrieve_term_corrections


def correction_options(config: Any) -> dict[str, Any]:
    """Convert a CorrectionConfig into the keyword contract used by the correction engine."""
    return {
        "enabled": config.enabled,
        "case_sensitive": config.case_sensitive,
        "alias_enabled": config.alias_enabled,
        "information_retrieval_enabled": config.information_retrieval_enabled,
        "nearest_neighbor_enabled": config.nearest_neighbor_enabled,
        "top_k": config.top_k,
        "max_ngram_tokens": config.max_ngram_tokens,
        "min_ir_score": config.min_ir_score,
        "min_nn_score": config.min_nn_score,
        "nn_backend": config.nn_backend,
        "nn_model_repo_id": config.nn_model_repo_id,
        "nn_model_revision": config.nn_model_revision,
        "nn_device": config.nn_device,
        "ir_weight": config.ir_weight,
        "nn_weight": config.nn_weight,
        "require_consensus": config.require_consensus,
        "min_score_margin": config.min_score_margin,
        "max_length_ratio": config.max_length_ratio,
    }


def resolved_correction_options(settings: Any) -> dict[str, Any]:
    """Return correction options with an immutable embedding revision when applicable."""
    options = correction_options(settings.correction)
    repo_id = settings.correction.nn_model_repo_id
    if (
        settings.correction.enabled
        and settings.correction.nearest_neighbor_enabled
        and settings.correction.nn_backend == "transformers"
        and repo_id
    ):
        from .models import load_model_lock

        lock = load_model_lock(settings)
        options["nn_model_revision"] = str(lock["models"][repo_id]["resolved_revision"])
    return options


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
    alias_enabled: bool = True,
    information_retrieval_enabled: bool = False,
    nearest_neighbor_enabled: bool = False,
    top_k: int = 3,
    max_ngram_tokens: int = 4,
    min_ir_score: float = 0.62,
    min_nn_score: float = 0.78,
    nn_backend: str = "char_ngram",
    nn_model_repo_id: str | None = None,
    nn_model_revision: str = "main",
    nn_device: str = "auto",
    ir_weight: float = 0.5,
    nn_weight: float = 0.5,
    require_consensus: bool = False,
    min_score_margin: float = 0.0,
    max_length_ratio: float = 4.0,
) -> pd.DataFrame:
    output = predictions.copy()
    pairs = _alias_pairs(terms)
    corrected_texts: list[str] = []
    details: list[str] = []

    for prediction in output["prediction_text"].astype(str):
        corrected = prediction
        applied: list[dict[str, Any]] = []
        for alias, canonical in pairs:
            if not enabled or not alias_enabled:
                break
            if case_sensitive:
                if alias in corrected:
                    corrected = corrected.replace(alias, canonical)
                    applied.append({"from": alias, "to": canonical, "method": "alias"})
            else:
                lower_text = corrected.lower()
                lower_alias = alias.lower()
                if lower_alias in lower_text:
                    start = lower_text.index(lower_alias)
                    end = start + len(alias)
                    corrected = f"{corrected[:start]}{canonical}{corrected[end:]}"
                    applied.append({"from": alias, "to": canonical, "method": "alias"})
        if enabled:
            corrected, retrieved = retrieve_term_corrections(
                corrected,
                terms,
                case_sensitive=case_sensitive,
                information_retrieval_enabled=information_retrieval_enabled,
                nearest_neighbor_enabled=nearest_neighbor_enabled,
                top_k=top_k,
                max_ngram_tokens=max_ngram_tokens,
                min_ir_score=min_ir_score,
                min_nn_score=min_nn_score,
                nn_backend=nn_backend,
                nn_model_repo_id=nn_model_repo_id,
                nn_model_revision=nn_model_revision,
                nn_device=nn_device,
                ir_weight=ir_weight,
                nn_weight=nn_weight,
                require_consensus=require_consensus,
                min_score_margin=min_score_margin,
                max_length_ratio=max_length_ratio,
            )
            applied.extend(retrieved)
        corrected_texts.append(corrected)
        details.append(json.dumps(applied, ensure_ascii=False))

    output["baseline_prediction_text"] = output["prediction_text"]
    output["prediction_text"] = corrected_texts
    output["correction_details"] = details
    output["correction_count"] = [len(json.loads(item)) for item in details]
    output["correction_methods"] = [
        "|".join(sorted({entry.get("method", "alias") for entry in json.loads(item)}))
        for item in details
    ]
    return output
