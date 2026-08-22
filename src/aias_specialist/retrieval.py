from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TermVariant:
    canonical: str
    variant: str
    category: str


@dataclass(frozen=True)
class RetrievalCandidate:
    start: int
    end: int
    source: str
    canonical: str
    ir_score: float
    nn_score: float
    combined_score: float
    method: str


def term_variants(terms: pd.DataFrame) -> list[TermVariant]:
    variants: list[TermVariant] = []
    seen: set[tuple[str, str]] = set()
    for record in terms.to_dict(orient="records"):
        canonical = str(record["canonical"]).strip()
        category = str(record.get("category", "")).strip()
        aliases = [
            item.strip() for item in str(record.get("aliases", "")).split("|") if item.strip()
        ]
        for variant in [canonical, *aliases]:
            key = (canonical, variant)
            if variant and key not in seen:
                variants.append(TermVariant(canonical, variant, category))
                seen.add(key)
    return variants


def _normalize(value: str, case_sensitive: bool) -> str:
    compact = re.sub(r"\s+", " ", value.strip())
    return compact if case_sensitive else compact.lower()


def _character_ngrams(value: str, case_sensitive: bool, sizes: tuple[int, ...]) -> list[str]:
    compact = re.sub(r"\s+", "", _normalize(value, case_sensitive))
    if not compact:
        return []
    output: list[str] = []
    for size in sizes:
        if len(compact) < size:
            output.append(compact)
        else:
            output.extend(compact[index : index + size] for index in range(len(compact) - size + 1))
    return output


def _bm25_scores(query: str, documents: list[str], case_sensitive: bool) -> list[float]:
    tokenized = [_character_ngrams(document, case_sensitive, (2, 3)) for document in documents]
    query_tokens = _character_ngrams(query, case_sensitive, (2, 3))
    if not query_tokens or not tokenized:
        return [0.0] * len(documents)

    document_count = len(tokenized)
    average_length = sum(len(tokens) for tokens in tokenized) / max(document_count, 1)
    frequencies: Counter[str] = Counter()
    for tokens in tokenized:
        frequencies.update(set(tokens))

    k1 = 1.5
    b = 0.75
    raw_scores: list[float] = []
    query_set = set(query_tokens)
    for tokens in tokenized:
        counts = Counter(tokens)
        length = max(len(tokens), 1)
        score = 0.0
        for token in query_set:
            frequency = counts[token]
            if not frequency:
                continue
            document_frequency = frequencies[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * length / max(average_length, 1.0))
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        raw_scores.append(score)

    maximum = max(raw_scores, default=0.0)
    if maximum <= 0.0:
        return [0.0] * len(documents)

    query_counter = Counter(query_tokens)
    normalized: list[float] = []
    for raw_score, tokens in zip(raw_scores, tokenized, strict=True):
        document_counter = Counter(tokens)
        overlap = sum((query_counter & document_counter).values())
        dice = 2.0 * overlap / max(sum(query_counter.values()) + sum(document_counter.values()), 1)
        normalized.append(0.4 * raw_score / maximum + 0.6 * dice)
    return normalized


def _char_ngram_embeddings(texts: list[str], case_sensitive: bool) -> np.ndarray:
    counters = [Counter(_character_ngrams(text, case_sensitive, (2, 3, 4))) for text in texts]
    vocabulary = sorted({token for counter in counters for token in counter})
    if not vocabulary:
        return np.zeros((len(texts), 1), dtype=np.float32)
    index = {token: position for position, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, counter in enumerate(counters):
        for token, count in counter.items():
            matrix[row, index[token]] = float(count)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _transformer_embeddings(
    texts: list[str],
    model_repo_id: str,
    model_revision: str,
    device: str,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer nearest-neighbor search requires the train dependencies. "
            "Install the project with '.[train]'."
        ) from exc

    runtime_device = device
    if runtime_device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id, revision=model_revision)
    model = AutoModel.from_pretrained(model_repo_id, revision=model_revision)
    model.to(runtime_device)
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(runtime_device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def _embeddings(
    texts: list[str],
    *,
    backend: str,
    case_sensitive: bool,
    model_repo_id: str | None,
    model_revision: str,
    device: str,
) -> np.ndarray:
    if backend == "char_ngram":
        return _char_ngram_embeddings(texts, case_sensitive)
    if not model_repo_id:
        raise ValueError("model_repo_id is required for transformers nearest-neighbor search")
    return _transformer_embeddings(texts, model_repo_id, model_revision, device)


def _spans(text: str, max_ngram_tokens: int) -> list[tuple[int, int, str]]:
    tokens = list(re.finditer(r"[0-9A-Za-z가-힣]+", text))
    spans: list[tuple[int, int, str]] = []
    for start in range(len(tokens)):
        for width in range(1, min(max_ngram_tokens, len(tokens) - start) + 1):
            first = tokens[start]
            last = tokens[start + width - 1]
            spans.append((first.start(), last.end(), text[first.start() : last.end()]))
    return spans


def retrieve_term_corrections(
    text: str,
    terms: pd.DataFrame,
    *,
    case_sensitive: bool,
    information_retrieval_enabled: bool,
    nearest_neighbor_enabled: bool,
    top_k: int,
    max_ngram_tokens: int,
    min_ir_score: float,
    min_nn_score: float,
    nn_backend: str,
    nn_model_repo_id: str | None,
    nn_model_revision: str,
    nn_device: str,
    ir_weight: float,
    nn_weight: float,
    require_consensus: bool,
    min_score_margin: float,
    max_length_ratio: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not information_retrieval_enabled and not nearest_neighbor_enabled:
        return text, []

    variants = term_variants(terms)
    spans = _spans(text, max_ngram_tokens)
    if not variants or not spans:
        return text, []

    variant_texts = [item.variant for item in variants]
    span_texts = [item[2] for item in spans]
    normalized_variants = {_normalize(item.variant, case_sensitive) for item in variants}
    ir_matrix: list[list[float]] = []
    if information_retrieval_enabled:
        ir_matrix = [_bm25_scores(span, variant_texts, case_sensitive) for span in span_texts]

    nn_matrix: np.ndarray | None = None
    if nearest_neighbor_enabled:
        all_embeddings = _embeddings(
            [*variant_texts, *span_texts],
            backend=nn_backend,
            case_sensitive=case_sensitive,
            model_repo_id=nn_model_repo_id,
            model_revision=nn_model_revision,
            device=nn_device,
        )
        variant_embeddings = all_embeddings[: len(variant_texts)]
        span_embeddings = all_embeddings[len(variant_texts) :]
        nn_matrix = span_embeddings @ variant_embeddings.T

    candidates: list[RetrievalCandidate] = []
    weight_total = (
        (ir_weight if information_retrieval_enabled else 0.0)
        + (nn_weight if nearest_neighbor_enabled else 0.0)
    )
    if weight_total <= 0.0:
        raise ValueError("At least one enabled retrieval weight must be greater than zero")

    for span_index, (start, end, source) in enumerate(spans):
        normalized_source = _normalize(source, case_sensitive)
        if normalized_source in normalized_variants or len(normalized_source.replace(" ", "")) < 2:
            continue
        best_by_canonical: dict[str, tuple[float, float]] = {}
        for variant_index, variant in enumerate(variants):
            ir_score = ir_matrix[span_index][variant_index] if ir_matrix else 0.0
            nn_score = float(nn_matrix[span_index, variant_index]) if nn_matrix is not None else 0.0
            current = best_by_canonical.get(variant.canonical, (0.0, 0.0))
            best_by_canonical[variant.canonical] = (
                max(current[0], ir_score),
                max(current[1], nn_score),
            )
        span_candidates: list[RetrievalCandidate] = []
        for canonical, (ir_score, nn_score) in best_by_canonical.items():
            ir_qualified = information_retrieval_enabled and ir_score >= min_ir_score
            nn_qualified = nearest_neighbor_enabled and nn_score >= min_nn_score
            if not ir_qualified and not nn_qualified:
                continue
            if (
                require_consensus
                and information_retrieval_enabled
                and nearest_neighbor_enabled
                and not (ir_qualified and nn_qualified)
            ):
                continue
            source_length = len(normalized_source.replace(" ", ""))
            canonical_length = len(_normalize(canonical, case_sensitive).replace(" ", ""))
            length_ratio = max(source_length, canonical_length) / max(
                min(source_length, canonical_length), 1
            )
            if length_ratio > max_length_ratio:
                continue
            combined = (
                (ir_weight * ir_score if information_retrieval_enabled else 0.0)
                + (nn_weight * nn_score if nearest_neighbor_enabled else 0.0)
            ) / weight_total
            method = "hybrid" if ir_qualified and nn_qualified else "ir" if ir_qualified else "nn"
            span_candidates.append(
                RetrievalCandidate(
                    start=start,
                    end=end,
                    source=source,
                    canonical=canonical,
                    ir_score=ir_score,
                    nn_score=nn_score,
                    combined_score=combined,
                    method=method,
                )
            )
        span_candidates.sort(key=lambda item: item.combined_score, reverse=True)
        if span_candidates:
            runner_up = span_candidates[1].combined_score if len(span_candidates) > 1 else 0.0
            if span_candidates[0].combined_score - runner_up >= min_score_margin:
                candidates.append(span_candidates[0])

    ranked = sorted(
        candidates,
        key=lambda item: (item.combined_score, item.end - item.start),
        reverse=True,
    )
    selected: list[RetrievalCandidate] = []
    for candidate in ranked:
        if len(selected) >= top_k:
            break
        if any(candidate.start < other.end and candidate.end > other.start for other in selected):
            continue
        selected.append(candidate)

    corrected = text
    details: list[dict[str, Any]] = []
    for candidate in sorted(selected, key=lambda item: item.start, reverse=True):
        corrected = (
            f"{corrected[: candidate.start]}{candidate.canonical}{corrected[candidate.end :]}"
        )
        details.append(
            {
                "from": candidate.source,
                "to": candidate.canonical,
                "method": candidate.method,
                "ir_score": round(candidate.ir_score, 6),
                "nn_score": round(candidate.nn_score, 6),
                "combined_score": round(candidate.combined_score, 6),
            }
        )
    details.reverse()
    return corrected, details
