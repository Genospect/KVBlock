"""Small output-quality metrics for benchmark reports."""

from __future__ import annotations

import re


def exact_match(prediction: str, answer: str) -> float:
    """Case-insensitive normalized exact match."""

    return float(_normalize_text(prediction) == _normalize_text(answer))


def contains_answer(prediction: str, answer: str) -> float:
    """Return 1 when the normalized answer appears in the prediction."""

    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return 0.0
    return float(normalized_answer in _normalize_text(prediction))


def token_f1(prediction: str, answer: str) -> float:
    """Token-level F1 for short-answer style evaluation."""

    pred_tokens = _normalize_text(prediction).split()
    answer_tokens = _normalize_text(answer).split()
    if not pred_tokens or not answer_tokens:
        return float(pred_tokens == answer_tokens)
    common = set(pred_tokens) & set(answer_tokens)
    overlap = sum(min(pred_tokens.count(tok), answer_tokens.count(tok)) for tok in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())
