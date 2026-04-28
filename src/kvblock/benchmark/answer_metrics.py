"""Answer normalization and QA scoring helpers for output benchmarks."""

from __future__ import annotations

import re
import string
from typing import Any, Callable, Sequence

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}

def normalize_answer(text: str) -> str:
    """Normalize an answer using HotpotQA-style text processing."""

    lowered = text.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    no_articles = _ARTICLES_RE.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match_score(prediction: str, gold: str) -> float:
    """Return normalized exact match as 0.0 or 1.0."""

    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> tuple[float, float, float]:
    """Return normalized token-level F1, precision, and recall."""

    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if (
        normalized_prediction in _SPECIAL_ANSWERS
        or normalized_gold in _SPECIAL_ANSWERS
    ) and normalized_prediction != normalized_gold:
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not prediction_tokens or not gold_tokens:
        score = float(prediction_tokens == gold_tokens)
        return score, score, score

    common = _token_overlap_count(prediction_tokens, gold_tokens)
    if common == 0:
        return 0.0, 0.0, 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def metric_max_over_ground_truths(
    prediction: str,
    golds: Sequence[str],
    metric_fn: Callable[[str, str], Any],
) -> Any:
    """Return the best metric value over all gold answers."""

    if not golds:
        raise ValueError("golds must not be empty")
    scores = tuple(metric_fn(prediction, gold) for gold in golds)
    return max(scores, key=_metric_sort_key)


def score_qa_answer(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    """Score a predicted short answer against one or more gold answers."""

    if not answers:
        raise ValueError("answers must not be empty")
    em = metric_max_over_ground_truths(prediction, answers, exact_match_score)
    f1, precision, recall = metric_max_over_ground_truths(
        prediction,
        answers,
        f1_score,
    )
    return {
        "em": float(em),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def _token_overlap_count(left: Sequence[str], right: Sequence[str]) -> int:
    counts: dict[str, int] = {}
    for token in left:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in right:
        count = counts.get(token, 0)
        if count <= 0:
            continue
        overlap += 1
        counts[token] = count - 1
    return overlap


def _metric_sort_key(value: float | tuple[float, float, float]) -> float:
    if isinstance(value, tuple):
        return value[0]
    return value
