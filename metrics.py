"""Retrieval metrics for a ranked list against a set of relevant ids.

Relevance is binary. Average precision walks the whole ranking and
divides by the number of relevant items, which is the standard
retrieval AP behind mAP.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def precision_at_k(retrieved: Sequence, relevant: Set, k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in list(retrieved)[:k] if item in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence, relevant: Set, k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item in list(retrieved)[:k] if item in relevant)
    return hits / len(relevant)


def average_precision(retrieved: Sequence, relevant: Set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(retrieved: Sequence, relevant: Set, k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(list(retrieved)[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def summarise(rankings: Iterable[Dict], ks: Sequence[int] = (1, 5, 10)) -> Dict:
    """Mean the per-query metrics.

    Each item is {"retrieved": [...], "relevant": set(...)}.
    """
    rows = list(rankings)
    if not rows:
        raise ValueError("No rankings to summarise")
    summary: Dict = {"queries": len(rows)}
    for k in ks:
        summary[f"precision@{k}"] = _mean(precision_at_k(row["retrieved"], set(row["relevant"]), k) for row in rows)
        summary[f"recall@{k}"] = _mean(recall_at_k(row["retrieved"], set(row["relevant"]), k) for row in rows)
    summary["mAP"] = _mean(average_precision(row["retrieved"], set(row["relevant"])) for row in rows)
    summary["ndcg@10"] = _mean(ndcg_at_k(row["retrieved"], set(row["relevant"]), 10) for row in rows)
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in summary.items()}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)
