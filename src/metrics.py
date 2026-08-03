"""검색 품질 지표. gold 는 정답 청크 id 집합, ranked 는 유사도 내림차순 청크 id 리스트."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """상위 k개 안에 들어온 정답 비율."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """상위 k개 안에 정답이 하나라도 있으면 1. 실무 RAG 에서 가장 체감되는 지표."""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def precision_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def mrr_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """첫 정답의 역순위. 1위에 맞히면 1.0, 5위면 0.2."""
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """이진 relevance 기준 nDCG. 정답을 얼마나 위쪽에 몰아놨는지."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(ranked[:k], start=1)
        if cid in gold
    )
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(ranked: list[int], gold: set[int], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"Hit@{k}"] = hit_at_k(ranked, gold, k)
    top = max(ks)
    out[f"MRR@{top}"] = mrr_at_k(ranked, gold, top)
    out[f"nDCG@{top}"] = ndcg_at_k(ranked, gold, top)
    out[f"Precision@{min(ks)}"] = precision_at_k(ranked, gold, min(ks))
    return out
