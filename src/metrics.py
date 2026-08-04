"""검색 품질 지표. gold 는 정답 청크 id 집합, ranked 는 유사도 내림차순 청크 id 리스트."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """상위 k개 안에 들어온 정답 비율."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """상위 k개 안에 정답이 하나라도 있으면 1. Success@k / Accuracy@k 라고도 한다.

    "정답이 상위 k개 안에 들어올 확률"이라 k 가 커지면 절대 낮아지지 않는다
    (ranked[:1] 은 ranked[:3] 의 부분집합이므로 Hit@1 <= Hit@3 이 항상 성립).
    실무 RAG 에서 가장 체감되는 지표이고, 이 프로젝트의 주 비교 기준이다.
    """
    return 1.0 if set(ranked[:k]) & gold else 0.0


def precision_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def f1_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """Precision@k 와 Recall@k 의 조화평균.

    주의: 질문당 정답 청크가 1개인데 k=3 이면 Precision 상한이 1/3 이라
    F1@3 의 이론적 최대값은 0.5 다. Hit@k 와 달리 k 가 커지면 오히려 내려가므로,
    @1 과 @3 을 세로로 비교하지 말고 같은 k 안에서 모델끼리만 비교할 것.
    """
    p = precision_at_k(ranked, gold, k)
    r = recall_at_k(ranked, gold, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


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
        out[f"Precision@{k}"] = precision_at_k(ranked, gold, k)
        out[f"Recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"F1@{k}"] = f1_at_k(ranked, gold, k)
        out[f"Hit@{k}"] = hit_at_k(ranked, gold, k)
    top = max(ks)
    out[f"MRR@{top}"] = mrr_at_k(ranked, gold, top)
    out[f"nDCG@{top}"] = ndcg_at_k(ranked, gold, top)
    return out
