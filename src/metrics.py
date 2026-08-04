"""검색 품질 지표. gold 는 정답 청크 id 집합, ranked 는 유사도 내림차순 청크 id 리스트.

Precision / Recall / F1 은 쓰지 않는다. 질문당 정답 청크가 1개인 데이터라
Recall@k 는 Hit@k 와 값이 같아 중복이고, Precision@k 는 상한이 1/k 로 묶여
k 를 키우면 오히려 내려가서 @1 과 @3 을 나란히 놓고 읽을 수 없기 때문이다.
"""

from __future__ import annotations

import math


def hit_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """상위 k개 안에 정답이 하나라도 있으면 1. Success@k / Accuracy@k 라고도 한다.

    "정답이 상위 k개 안에 들어올 확률"이라 k 가 커지면 절대 낮아지지 않는다
    (ranked[:1] 은 ranked[:3] 의 부분집합이므로 Hit@1 <= Hit@3 이 항상 성립).
    실무 RAG 에서 가장 체감되는 지표이고, 이 프로젝트의 주 비교 기준이다.
    """
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """첫 정답의 역순위. 1위에 맞히면 1.0, 5위면 0.2.

    Hit@k 가 구분하지 못하는 것 — 1등으로 맞혔는지 3등으로 겨우 맞혔는지 — 을 가른다.
    """
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """이진 relevance 기준 nDCG. 정답을 얼마나 위쪽에 몰아놨는지.

    MRR 은 첫 정답만 보지만 nDCG 는 정답이 여러 개일 때 나머지 위치까지 반영한다.
    """
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
    out: dict[str, float] = {f"Hit@{k}": hit_at_k(ranked, gold, k) for k in ks}
    top = max(ks)
    out[f"MRR@{top}"] = mrr_at_k(ranked, gold, top)
    out[f"nDCG@{top}"] = ndcg_at_k(ranked, gold, top)
    return out
