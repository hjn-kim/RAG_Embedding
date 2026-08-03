"""임베딩 모델 검색 성능 비교 벤치마크.

사용법:
    python -m src.run_benchmark
    python -m src.run_benchmark --models kure-v1 bge-m3
    python -m src.run_benchmark --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .chunker import chunk_blocks
from .gold import label_gold, load_questions
from .loaders import load_documents
from .metrics import evaluate_ranking
from .models import ModelSpec, load_encoder, resolve_device

ROOT = Path(__file__).resolve().parent.parent


def build_corpus(cfg: dict):
    print("[1/4] 문서 로드")
    blocks = load_documents(ROOT / cfg["paths"]["data_dir"])

    print("[2/4] 청킹")
    ck = cfg["chunking"]
    chunks = chunk_blocks(
        blocks,
        chunk_size=ck["chunk_size"],
        chunk_overlap=ck["chunk_overlap"],
        min_chunk_chars=ck["min_chunk_chars"],
    )
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f"  총 {len(chunks)} chunks  {by_source}")
    return chunks


def rank_from_scores(scores: np.ndarray, allowed: np.ndarray | None, top_k: int) -> list[int]:
    """점수 벡터 → 상위 top_k 인덱스. allowed 는 후보 제한용 bool 마스크."""
    s = scores.copy()
    if allowed is not None:
        s[~allowed] = -np.inf
    k = min(top_k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])].tolist()


def run_model(spec: ModelSpec, cfg: dict, chunks, questions, gold):
    """모델 하나를 평가하고 (요약 행 리스트, 질문별 상세 리스트) 반환."""
    runtime = cfg["runtime"]
    ret = cfg["retrieval"]
    bs = runtime["batch_size"]
    ks = ret["ks"]
    top_k = max(ret["top_k"], max(ks))

    enc = load_encoder(spec, runtime)

    corpus_texts = [c.text for c in chunks]
    t0 = time.time()
    doc_vecs = enc.encode_passages(corpus_texts, bs)
    encode_sec = time.time() - t0

    q_texts = [q.question for q in questions]
    t0 = time.time()
    q_vecs = enc.encode_queries(q_texts, bs)
    query_sec = time.time() - t0

    dense_scores = q_vecs @ doc_vecs.T  # 정규화되어 있으므로 내적 = 코사인

    # BGE-M3 하이브리드
    variants = {"dense": dense_scores}
    sparse = getattr(enc, "sparse_score_matrix", lambda: None)()
    if sparse is not None and sparse.shape == dense_scores.shape:
        h = spec.hybrid
        variants["hybrid(dense+sparse)"] = (
            h.get("dense_weight", 1.0) * dense_scores
            + h.get("sparse_weight", 0.3) * sparse
        )

    # scope=own_doc 이면 질문이 속한 문서로 후보를 제한
    sources = np.array([c.source for c in chunks])
    masks: list[np.ndarray | None] = []
    for q in questions:
        if ret["scope"] == "own_doc" and q.source:
            masks.append(sources == q.source)
        else:
            masks.append(None)

    summaries, details = [], []
    for variant, scores in variants.items():
        label = spec.key if variant == "dense" and len(variants) == 1 else f"{spec.key} [{variant}]"
        per_q = []
        for qi, q in enumerate(questions):
            ranked = rank_from_scores(scores[qi], masks[qi], top_k)
            m = evaluate_ranking(ranked, gold[q.id], ks)
            per_q.append(m)
            details.append(
                {
                    "model": label,
                    "qid": q.id,
                    "question": q.question,
                    "n_gold": len(gold[q.id]),
                    **{k: round(v, 4) for k, v in m.items()},
                    "top5": [
                        {
                            "chunk_id": cid,
                            "score": round(float(scores[qi][cid]), 4),
                            "is_gold": cid in gold[q.id],
                            "source": chunks[cid].source,
                            "locator": chunks[cid].locator,
                            "preview": chunks[cid].text[:120],
                        }
                        for cid in ranked[:5]
                    ],
                }
            )

        row = {"model": label, "hf_id": spec.hf_id, "dim": int(doc_vecs.shape[1])}
        for metric in per_q[0]:
            row[metric] = float(np.mean([m[metric] for m in per_q]))
        row["corpus_encode_s"] = round(encode_sec, 2)
        row["chunks_per_s"] = round(len(chunks) / encode_sec, 1) if encode_sec else 0.0
        row["query_ms"] = round(query_sec / max(len(questions), 1) * 1000, 1)
        row["index_MB"] = round(doc_vecs.nbytes / 1024 / 1024, 2)
        summaries.append(row)

    enc.close()
    return summaries, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--models", nargs="*", default=None, help="평가할 모델 key (기본: 전체)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_corpus(cfg)
    (results_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/4] 질문 로드 및 정답 라벨링")
    questions = load_questions(ROOT / cfg["paths"]["questions_file"])
    gold, unmatched, too_broad = label_gold(questions, chunks)
    print(f"  질문 {len(questions)}개, 질문당 평균 정답 청크 "
          f"{np.mean([len(g) for g in gold.values()]):.1f}개")
    if too_broad:
        print(f"  [경고] 정답 청크가 너무 많은 질문 (전체 {len(chunks)}개 중):")
        for qid, n in too_broad:
            print(f"         {qid}: {n}개")
        print("         → 조건이 헐렁해서 아무 모델이나 맞히게 됩니다. must_include 문구를"
              " 더 길고 고유하게 쓰거나 must_exclude 로 걸러내세요.")
    if unmatched:
        print(f"  [경고] 정답 청크를 못 찾은 질문: {unmatched}")
        print("         → must_include 문구가 문서 원문과 정확히 일치하는지 확인하세요.")
        print("         → results/chunks.json 에서 실제 청크 텍스트를 볼 수 있습니다.")
        questions = [q for q in questions if gold[q.id]]
        if not questions:
            raise SystemExit("평가 가능한 질문이 없습니다. questions.json 을 수정해주세요.")

    specs = [ModelSpec.from_config(m) for m in cfg["models"]]
    if args.models:
        specs = [s for s in specs if s.key in args.models]
    if not specs:
        raise SystemExit("평가할 모델이 없습니다.")

    print(f"[4/4] 모델 평가 (device={resolve_device(cfg['runtime']['device'])})")
    all_rows, all_details = [], []
    for spec in specs:
        print(f"\n── {spec.key}  ({spec.hf_id})")
        try:
            rows, details = run_model(spec, cfg, chunks, questions, gold)
        except Exception as e:  # 한 모델이 죽어도 나머지는 계속
            print(f"  [실패] {type(e).__name__}: {e}")
            continue
        all_rows.extend(rows)
        all_details.extend(details)
        for r in rows:
            top = max(cfg["retrieval"]["ks"])
            print(f"  → Hit@1 {r['Hit@1']:.3f} | Recall@5 {r.get('Recall@5', float('nan')):.3f} "
                  f"| nDCG@{top} {r[f'nDCG@{top}']:.3f} | {r['chunks_per_s']} chunks/s")

    if not all_rows:
        raise SystemExit("성공한 모델이 없습니다.")

    top = max(cfg["retrieval"]["ks"])
    df = pd.DataFrame(all_rows).sort_values(f"nDCG@{top}", ascending=False)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(4)

    df.to_csv(results_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (results_dir / "details.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "summary.md").write_text(
        "# 임베딩 모델 비교 결과\n\n"
        f"- 청크 수: {len(chunks)}\n- 질문 수: {len(questions)}\n"
        f"- scope: {cfg['retrieval']['scope']}\n"
        f"- chunk_size / overlap: {cfg['chunking']['chunk_size']} / "
        f"{cfg['chunking']['chunk_overlap']}\n\n"
        + df.to_markdown(index=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"\n결과 저장: {results_dir}/summary.csv, summary.md, details.json")
    print("모델이 왜 틀렸는지는 details.json 의 top5 를 보세요.")


if __name__ == "__main__":
    main()
