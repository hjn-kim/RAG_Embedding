"""임베딩 모델 검색 성능 비교 벤치마크.

사용법:
    python -m src.run_benchmark
    python -m src.run_benchmark --models kure-v1 bge-m3
    python -m src.run_benchmark --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .chunker import chunk_blocks
from .gold import label_gold, load_questions, match_context
from .loaders import load_documents
from .metadata import attach_metadata
from .metrics import evaluate_ranking
from .models import ModelSpec, load_encoder, resolve_device

ROOT = Path(__file__).resolve().parent.parent

METRIC_NOTE = (
    "- **Hit@k**  상위 k개 안에 정답이 하나라도 있던 질문 비율. "
    "k 가 커지면 절대 낮아지지 않는다 (Hit@1 <= Hit@3)\n"
    "- **정답@k**  같은 값을 비율이 아니라 실제 맞힌 문제 개수로 센 것\n"
    "- **MRR@k**  첫 정답의 등수 역수 (1등=1.0, 2등=0.5, 3등=0.33, 밖=0). "
    "'찾았나'와 '얼마나 위에 올렸나'를 한 숫자로 묶은 것\n"
    "- **nDCG@k**  정답을 얼마나 위쪽에 몰아놨는지. 정답이 여러 개일 때도 제대로 계산된다 "
    "(최종 정렬 기준)\n"
    "- **VRAM_MB**  모델 로드 + 인코딩 중 최대 GPU 메모리 사용량\n"
    "- **index_MB**  청크 벡터를 전부 담은 인덱스 용량 (차원에 비례)\n"
    "- **chunks_per_s / query_ms**  워밍업 후 측정한 인코딩 처리량 / 질문 1개당 지연\n"
)


def build_corpus(cfg: dict):
    print("[1/4] 문서 로드")
    blocks = load_documents(ROOT / cfg["paths"]["data_dir"])

    # 메타데이터는 기록 전용이다. 검색 후보를 거르는 데 쓰지 않으므로
    # 모델 간 비교 조건은 그대로 유지된다.
    doc_meta = attach_metadata(blocks)
    for source, m in doc_meta.items():
        print(f"    {source}: {m['doc_type']} / {m['기관'] or '기관?'} / "
              f"{m['작성일'] or '작성일?'} / 섹션 {m['섹션수']}개")

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


def _cuda_sync() -> None:
    """CUDA 커널은 비동기로 큐잉된다. 동기화하지 않으면 '제출한 시간'을 재게 된다."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_mem_mb() -> float:
    """모델 로드 + 인코딩 동안 찍은 최대 GPU 메모리(MB). CPU 실행이면 0."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def measure_throughput(fn, texts: list[str], batch_size: int,
                       min_texts: int, warmup: int, repeat: int) -> tuple[float, int]:
    """초당 처리 텍스트 수를 측정한다. (texts_per_s, 실제 측정에 쓴 텍스트 수)

    코퍼스가 작으면 CUDA 컨텍스트 초기화·cuBLAS 커널 오토튜닝 같은 '고정 비용'이
    실제 연산 시간을 압도해서, 먼저 실행된 모델만 손해를 보는 무의미한 숫자가 나온다.
    (22개 청크에서 KURE-v1 51 chunks/s vs 같은 크기의 e5-large 196 chunks/s 처럼)

    그래서 세 가지를 한다:
      1) 텍스트를 min_texts 개까지 복제해 고정 비용이 묻힐 만큼 작업량을 늘린다
      2) 워밍업을 먼저 돌려 초기화 비용을 측정 밖으로 뺀다
      3) repeat 번 재서 최솟값(외부 간섭이 가장 적었던 실행)을 쓴다
    """
    if not texts or repeat < 1:
        return 0.0, 0

    n = max(min_texts, len(texts))
    bench = (texts * math.ceil(n / len(texts)))[:n]

    for _ in range(max(warmup, 0)):
        fn(bench[: min(len(bench), batch_size * 2)], batch_size)
    _cuda_sync()

    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(bench, batch_size)
        _cuda_sync()
        best = min(best, time.perf_counter() - t0)

    return (len(bench) / best if best > 0 else 0.0), len(bench)


def run_model(spec: ModelSpec, cfg: dict, chunks, questions, gold):
    """모델 하나를 평가하고 (요약 행 리스트, 질문별 상세 리스트) 반환."""
    runtime = cfg["runtime"]
    ret = cfg["retrieval"]
    bs = runtime["batch_size"]
    ks = ret["ks"]
    top_k = max(ret["top_k"], max(ks))

    _reset_peak_mem()  # 모델 로드부터의 피크 VRAM 을 재기 위해 직전에 초기화
    enc = load_encoder(spec, runtime)

    corpus_texts = [c.text for c in chunks]
    doc_vecs = enc.encode_passages(corpus_texts, bs)

    q_texts = [q.question for q in questions]
    q_vecs = enc.encode_queries(q_texts, bs)

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

    # 속도 측정은 정확도 계산이 끝난 뒤에 한다.
    # (BGE-M3 는 encode 할 때마다 내부 sparse 상태를 덮어쓰므로 순서를 바꾸면 하이브리드가 깨진다)
    min_texts = runtime.get("speed_min_texts", 512)
    warmup = runtime.get("speed_warmup", 1)
    repeat = runtime.get("speed_repeat", 3)
    chunks_per_s, n_bench = measure_throughput(
        enc.encode_passages, corpus_texts, bs, min_texts, warmup, repeat
    )
    q_per_s, _ = measure_throughput(
        enc.encode_queries, q_texts, bs, min_texts, warmup, repeat
    )
    query_ms = 1000.0 / q_per_s if q_per_s else 0.0
    vram_mb = _peak_mem_mb()
    if repeat >= 1:
        print(f"  속도 측정: {n_bench}개 텍스트 × {repeat}회 (워밍업 {warmup}회) → "
              f"{chunks_per_s:.1f} chunks/s, 피크 VRAM {vram_mb:.0f}MB")

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
                            "섹션": chunks[cid].meta.get("섹션"),
                            # gold 청크는 '왜 정답인지' 걸린 문구 주변을 보여준다
                            "match": (
                                match_context(
                                    chunks[cid].text, q.must_include + q.any_include
                                )
                                if cid in gold[q.id]
                                else None
                            ),
                            "preview": chunks[cid].text[:200],
                        }
                        for cid in ranked[:5]
                    ],
                }
            )

        row = {
            "model": label,
            "hf_id": spec.hf_id,
            "dim": int(doc_vecs.shape[1]),
            "청크수": len(chunks),
        }
        for metric in per_q[0]:
            row[metric] = float(np.mean([m[metric] for m in per_q]))
        # Hit@k 평균(비율) 대신 실제로 맞힌 문제 개수 — 표본이 작을 때 체감이 정확하다
        for k in ks:
            row[f"정답@{k}"] = int(sum(m[f"Hit@{k}"] for m in per_q))
        row["VRAM_MB"] = round(vram_mb, 0)
        row["index_MB"] = round(doc_vecs.nbytes / 1024 / 1024, 2)
        row["chunks_per_s"] = round(chunks_per_s, 1)
        row["query_ms"] = round(query_ms, 2)
        # 틀린 문제는 표에 넣기엔 길어서 별도 리포트로 뺀다 (main 에서 pop)
        row["_misses"] = {
            k: [questions[i].id for i, m in enumerate(per_q) if m[f"Hit@{k}"] == 0.0]
            for k in ks
        }
        summaries.append(row)

    enc.close()
    return summaries, details


def build_miss_report(misses: dict, questions, gold, ks: list[int]) -> str:
    """모델별로 못 맞힌 질문을 나열한다.

    맨 앞의 '모든 모델이 틀린 문제' 가 가장 중요하다. 어느 모델도 못 맞혔다면
    모델 성능 문제가 아니라 질문 문장이나 must_include 라벨을 의심해야 한다.
    """
    qtext = {q.id: q.question for q in questions}
    n = len(questions)
    lines = [f"# 틀린 문제 목록  (질문 {n}개 기준)", ""]

    for k in ks:
        sets = [set(m.get(k, [])) for m in misses.values() if m]
        common = set.intersection(*sets) if sets else set()
        lines += [f"## 모든 모델이 @{k} 에서 틀린 문제 — {len(common)}개", ""]
        if common:
            lines += ["모델 탓이 아닐 가능성이 높다. 질문 문장 또는 정답 라벨을 점검할 것.", ""]
            lines += [
                f"- `{qid}` (정답 청크 {len(gold.get(qid, []))}개)  {qtext.get(qid, '')}"
                for qid in sorted(common)
            ]
        else:
            lines.append("없음")
        lines.append("")

    lines += ["---", ""]
    for model, m in misses.items():
        lines += [f"## {model}", ""]
        for k in ks:
            ids = m.get(k, [])
            lines += [f"### 틀린@{k} — {len(ids)}/{n}개", ""]
            lines += (
                [f"- `{qid}`  {qtext.get(qid, '')}" for qid in ids] if ids else ["없음"]
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    # Windows 콘솔은 기본이 cp949 라 '—' 같은 기호에서 UnicodeEncodeError 로 죽는다.
    # (RunPod/Linux 는 원래 UTF-8 이라 무해)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

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
            lo = min(cfg["retrieval"]["ks"])
            hi = max(cfg["retrieval"]["ks"])
            print(f"  → 정답 {r[f'정답@{lo}']}/{len(questions)}(@{lo}) → "
                  f"{r[f'정답@{hi}']}/{len(questions)}(@{hi}) | "
                  f"Hit@{lo} {r[f'Hit@{lo}']:.3f} | Hit@{hi} {r[f'Hit@{hi}']:.3f} | "
                  f"{r['chunks_per_s']} chunks/s")

    if not all_rows:
        raise SystemExit("성공한 모델이 없습니다.")

    ks = cfg["retrieval"]["ks"]
    lo, hi = min(ks), max(ks)

    # 틀린 문제 목록은 표에 넣기엔 길어서 별도 리포트로 뺀다
    misses = {r["model"]: r.pop("_misses", {}) for r in all_rows}

    df = pd.DataFrame(all_rows).sort_values(f"nDCG@{hi}", ascending=False)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(4)

    # summary.csv 에는 hf_id 까지 포함한 전체 컬럼을, 화면과 summary.md 에는 추린 표를 쓴다.
    display_cols = ["model", "dim", "청크수"]
    display_cols += [f"Hit@{k}" for k in ks]
    display_cols += [f"MRR@{hi}", f"nDCG@{hi}"]
    display_cols += [f"정답@{k}" for k in ks]
    display_cols += ["VRAM_MB", "index_MB", "chunks_per_s", "query_ms"]
    view = df[[c for c in display_cols if c in df.columns]]

    miss_report = build_miss_report(misses, questions, gold, ks)
    (results_dir / "misses.md").write_text(miss_report, encoding="utf-8")

    df.to_csv(results_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (results_dir / "details.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "summary.md").write_text(
        "# 임베딩 모델 비교 결과\n\n"
        f"- 청크 수: {len(chunks)}\n- 질문 수: {len(questions)}  (정답@k 는 이 중 몇 개를 맞혔는지)\n"
        f"- scope: {cfg['retrieval']['scope']}\n"
        f"- chunk_size / overlap: {cfg['chunking']['chunk_size']} / "
        f"{cfg['chunking']['chunk_overlap']}\n"
        f"- chunks_per_s: 워밍업 후 최소 {cfg['runtime'].get('speed_min_texts', 512)}개 텍스트를 "
        f"{cfg['runtime'].get('speed_repeat', 3)}회 인코딩한 최고 기록\n\n"
        + view.to_markdown(index=False)
        + "\n\n" + METRIC_NOTE
        + "\n<details><summary>hf_id 포함 전체 컬럼</summary>\n\n"
        + df.to_markdown(index=False)
        + "\n\n</details>\n\n---\n\n"
        + miss_report,
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print(view.to_string(index=False))
    print("=" * 110)
    print(f"질문 {len(questions)}개 / 청크 {len(chunks)}개 기준")
    print(METRIC_NOTE)

    # 틀린 문제 — 화면에는 요약만, 전체 목록은 misses.md
    print("── 틀린 문제 " + "─" * 60)
    for k in ks:
        sets = [set(m.get(k, [])) for m in misses.values() if m]
        common = sorted(set.intersection(*sets)) if sets else []
        print(f"  모든 모델이 @{k} 에서 틀림: {len(common)}개"
              + (f"  {', '.join(common)}" if common else ""))
    for model, m in misses.items():
        parts = [f"@{k} {len(m.get(k, []))}개" for k in ks]
        worst = m.get(max(ks), [])
        print(f"  {model:32s} " + " / ".join(parts)
              + (f"  →  {', '.join(worst)}" if worst else ""))

    print(f"\n결과 저장: {results_dir}/summary.csv, summary.md, misses.md, details.json")
    print("  · misses.md    질문별로 어느 모델이 틀렸는지 전체 목록")
    print("  · details.json 왜 틀렸는지 — 질문마다 실제로 뽑아온 상위 5개 청크")


if __name__ == "__main__":
    main()
