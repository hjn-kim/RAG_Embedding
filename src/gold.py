"""질문 로드 + 정답(gold) 청크 자동 라벨링.

수작업으로 "정답 청크 번호"를 적는 건 청킹 설정을 바꿀 때마다 다시 해야 해서 비현실적이다.
대신 질문마다 '정답에 반드시 들어있는 문구'(must_include)를 적어두면,
그 문구를 포함한 청크를 자동으로 gold 로 라벨링한다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk


@dataclass
class Question:
    id: str
    question: str
    source: str | None = None       # 어느 문서에 답이 있는지 (scope=own_doc 일 때 사용)
    must_include: list[str] = None  # 전부 포함되어야 gold
    any_include: list[str] = None   # 하나라도 포함되면 gold
    must_exclude: list[str] = None  # 하나라도 있으면 gold 에서 제외 (must_include 의 반대)
    note: str = ""


# questions.json 에서 허용하는 키. 오타를 조용히 무시하지 않기 위해 검사한다.
ALLOWED_KEYS = {
    "id", "question", "source",
    "must_include", "any_include", "must_exclude",
    "note",
}


def _key(text: str) -> str:
    """매칭용 정규화: 한글 PDF는 띄어쓰기가 깨지는 경우가 많아 공백을 전부 제거한다."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", text)


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    questions: list[Question] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(items, start=1):
        qid = str(item.get("id", f"q{i}"))

        unknown = set(item) - ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"질문 {qid}: 알 수 없는 키 {sorted(unknown)}. "
                f"사용 가능: {sorted(ALLOWED_KEYS)}"
            )
        if qid in seen_ids:
            raise ValueError(f"질문 id 가 중복됩니다: {qid}")
        seen_ids.add(qid)

        questions.append(
            Question(
                id=qid,
                question=item["question"],
                source=item.get("source"),
                must_include=item.get("must_include") or [],
                any_include=item.get("any_include") or [],
                must_exclude=item.get("must_exclude") or [],
                note=item.get("note", ""),
            )
        )
    return questions


def match_context(text: str, phrases: list[str], width: int = 90) -> str | None:
    """청크에서 must_include/any_include 문구가 실제로 걸린 지점 주변을 잘라 반환.

    청크 앞부분만 보여주는 preview 로는 "왜 이게 정답 청크인지"를 확인할 수 없다.
    (예: 566자 청크의 290자 지점에 답이 있으면 preview 200자로는 안 보인다)

    매칭은 공백을 지운 문자열에서 하므로, 원문 위치로 되돌리기 위해
    '공백 제거 문자열의 i번째 글자 → 원문의 몇 번째 글자' 인덱스를 같이 만든다.
    """
    norm = unicodedata.normalize("NFKC", text)
    lowered = norm.lower()
    packed, back = [], []
    for i, ch in enumerate(lowered):
        if not ch.isspace():
            packed.append(ch)
            back.append(i)
    key = "".join(packed)

    for phrase in phrases:
        pk = _key(phrase)
        if not pk:
            continue
        pos = key.find(pk)
        if pos == -1:
            continue
        start, end = back[pos], back[min(pos + len(pk), len(back)) - 1] + 1
        left, right = max(0, start - width), min(len(norm), end + width)
        snippet = re.sub(r"\s+", " ", norm[left:right]).strip()
        return ("…" if left > 0 else "") + snippet + ("…" if right < len(norm) else "")
    return None


# gold 청크가 전체 코퍼스의 이 비율을 넘으면 조건이 너무 헐렁하다고 본다.
BROAD_RATIO = 0.25


def label_gold(
    questions: list[Question], chunks: list[Chunk]
) -> tuple[dict[str, set[int]], list[str], list[tuple[str, int]]]:
    """질문 id → 정답 청크 id 집합.

    반환: (gold, gold 를 못 찾은 질문 id, 조건이 너무 넓은 질문 [(id, gold 수)])
    """
    chunk_keys = [(c.id, _key(c.text), c.source) for c in chunks]
    gold: dict[str, set[int]] = {}
    unmatched: list[str] = []
    too_broad: list[tuple[str, int]] = []

    for q in questions:
        must = [_key(s) for s in q.must_include if s.strip()]
        any_ = [_key(s) for s in q.any_include if s.strip()]
        excl = [_key(s) for s in q.must_exclude if s.strip()]
        if not must and not any_:
            raise ValueError(
                f"질문 {q.id}: must_include 또는 any_include 중 하나는 반드시 필요합니다. "
                f"(must_exclude 는 단독으로 쓸 수 없습니다 — 제외 조건일 뿐이라 "
                f"'답이 없는 모든 청크'가 정답이 되어버립니다)"
            )

        matched = set()
        for cid, ckey, csource in chunk_keys:
            if q.source and csource != q.source:
                continue
            if must and not all(m in ckey for m in must):
                continue
            if any_ and not any(a in ckey for a in any_):
                continue
            if excl and any(e in ckey for e in excl):
                continue
            matched.add(cid)

        gold[q.id] = matched
        if not matched:
            unmatched.append(q.id)
        elif len(matched) > max(2, BROAD_RATIO * len(chunks)):
            too_broad.append((q.id, len(matched)))

    return gold, unmatched, too_broad
