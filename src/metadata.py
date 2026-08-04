"""문서·청크 메타데이터 추출 (규칙 기반).

LLM 없이 정규식과 휴리스틱만 쓴다. 검색 후보를 거르는 데는 관여하지 않고
chunks.json 에 기록만 남긴다 — 결과를 볼 때 "이 청크가 문서 어디서 나온 건지"를
바로 알기 위한 것이다. 검색 필터로 쓰면 모델 간 비교가 오염되므로 일부러 분리해 뒀다.

단독 실행:
    python -m src.metadata           # data/ 문서들의 메타데이터를 뽑아 출력
    python -m src.metadata --json    # JSON 으로 출력
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .loaders import Block, load_documents

ROOT = Path(__file__).resolve().parent.parent


# ── 문서 종류 ──────────────────────────────────────────────────
# 공백을 전부 지운 텍스트에서 찾는다. 한국 공문서는 "고 소 장" 처럼 글자를
# 벌려 쓰는 경우가 많아 원문 그대로 매칭하면 놓친다.
DOC_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("보도자료", ("보도자료", "보도가능", "대변인실", "자료문의")),
    ("고소장", ("고소장", "피고소인", "고소취지", "고소사실")),
    ("고발장", ("고발장", "피고발인", "고발취지")),
    ("공소장", ("공소장", "공소사실", "적용법조")),
    ("판결문", ("청구취지", "변론종결", "판결을선고", "주문")),
    ("의견서", ("의견서", "변호인의견")),
    ("계약서", ("계약서", "계약당사자", "본계약")),
]

# ── 날짜 ───────────────────────────────────────────────────────
DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})"),
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
]
# "’26년 상반기" 처럼 연도만 두 자리로 쓴 경우
YEAR_ONLY_PATTERN = re.compile(r"[’'’]\s*(\d{2})\s*년")

# ── 발신 기관 ──────────────────────────────────────────────────
ORG_PATTERN = re.compile(
    r"(대검찰청"
    r"|[가-힣]{2,5}고등검찰청"
    r"|[가-힣]{2,5}지방검찰청(?:\s*[가-힣]{2,5}지청)?"
    r"|[가-힣]{2,5}지검(?:\s*[가-힣]{2,5}지청)?"
    r"|[가-힣]{2,5}고등법원"
    r"|[가-힣]{2,5}지방법원"
    r"|[가-힣]{2,6}경찰서"
    r"|[가-힣]{2,5}지방경찰청"
    r"|국가수사본부)"
)

# ── 제목 ───────────────────────────────────────────────────────
# 제목이 줄바꿈으로 끊기는 경우가 많아 넉넉히 잡은 뒤 _tidy_title 로 다듬는다.
TITLE_PATTERNS = [
    re.compile(r"제\s*목\s*[:：]?\s*(.{4,200})", re.DOTALL),
    re.compile(r"([「『][^」』]{4,80}[」』])"),
]

# ── 섹션(소제목) ───────────────────────────────────────────────
# 청크가 문서의 어느 대목에서 나왔는지 표시한다. 문서 성격에 따라 잘 안 맞으면
# 여기에 패턴을 추가하면 된다.
SECTION_PATTERNS = [
    re.compile(r"^\s*[▣■□◆◇○●▶▷]\s*(.{2,60}?)\s*$"),          # ▣ 소제목
    re.compile(r"^\s*(제\s*\d+\s*[조장절](?:\s*\([^)]{1,30}\))?)"),  # 제3조(목적)
    re.compile(r"^\s*([가-힣]{2,5}지검(?:\s+[가-힣]{2,5}지청)?\s+\S{2,10}부)\s*$"),
    re.compile(r"^\s*(\d{1,2}\.\s+\S.{1,48})\s*$"),               # 1. 사건의 개요
]


def _despace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def detect_doc_type(full_text: str) -> str:
    """키워드 적중 개수가 가장 많은 종류로 판정. 하나도 없으면 '기타'."""
    flat = _despace(full_text)
    best, best_hits = "기타", 0
    for label, keywords in DOC_TYPE_RULES:
        hits = sum(1 for kw in keywords if kw in flat)
        if hits > best_hits:
            best, best_hits = label, hits
    return best


def extract_date(text: str) -> str | None:
    """가장 먼저 나오는 완전한 날짜를 ISO(YYYY-MM-DD)로. 없으면 연도만이라도."""
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            year, month, day = (int(g) for g in m.groups())
            if 1900 <= year <= 2200 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
    m = YEAR_ONLY_PATTERN.search(text)
    if m:
        return f"20{m.group(1)}"  # '26 → 2026
    return None


def extract_org(blocks: list[Block]) -> str | None:
    """머리말에 나온 기관을 우선하고, 없으면 문서 전체에서 최빈값을 쓴다."""
    if not blocks:
        return None
    head = ORG_PATTERN.search(blocks[0].text)
    if head:
        return re.sub(r"\s+", " ", head.group(1)).strip()

    counts = Counter(
        re.sub(r"\s+", " ", m.group(1)).strip()
        for b in blocks
        for m in ORG_PATTERN.finditer(b.text)
    )
    return counts.most_common(1)[0][0] if counts else None


def _tidy_title(raw: str) -> str | None:
    """넉넉히 잡아온 문자열에서 제목만 남긴다.

    '제 목' 뒤 내용은 줄바꿈으로 끊기거나("… 우수부서」 및") 본문까지 딸려온다.
    「」 가 있으면 마지막 닫는 괄호까지 남기고, 그 뒤에 붙는 짧은 서술어
    ("선정", "발표")만 한 단어 더 붙인다.
    """
    title = re.sub(r"\s+", " ", raw).strip()
    end = title.rfind("」")
    if end != -1:
        tail = re.match(r"\s*([가-힣]{2,6})", title[end + 1 :])
        title = title[: end + 1] + (f" {tail.group(1)}" if tail else "")
    else:
        title = title.splitlines()[0] if "\n" in raw else title
    title = title[:120].strip()
    return title if len(title) >= 3 else None


def extract_title(blocks: list[Block]) -> str | None:
    """'제 목' 뒤 → 「」 안 → 첫 줄 순으로 시도."""
    if not blocks:
        return None
    head = "\n".join(b.text for b in blocks[:2])
    for pattern in TITLE_PATTERNS:
        m = pattern.search(head)
        if m:
            title = _tidy_title(m.group(1))
            if title:
                return title

    first_line = blocks[0].text.splitlines()[0].strip() if blocks[0].text else ""
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line[:120] if len(first_line) >= 3 else None


def _is_sentence(text: str) -> bool:
    """소제목이 아니라 문장인지. 고소장 양식의 '□ …없습니다.' 같은 체크박스 줄을 거른다."""
    if re.search(r"(니다|습니다|였다|한다|이다)\.?$", text):
        return True  # 종결어미로 끝나는 건 길이와 무관하게 문장이다
    return len(text) > 15 and text.endswith(".")


# "1. 차용증 사본 1부" 처럼 수량으로 끝나는 건 증거목록 항목이지 소제목이 아니다
LIST_ITEM_TAIL = re.compile(r"\d+\s*(부|통|장|매|개|건|점)$")


def detect_section(line: str) -> str | None:
    """한 줄이 소제목처럼 생겼으면 그 소제목을 반환."""
    for pattern in SECTION_PATTERNS:
        m = pattern.match(line)
        if m:
            found = re.sub(r"\s+", " ", m.group(1)).strip()
            if _is_sentence(found) or LIST_ITEM_TAIL.search(found):
                return None
            return found
    return None


def sections_in(text: str) -> tuple[str | None, str | None]:
    """블록 안의 (첫 줄 소제목, 마지막 소제목).

    PDF 는 한 페이지가 통째로 한 블록이라 소제목이 블록 중간에 묻힌다.
    첫 줄만 보면 못 찾으므로 모든 줄을 훑는다.
    """
    lines = text.splitlines()
    if not lines:
        return None, None
    first = detect_section(lines[0])
    found = [s for s in (detect_section(line) for line in lines) if s]
    return first, (found[-1] if found else None)


def extract_doc_meta(blocks: list[Block]) -> dict:
    """문서 하나(같은 source 블록들) → 문서 수준 메타데이터."""
    full_text = "\n".join(b.text for b in blocks)
    return {
        "doc_type": detect_doc_type(full_text),
        "기관": extract_org(blocks),
        "작성일": extract_date(full_text),
        "제목": extract_title(blocks),
    }


def attach_metadata(blocks: list[Block]) -> dict[str, dict]:
    """모든 블록에 meta 를 채우고, 문서별 메타데이터 요약을 반환한다.

    섹션은 앞에서부터 흘려 채운다(forward fill). 소제목 블록을 만나면 그 이후
    블록들은 다음 소제목이 나올 때까지 그 섹션에 속한 것으로 본다.
    """
    groups: dict[str, list[Block]] = {}
    for b in blocks:
        groups.setdefault(b.source, []).append(b)

    report: dict[str, dict] = {}
    for source, group in groups.items():
        doc_meta = extract_doc_meta(group)

        section: str | None = None
        section_counts: Counter[str] = Counter()
        for b in group:
            first, last = sections_in(b.text)
            # 블록이 소제목으로 시작하면 그 블록은 새 섹션 소속,
            # 아니면 직전까지 이어지던 섹션을 물려받는다.
            b.meta = {**doc_meta, "섹션": first or section}
            if last:
                section = last
            if b.meta["섹션"]:
                section_counts[b.meta["섹션"]] += 1

        report[source] = {
            **doc_meta,
            "블록수": len(group),
            "글자수": sum(len(b.text) for b in group),
            "섹션수": len(section_counts),
            "섹션": list(section_counts),
        }
    return report


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description="data/ 문서의 메타데이터를 규칙 기반으로 추출")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--json", action="store_true", help="JSON 으로 출력")
    args = ap.parse_args()

    blocks = load_documents(Path(args.data_dir))
    report = attach_metadata(blocks)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    for source, meta in report.items():
        print(f"\n── {source}")
        print(f"  문서종류 : {meta['doc_type']}")
        print(f"  기관     : {meta['기관'] or '(못 찾음)'}")
        print(f"  작성일   : {meta['작성일'] or '(못 찾음)'}")
        print(f"  제목     : {meta['제목'] or '(못 찾음)'}")
        print(f"  규모     : {meta['블록수']} blocks / {meta['글자수']:,}자")
        print(f"  섹션     : {meta['섹션수']}개")
        for name in meta["섹션"][:10]:
            print(f"             · {name}")
        if meta["섹션수"] > 10:
            print(f"             … 외 {meta['섹션수'] - 10}개")

    missing = [s for s, m in report.items() if not m["기관"] or not m["작성일"]]
    if missing:
        print(f"\n[참고] 일부 항목을 못 찾은 문서: {', '.join(missing)}")
        print("       src/metadata.py 상단의 정규식 패턴을 문서 형식에 맞게 추가하세요.")


if __name__ == "__main__":
    main()
