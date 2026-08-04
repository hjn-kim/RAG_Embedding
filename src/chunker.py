"""공통 청킹. 모든 모델이 '똑같은' 청크를 쓰게 하는 것이 공정 비교의 전제다."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .loaders import Block

# 큰 단위부터 차례로 시도하는 분할 기준
SEPARATORS = ["\n\n", "\n", ". ", "다. ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    locator: str
    meta: dict = field(default_factory=dict)  # 부모 블록에서 물려받는다

    def to_dict(self) -> dict:
        return asdict(self)


def _split_recursive(text: str, chunk_size: int, seps: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not seps:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    pieces: list[str] = []
    buf = ""
    for i, part in enumerate(parts):
        candidate = part if i == len(parts) - 1 else part + sep
        if len(buf) + len(candidate) <= chunk_size:
            buf += candidate
        else:
            if buf:
                pieces.append(buf)
            if len(candidate) > chunk_size:
                pieces.extend(_split_recursive(candidate, chunk_size, rest))
                buf = ""
            else:
                buf = candidate
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        out.append(tail + cur)
    return out


def chunk_blocks(
    blocks: list[Block],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_chars: int = 30,
) -> list[Chunk]:
    """Block 리스트 → Chunk 리스트.

    같은 source 안에서 인접 블록을 chunk_size 까지 이어붙인 뒤,
    넘치면 재귀 분할하고 overlap 을 적용한다.
    """
    chunks: list[Chunk] = []
    next_id = 0

    # source 별로 순서를 유지한 채 그룹핑
    groups: dict[str, list[Block]] = {}
    for b in blocks:
        groups.setdefault(b.source, []).append(b)

    for source, group in groups.items():
        buf_text, buf_locators, buf_metas = "", [], []

        def flush() -> None:
            nonlocal buf_text, buf_locators, buf_metas, next_id
            if not buf_text.strip():
                buf_text, buf_locators, buf_metas = "", [], []
                return
            pieces = _split_recursive(buf_text.strip(), chunk_size, SEPARATORS)
            pieces = _apply_overlap(pieces, chunk_overlap)
            locator = (
                buf_locators[0]
                if len(set(buf_locators)) == 1
                else f"{buf_locators[0]}~{buf_locators[-1]}"
            )
            # 여러 블록이 합쳐졌으면 첫 블록의 meta 를 대표로 쓴다.
            # (섹션은 앞에서부터 흘려 채워지므로 첫 블록 것이 그 덩어리의 섹션이다)
            meta = dict(buf_metas[0]) if buf_metas else {}
            for piece in pieces:
                piece = re.sub(r"\s+", " ", piece).strip()
                if len(piece) < min_chunk_chars:
                    continue
                chunks.append(Chunk(next_id, piece, source, locator, dict(meta)))
                next_id += 1
            buf_text, buf_locators, buf_metas = "", [], []

        for b in group:
            if len(buf_text) + len(b.text) + 1 > chunk_size and buf_text:
                flush()
            buf_text = f"{buf_text}\n{b.text}" if buf_text else b.text
            buf_locators.append(b.locator)
            buf_metas.append(b.meta)
            if len(buf_text) >= chunk_size:
                flush()
        flush()

    return chunks
