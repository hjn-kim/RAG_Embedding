"""질문을 만들기 전에 문서가 어떻게 청킹됐는지 눈으로 확인하는 헬퍼.

    python -m src.inspect_chunks              # 전체를 results/chunks_preview.txt 로 덤프
    python -m src.inspect_chunks --grep 계약   # 특정 단어가 든 청크만 보기
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .run_benchmark import ROOT, build_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--grep", default=None, help="이 문자열을 포함한 청크만 출력")
    ap.add_argument("--limit", type=int, default=0, help="출력할 최대 개수 (0=전체)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    chunks = build_corpus(cfg)

    if args.grep:
        needle = args.grep.replace(" ", "").lower()
        chunks = [c for c in chunks if needle in c.text.replace(" ", "").lower()]
        print(f"\n'{args.grep}' 포함 청크: {len(chunks)}개")
    if args.limit:
        chunks = chunks[: args.limit]

    lines = []
    for c in chunks:
        lines.append(f"[{c.id}] ({c.source} / {c.locator}) {len(c.text)}자")
        lines.append(c.text)
        lines.append("-" * 80)
    text = "\n".join(lines)

    out = ROOT / cfg["paths"]["results_dir"] / "chunks_preview.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text[:5000])
    print(f"\n(전체 {len(chunks)}개 → {out})")


if __name__ == "__main__":
    main()
