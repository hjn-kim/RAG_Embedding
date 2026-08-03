"""문서 로더: .docx / .pdf 에서 텍스트와 메타데이터를 추출한다."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Block:
    """문서에서 뽑아낸 텍스트 한 덩어리 (문단 / 표 행 / 페이지)."""

    text: str
    source: str  # 파일명
    locator: str  # "p.3", "para.12", "table.2" 처럼 원문 위치


def normalize_text(text: str) -> str:
    """전각/호환 문자 정리 + 공백 정규화. 한글 PDF 추출물 정리에 필요."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t 　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_docx(path: Path) -> list[Block]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks: list[Block] = []

    # 본문 순서대로(문단/표 섞여 있음) 훑기 위해 body XML 요소를 직접 순회한다.
    para_i, table_i = 0, 0
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para_i += 1
            text = normalize_text(Paragraph(child, doc).text)
            if text:
                blocks.append(Block(text, path.name, f"para.{para_i}"))
        elif tag == "tbl":
            table_i += 1
            table = Table(child, doc)
            for row_i, row in enumerate(table.rows, start=1):
                cells = [normalize_text(c.text) for c in row.cells]
                # 같은 셀이 병합으로 중복되는 경우 제거
                deduped, seen = [], set()
                for c in cells:
                    if c and c not in seen:
                        seen.add(c)
                        deduped.append(c)
                line = " | ".join(deduped)
                if line:
                    blocks.append(
                        Block(line, path.name, f"table.{table_i}.row.{row_i}")
                    )

    return blocks


def load_pdf(path: Path) -> list[Block]:
    blocks: list[Block] = []

    try:
        import fitz  # PyMuPDF

        with fitz.open(str(path)) as doc:
            for page_i, page in enumerate(doc, start=1):
                text = normalize_text(page.get_text("text"))
                if text:
                    blocks.append(Block(text, path.name, f"p.{page_i}"))
        if blocks:
            return blocks
    except ImportError:
        pass

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for page_i, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            blocks.append(Block(text, path.name, f"p.{page_i}"))

    return blocks


def load_documents(data_dir: Path) -> list[Block]:
    """data_dir 안의 모든 .docx / .pdf 를 읽어 Block 리스트로 반환."""
    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.suffix.lower() in {".docx", ".pdf"} and not p.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(
            f"{data_dir} 안에 .docx 또는 .pdf 파일이 없습니다. 문서를 넣어주세요."
        )

    blocks: list[Block] = []
    for path in files:
        loaded = load_docx(path) if path.suffix.lower() == ".docx" else load_pdf(path)
        if not loaded:
            print(f"  [경고] {path.name} 에서 텍스트를 추출하지 못했습니다 "
                  f"(스캔 이미지 PDF일 수 있음 → OCR 필요).")
        print(f"  {path.name}: {len(loaded)} blocks")
        blocks.extend(loaded)

    return blocks
