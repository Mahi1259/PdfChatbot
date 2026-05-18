"""PDF text extraction and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber


class PDFExtractionError(Exception):
    pass


@dataclass
class Chunk:
    text: str
    page: int
    index: int


def extract_text_by_page(pdf_file: BinaryIO) -> list[str]:
    pages: list[str] = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            parts: list[str] = []

            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)

            for table in page.extract_tables() or []:
                rendered_rows: list[str] = []
                for row in table:
                    cells = [(cell or "").strip() for cell in row]
                    if any(cells):
                        rendered_rows.append(" | ".join(cells))
                if rendered_rows:
                    parts.append("\n".join(rendered_rows))

            pages.append("\n\n".join(parts).strip())

    if not any(p.strip() for p in pages):
        raise PDFExtractionError(
            "No extractable text found. The PDF may be scanned/image-based — OCR is required."
        )
    return pages


def chunk_text(
    pages: list[str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    counter = 0

    for page_num, page_text in enumerate(pages, start=1):
        text = " ".join(page_text.split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, index=counter))
                counter += 1
            if end == len(text):
                break
            start += step

    return chunks


def process_pdf(
    pdf_file: BinaryIO,
    chunk_size: int = 500,
    overlap: int = 50,
) -> tuple[list[Chunk], int]:
    pages = extract_text_by_page(pdf_file)
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
    return chunks, len(pages)
