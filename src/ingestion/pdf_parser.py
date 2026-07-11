"""
AutoMind — PDF parsing.
Extracts raw text from owner-manual PDFs, page by page, preserving page
numbers as metadata. This lets the RAG system later cite "see page 42 of
the Corolla manual" — a nice touch for demo day / interviews.

Requires: pip install pymupdf
"""

import json
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF

from src.config import MANUALS_DIR, PROCESSED_DIR


def extract_pages(pdf_path: Path) -> List[Dict]:
    """
    Extracts text from every page of a PDF.

    Returns a list of dicts:
        {"page_number": int, "text": str, "source_file": str}
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue  # skip blank / image-only pages (common in manuals' diagram pages)

        pages.append({
            "page_number": page_num,
            "text": text,
            "source_file": pdf_path.name,
        })

    doc.close()
    return pages


def parse_all_manuals() -> Dict[str, List[Dict]]:
    """
    Parses every PDF found in data/manuals/ and saves raw page-level
    JSON to data/processed/<manual_name>_raw.json

    Returns a dict mapping filename -> list of page dicts, for immediate
    use by the chunker without re-reading from disk.
    """
    pdf_files = list(MANUALS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {MANUALS_DIR}. Drop your owner manual "
              f"PDFs there first (e.g. corolla_2023.pdf, civic_2022.pdf).")
        return {}

    all_results = {}
    for pdf_path in pdf_files:
        print(f"📄 Parsing {pdf_path.name} ...")
        pages = extract_pages(pdf_path)
        all_results[pdf_path.name] = pages

        out_path = PROCESSED_DIR / f"{pdf_path.stem}_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        print(f"   -> {len(pages)} pages extracted, saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parse_all_manuals()