#!/usr/bin/env python3
"""
High-fidelity PDF → clean text extractor for LLM training.

Features:
- Hybrid extraction pipeline (text layer → pdfminer → OCR fallback)
- Adaptive paragraph reconstruction using statistical line-joining
- Header/footer removal using cross-page frequency heuristics
- Unicode normalization & OCR noise cleanup
- Optional JSONL output for large-scale training prep
"""

import os
import re
import json
import argparse
import unicodedata
from collections import Counter, defaultdict

from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract

# -------------------------------------------------------------
#  Text Layer Extraction
# -------------------------------------------------------------

def extract_with_text_layer(pdf_path):
    """Extract text using pdfminer (works when PDF has a real text layer)."""
    try:
        text = extract_text(pdf_path)
        if text and len(text.strip()) > 25:
            return text
    except Exception:
        pass
    return None


# -------------------------------------------------------------
#  OCR Extraction Fallback
# -------------------------------------------------------------

def extract_with_ocr(pdf_path):
    """Fallback OCR using Tesseract."""
    pages = convert_from_path(pdf_path, dpi=300)
    text_pages = []

    for i, page in enumerate(pages):
        try:
            txt = pytesseract.image_to_string(page)
            text_pages.append(txt)
        except Exception as e:
            text_pages.append("")
    return "\n\n".join(text_pages)


# -------------------------------------------------------------
#  Header/Footer Removal Heuristics
# -------------------------------------------------------------

def detect_repeating_lines(pages, max_line_len=80, min_freq=0.25):
    """
    Identify lines that appear frequently across pages (typical headers/footers).
    - max_line_len: avoid capturing long real content
    - min_freq: if a line appears in >25% of pages, treat as boilerplate
    """
    counter = Counter()

    for page in pages:
        for line in page.split("\n"):
            striped = line.strip()
            if 0 < len(striped) <= max_line_len:
                counter[striped] += 1

    threshold = max(2, int(len(pages) * min_freq))
    boilerplate = {line for line, count in counter.items() if count >= threshold}
    return boilerplate


def remove_headers_footers(pages, boilerplate):
    cleaned = []
    for page in pages:
        new = "\n".join(
            line for line in page.split("\n")
            if line.strip() not in boilerplate
        )
        cleaned.append(new)
    return cleaned


# -------------------------------------------------------------
#  Paragraph Reconstruction (Smart Line Joining)
# -------------------------------------------------------------

def join_lines_smart(text):
    """
    Join broken PDF lines into clean paragraphs.
    Uses a statistical heuristic:
    - If line ends without punctuation and next line is lowercase → join.
    - If line is unusually short compared to median → join (likely wrapped).
    """
    lines = [l.rstrip() for l in text.split("\n")]

    lens = [len(l) for l in lines if len(l) > 0]
    median_len = sorted(lens)[len(lens) // 2] if lens else 80

    out = []
    buff = ""

    for i in range(len(lines)):
        l = lines[i].strip()

        if not l:
            if buff:
                out.append(buff)
            buff = ""
            continue

        if not buff:
            buff = l
            continue

        join_condition = (
            (not buff.endswith((".", "!", "?", ":", ";"))) and
            (len(l) < median_len * 0.8 or l[:1].islower())
        )

        if join_condition:
            buff += " " + l
        else:
            out.append(buff)
            buff = l

    if buff:
        out.append(buff)

    return "\n\n".join(out)


# -------------------------------------------------------------
#  Unicode Normalization & Cleanup
# -------------------------------------------------------------

def normalize_text(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = t.replace("\uf0b7", "•")  # common OCR bullet fix
    return t.strip()


# -------------------------------------------------------------
#  Main Pipeline
# -------------------------------------------------------------

def clean_pdf(pdf_path):
    # 1. Try text-layer extraction
    text = extract_with_text_layer(pdf_path)

    # 2. OCR if text layer missing
    if not text or len(text.strip()) < 50:
        text = extract_with_ocr(pdf_path)

    # 3. Split pages for boilerplate analysis
    pages = text.split("\f") if "\f" in text else text.split("\n\n\n")

    # 4. Identify frequent headers/footers
    boiler = detect_repeating_lines(pages)

    # 5. Remove them
    pages = remove_headers_footers(pages, boiler)

    # 6. Re-join pages
    text = "\n\n".join(pages)

    # 7. Smart paragraph reconstruction
    text = join_lines_smart(text)

    # 8. Unicode normalization & noise cleanup
    text = normalize_text(text)

    return text


# -------------------------------------------------------------
#  CLI
# -------------------------------------------------------------

def save_output(out_path, content):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF file to clean")
    parser.add_argument("-o", "--out", help="Output TXT path")
    parser.add_argument("--jsonl", action="store_true", help="Also output JSONL format")

    args = parser.parse_args()

    cleaned = clean_pdf(args.pdf)

    out_path = args.out or (os.path.splitext(args.pdf)[0] + ".clean.txt")
    save_output(out_path, cleaned)

    if args.jsonl:
        jsonl_path = out_path.replace(".txt", ".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for paragraph in cleaned.split("\n\n"):
                if paragraph.strip():
                    f.write(json.dumps({"text": paragraph.strip()}, ensure_ascii=False) + "\n")

    print(f"Cleaned text written to: {out_path}")


if __name__ == "__main__":
    main()
