#!/usr/bin/env python
"""
convert_to_clean_txt.py
Convert all PDFs and EPUBs in a directory to cleaned .txt files.
Handles digital, scanned, and EPUB sources.
"""

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --------------------------------------------------------------
# 1. INSTALL REQUIRED LIBS (once)
# --------------------------------------------------------------
# pip install pymupdf ebooklib beautifulsoup4 pdf2image pillow pytesseract tqdm

import fitz  # pymupdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import pytesseract

# --- English word list (top 5k) for quick lookup ---
# Download once: curl -L -o words.txt https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
WORD_SET = set()
try:
    with open("./words.txt", "r", encoding="utf-8") as f:
        WORD_SET = {w.strip().lower() for w in f if 3 <= len(w) <= 15}
except FileNotFoundError as e:
    raise FileNotFoundError("Could not find words set file")

# --------------------------------------------------------------
# 2. TEXT EXTRACTION FUNCTIONS
# --------------------------------------------------------------

def pdf_digital_to_text(pdf_path: Path) -> str:
    """Extract text from digital (searchable) PDFs using pymupdf."""
    try:
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                txt = b[4].strip()
                if txt:
                    text_blocks.append(txt)
        return "\n\n".join(text_blocks)
    except Exception as e:
        raise RuntimeError(f"Failed to read digital PDF {pdf_path}: {e}")

def pdf_scanned_to_text(pdf_path: Path, dpi: int = 300) -> str:
    """OCR scanned PDFs using Tesseract."""
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        text_pages = []
        for i, page in enumerate(pages):
            txt = pytesseract.image_to_string(page, lang='eng')
            text_pages.append(f"--- Page {i+1} ---\n{txt}")
        return "\n\n".join(text_pages)
    except Exception as e:
        raise RuntimeError(f"OCR failed on {pdf_path}: {e}")

def pdf_to_text(pdf_path: Path) -> str:
    """Auto-detect: digital or scanned."""
    try:
        # Quick check: if pymupdf finds text → digital
        doc = fitz.open(pdf_path)
        sample = doc[0].get_text()
        if sample.strip():
            return pdf_digital_to_text(pdf_path)
    except:
        pass
    # Fall back to OCR
    return pdf_scanned_to_text(pdf_path)

def epub_to_text(epub_path: Path) -> str:
    """Extract clean text from EPUB using ebooklib + BeautifulSoup."""
    try:
        book = epub.read_epub(str(epub_path))
        text_parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                # Remove scripts, styles, nav
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"Failed to read EPUB {epub_path}: {e}")

# --------------------------------------------------------------
# 3. UNIVERSAL CLEANING
# --------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove TOC, footnotes, page numbers, OCR garbage, citations."""
    lines = text.splitlines()
    cleaned = []
    in_toc = False

    for line in lines:
        l = line.strip()
        low = line.lower()

        # --- Skip TOC ---
        if re.match(r'^(table of contents|contents).*', low, re.I):
            in_toc = True
            continue
        if in_toc and l and not re.search(r'\d+\s*$', l):
            in_toc = False
        if in_toc:
            continue

        # --- Skip page numbers, headers ---
        if re.match(r'^\s*\d+\s*$', l):
            continue
        if re.match(r'^(chapter|section|part) \d+.*?\d*$', l, re.I):
            continue

        # --- Remove footnote markers ---
        l = re.sub(r'[\[\(]?\d+[\]\)]?', '', l)
        l = re.sub(r'[\u2070-\u209F]', '', l)  # superscript

        # --- Remove in-text citations ---
        l = re.sub(r'\((?:[A-Z][a-z]+(?: and [A-Z][a-z]+)?, )?\d{4}[a-z]?\)', '', l)

        # --- OCR fixes ---
        replacements = {
            'ß': 'ss', 'Gríßhma': 'Grishma', 'Varßha': 'Varsha',
            'y a M': 'May', 'lu J': 'Jul', 'tp e S': 'Sep', 'v o N': 'Nov',
            'ce D': 'Dec', 'b e F': 'Feb', 'r a M': 'Mar', 'n a J': 'Jan',
            'h ß o D': 'Dosha', 'et al u m u c c A': 'Accumulation',
            'et a v a r g g A': 'Aggravation', 'k a e w': 'weak',
            'n oits e gid': 'digestion', 'niar cidic a': 'acidic rain',
        }
        for bad, good in replacements.items():
            l = l.replace(bad, good)

        if l:
            cleaned.append(l)

    # --- Collapse whitespace ---
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def remove_repeating_headers(text: str, min_repeats: int = 3) -> str:
    """
    Detect and remove lines that appear on ≥ min_repeats pages.
    Works on page-separated OCR output.
    """
    # Split by page markers (from your OCR output: "--- Page X ---")
    page_marker = re.compile(r'--- Page \d+ ---')
    pages = page_marker.split(text)
    if len(pages) <= 1:
        return text  # no page breaks

    # Extract clean lines per page
    page_lines = []
    for page in pages[1:]:  # first split is empty
        lines = [l.strip() for l in page.splitlines() if l.strip()]
        page_lines.append(lines)

    # Count line frequency across pages
    from collections import Counter
    line_counts = Counter()
    for lines in page_lines:
        for line in lines:
            # Normalize: collapse spaces, ignore case for short lines
            norm = re.sub(r'\s+', ' ', line.upper())
            if len(norm) > 10:  # ignore short noise
                line_counts[norm] += 1

    # Identify repeating headers (appear on ≥ min_repeats pages)
    repeating = {orig for norm, count in line_counts.items()
                 if count >= min_repeats
                 for page in page_lines
                 for orig in page if re.sub(r'\s+', ' ', orig.upper()) == norm}

    # Rebuild text without repeating lines
    cleaned_pages = []
    for lines in page_lines:
        filtered = [l for l in lines if l not in repeating]
        if filtered:
            cleaned_pages.append('\n'.join(filtered))

    return '\n\n'.join(cleaned_pages)

def remove_ocr_junk(text: str) -> str:
    """
    Remove OCR noise: single chars, symbol spam, vertical fragments, etc.
    """
    lines = text.splitlines()
    cleaned = []
    seen_headers = set()  # for deduplication across pages

    # --- WHITELIST: short ALL-CAPS you want to keep ---
    WHITELIST = {
        'Q:', 'A:', 'YES', 'NO', 'FIG.', 'TAB.', 'REF.',
        'DOI:', 'ISBN:', 'URL:', 'HTTP:', 'HTTPS:'
    }

    # --- 1. Define junk patterns ---
    junk_patterns = [
        r'^[~—\-=_*%.¢«»]{2,}$',           # 2+ symbols only
        r'^[A-Za-z0-9]{1,2}$',            # 1–2 char "words"
        r'^[A-Za-z]{1,3}$',               # 1–3 letter fragments
        r'^[A-Z]{2,4}$',                  # ALL-CAPS acronyms <4 letters
        r'^[^\w\s]{1,4}$',                # 1–4 non-alphanumeric
        r'^\s*[~—\-=*%¢«»]+\s*$',         # lines of symbols
    ]
    junk_regex = re.compile('|'.join(junk_patterns))

    # --- 2. Filter line-by-line ---
    for line in lines:
        l = line.strip()

        # Skip obvious junk
        if junk_regex.match(l):
            continue

        # Skip lines with >50% non-alphanumeric (e.g., "Sy ~~ v")
        if sum(1 for c in l if not c.isalnum() and c not in ' ') / max(len(l), 1) > 0.5:
            continue

        # Skip lines with too many spaces (broken OCR)
        if l.count(' ') > len(l) * 0.6:
            continue

        # Skip ALL-CAPS lines (unless whitelisted) ---
        if l.isupper() and len(l) > 5:  # >5 to avoid "NO", "YES"
            if l not in WHITELIST:
                # Optional: keep first occurrence as section marker
                if l not in seen_headers:
                    seen_headers.add(l)
                    cleaned.append(f"### {l}")  # mark as heading
                continue

        # Keep meaningful lines
        if l:
            cleaned.append(l)

    # --- 3. Re-join with single blank lines ---
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_garbage_line(line: str, max_repeat_ratio: float = 0.4, min_entropy: float = 1.2) -> bool:
    """
    Return True if line is OCR garbage.
    Criteria:
      1. >40% repeated characters (e.g., 'nnnnn')
      2. <1.2 Shannon entropy (too uniform)
      3. No real English words
      4. >50% non-alphanumeric
      5. ALL-CAPS + >20 chars (random header)
    """
    l = line.strip()
    if not l or len(l) < 5:
        return True

    # 1. Repeated char ratio
    counts = collections.Counter(l.lower())
    max_repeat = max(counts.values(), default=0)
    if max_repeat / len(l) > max_repeat_ratio:
        return True

    # 2. Shannon entropy
    import math
    probs = [count / len(l) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    if entropy < min_entropy:
        return True

    # 3. No real words
    words = re.findall(r'[a-zA-Z]+', l.lower())
    real_words = sum(1 for w in words if w in WORD_SET)
    if real_words == 0 and len(words) > 2:
        return True

    # 4. High symbol ratio
    symbols = sum(1 for c in l if not c.isalnum() and c not in ' .,!?')
    if symbols / len(l) > 0.5:
        return True

    # 5. ALL-CAPS long garbage
    if l.isupper() and len(l) > 20:
        return True

    return False

def remove_garbage_lines(text: str) -> str:
    """Remove all garbage lines from OCR output."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if not is_garbage_line(line):
            cleaned.append(line.strip())
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
# --------------------------------------------------------------
# 4. SINGLE FILE PROCESSOR
# --------------------------------------------------------------
def process_file(input_path: Path, output_dir: Path) -> Path:
    """Convert one PDF/EPUB → cleaned .txt"""
    if input_path.suffix.lower() == '.pdf':
        raw_text = pdf_to_text(input_path)
    elif input_path.suffix.lower() == '.epub':
        raw_text = epub_to_text(input_path)
    else:
        raise ValueError(f"Unsupported file: {input_path}")

    # -- cleaning pipeline here --
    raw_text = remove_repeating_headers(raw_text)
    raw_text = remove_ocr_junk(raw_text)
    clean_text_content = clean_text(raw_text)

    output_path = output_dir / f"{input_path.stem}.txt"
    output_path.write_text(clean_text_content, encoding='utf-8')
    return output_path

# --------------------------------------------------------------
# 5. BATCH PROCESSOR
# --------------------------------------------------------------
def convert_directory(source_dir: str, target_dir: str, max_workers: int = 4):
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if not source.is_dir():
        raise ValueError(f"Source directory not found: {source}")

    files = list(source.glob("*.pdf")) + list(source.glob("*.epub"))
    if not files:
        print("No PDF or EPUB files found.")
        return

    print(f"Found {len(files)} files. Converting with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, target): f.name
            for f in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            src_name = futures[future]
            try:
                out_path = future.result()
                print(f"Success: {src_name} → {out_path.name}")
            except Exception as e:
                print(f"Failed: {src_name} | {e}")

# --------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs and EPUBs to cleaned .txt files."
    )
    parser.add_argument("source", help="Source directory with PDFs/EPUBs")
    parser.add_argument("target", help="Target directory for .txt outputs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    try:
        convert_directory(args.source, args.target, max_workers=args.workers)
        print(f"\nAll done! Cleaned files in: {args.target}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()