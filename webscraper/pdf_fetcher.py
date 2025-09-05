#!/usr/bin/env python3

"""This module provides utility functions related to PDF retrieval and text extraction.
"""
import requests
import config
import re
import statistics
from collections import Counter, defaultdict
import fitz  # PyMuPDF

def download_pdf(url, output_path):
    """Download a PDF file from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {output_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    title = ""
    try:
        # Open the PDF file
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            title = metadata.get("title")

            # Iterate through each page
            for page in doc:
                # Extract text from the page
                text += page.get_text()
        print("Text extracted successfully.")
    except Exception as e:
        print(f"Error extracting text: {e}")
    return title, text


def save_text_to_file(title, text, output_path):
    """Save extracted text to a file."""
    if not title:
        title = "no_title"

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(title + "\n\n" + text)
    print(f"Text saved to {output_path}")


DIGITS_RE = re.compile(r"^\d{1,4}$")
FOOTNOTE_MARK_RE = re.compile(r"^(\d+|[*†‡])\s")  # simple heuristic

def is_probable_page_number(text, bbox, page_w, page_h):
    text = text.strip()
    if not DIGITS_RE.match(text):
        return False
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    # near bottom or top and roughly centered
    centered = abs(mid_x - page_w/2) < page_w * 0.2
    top_or_bottom = (mid_y < page_h * 0.08) or (mid_y > page_h * 0.92)
    return centered and top_or_bottom

def learn_header_footer_bands(pages_dicts, n=5, band_frac=0.10):
    """Infer top/bottom bands and repeated lines in them from first n pages."""
    top_texts = Counter()
    bot_texts = Counter()
    top_positions = []
    bot_positions = []
    for p in pages_dicts[:n]:
        w, h = p["width"], p["height"]
        top_y = h * band_frac
        bot_y = h * (1 - band_frac)
        for b in p["blocks"]:
            y0 = b["bbox"][1]
            if y0 < top_y:
                s = normalize_inline_text(b)
                if s.strip():
                    top_texts[s.strip()] += 1
                    top_positions.append(y0)
            if y0 > bot_y:
                s = normalize_inline_text(b)
                if s.strip():
                    bot_texts[s.strip()] += 1
                    bot_positions.append(y0)
    # Repeated header/footer candidates: appear on majority of sampled pages
    min_count = max(2, n // 2)  # conservative
    headers = {t for t,c in top_texts.items() if c >= min_count}
    footers = {t for t,c in bot_texts.items() if c >= min_count}
    return headers, footers

def normalize_inline_text(block):
    # flatten lines->spans into simple text for signatures / duplicate matching
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return " ".join(" ".join(p.split()) for p in parts)

def page_font_stats(pdict):
    sizes = []
    for b in pdict["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if sp.get("size"):
                    sizes.append(sp["size"])
    median = statistics.median(sizes) if sizes else 10.0
    return median

def block_avg_font(block):
    sizes = []
    for ln in block.get("lines", []):
        for sp in ln.get("spans", []):
            if sp.get("size"):
                sizes.append(sp["size"])
    return (sum(sizes)/len(sizes)) if sizes else 0.0

def should_drop_block(block, page_w, page_h, med_font, headers, footers):
    text = normalize_inline_text(block).strip()
    if not text:
        return True
    # Drop obvious page numbers
    if is_probable_page_number(text, block["bbox"], page_w, page_h):
        return True
    # Drop repeated headers/footers we learned
    if text in headers or text in footers:
        return True
    # Drop small-font footnotes in bottom band or explicit footnote lines
    y0, y1 = block["bbox"][1], block["bbox"][3]
    avg_font = block_avg_font(block)
    is_bottom_band = y0 > page_h * 0.85
    looks_like_footnote = FOOTNOTE_MARK_RE.match(text) is not None
    much_smaller_font = avg_font and avg_font < 0.8 * med_font
    if is_bottom_band and (much_smaller_font or looks_like_footnote):
        return True
    return False

def sort_blocks_reading_order(blocks):
    # Sort by top y, then by left x (helps keep columns separate)
    return sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))

def lines_from_block(block):
    # Return a list of lines as strings from the block, preserving within-line order
    lines = []
    for ln in block.get("lines", []):
        spans = [sp.get("text", "") for sp in ln.get("spans", [])]
        s = "".join(spans)
        lines.append(s)
    return lines

def dehyphenate_and_reflow(lines):
    out = []
    buf = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            # paragraph break
            if buf:
                out.append(buf.strip())
                buf = ""
            continue
        if buf:
            # heuristic: if previous ends with hyphen, join without space
            if buf.endswith('-'):
                # common fix: remove the hyphen and join if next starts lowercase
                if line and line[:1].islower():
                    buf = buf[:-1] + line.lstrip()
                else:
                    # keep hyphen if next line clearly starts a proper noun or new sentence
                    buf = buf[:-1] + "-" + line.lstrip()
            else:
                # join with space unless current line looks like a new paragraph bullet/number
                if re.match(r"^(\s*[\-\u2022*]|\s*\d+[\.\)])\s", line):
                    out.append(buf.strip())
                    buf = line.strip()
                else:
                    buf += " " + line.strip()
        else:
            buf = line.strip()
    if buf:
        out.append(buf.strip())
    # also collapse double spaces
    out = [" ".join(p.split()) for p in out]
    return "\n\n".join(out)

def extract_clean_pdf_text(pdf_path, max_pages=None, learn_pages=5):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    title = metadata.get("title")

    pages = []
    for i, page in enumerate(doc):
        if (max_pages is not None) and (i >= max_pages):
            break
        # Prefer dict to get bbox, lines, spans, fonts
        # If your PyMuPDF supports flags, you can try:
        # page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES)
        pdict = page.get_text("dict")
        pdict["width"] = page.rect.width
        pdict["height"] = page.rect.height
        pages.append(pdict)

    headers, footers = learn_header_footer_bands(pages, n=min(learn_pages, len(pages)))

    all_blocks = []
    for p in pages:
        med_font = page_font_stats(p)
        filtered = []
        for b in p["blocks"]:
            if b.get("type", 0) != 0:  # 0 = text, skip images etc.
                continue
            if should_drop_block(b, p["width"], p["height"], med_font, headers, footers):
                continue
            filtered.append(b)
        all_blocks.extend(sort_blocks_reading_order(filtered))

    # Convert kept blocks -> lines, then reflow & dehyphenate
    lines = []
    for b in all_blocks:
        lines.extend(lines_from_block(b))
        # add a blank line between blocks to mark paragraph-ish boundary
        lines.append("")

    text = dehyphenate_and_reflow(lines)
    return title, text


# Example usage
if __name__ == "__main__":
    # URL of the PDF file
    pdf_url = config.TEST_PDF_URL

    # Local paths
    pdf_output_path = config.CORPUS_FOLDER_LOCATION + "sample.pdf"  # Path to save the downloaded PDF
    text_output_path = config.CORPUS_FOLDER_LOCATION + "output.txt"  # Path to save the extracted text

    # Step 1: Download the PDF
    download_pdf(pdf_url, pdf_output_path)

    # Step 2: Extract text from the PDF
    title, extracted_text = extract_text_from_pdf(pdf_output_path)

    # Step 3: Save the extracted text to a file
    save_text_to_file(title, extracted_text, text_output_path)

    # Print the extracted text (optional)
    print("Extracted Text:\n", extracted_text)
