"""
Extraction utilities:
- text_layer_validity: quick checks whether the PDF has a usable text layer
- extract_poppler: wrapper for `pdftotext` (if installed)
- extract_pdfminer: uses pdfminer.six to extract with positions
- ocr_page: uses pytesseract to OCR a PIL image of a page
- hybrid_extract: orchestration that chooses text-layer vs OCR per page
Outputs an intermediate JSON structure with metadata per page.
"""
import shutil
import subprocess
import tempfile
import os
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

def text_layer_validity(pdf_path, max_pages=5):
    """Quick heuristic: extract text using pdfminer for first N pages and measure printable density."""
    text = []
    try:
        for i, page_layout in enumerate(extract_pages(pdf_path, laparams=LAParams()), start=1):
            page_text = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text.append(element.get_text())
            text.append("\n".join(page_text))
            if i >= max_pages:
                break
    except Exception:
        return False
    joined = "\n".join(text)
    # Measure ratio of printable to total characters
    printable = sum(1 for c in joined if c.isprintable())
    total = max(1, len(joined))
    density = printable / total
    # Also check for minimum text length
    return (density > 0.6) and (len(joined) > 200)

def extract_poppler(pdftotext_path, pdf_path, page=None, encoding='utf-8'):
    """Use poppler's pdftotext if available. `page` is 1-indexed."""
    if not shutil.which(pdftotext_path):
        raise FileNotFoundError(f"{pdftotext_path} not found on PATH.")
    cmd = [pdftotext_path, '-enc', encoding, pdf_path, '-']
    if page is not None:
        cmd = [pdftotext_path, '-f', str(page), '-l', str(page), '-enc', encoding, pdf_path, '-']
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode('utf-8', errors='replace'))
    return out.stdout.decode(encoding, errors='replace')

def extract_pdfminer(pdf_path, page_numbers=None):
    """Extract text plus basic positional metadata using pdfminer.six for selected pages (1-indexed list)."""
    results = []
    for i, page_layout in enumerate(extract_pages(pdf_path, laparams=LAParams()), start=1):
        if page_numbers and i not in page_numbers:
            continue
        page_lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                # basic cleaning
                page_lines.append({
                    'text': text,
                    'x0': getattr(element, 'x0', None),
                    'y0': getattr(element, 'y0', None),
                    'x1': getattr(element, 'x1', None),
                    'y1': getattr(element, 'y1', None),
                })
        results.append({'page': i, 'blocks': page_lines})
    return results

def ocr_page_image(pil_image, lang=None, oem=3, psm=1):
    """OCR a PIL image using pytesseract, returning text and a confidence heuristic."""
    config = f"--oem {oem} --psm {psm}"
    if lang:
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        # confidence metrics require tsv output; keep simple
    else:
        text = pytesseract.image_to_string(pil_image, config=config)
    # crude confidence proxy: length of text / image size
    w, h = pil_image.size
    density = len(text) / (w*h + 1)
    return {'text': text, 'density': density}

def hybrid_extract(pdf_path, output_json_path, pdftotext_path='pdftotext', ocr_lang=None):
    """Main orchestration. Produces intermediate JSON where each page has:
       - source: 'text' or 'ocr'
       - text
       - meta: bounding boxes or simple confidence
    """
    pages_out = []
    # decide on text-layer validity
    has_text = text_layer_validity(pdf_path)
    # load images once if OCR fallback is needed
    images = None
    if not has_text:
        images = convert_from_path(pdf_path, dpi=300)
    # Try extracting with poppler if available and text layer detected
    use_poppler = shutil.which(pdftotext_path) is not None and has_text
    try:
        if use_poppler:
            # extract whole document as text and split by form feeds
            txt = extract_poppler(pdftotext_path, pdf_path)
            pages = txt.split('\f')
            for i, page_text in enumerate(pages, start=1):
                pages_out.append({'page': i, 'source': 'text', 'text': page_text.strip(), 'confidence': 1.0})
        elif has_text:
            # fallback to pdfminer structured extraction
            pages = extract_pdfminer(pdf_path)
            for p in pages:
                joined = "\n".join(b['text'] for b in p['blocks'])
                pages_out.append({'page': p['page'], 'source': 'text', 'text': joined.strip(), 'confidence': 1.0})
        else:
            for i, img in enumerate(images, start=1):
                ocr_res = ocr_page_image(img, lang=ocr_lang)
                pages_out.append({'page': i, 'source': 'ocr', 'text': ocr_res['text'].strip(), 'confidence': ocr_res['density']})
    finally:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump({'pdf_path': pdf_path, 'pages': pages_out}, f, ensure_ascii=False, indent=2)
    return output_json_path
