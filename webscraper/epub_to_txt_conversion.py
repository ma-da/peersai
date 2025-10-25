# webscraper/epub_to_txt_conversion.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import re

from ebooklib import epub
from bs4 import BeautifulSoup

import cleaning_utils as cu

# Keep short paragraphs that are likely prose; tune as needed
DEFAULT_MIN_PARAGRAPH_LEN = 30

def _extract_epub_items_in_spine(book: epub.EpubBook) -> Iterable[Tuple[str, bytes]]:
    """
    Yield (idref, content_bytes) in the order of the spine so chapters stay in reading order.
    Falls back to all ITEM_DOCUMENTs if spine is missing.
    """
    id_map = {item.get_id(): item for item in book.get_items() if item.get_type() == 9}
    spine = [i[0] if isinstance(i, tuple) else i for i in (book.spine or [])]
    seen = set()
    for idref in spine:
        it = id_map.get(idref)
        if it and idref not in seen:
            seen.add(idref)
            yield idref, it.get_body_content()
    # Fallback: include any remaining documents not referenced by spine
    for item in book.get_items():
        if item.get_type() == 9 and item.get_id() not in seen:
            yield item.get_id(), item.get_body_content()

def convert_epub_to_clean_text(
    epub_path: Path,
    min_paragraph_len: int = DEFAULT_MIN_PARAGRAPH_LEN,
) -> tuple[str, str] | None:
    """
    Returns (title, clean_text) or None if rejected by natural-language gates.
    """
    book = epub.read_epub(str(epub_path))
    title = (book.get_metadata('DC', 'title') or [[epub_path.stem]])[0][0] or ""
    paragraphs: list[str] = []

    for _id, html_bytes in _extract_epub_items_in_spine(book):
        # BeautifulSoup expects bytes/str; handle bytes explicitly
        soup = BeautifulSoup(html_bytes, "lxml")

        # Drop non-content elements commonly found in EPUB XHTML
        for tag in soup(["script", "style", "nav", "aside", "header", "footer", "figure", "svg", "img"]):
            tag.decompose()

        # Get visible text, keep paragraph boundaries
        text = soup.get_text(separator="\n", strip=True)

        # Normalize multiple newlines; split into candidate paragraphs
        text = re.sub(r"\n{2,}", "\n\n", text)
        candidates = [p.strip() for p in text.split("\n")]

        # Keep only substantive paragraphs (avoid captions, running heads, etc.)
        for p in candidates:
            if len(p) >= min_paragraph_len:
                paragraphs.append(p)

    # Join and run through the same cleaner used elsewhere
    raw_joined = "\n\n".join(paragraphs)

    # EPUB content is already HTML-derived; final passes:
    clean = cu.normalize_and_clean(raw_joined)

    # Accept/reject gates (same heuristics you use for HTML/PDF)
    if not cu.has_natural_language_run(clean) or cu.is_garbled(clean):
        return None

    return (title, clean)
