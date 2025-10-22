# webscraper/cleaning_utils.py
import re, html, unicodedata
from bs4 import BeautifulSoup
from ftfy import fix_text
import chardet

# --- Tunables (can also be surfaced as CLI flags) ---
WORDS_IN_A_ROW_THRESHOLD = 60
ALPHA_TOKEN_MIN_FRACTION = 0.80
MAX_NONASCII_FRACTION    = 0.20
MIN_CHARS_FOR_DOC        = 400

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def detect_and_decode(raw_bytes: bytes) -> str:
    guess = chardet.detect(raw_bytes) or {}
    enc = guess.get("encoding") or "utf-8"
    try:
        return raw_bytes.decode(enc, errors="replace")
    except LookupError:
        return raw_bytes.decode("utf-8", errors="replace")

def strip_html_if_needed(text: str, *, force_html: bool=False) -> str:
    # Heuristic: treat content as HTML if it looks like HTML or caller forces it
    looks_like_html = ("<html" in text[:1000].lower()) or ("</p>" in text.lower()) or ("<body" in text.lower())
    if force_html or looks_like_html:
        soup = BeautifulSoup(text, "lxml")  # falls back to html.parser if lxml missing
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    return html.unescape(text)

def normalize_and_clean(text: str) -> str:
    # Fix mojibake / unicode wonkiness and normalize
    text = fix_text(text)
    text = unicodedata.normalize("NFC", text)

    # Remove control chars (keep basic whitespace)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t ") else " " for ch in text)

    # Light de-hyphenation from PDFs: word-\nword -> wordword; join wrapped lines
    text = re.sub(r"(\w)-\s*\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # normalize single newlines
    # Preserve paragraphs but collapse extra vertical space
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace intrusive punctuation runs with spaces; keep single sentence punctuation
    text = re.sub(r"[^\w\s'\.\,\;\:\!\?\-]", " ", text)         # remove odd symbols
    text = re.sub(r"[_\-+=~^`|\\/<>{}\[\]*#%$@]{2,}", " ", text) # collapse runs

    # Normalize whitespace to single spaces except paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def has_natural_language_run(clean_text: str,
                             window_size: int = WORDS_IN_A_ROW_THRESHOLD,
                             alpha_min_frac: float = ALPHA_TOKEN_MIN_FRACTION,
                             max_nonascii_frac: float = MAX_NONASCII_FRACTION) -> bool:
    tokens = _word_re.findall(clean_text)
    if len(tokens) < window_size:
        return False
    is_alpha = [t.isalpha() for t in tokens]
    token_nonascii_frac = [sum(ord(c) > 127 for c in t)/max(1,len(t)) for t in tokens]

    alpha_count = sum(is_alpha[:window_size])
    nonascii_avg = sum(token_nonascii_frac[:window_size]) / window_size
    if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
        return True

    for i in range(window_size, len(tokens)):
        alpha_count += is_alpha[i] - is_alpha[i - window_size]
        nonascii_avg += (token_nonascii_frac[i] - token_nonascii_frac[i - window_size]) / window_size
        if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
            return True
    return False

def is_garbled(clean_text: str, *, min_chars: int = MIN_CHARS_FOR_DOC) -> bool:
    if len(clean_text) < min_chars:
        return True
    nospace = clean_text.replace(" ", "")
    if not nospace:
        return True
    alpha = sum(c.isalpha() for c in nospace)
    return (alpha / len(nospace)) < 0.55

def final_clean_htmlish(raw_text: str) -> str:
    """Convenience: for HTML-ish sources—strip tags, then normalize/clean."""
    stripped = strip_html_if_needed(raw_text, force_html=True)
    return normalize_and_clean(stripped)

def final_clean_plain(raw_text: str) -> str:
    """Convenience: for already-plain sources (e.g., PDF text)."""
    return normalize_and_clean(raw_text)
