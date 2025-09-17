import hashlib

import config
from config import *
from pathlib import Path
import json, re, unicodedata
import numpy as np
import chardet
import html
from ftfy import fix_text
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from typing import Optional
import tldextract

def read_text_with_detection(p: Path) -> str:
    raw = p.read_bytes()
    guess = chardet.detect(raw) or {}
    enc = guess.get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")

def strip_html(text: str, ext: str) -> str:
    # Heuristic: if looks like HTML or extension is html/htm -> parse
    looks_like_html = ("<html" in text[:1000].lower()) or ("</p>" in text.lower()) or ("<body" in text.lower())
    if ext in {".html", ".htm"} or looks_like_html:
        soup = BeautifulSoup(text, "lxml")  # fallbacks to html.parser if lxml missing
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    # Unescape entities
    return html.unescape(text)

def normalize_and_clean(text: str) -> str:
    # Fix mojibake and weird unicode
    text = fix_text(text)
    # Normalize to NFC
    text = unicodedata.normalize("NFC", text)
    # Remove control / non-printing (keep basic whitespace)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t ") else " " for ch in text)
    # Replace intraword punctuation like "securIty^^^>^^" with spaces around runs
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse runs of underscores/dashes/etc. to a space
    text = re.sub(r"[_\-+=~^`|\\/<>{}\[\]()*#%$@:;.,!?]{2,}", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def has_natural_language_run(clean_text: str,
                             window_size: int = WORDS_IN_A_ROW_THRESHOLD,
                             alpha_min_frac: float = ALPHA_TOKEN_MIN_FRACTION,
                             max_nonascii_frac: float = MAX_NONASCII_FRACTION) -> bool:
    # Tokenize to words
    tokens = WORD_RE.findall(clean_text)
    if len(tokens) < window_size:
        return False

    # Precompute for sliding window: mark alpha-only tokens
    is_alpha = [t.isalpha() for t in tokens]

    # For non-ASCII fraction per window, work on character slices:
    # Make a quick index of cumulative counts to avoid O(n^2).
    chars = clean_text
    # To approximate per-window non-ASCII, we map token indices to char spans.
    # Simpler heuristic: compute on token strings themselves.
    token_nonascii_frac = [sum(ord(c) > 127 for c in t)/max(1,len(t)) for t in tokens]

    # Slide 100-word window
    alpha_count = sum(is_alpha[:window_size])
    nonascii_avg = sum(token_nonascii_frac[:window_size]) / window_size

    if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
        return True

    for i in range(window_size, len(tokens)):
        # remove left, add right
        alpha_count += is_alpha[i] - is_alpha[i - window_size]
        nonascii_avg += (token_nonascii_frac[i] - token_nonascii_frac[i - window_size]) / window_size
        if alpha_count / window_size >= alpha_min_frac and nonascii_avg <= max_nonascii_frac:
            return True

    return False

def is_garbled(clean_text: str) -> bool:
    # Reject if too short overall
    if len(clean_text) < 400:  # ~a few sentences
        return True
    # Ratio of letters to all non-space chars; if extremely low, it's junk
    nospace = clean_text.replace(" ", "")
    if not nospace:
        return True
    alpha = sum(c.isalpha() for c in nospace)
    if alpha / len(nospace) < 0.55:
        return True
    return False

def relative_output_path(src: Path) -> Path:
    rel = src.relative_to(INPUT_DIR)
    return FILTERED_CORPUS_OUTPUT_DIR / rel.with_suffix(".txt")  # save everything as clean .txt

# --- Extract registered domain (SLD + TLD) ---
def extract_domain(src: str) -> str:
    if not isinstance(src, str) or not src:
        return None
    ext = tldextract.extract(src)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return None

def _tokenize_like_vectorizer(s: str):
    # Apply SAME preprocessor first (phrase stripping, etc.)
    s = preprocessor(s).lower()
    # Tokenize
    toks = TOKEN_RE.findall(s)
    # Remove stopwords
    toks = [t for t in toks if t not in EN_STOP]
    return toks

def tfidf_query_vector(inv_vocab, idf_vals, tok2idx, query: str) -> np.ndarray:
    """Return unit-norm TF-IDF vector aligned with vocabulary indices."""
    toks = _tokenize_like_vectorizer(query)
    if not toks:
        return np.zeros(len(inv_vocab), dtype=np.float32)
    # term frequency over known vocab
    tf = {}
    for t in toks:
        i = tok2idx.get(t)
        if i is not None:
            tf[i] = tf.get(i, 0) + 1
    if not tf:
        return np.zeros(len(inv_vocab), dtype=np.float32)
    # build dense vector and apply IDF (scikit's default normalization is L2 on TF-IDF)
    qv = np.zeros(len(inv_vocab), dtype=np.float32)
    for i, f in tf.items():
        qv[i] = f * idf_vals[i]
    # L2 normalize
    n = np.linalg.norm(qv) + 1e-12
    qv /= n
    return qv

def clean_text(s: str) -> str:
    s = NONPRINTABLE_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def chunk_by_words(text: str, chunk_words: int) -> list[str]:
    toks = text.split()
    return [" ".join(toks[i:i+chunk_words]) for i in range(0, len(toks), chunk_words) if toks]

def source_from_filename(path: Path) -> str:
    base = path.stem
    if any(tld in base for tld in (".com", ".org", ".info")):
        return base.replace("_", "/")
    return base

def looks_like_isbn(s: str) -> bool:
    return bool(ISBN_RE.search(s or ""))

def looks_like_trineday_sitb(s: str) -> bool:
    return isinstance(s, str) and "_SITB" in s

def looks_like_title(s: str) -> bool:
    if not isinstance(s, str): return False
    if any(c in s for c in "/\\?.#@:"):  # URL-ish punctuation
        return False
    # Heuristic: titles have spaces and letters
    return any(ch.isalpha() for ch in s) and " " in s

def normalize_isbn(src: str) -> Optional[str]:
    """Strip _SITB and non-digits, return ISBN if length 10 or 13."""
    if not isinstance(src, str):
        return None
    base = src.split("_SITB")[0]
    digits = re.sub(r"[^0-9Xx]", "", base)
    if len(digits) in (10, 13):
        return digits.upper()
    return None

def search_trineday(query: str) -> Optional[str]:
    """Return the first matching product URL from TrineDay search."""
    url = f"{TRINEDAY_BASE_SHOP}/search?q={quote(query)}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Look for product links
    for a in soup.select('a[href*="/products/"]'):
        href = a.get("href", "")
        if "/products/" in href:
            return TRINEDAY_BASE_SHOP + href if href.startswith("/") else href
    return None


# --- Extract registered domain (SLD + TLD) ---
def extract_domain(src: str) -> str:
    if not isinstance(src, str) or not src:
        return None
    ext = tldextract.extract(src)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return None

def preprocessor(text: str) -> str:
    """Strip custom phrases & control chars, collapse whitespace."""
    s = "" if text is None else str(text)
    s = PHRASE_RE.sub(" ", s)      # remove custom phrases
    s = CTRL_RE.sub(" ", s)        # strip control chars
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def float4(x: float) -> float:
    """Round to 4 decimals to shrink JSON size."""
    return float(f"{x:.4f}")

def preprocess_text(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    for r in config.PHRASE_RES:
        t = r.sub(" ", t)
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)  # control chars
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def tokenize(s: str):
    return TOKEN_RE.findall(preprocess_text(s))

# --- Build unit-norm TF-IDF for a query ---
def tfidf_query_vector_stopwords(stopwords, token_to_idx, idf, V, q: str) -> np.ndarray:
    toks = tokenize(q)
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    if not toks:
        return np.zeros(V, dtype=np.float32)
    tf = {}
    for t in toks:
        i = token_to_idx.get(t)
        if i is not None:
            tf[i] = tf.get(i, 0) + 1
    if not tf:
        return np.zeros(V, dtype=np.float32)
    qv = np.zeros(V, dtype=np.float32)
    for i, f in tf.items():
        qv[i] = f * idf[i]
    n = np.linalg.norm(qv) + 1e-12
    qv /= n
    return qv

# --- Centroid routing (cosine since both sides are L2-normalized) ---
def topk_centroids(C, qv: np.ndarray, k: int = 3):
    sims = C @ qv                      # (K,)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

# --- Search inside specific groups (use local JSON shards) ---
def search_in_groups(qv: np.ndarray, group_ids, manifest, top_docs: int = 5):
    results = []
    for gid in group_ids:
        group_name = manifest["groups"][gid]     # e.g., 'group_012.json'
        with open(PREPROCESSING_OUT_DIR / group_name, "r", encoding="utf-8") as f:
            rows = json.load(f)                  # each row has: tfidf [[idx, w]...], norm, title, source, text, row_id
        for r in rows:
            # sparse dot
            dot = 0.0
            for idx, w in r["tfidf"]:
                dot += w * qv[idx]
            sim = dot / (r["norm"])             # qv is unit-norm
            results.append((sim, gid, r))
    results.sort(key=lambda x: -x[0])
    return results[:top_docs]

# --- Convenience: end-to-end search ---
def search_json(C, query: str, k_centroids: int = 5, top_docs: int = 5):
    qv = tfidf_query_vector(query)
    if np.all(qv == 0):
        return "[]"
    centroids = topk_centroids(qv, k=k_centroids)
    gids = [cid for cid, _ in centroids]
    hits = search_in_groups(qv, gids, top_docs=top_docs)

    # Build list of {source, snippet}
    results = []
    for score, gid, r in hits:
        results.append({
            "source": r.get("source"),
            "snippet": (r.get("text") or "")[:240]
        })

    return json.dumps(results, indent=2, ensure_ascii=False)

def is_url_like(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    ext = tldextract.extract(s)
    return bool(ext.suffix)  # has a public suffix => looks URL-like

def rpc(url, method, params):
    r = requests.post(url, json={"jsonrpc":"2.0","id":1,"method":method,"params":params})
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["result"]
