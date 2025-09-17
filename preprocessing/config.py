import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import text as sklearn_text

#BASE = Path(r"C:/datasources")
BASE = Path(r"/Volumes/SSK/peers_dev/corpus")
ARTICLES_JSON = BASE / "ai" / "articles_df.json"

#INPUT_DIR     = BASE / "ai_corpus_slimmer/corpus_beta_set_slim"   # <- change me
INPUT_DIR     = BASE / "wtk_beta_txt_corpus"   # <- change me

FILTERED_CORPUS_OUTPUT_DIR    = INPUT_DIR / "filtered_beta_corpus"  # <- change me

PREPROCESSING_OUT_DIR     = INPUT_DIR / "preprocessing_output"   # <- change me

#TEXT_DIR      = BASE / "ai_corpus_slimmer" / "ai_corpus_slimmer_clean"
TEXT_DIR       = FILTERED_CORPUS_OUTPUT_DIR

REJECT_DIR    = PREPROCESSING_OUT_DIR      / "rejected"
LOG_CSV       = FILTERED_CORPUS_OUTPUT_DIR / "clean_log.csv"
EXTS          = {".txt", ".html", ".htm"}

WORDS_IN_A_ROW_THRESHOLD = 60
ALPHA_TOKEN_MIN_FRACTION = 0.80   # in a 100-word window, ≥80% tokens should be alphabetic
MAX_NONASCII_FRACTION    = 0.20   # in a 100-word window, ≤20% chars non-ASCII
DRY_RUN = False  # True = do not write/move; just report

TRINEDAY_SOURCE_CANDIDATES_FILE = BASE / "ai" / "trineday_source_candidates.csv"
TRINEDAY_SOURCES_DF = pd.read_csv(TRINEDAY_SOURCE_CANDIDATES_FILE)

TRINEDAY_BASE_SHOP = "https://trineday.myshopify.com"
TRINEDAY_MAPPING_PATH = str(BASE / "ai/trineday_source_mapping.csv")

ARTICLES_CHUNK_MAPPING_PATH = str(BASE / "ai" / "articles_plus_chunks_mapped.csv")

OUT_CORPUS_LIST_PATH = str(PREPROCESSING_OUT_DIR / "beta_slim_final.csv")

ISBN_RE = re.compile(r"""
    (?<!\d)                     # not preceded by digit
    (?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX]  # ISBN-10/13-ish
    (?!\d)                      # not followed by digit
""", re.VERBOSE | re.IGNORECASE)

# --- Custom phrase removal (case-insensitive) ---
CUSTOM_PHRASES = [
    "Click here",
    "more along these lines",
    "About us",
]

# build one regex, ignore case
PHRASE_RE = re.compile("|".join(re.escape(p) for p in CUSTOM_PHRASES), flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # alpha tokens; allow simple apostrophes
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # non-printing control chars
NONPRINTABLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# --- Stopword set (scikit-learn English) ---
EN_STOP = sorted(w.lower() for w in sklearn_text.ENGLISH_STOP_WORDS)  # <- list, not set

# ---------- PATHS ----------
BETA_LIST_CSV_PATH  = BASE / "ai" / "beta_slim_final.csv"
TEXT_COL  = "__text_for_tfidf__"

# All artifacts go here:
PREPROCESSING_OUT_DIR   = BASE / "betaslim"
PREPROCESSING_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Individual files
OUT_VOCAB = PREPROCESSING_OUT_DIR / "vocabulary.json"
OUT_IDF   = PREPROCESSING_OUT_DIR / "idf.json"
OUT_STOP  = PREPROCESSING_OUT_DIR / "stopwords.json"     # (only if you also export stopwords)
OUT_CENTS = PREPROCESSING_OUT_DIR / "centroids.json"
OUT_INDEX = PREPROCESSING_OUT_DIR / "centroids_index.json"

VOCAB_FN  = OUT_VOCAB
IDF_FN    = OUT_IDF
CENTS_FN  = OUT_CENTS
INDEX_FN  = OUT_INDEX
STOP_FN   = OUT_STOP

# --------- KMEANS CONFIG ---------
CAP = 1500           # max items per final group
MIN_SPLIT = 2        # minimum k when we split a cluster
MAX_SPLIT = 10       # don't split any single cluster into more than this many at once
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300

# --------- IPFS CONFIG ---------
IPFS_FOLDER = INPUT_DIR
MANIFEST_PATH = IPFS_FOLDER / "betaslim_manifest.byfile.json"

IPFS_FILES_TO_PIN = [
    IPFS_FOLDER / "betaslim_manifest.byfile.json",
    IPFS_FOLDER / "groups_urls.json",
]

# 1) JWT (recommended)
PINATA_JWT = "your jwt"
HEADERS_AUTH = {"Authorization": f"Bearer {PINATA_JWT}"}

# 2) OR API key/secret:
# PINATA_API_KEY = "..."
# PINATA_API_SECRET = "..."
# HEADERS_AUTH = {"pinata_api_key": PINATA_API_KEY, "pinata_secret_api_key": PINATA_API_SECRET}

IPFS_DEDICATED = "https://peers.mypinata.cloud/ipfs/"
IPFS_CUSTOM    = "https://ai.peerservice.org/ipfs/"

# Optional: basic retry settings
IPFS_MAX_RETRIES = 3
IPFS_BACKOFF_SEC = 2.0

HIVE_AUTHOR = "wanttoknow"
HIVE_PERMLINK = "seeds-of-truth-index-registration-0-1"

HIVE_RPC = "https://api.hive.blog"