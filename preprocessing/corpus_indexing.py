import math

from sklearn.feature_extraction.text import TfidfVectorizer

from config import *
from utils import *

def do_corpus_indexing():
    # --- Helpers ---
    try:
        articles_df = pd.read_json(ARTICLES_JSON)
    except ValueError:
        articles_df = pd.read_json(ARTICLES_JSON, lines=True)

    if "__text_for_tfidf__" not in articles_df.columns:
        raise ValueError("`__text_for_tfidf__` not found in articles_df.json")

    # --- Derive target chunk size (words) ---
    word_counts = articles_df["__text_for_tfidf__"].dropna().astype(str).map(lambda s: len(s.split()))
    target_words = int(np.clip(np.median(word_counts) if len(word_counts) else 480, 200, 900))
    print(f"Target chunk size (words): ~{target_words}")

    # --- ID assignment ---
    existing_max_id = (
        articles_df["ID"].max()
        if "ID" in articles_df.columns and pd.api.types.is_numeric_dtype(articles_df["ID"])
        else None
    )
    start_id = max(30001, int(existing_max_id) + 1) if existing_max_id is not None and math.isfinite(existing_max_id) else 30001
    next_id = start_id
    print(f"Starting new chunk IDs at: {start_id}")

    # --- Build new rows with the same columns as articles_df ---
    all_cols = list(articles_df.columns)  # preserve everything
    new_rows = []

    for p in sorted(TEXT_DIR.rglob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                txt = p.read_text(encoding="latin-1", errors="replace")
            except Exception:
                continue

        txt = clean_text(txt)
        if not txt:
            continue

        for chunk in chunk_by_words(txt, target_words):
            row = {col: None for col in all_cols}  # fill with None for missing fields
            row["ID"] = next_id
            row["Source"] = source_from_filename(p)
            row["__text_for_tfidf__"] = chunk
            new_rows.append(row)
            next_id += 1

    print(f"Created {len(new_rows)} chunk rows.")

    # --- Combine ---
    chunks_df = pd.DataFrame(new_rows, columns=all_cols)
    combined_df = pd.concat([articles_df, chunks_df], ignore_index=True)

    print(f"Final combined rows: {len(combined_df)}")
    combined_df.head(3)

    # --- Replace any '|' with 'I' in all string columns ---
    for col in combined_df.columns:
        if combined_df[col].dtype == object:
            combined_df[col] = combined_df[col].astype(str).str.replace("|", "I", regex=False)

    # --- Save to pipe-separated CSV ---
    OUT_PATH = BASE / "ai" / "beta_slim.csv"
    combined_df.to_csv(OUT_PATH, sep="|", index=False, encoding="utf-8")

    print(f"Saved combined dataframe to {OUT_PATH}")

    combined_df["Domain"] = combined_df["Source"].map(extract_domain)

    # --- Count by domain ---
    domain_counts = combined_df["Domain"].value_counts().head(20)

    # Find rows where Domain could not be extracted
    non_url_entries = combined_df[combined_df["Domain"].isna() | (combined_df["Domain"] == "")]

    print(f"Found {len(non_url_entries)} entries where Source is not a URL.")

    return combined_df


def process_trineday_corpus_and_generate_mapping(combined_df):
    candidates = combined_df["Source"].dropna().astype(str).unique().tolist()
    bookish = [s for s in candidates
               if not is_url_like(s)
               and (looks_like_isbn(s) or looks_like_trineday_sitb(s) or looks_like_title(s))]

    key_candidates_df = pd.DataFrame({"Source_original": sorted(set(bookish))})
    OUT_LIST = r"C:/datasources/ai/trineday_source_candidates.csv"
    key_candidates_df.to_csv(OUT_LIST, index=False, encoding="utf-8")
    print(f"Wrote {len(key_candidates_df)} candidates to {OUT_LIST}")
    key_candidates_df.head(10)

    # Fix one specific Source string
    combined_df["Source"] = combined_df["Source"].replace(
        "www.law.cornell.edu_uscode_text_18_1001",
        "https://www.law.cornell.edu/uscode/text/18/1001"
    )

    # Drop unwanted Sources
    to_drop = [
        "WL0003SearchInside",
        "http://66.102.7.104/search?q=cache:SV_nwSXUZL0J:news.yahoo.com/s/nm/20051124/film_nm/arts_film_truce_dc"
    ]
    before = len(combined_df)
    combined_df = combined_df[~combined_df["Source"].isin(to_drop)].reset_index(drop=True)
    after = len(combined_df)

    print(f"Removed {before - after} rows. New total: {after}")

    rows = []
    for src in TRINEDAY_SOURCES_DF["Source_original"].astype(str):
        isbn = normalize_isbn(src)
        query = isbn if isbn else src
        try:
            url = search_trineday(query)
        except Exception as e:
            url = None
        rows.append({"Source_original": src, "Trineday_URL": url})

    mapping_df = pd.DataFrame(rows)
    out_path = TRINEDAY_MAPPING_PATH
    mapping_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Mapping saved to {out_path}")
    mapping_df.head(10)

    # Merge with your combined_df
    combined_df = combined_df.merge(
        mapping_df,
        left_on="Source",  # current values in your dataframe
        right_on="Source_original",
        how="left"
    )

    # Replace Source with the Trineday_URL if present
    combined_df["Source"] = combined_df["Trineday_URL"].combine_first(combined_df["Source"])

    # Drop helper columns if you don’t need them anymore
    combined_df = combined_df.drop(columns=["Source_original", "Trineday_URL"])

    # --- Clean up pipe characters in all string columns ---
    for col in combined_df.columns:
        if combined_df[col].dtype == object:
            combined_df[col] = combined_df[col].astype(str).str.replace("|", "I", regex=False)

    # --- Save as pipe-separated CSV ---
    combined_df.to_csv(ARTICLES_CHUNK_MAPPING_PATH, sep="|", index=False, encoding="utf-8")

    print(f"Updated dataframe with mapped Sources saved to {out_path}")
    print("Applied mapping. Sample rows:")
    print(combined_df.head())

    combined_df["Domain"] = combined_df["Source"].map(extract_domain)

    # --- Count by domain ---
    domain_counts = combined_df["Domain"].value_counts().head(20)  # top 20 for clarity

    # Replace myshopify.com with "Trineday Press" in the counts index
    domain_counts_renamed = domain_counts.rename(index={"myshopify.com": "Trineday Press"})

    # Count publications, drop NaN, blanks, and literal "None"
    pub_series = combined_df["Publication"].dropna().astype(str)
    pub_series = pub_series[pub_series.str.strip().ne("")]
    pub_series = pub_series[pub_series.str.strip().ne("None")]  # drop string "None"

    pub_counts = pub_series.value_counts()

    # Filter: keep only those with >2 occurrences
    pub_counts = pub_counts[pub_counts > 2]

    out_path = r"C:/datasources/ai/beta_slim_final.csv"
    combined_df.to_csv(out_path, sep="|", index=False, encoding="utf-8")

    return combined_df


def _tokenize_like_vectorizer(s: str):
    # Apply SAME preprocessor first (phrase stripping, etc.)
    s = preprocessor(s).lower()
    # Tokenize
    toks = TOKEN_RE.findall(s)
    # Remove stopwords
    toks = [t for t in toks if t not in EN_STOP]
    return toks


def tfidf_query_vector(inv_vocab, tok2idx, idf_vals, query: str) -> np.ndarray:
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


def do_tf_idf_vectorizer_build():
    # --- Load data ---
    df = pd.read_csv(BETA_LIST_CSV_PATH, sep="|", encoding="utf-8")
    assert TEXT_COL in df.columns, f"Missing column: {TEXT_COL}"

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=EN_STOP,
        preprocessor=preprocessor,
        token_pattern=r"(?u)\b\w+\b",
        max_df=0.95,
        min_df=2,
        dtype=np.float32,
    )

    # --- Fit & transform ---
    X = vectorizer.fit_transform(df[TEXT_COL].fillna(""))
    N, V = X.shape
    print(f"TF-IDF built: docs={N}, vocab={V}")

    # --- Export artifacts for client parity ---

    # 1) stopwords.json
    with open(OUT_STOP, "w", encoding="utf-8") as f:
        json.dump(sorted(EN_STOP), f, ensure_ascii=False)
    print("Wrote", OUT_STOP)

    # 2) vocabulary.json (index -> token array)
    # scikit stores token->index; invert it to a list ordered by index
    tok2idx = vectorizer.vocabulary_
    inv_vocab = [""] * len(tok2idx)
    for tok, i in tok2idx.items():
        inv_vocab[i] = tok
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(inv_vocab, f, ensure_ascii=False)
    print("Wrote", OUT_VOCAB)

    # 3) idf.json (aligned with vocabulary indices)
    idf_vals = vectorizer.idf_.astype(np.float32)
    # optional: round for smaller JSON
    idf_out = [float(f"{v:.4f}") for v in idf_vals]
    with open(OUT_IDF, "w", encoding="utf-8") as f:
        json.dump(idf_out, f, ensure_ascii=False)
    print("Wrote", OUT_IDF)

    # Note: TfidfVectorizer handles analyzer steps internally; we nreplicate only what we need.

    # --- Quick sanity test ---
    q = "Click here to learn more about us and cia covert operations"
    qv = tfidf_query_vector(inv_vocab, idf_vals, tok2idx, q)
    print("Query vector shape:", qv.shape, "norm:", np.linalg.norm(qv))

    return vectorizer, X, inv_vocab

def do_local_tf_idf_search_test():
    with open(VOCAB_FN, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)  # index -> token
    token_to_idx = {t: i for i, t in enumerate(vocab_list)}
    V = len(vocab_list)

    with open(IDF_FN, "r", encoding="utf-8") as f:
        idf = np.array(json.load(f), dtype=np.float32)  # length V

    with open(CENTS_FN, "r", encoding="utf-8") as f:
        C = np.array(json.load(f), dtype=np.float32)  # (K, V) already L2-normalized

    with open(INDEX_FN, "r", encoding="utf-8") as f:
        manifest = json.load(f)  # {"k":K, "groups":[... filenames ...]}

    # Stopwords (optional but recommended for full parity)
    try:
        with open(STOP_FN, "r", encoding="utf-8") as f:
            stopwords = set(json.load(f))
    except FileNotFoundError:
        stopwords = set()  # fallback if you didn’t export

    # --- Example ---
    print(search_json(C, "cia covert operations in southeast asia and media propaganda"))