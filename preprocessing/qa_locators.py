# qa_locators.py
import re, json, sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])  # keep tagger+parser+ner
nlp.max_length = 2_000_000  # optional if some lines are very long

# --------------------------
# Keyword extraction helpers
# --------------------------
STOP_PHRASES = set([
    "introduction", "conclusion", "summary", "background", "results", "discussion",
    "in this article", "this paper", "this study"
])

def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def top_tfidf_phrases(texts: List[str], k: int = 8) -> List[List[str]]:
    """Return top k n-grams (1–3) by TF-IDF for each text."""
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,3),
        stop_words="english",
        max_features=50000,
        min_df=1
    )
    X = vec.fit_transform(texts)
    feats = vec.get_feature_names_out()
    out = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        idxs = row.indices
        vals = row.data
        pairs = sorted(zip(idxs, vals), key=lambda p: p[1], reverse=True)[:k*3]  # oversample, filter later
        cands = [feats[j] for j,_ in pairs]
        # filter generic/stop phrases, keep diversity & length
        uniq = []
        seen = set()
        for p in cands:
            if p in seen: continue
            if p in STOP_PHRASES: continue
            if len(p) < 3: continue
            seen.add(p)
            uniq.append(p)
        out.append(uniq[:k])
    return out

def spacy_key_entities(text: str, k: int = 8) -> List[str]:
    doc = nlp(text)
    # Prefer specific entities & long noun chunks
    ents = [e.text for e in doc.ents if e.label_ not in {"CARDINAL","QUANTITY","PERCENT","MONEY"}]
    chunks = [ch.text for ch in doc.noun_chunks if len(ch.text) > 3]
    # normalize & count
    def norm(s):
        s = re.sub(r"\s+", " ", s.strip())
        return s.strip(".,;:!?)(").lower()
    counts = Counter([norm(x) for x in ents + chunks])
    ordered = [t for t,_ in counts.most_common(k*3)]
    # keep reasonably specific phrases
    filtered = [p for p in ordered if len(p.split()) <= 5 and p not in STOP_PHRASES]
    # restore original casing by picking first occurrence
    restore_map = {}
    for e in ents + chunks:
        n = norm(e)
        if n not in restore_map:
            restore_map[n] = e.strip()
    return [restore_map.get(p, p) for p in filtered[:k]]

# --------------------------
# Question composition
# --------------------------
TEMPLATES = [
    "Which passage covers {kw}?",
    "Where in the material is {kw} discussed?",
    "Find the section about {kw}. What does it say?",
    "Which excerpt addresses {kw} and related details?",
    "Locate the part concerning {kw}.",
]

def make_questions(keywords: List[str], max_q: int = 4) -> List[str]:
    qs = []
    used = set()
    for i, kw in enumerate(keywords):
        # dedupe near-duplicates by lowercase
        low = kw.lower()
        if low in used: continue
        used.add(low)
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        q = tmpl.format(kw=kw)
        qs.append(q)
        if len(qs) >= max_q:
            break
    return qs

# --------------------------
# Main
# --------------------------
def main(infile: str, outfile: str, max_q=4, limit = 0):
    lines = [clean(x) for x in Path(infile).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        print("No content found."); return

    # Pre-compute tf-idf phrases to blend with NER/noun-chunk signals
    tfidf_list = top_tfidf_phrases(lines, k=8)

    with Path(outfile).open("w", encoding="utf-8") as w:
        num_processed = 0
        for i, ctx in enumerate(tqdm(lines, desc="Building locators")):
            # hybrid keywords = top entities/chunks + tfidf phrases
            ents = spacy_key_entities(ctx, k=8)
            tfk  = tfidf_list[i]
            # blend, keeping order: entities first (more “named”) then tf-idf diversity
            seen = set()
            blended = []
            for kw in ents + tfk:
                norm = kw.lower()
                if norm in seen: continue
                seen.add(norm)
                blended.append(kw)
            if not blended:
                blended = ["the main topic of this passage"]

            questions = make_questions(blended, max_q=max_q)

            # Emit: full content preserved as answer and context
            for q in questions:
                rec = {
                    "id": f"line-{i}",
                    "question": q,
                    "answer": ctx,          # full passage as the answer
                    "context": ctx,         # duplicate for clarity / future use
                    "keywords": blended[:8] # optional helpers (not required by trainers)
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

            num_processed = num_processed + 1
            if 0 < limit < num_processed:
                print(f"** Processed up to limit {limit} number of lines. Stopping. **")
                return

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python qa_locators.py input.txt output.jsonl [max_q_per_line]")
        sys.exit(1)
    infile, outfile = sys.argv[1], sys.argv[2]
    max_q = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    main(infile, outfile, max_q, 4)
