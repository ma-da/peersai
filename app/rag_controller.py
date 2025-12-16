"""
Refactored retrieval pipeline
-----------------------------
Key changes vs original:
- No global mutable state
- Explicit RetrievalState object
- Clear stage boundaries (boot, sparse retrieval, routing, shard fetch, rerank, context build)
- Consistent error handling
- Async used only for I/O

This keeps your logic and assumptions intact, but makes the system testable,
composable, and much easier to reason about.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import requests
from collections import Counter

# ------------------ CONFIG ------------------
HIVE_RPC = "https://api.hive.blog"
AUTHOR = "wanttoknow"
PERMLINK = "seeds-of-truth-index-registration-0-1"
FLASK_PROXY_URL = "https://fixingbrokenrobots.pythonanywhere.com/chat"
MAX_QUESTION_WORDS = 400
TOP_DOCS = 20
# --------------------------------------------


# ================== STATE ==================
@dataclass
class RetrievalState:
    vocab: List[str]
    token_to_idx: Dict[str, int]
    idf: np.ndarray
    centroids: List[np.ndarray]
    stopwords: Set[str]
    group_list: List[str]
    groups_url_map: Dict[str, str]


# ================== BOOT ==================

def hive_get_content(author: str, permlink: str) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "condenser_api.get_content",
        "params": [author, permlink],
    }
    r = requests.post(HIVE_RPC, json=payload)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["result"]


def boot() -> RetrievalState:
    """Download registry + artifacts and build immutable retrieval state."""
    print("Booting retrieval system…")

    post = hive_get_content(AUTHOR, PERMLINK)
    print("Boot: got hive post")

    registry_json = re.sub(r"```(?:json)?", "", post["body"]).strip()
    print("Boot: extracted registry json")

    registry = json.loads(registry_json)
    print("Boot: parsed registry")

    def pick(obj_key: str) -> str:
        obj = registry.get(obj_key)
        if not obj:
            raise KeyError(f"Missing registry entry: {obj_key}")
        return obj.get("url_custom") or obj.get("url_dedicated")

    manifest = requests.get(pick("betaslim_manifest.byfile.json")).json()
    print("Boot: got manifest")

    groups_url_map = requests.get(pick("groups_urls.json")).json()
    print("Boot: got groups_url_map")

    files = {f["name"]: f for f in manifest.get("files", [])}

    def file_url(name: str) -> str:
        f = files.get(name)
        if not f:
            raise KeyError(f"Missing file in manifest: {name}")
        return f.get("url_custom") or f.get("url_dedicated")

    vocab = requests.get(file_url("vocabulary.json")).json()
    print("Boot: got vocab")

    idf = np.array(requests.get(file_url("idf.json")).json(), dtype=np.float32)
    centroids = [np.array(row, dtype=np.float32)
                 for row in requests.get(file_url("centroids.json")).json()]
    print("Boot: got centroids")

    stopwords = set(w.lower() for w in requests.get(file_url("stopwords.json")).json())
    print("Boot: got stopwords")

    index_obj = requests.get(
        file_url(manifest.get("roles", {}).get("index", "centroids_index.json"))
    ).json()
    print("Boot: got roles")

    state = RetrievalState(
        vocab=vocab,
        token_to_idx={t: i for i, t in enumerate(vocab)},
        idf=idf,
        centroids=centroids,
        stopwords=stopwords,
        group_list=index_obj["groups"],
        groups_url_map=groups_url_map,
    )

    print(f"✓ Boot complete: {len(vocab):,} vocab | {len(centroids)} centroids")
    return state


# ================== RETRIEVAL ==================

def truncate_question(q: str) -> tuple[str, bool]:
    words = q.split()
    if len(words) <= MAX_QUESTION_WORDS:
        return q.strip(), False
    return " ".join(words[:MAX_QUESTION_WORDS]), True


def build_query_vector(state: RetrievalState, query: str) -> Optional[np.ndarray]:
    tokens = [t.lower() for t in query.split()
              if t.lower() not in state.stopwords and len(t) > 2]
    if not tokens:
        return None

    tf = Counter(tokens)
    qv = np.zeros(len(state.vocab), dtype=np.float32)

    for tok, freq in tf.items():
        idx = state.token_to_idx.get(tok)
        if idx is not None:
            qv[idx] = freq * state.idf[idx]

    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm
    return qv


def top_k_centroids(state: RetrievalState, qv: np.ndarray, k: int = 9) -> List[int]:
    sims = [float(np.dot(qv, c)) for c in state.centroids]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


async def fetch_json(url: str) -> Any:
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(None, requests.get, url)
    r.raise_for_status()
    return r.json()


async def sparse_retrieve(state: RetrievalState, query: str) -> List[Dict[str, Any]]:
    qv = build_query_vector(state, query)
    if qv is None:
        return []

    cent_ids = top_k_centroids(state, qv)
    shard_names = [state.group_list[i] for i in cent_ids]

    shards = await asyncio.gather(
        *[fetch_json(state.groups_url_map[name]) for name in shard_names]
    )

    scored: List[Dict[str, Any]] = []
    for shard in shards:
        for row in shard:
            sim = sum(weight * qv[idx] for idx, weight in row.get("tfidf", []))
            norm = row.get("norm", 1.0)
            score = sim / norm if norm > 0 else 0.0

            scored.append({
                "row_id": row.get("row_id"),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "score_sparse": score,
            })

    scored.sort(key=lambda d: d["score_sparse"], reverse=True)
    return scored


# ================== CONTEXT ==================

def build_context(docs: List[Dict[str, Any]], query: str) -> str:
    terms = set(query.lower().split())
    used_trigrams: Set[str] = set()
    blocks: List[str] = []

    def trigrams(tokens: List[str]) -> List[str]:
        return [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

    for doc in docs[:TOP_DOCS]:
        sentences = [s.strip() for s in doc["text"].split(".") if s.strip()]
        kept: List[str] = []

        for sent in sentences:
            if not any(t in sent.lower() for t in terms):
                continue
            tris = trigrams(sent.lower().split())
            if any(tri in used_trigrams for tri in tris):
                continue
            kept.append(sent)
            used_trigrams.update(tris)
            if len(kept) >= 5:
                break

        if kept:
            blocks.append(
                f'<doc id="{doc.get("row_id")}" url="{doc.get("source")}">\n'
                + "\n".join(f"- {s}" for s in kept)
                + "\n</doc>"
            )

    return "\n\n".join(blocks)


# ================== ORCHESTRATION ==================

async def ask(state: RetrievalState, question: str, *, verbose: bool = True) -> str:
    q, truncated = truncate_question(question)

    if verbose:
        print("Searching corpus…")
    t0 = time.time()

    docs = await sparse_retrieve(state, q)
    context = build_context(docs, q)

    if verbose:
        print(f"Retrieved context in {time.time() - t0:.2f}s")

    payload = {
        "query": q,
        "context": context,
        "temperature": 0.3,
        "max_tokens": 3072,
    }

    r = requests.post(FLASK_PROXY_URL, json=payload)
    r.raise_for_status()
    data = r.json()

    answer = data.get("answer", "")
    if truncated:
        answer = "(Question truncated)\n\n" + answer
    return answer.strip()

async def search_references(
    state: RetrievalState,
    query: str,
    *,
    top_k: int = 20,
    max_per_cluster: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run retrieval only (no LLM) and return results as JSON.

    Returns:
      {
        "query": "...",
        "results": [
          {
            "row_id": "...",
            "title": "...",
            "source": "...",
            "score": 0.123,
            "text": "..."
          },
          ...
        ]
      }
    """

    if verbose:
        print("Running reference search…")

    qv = build_query_vector(state, query)
    if qv is None:
        return {
            "query": query,
            "results": [],
            "warning": "Query contained no indexable terms"
        }

    # ---- centroid routing ----
    centroid_ids = top_k_centroids(state, qv, k=9)
    shard_names = [state.group_list[i] for i in centroid_ids]

    # ---- fetch shards in parallel ----
    shards = await asyncio.gather(
        *[fetch_json(state.groups_url_map[name]) for name in shard_names]
    )

    scored: List[Dict[str, Any]] = []

    # ---- sparse scoring ----
    for shard in shards:
        for row in shard[:max_per_cluster]:
            sim = sum(weight * qv[idx] for idx, weight in row.get("tfidf", []))
            norm = row.get("norm", 1.0)
            score = sim / norm if norm > 0 else 0.0

            if score <= 0:
                continue

            scored.append({
                "row_id": row.get("row_id"),
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "score": float(score),
                "text": row.get("text", ""),
            })

    # ---- global ranking ----
    scored.sort(key=lambda r: r["score"], reverse=True)

    results = scored[:top_k]

    return {
        "query": query,
        "num_results": len(results),
        "results": results,
    }

# ================== MAIN ==================

async def main():
    state = boot()

    # Test ask method
    #answer = await ask(
    #    state,
    #    "What really happened on 9/11 according to declassified documents and whistleblowers?",
    #)
    #print("\nANSWER:\n" + "=" * 40)
    #print(answer)

    # Test search_references method
    refs = await search_references(
        state,
        "What do declassified documents say about JFK assassination planning?",
        top_k=10,
        verbose=True,
    )
    print(json.dumps(refs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
