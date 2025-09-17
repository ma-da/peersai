import json
from collections import deque

from scipy.sparse import csr_matrix
from tqdm import tqdm

from config import *
from sklearn.cluster import KMeans
import numpy as np
import math

from utils import float4
import pandas as pd

def _choose_split_k(n_items, cap=CAP, min_k=MIN_SPLIT, max_k=MAX_SPLIT):
    """
    For a cluster of size n_items, choose a k that should bring subclusters under 'cap' on average.
    We clamp k to [min_k, max_k].
    """
    k = int(math.ceil(n_items / cap))
    k = max(min_k, min(max_k, k))
    return k

def _kmeans_split(X_sub, k, random_state=RANDOM_STATE):
    """
    Run KMeans on the provided sub-matrix X_sub (rows = items in this cluster).
    Returns labels (len = X_sub.shape[0]) and the fitted model.
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=random_state,
        verbose=0
    )
    labels = km.fit_predict(X_sub)
    return labels, km

def recursive_kmeans_cap(X, cap=CAP, random_state=RANDOM_STATE, verbose=True):
    """
    Recursively split clusters with KMeans until every cluster contains <= cap items.
    Returns:
      - groups: list of np.ndarray index arrays (global indices into X)
      - centroids: list of centroid vectors (np.ndarray), one per final group
    """
    N = X.shape[0]
    initial_indices = np.arange(N)
    queue = deque([initial_indices])
    final_groups = []

    while queue:
        idxs = queue.popleft()
        n = len(idxs)

        if n <= cap:
            # This group is small enough; accept as final
            final_groups.append(idxs)
            if verbose:
                print(f"✓ group accepted: size={n}")
            continue

        # Need to split
        k = _choose_split_k(n, cap=cap)
        if verbose:
            print(f"→ splitting group of size {n} with k={k}")

        # Slice X to the rows of this group
        X_sub = X[idxs]

        # Fit KMeans for this subset
        labels, km = _kmeans_split(X_sub, k, random_state=random_state)

        # Push each subcluster back to queue (by global indices)
        for c in range(k):
            sub_idxs = idxs[labels == c]
            if len(sub_idxs) == 0:
                continue
            queue.append(sub_idxs)

    # Compute centroids for the final groups (as mean TF-IDF vector per group)
    # If X is sparse, this stays efficient.
    centroids = []
    for gidxs in final_groups:
        Xg = X[gidxs]
        # centroid = mean along rows
        if hasattr(Xg, "mean"):  # works for numpy arrays and scipy sparse
            c = Xg.mean(axis=0)
            # bring to 1D np.array
            c = np.asarray(c).ravel()
        else:
            # Fallback (shouldn't happen): convert to dense
            c = np.mean(np.asarray(Xg), axis=0)
        # (Optional) L2-normalize centroid now if you'll use cosine
        norm = np.linalg.norm(c) + 1e-12
        centroids.append((c / norm).astype(np.float32))
    centroids = np.vstack(centroids).astype(np.float32)

    return final_groups, centroids

def calculate_kmeans(X):
    # ------------------ USAGE ------------------
    # X is your TF-IDF matrix (dense or CSR sparse). N = X.shape[0].

    groups, centroids_unit = recursive_kmeans_cap(X, cap=1500, random_state=RANDOM_STATE, verbose=True)
    print(f"Final number of groups: {len(groups)}")
    sizes = [len(g) for g in groups]
    print(f"Min/Max group size: {min(sizes)} / {max(sizes)}")

    # Build a labels array (cluster id per document) if you need it downstream
    labels_balanced = np.empty(X.shape[0], dtype=np.int32)
    for gid, gidxs in enumerate(groups):
        labels_balanced[gidxs] = gid

    # 'centroids_unit' are L2-normalized centroids per final group (good for cosine).
    # If you prefer unnormalized, just remove the normalization block above.

    return groups, centroids_unit

def build_index_shards(X, vectorizer, groups, centroids_unit, df):
    # ---------- 1) vocabulary.json (index -> token) ----------
    tok2idx = vectorizer.vocabulary_  # token -> index
    inv_vocab = [""] * len(tok2idx)
    for tok, idx in tok2idx.items():
        inv_vocab[idx] = tok
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(inv_vocab, f, ensure_ascii=False)

    # ---------- 2) idf.json (aligned with vocabulary indices) ----------
    idf_arr = vectorizer.idf_.astype(np.float32)  # np.array shape (V,)
    idf_out = [float4(v) for v in idf_arr.tolist()]
    with open(OUT_IDF, "w", encoding="utf-8") as f:
        json.dump(idf_out, f, ensure_ascii=False)

    # ---------- 3) centroids.json (already unit-norm from recursive_kmeans_cap) ----------
    # centroids_unit: np.ndarray shape (K, V), L2-normalized
    centroids_json = [[float4(v) for v in row] for row in centroids_unit]
    with open(OUT_CENTS, "w", encoding="utf-8") as f:
        json.dump(centroids_json, f, ensure_ascii=False)

    # ---------- 4) groups: one JSON per final cluster with sparse tf-idf + doc info ----------
    # Ensure CSR for fast row ops
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # Precompute L2 norms for cosine
    doc_norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1 + 1e-12  # shape (N,)

    group_files = []
    for g, idxs in tqdm(list(enumerate(groups)), desc="Writing groups"):
        Xg = X[idxs]  # CSR view

        rows = []
        for i_local, i_doc in enumerate(idxs):
            row = Xg.getrow(i_local)  # 1xV CSR
            cols = row.indices  # nonzero column indices
            data = row.data  # corresponding tf-idf weights

            pairs = [[int(c), float4(w)] for c, w in zip(cols, data)]

            # prefer your dataset ID if present; else fallback to i_doc
            if "ID" in df.columns and pd.notna(df.iloc[i_doc].get("ID", np.nan)):
                row_id_val = int(df.iloc[i_doc]["ID"])
            else:
                row_id_val = int(i_doc)

            rows.append({
                "row_id": row_id_val,
                "source": df.iloc[i_doc].get("Source", None),
                "title": df.iloc[i_doc].get("Title", None),
                "text": df.iloc[i_doc].get(TEXT_COL, None),  # optional: remove to shrink shard
                "norm": float4(doc_norms[i_doc]),
                "tfidf": pairs
            })

        out_path = PREPROCESSING_OUT_DIR / f"group_{g:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        group_files.append(out_path.name)  # store only filename in manifest

    # ---------- 5) centroids_index.json (manifest) ----------
    K = len(groups)
    manifest = {"version": 1, "k": K, "groups": group_files}
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    print("Exported to:", PREPROCESSING_OUT_DIR)
    print(" Files: vocabulary.json, idf.json, centroids.json, centroids_index.json, and group_XXX.json shards")