"""Candidate generation ("blocking") — memory-bounded via feature hashing.

Every record becomes ONE sparse vector = weighted concat of five signals:
    addr word tokens | addr char 3-4grams | name word tokens |
    name char 3grams | name consonant-skeleton char 3-4grams
Each signal is a HashingVectorizer (fixed feature space, no vocabulary in RAM,
so memory does not grow with corpus size) followed by an IDF scaling fitted on
a sample. Cosine similarity = dot product of L2-normalised vectors, so the top-k
sparse matmul returns a weighted blend of the five cosines.

Nothing is country-specific: the same hashing + IDF is fit per country on its pool,
so the model and blocker transfer to unseen France. The consonant-skeleton block is
what recovers transliterated (Devanagari->ASCII) Indian names whose raw n-grams differ.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

# (column, analyzer, ngram_range, n_features_bits, weight)
BLOCKS = [
    ("addr_n",   "word",    (1, 1), 20, 0.32),
    ("addr_n",   "char_wb", (3, 4), 21, 0.15),
    ("name_n",   "word",    (1, 1), 19, 0.13),
    ("name_sq",  "char",    (3, 3), 20, 0.16),
    ("name_skel","char",    (3, 4), 20, 0.24),
]


class _Block:
    def __init__(self, col, analyzer, ngram, bits, weight):
        self.col, self.w = col, np.sqrt(weight)
        self.h = HashingVectorizer(analyzer=analyzer, ngram_range=ngram,
                                   n_features=2 ** bits, alternate_sign=False,
                                   norm=None, dtype=np.float32,
                                   token_pattern=r"\S+" if analyzer == "word" else None)
        self.idf = TfidfTransformer(sublinear_tf=True)

    def fit(self, df):
        self.idf.fit(self.h.transform(df[self.col].values))
        return self

    def transform(self, df):
        return self.idf.transform(self.h.transform(df[self.col].values)) * self.w


class Blocker:
    def __init__(self):
        self.blocks = [_Block(*b) for b in BLOCKS]

    def fit(self, pool, max_docs=400_000, seed=0):
        fit_df = pool if len(pool) <= max_docs else pool.sample(max_docs, random_state=seed)
        for b in self.blocks:
            b.fit(fit_df)
        return self

    def transform(self, df):
        M = sp.hstack([b.transform(df) for b in self.blocks], format="csr", dtype=np.float32)
        return normalize(M, norm="l2", copy=False)   # so dot product == cosine blend

    def topk(self, q_mat, p_mat_T, k=15, thr=0.05):
        R = sp_matmul_topn(q_mat, p_mat_T, top_n=k, threshold=thr, sort=True, n_threads=2).tocoo()
        return R.row, R.col, R.data


def search(blocker, q, pool, k=15, thr=0.05, n_shards=6, q_chunk=100_000, log=print):
    """Top-k pool neighbours for every row of `q`, sharding the pool to bound memory.
    Returns DataFrame[qi, pi, score] (positional indices), best-first, <=k per query."""
    import pandas as pd, time
    n = len(pool)
    bounds = np.linspace(0, n, n_shards + 1).astype(int)
    parts = []
    for s in range(n_shards):
        t = time.time()
        lo, hi = bounds[s], bounds[s + 1]
        PT = blocker.transform(pool.iloc[lo:hi]).T.tocsr()
        for qs in range(0, len(q), q_chunk):
            Q = blocker.transform(q.iloc[qs:qs + q_chunk])
            r, c, sc = blocker.topk(Q, PT, k=k, thr=thr)
            parts.append(pd.DataFrame({"qi": r.astype(np.int64) + qs,
                                       "pi": c.astype(np.int64) + lo, "score": sc}))
        del PT
        log(f"shard {s+1}/{n_shards} done in {time.time()-t:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["qi", "score"], ascending=[True, False], kind="stable")
    df["rank"] = df.groupby("qi").cumcount() + 1
    return df[df["rank"] <= k].reset_index(drop=True)
