"""Dense-embedding blocking: hashed 5-signal TF-IDF -> TruncatedSVD -> FAISS ANN.

Bounded memory (hashing dims are fixed; SVD components are small) and fast search
(FAISS IVF), so it scales to the full 1.7M x 10M test on a 2-core box.
"""
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import faiss

BLOCKS = [  # col, analyzer, ngram, bits, weight
    ("addr_n",   "word",    (1, 1), 18, 0.32),
    ("addr_n",   "char_wb", (3, 4), 19, 0.15),
    ("name_n",   "word",    (1, 1), 17, 0.13),
    ("name_sq",  "char",    (3, 3), 18, 0.16),
    ("name_skel","char",    (3, 4), 18, 0.24),
]

class Embedder:
    def __init__(self, dim=160):
        self.dim = dim
        self.vecs = []
        for col, an, ng, bits, w in BLOCKS:
            h = HashingVectorizer(analyzer=an, ngram_range=ng, n_features=2**bits,
                                  alternate_sign=False, norm=None, dtype=np.float32,
                                  token_pattern=r"\S+" if an == "word" else None)
            self.vecs.append((col, h, TfidfTransformer(sublinear_tf=True), np.sqrt(w)))
        self.svd = TruncatedSVD(n_components=dim, random_state=0)

    def _sparse(self, df):
        return sp.hstack([t.transform(h.transform(df[c].values)) * w
                          for c, h, t, w in self.vecs], format="csr", dtype=np.float32)

    def fit(self, pool, idf_docs=400_000, svd_docs=250_000, seed=0):
        s = pool.sample(min(idf_docs, len(pool)), random_state=seed)
        for c, h, t, w in self.vecs:
            t.fit(h.transform(s[c].values))
        sv = pool.sample(min(svd_docs, len(pool)), random_state=seed + 1)
        self.svd.fit(self._sparse(sv))
        return self

    def embed(self, df, batch=150_000, out=None):
        """Embed rows into `out` (an np.ndarray or np.memmap of shape [len(df), dim]).
        Allocates a RAM array if out is None. Rows are L2-normalised."""
        if out is None:
            out = np.empty((len(df), self.dim), dtype=np.float32)
        for i in range(0, len(df), batch):
            X = self._sparse(df.iloc[i:i + batch])
            e = self.svd.transform(X).astype(np.float32)
            normalize(e, copy=False)
            out[i:i + batch] = e
        return out


def build_ivfpq(P, dim, nlist=4096, m=32, nbits=8, train_n=300_000, seed=0):
    """IVFPQ index (compressed, low-memory) over embedding matrix/memmap P."""
    quant = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quant, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.RandomState(seed)
    tr = P[rng.choice(len(P), min(train_n, len(P)), replace=False)]
    idx.train(np.ascontiguousarray(tr, dtype=np.float32))
    for i in range(0, len(P), 500_000):
        idx.add(np.ascontiguousarray(P[i:i + 500_000], dtype=np.float32))
    return idx
