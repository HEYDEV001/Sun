"""Pairwise features for (S1 record, candidate) scoring.

All features are symmetric string-similarity signals plus the blocking score.
None are country-specific, so the model transfers to unseen France.
"""
import numpy as np
from rapidfuzz import fuzz, distance

def _set(s):
    return set(s.split())

def _jacc(a, b):
    if not a or not b:
        return 0.0
    A, B = _set(a), _set(b)
    u = len(A | B)
    return len(A & B) / u if u else 0.0

def _num_overlap(a, b):
    A, B = _set(a), _set(b)
    if not A and not B:
        return 1.0            # both addresses have no house/number tokens -> not contradictory
    if not A or not B:
        return 0.5
    return len(A & B) / len(A | B)

def pair_features(q, p):
    """q, p: dict-like rows with keys name_n, name_sq, name_skel, addr_n, nums,
    is_dom, nonlatin. Returns list[float]."""
    qn, pn = q["name_n"], p["name_n"]
    qa, pa = q["addr_n"], p["addr_n"]
    f = [
        fuzz.token_sort_ratio(qn, pn) / 100.0,
        fuzz.token_set_ratio(qn, pn) / 100.0,
        fuzz.partial_ratio(qn, pn) / 100.0,
        fuzz.ratio(q["name_sq"], p["name_sq"]) / 100.0,
        fuzz.ratio(q["name_skel"], p["name_skel"]) / 100.0,      # cross-script consonant match
        _jacc(qn, pn),
        1.0 - distance.JaroWinkler.distance(q["name_sq"], p["name_sq"]),
        # address
        fuzz.token_sort_ratio(qa, pa) / 100.0 if (qa and pa) else 0.0,
        fuzz.token_set_ratio(qa, pa) / 100.0 if (qa and pa) else 0.0,
        _jacc(qa, pa),
        _num_overlap(q["nums"], p["nums"]),
        # meta
        float(p["is_dom"]),
        float(q["nonlatin"] or p["nonlatin"]),
        float(len(pa) == 0),                                     # candidate has no address
        abs(len(qn) - len(pn)) / (max(len(qn), len(pn)) + 1),
        float(pn.startswith("s3-")),  # placeholder, overwritten by source flag below
    ]
    return f

FEATURE_NAMES = [
    "name_token_sort", "name_token_set", "name_partial", "name_sq_ratio", "name_skel_ratio",
    "name_jaccard", "name_jw", "addr_token_sort", "addr_token_set", "addr_jaccard",
    "num_overlap", "is_domain", "nonlatin", "cand_no_addr", "len_ratio", "src_is_s3", "block_score",
]

def build_matrix(qrows, prows, block_scores, src_is_s3):
    import numpy as np
    M = np.empty((len(qrows), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (q, p, bs, s3) in enumerate(zip(qrows, prows, block_scores, src_is_s3)):
        row = pair_features(q, p)
        row[15] = float(s3)
        row.append(float(bs))
        M[i] = row
    return M
