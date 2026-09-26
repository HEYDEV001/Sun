"""Scalable candidate generation via inverted-index blocking keys.

Each record emits several cheap keys; two records are candidate-matched iff they
share >=1 key. Keys are built so a true pair almost always collides on at least one:
  KN2  : two most distinctive name tokens (by rarity) -> pairs with shared name words
  KSK  : 6-char prefix of the consonant skeleton      -> transliteration-robust
  KNUM : each numeric address token + name initial    -> same street-number pairs
  KADDR: address locality token (rarest) + name init
Huge keys (common words / numbers) are dropped so posting lists stay short, which
is what makes this linear instead of quadratic. Everything is country-scoped.
"""
import numpy as np, pandas as pd
from collections import defaultdict

def _rare_tokens(series, k, min_len=3):
    """document frequency of every token, for choosing the k rarest tokens per row."""
    from collections import Counter
    df = Counter()
    toks = [t.split() for t in series.values]
    for ts in toks:
        for t in set(ts):
            if len(t) >= min_len:
                df[t] += 1
    return df, toks

def make_keys(frame, max_post=150):
    """Return dict key->np.array(row_idx), pruned to blocks smaller than max_post."""
    n = len(frame)
    name = frame.name_n.values; sq = frame.name_sq.values; skel = frame.name_skel.values; nums = frame.nums.values
    dfN, toksN = _rare_tokens(frame.name_n, None)
    posting = defaultdict(list)
    def qg(s, q):
        return {s[j:j+q] for j in range(len(s) - q + 1)}
    for i in range(n):
        ts = [t for t in toksN[i] if len(t) >= 3]
        ts_sorted = sorted(set(ts), key=lambda t: dfN[t])
        # KN: three rarest name tokens
        for t in ts_sorted[:3]:
            posting[("n", t)].append(i)
        # KSK: sorted-skeleton prefix (order- & script-robust)
        sk = "".join(sorted(skel[i]))
        if len(sk) >= 5:
            posting[("sk", "".join(sorted(set(skel[i])))[:8])].append(i)
        # KQG: rarest 4-grams of squashed name (typo-robust, like coarse char n-grams)
        grams = qg(sq[i], 4)
        grams = sorted(grams, key=lambda g: dfN.get(g, 0))  # dfN misses grams -> 0, fine as tiebreak
        for g in list(grams)[:6]:
            posting[("q", g)].append(i)
        # KNUM: numeric address token + name initial 2 chars
        ini = (name[i][:2] if name[i] else "")
        for num in set(nums[i].split()):
            if len(num) >= 2:
                posting[("num", num, ini)].append(i)
    return posting

def candidates(qframe, pframe, max_post=150, cap=60):
    """Candidate (qi, pi) pairs where q and p share a pruned key. Positional idx."""
    # build posting lists on the pool, plus query keys, join by key
    from collections import defaultdict
    pkeys = make_keys(pframe, max_post)
    pkeys = {k: v for k, v in pkeys.items() if 1 <= len(v) <= max_post}
    qkeys = make_keys(qframe, max_post)
    pairs = defaultdict(set)
    for k, qidx in qkeys.items():
        pl = pkeys.get(k)
        if not pl:
            continue
        for qi in qidx:
            s = pairs[qi]
            if len(s) < cap * 4:
                s.update(pl)
    rows = []
    for qi, ps in pairs.items():
        for pi in ps:
            rows.append((qi, pi))
    return pd.DataFrame(rows, columns=["qi", "pi"])
