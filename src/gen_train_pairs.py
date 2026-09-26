"""Run blocking on a train S1 sample, label candidates against ground truth,
extract pair features, and persist a training table per country."""
import sys, time
import numpy as np, pandas as pd
from blocking import Blocker, search
from features import build_matrix, FEATURE_NAMES
import config

D = config.DATA + "/"
OUT = config.TRAINPAIRS
COLS = ["name_n", "name_sq", "name_skel", "addr_n", "nums", "is_dom", "nonlatin"]

def run(country, n_sample, K, shards):
    t0 = time.time()
    s1 = pd.read_parquet(D + "norm/train_source1.parquet"); s1 = s1[s1.country == country]
    s1 = s1.sample(min(n_sample, len(s1)), random_state=7).reset_index(drop=True)
    pool = pd.concat([pd.read_parquet(D + f"norm/train_source{i}.parquet").assign(src=i) for i in (2, 3)])
    pool = pool[pool.country == country].reset_index(drop=True)
    gt = pd.read_parquet(D + "pq/train_ground_truth.parquet").set_index("source1_entity_id").matched_entity_ids
    print(f"[{country}] q={len(s1):,} pool={len(pool):,} load={time.time()-t0:.0f}s", flush=True)

    b = Blocker().fit(pool)
    res = search(b, s1, pool, k=K, thr=0.02, n_shards=shards, log=lambda m: print("  " + m, flush=True))

    pid = pool.entity_id.values
    res["pid"] = pid[res.pi.values]
    res["qeid"] = s1.entity_id.values[res.qi.values]
    # label
    truth = {q: set(x.split(",")) if isinstance(x, str) and x else set() for q, x in gt.loc[s1.entity_id].items()}
    res["y"] = [1 if r.pid in truth[r.qeid] else 0 for r in res.itertuples()]

    qrows = s1.iloc[res.qi.values][COLS].to_dict("records")
    prows = pool.iloc[res.pi.values][COLS].to_dict("records")
    s3flag = (pool.src.values[res.pi.values] == 3).astype(int)
    M = build_matrix(qrows, prows, res.score.values, s3flag)
    out = pd.DataFrame(M, columns=FEATURE_NAMES)
    out["y"] = res.y.values; out["qeid"] = res.qeid.values; out["country"] = country
    out.to_parquet(f"{OUT}{country}.parquet")
    # recall ceiling of this candidate set
    got = res[res.y == 1].groupby("qeid").size()
    tot = pd.Series({q: len(v) for q, v in truth.items()})
    nz = tot[tot > 0]
    rec = (got.reindex(nz.index).fillna(0) / nz).mean()
    print(f"[{country}] pairs={len(out):,} pos={out.y.mean():.3f} recall_ceiling={rec:.4f} "
          f"cand/q={len(out)/len(s1):.1f} total={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    run("US", 45000, 20, 6)
    run("India", 45000, 20, 5)
