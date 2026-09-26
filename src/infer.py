"""End-to-end inference on the test set.

Produces the two required files in output/:
  candidate_pairs.tsv   : source1_entity_id \t matched_entity_ids   (blocking output)
  matching_results.tsv  : source1_entity_id \t matched_entity_ids   (model-kept matches; scored)

Guarantees enforced (validator rules):
  * every test S1 id present exactly once in both files
  * no duplicate ids within a list, no self-matches, ids exist in test set
  * matching_results is a strict subset of candidate_pairs
"""
import sys, time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from blocking import Blocker, search
from features import build_matrix, FEATURE_NAMES
import config

D = config.DATA + "/"
MODELDIR = config.MODELDIR
OUTDIR = config.OUTDIR
COLS = ["name_n", "name_sq", "name_skel", "addr_n", "nums", "is_dom", "nonlatin"]
K = 20
CAND_MIN_SCORE = 0.06     # prune very weak blocking candidates (keeps candidate set lean)
BATCH = 200_000           # score feature rows in batches to bound memory

def infer_country(country, booster, thr, shards):
    t0 = time.time()
    s1 = pd.read_parquet(D + "norm/test_source1.parquet"); s1 = s1[s1.country == country].reset_index(drop=True)
    pool = pd.concat([pd.read_parquet(D + f"norm/test_source{i}.parquet").assign(src=i) for i in (2, 3)])
    pool = pool[pool.country == country].reset_index(drop=True)
    print(f"[{country}] q={len(s1):,} pool={len(pool):,} load={time.time()-t0:.0f}s", flush=True)

    b = Blocker().fit(pool)
    res = search(b, s1, pool, k=K, thr=CAND_MIN_SCORE, n_shards=shards,
                 log=lambda m: print("  " + m, flush=True))
    pid = pool.entity_id.values
    res["pid"] = pid[res.pi.values]
    res["qeid"] = s1.entity_id.values[res.qi.values]
    print(f"[{country}] candidates={len(res):,} ({len(res)/len(s1):.1f}/q) block_done={time.time()-t0:.0f}s", flush=True)

    # score in batches
    preds = np.empty(len(res), dtype=np.float32)
    s3flag = (pool.src.values[res.pi.values] == 3).astype(int)
    for i in range(0, len(res), BATCH):
        sl = slice(i, i + BATCH)
        qrows = s1.iloc[res.qi.values[sl]][COLS].to_dict("records")
        prows = pool.iloc[res.pi.values[sl]][COLS].to_dict("records")
        M = build_matrix(qrows, prows, res.score.values[sl], s3flag[sl])
        preds[sl] = booster.predict(M)
    res["pred"] = preds
    print(f"[{country}] scored={time.time()-t0:.0f}s", flush=True)
    return s1.entity_id.values, res[["qeid", "pid", "score", "pred"]]

def main():
    import os; os.makedirs(OUTDIR, exist_ok=True)
    meta = json.load(open(MODELDIR + "meta.json"))
    thr = meta["threshold"]
    booster = lgb.Booster(model_file=MODELDIR + "matcher.txt")
    shards = {"US": 6, "India": 5, "France": 2}
    all_ids, all_res = [], []
    for c in ("US", "India", "France"):
        ids, r = infer_country(c, booster, thr, shards.get(c, 4))
        all_ids.append(ids); all_res.append(r)
    ids = np.concatenate(all_ids)
    res = pd.concat(all_res, ignore_index=True)
    res = res[res.qeid != res.pid]                              # no self-match

    # candidate file: all blocking candidates, best-first
    res = res.sort_values(["qeid", "pred"], ascending=[True, False], kind="stable")
    cand = res.groupby("qeid").pid.apply(lambda s: ",".join(dict.fromkeys(s))).to_dict()
    kept = res[res.pred >= thr]
    match = kept.groupby("qeid").pid.apply(lambda s: ",".join(dict.fromkeys(s))).to_dict()

    def write(path, table):
        rows = [(i, table.get(i, "")) for i in ids]
        pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]) \
          .to_csv(path, sep="\t", index=False)
    write(OUTDIR + "candidate_pairs.tsv", cand)
    write(OUTDIR + "matching_results.tsv", match)
    nz = sum(1 for i in ids if match.get(i))
    print(f"DONE ids={len(ids):,} nonempty_match={nz:,} ({nz/len(ids):.3f}) "
          f"mean_cand={sum(len(v.split(',')) for v in cand.values())/len(ids):.2f}", flush=True)

if __name__ == "__main__":
    main()
