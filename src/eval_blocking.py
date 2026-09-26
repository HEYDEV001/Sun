"""Measure blocking recall / candidate-set size on a train sample."""
import sys, time
import numpy as np, pandas as pd
from blocking import Blocker
import config

country, n_sample, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D = config.DATA + "/"
t0 = time.time()
s1 = pd.read_parquet(D + "norm/train_source1.parquet"); s1 = s1[s1.country == country]
pool = pd.concat([pd.read_parquet(D + f"norm/train_source{i}.parquet") for i in (2, 3)])
pool = pool[pool.country == country].reset_index(drop=True)
gt = pd.read_parquet(D + "pq/train_ground_truth.parquet").set_index("source1_entity_id").matched_entity_ids
q = s1.sample(n_sample, random_state=1).reset_index(drop=True)
print(f"{country}: pool={len(pool):,} sample={len(q):,} load={time.time()-t0:.0f}s", flush=True)

from blocking import search
t = time.time(); b = Blocker().fit(pool)
print(f"fit {time.time()-t:.0f}s", flush=True)
t = time.time(); res = search(b, q, pool, k=K, thr=0.02, n_shards=int(sys.argv[4]) if len(sys.argv) > 4 else 6,
                              log=lambda m: print(m, flush=True))
print(f"search {time.time()-t:.0f}s  ({(time.time()-t)/len(q)*1000:.1f} ms/query)", flush=True)
pid = pool.entity_id.values
cand = pd.DataFrame({"qi": res.qi.values, "pid": pid[res.pi.values], "score": res.score.values, "rank": res["rank"].values})
truth = pd.DataFrame({"qi": np.arange(len(q)), "ids": [x.split(",") if isinstance(x, str) and x else [] for x in gt.loc[q.entity_id].values]})
tp = truth.explode("ids").dropna(); tp = tp[tp.ids != ""].rename(columns={"ids": "pid"})
m = tp.merge(cand, on=["qi", "pid"], how="left")
print(f"true pairs={len(tp):,}  singletons={(truth.ids.str.len()==0).mean():.3f}")
for k in (3, 5, 8, 10, 15, K):
    print(f"  pair recall@{k:>2}: {(m['rank'] <= k).mean():.4f}")
allin = m.assign(hit=m["rank"] <= K).groupby("qi").hit.all()
print(f"  entities with ALL matches inside top{K}: {allin.mean():.4f}")
print("  mean candidates/query:", len(cand) / len(q))
miss = m[m["rank"].isna() | (m["rank"] > K)].head(6)
pool_i = pool.set_index("entity_id"); q_i = q
for _, x in miss.iterrows():
    a = q.iloc[x.qi]; b_ = pool_i.loc[x.pid]
    print(f"MISS  S1: {a.name_n} | {a.addr_n}\n      S23: {b_.name_n} | {b_.addr_n}")
