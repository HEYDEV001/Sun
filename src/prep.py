"""Normalise every raw parquet (train/test, S1/S2/S3) into a compact frame.

Output columns: entity_id, country, name_n, name_sq, name_skel, is_dom, nonlatin,
                addr_n, nums, has_addr
"""
import sys, time
import pandas as pd
from multiprocessing import Pool
from normalize import norm_name, norm_addr, skeleton
import config

RAW = config.RAW
OUT = config.NORM

def _chunk(args):
    names, addrs = args
    n = [norm_name(x) for x in names]
    a = [norm_addr(x) for x in addrs]
    return (
        [x[0] for x in n], [x[1] for x in n], [skeleton(x[1]) for x in n],
        [x[2] for x in n], [x[3] for x in n], [x[0] for x in a], [x[1] for x in a],
    )

def run(name, pool):
    t = time.time()
    df = pd.read_parquet(f"{RAW}{name}.parquet")
    step = 100_000
    jobs = [(df.business_name.values[i:i+step], df.business_address.values[i:i+step])
            for i in range(0, len(df), step)]
    res = pool.map(_chunk, jobs)
    cols = list(zip(*res))
    flat = [sum((list(c) for c in col), []) for col in cols]
    out = pd.DataFrame({
        "entity_id": df.entity_id.values, "country": df.country.values,
        "name_n": flat[0], "name_sq": flat[1], "name_skel": flat[2],
        "is_dom": flat[3], "nonlatin": flat[4], "addr_n": flat[5], "nums": flat[6],
    })
    out["is_dom"] = out.is_dom.astype("int8"); out["nonlatin"] = out.nonlatin.astype("int8")
    out["has_addr"] = (out.addr_n.str.len() > 0).astype("int8")
    out.to_parquet(f"{OUT}{name}.parquet")
    print(name, len(out), f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    workers = int(os.environ.get("PREP_WORKERS", "6"))   # 8-core box; leave headroom
    with Pool(workers) as pool:
        for n in ["test_source1", "train_source1", "test_source2", "test_source3", "train_source2", "train_source3"]:
            run(n, pool)
