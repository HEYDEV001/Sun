"""Convert the raw TSVs (data/trainpairs/pq/{train,test}/*.tsv) into the flat
parquet files the pipeline expects at data/pq/<name>.parquet.

Streamed batch-by-batch via pyarrow so peak memory stays well under the 8 GB box.
All columns are read as strings (entity_id / ids must not be coerced to numbers,
and empty matched_entity_ids must survive as "").
"""
import time
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import config

# (source tsv path, output parquet basename, string columns)
SRC_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]

JOBS = [
    (config.TSV_TRAIN + "train_source1.tsv", "train_source1", SRC_COLS),
    (config.TSV_TRAIN + "train_source2.tsv", "train_source2", SRC_COLS),
    (config.TSV_TRAIN + "train_source3.tsv", "train_source3", SRC_COLS),
    (config.TSV_TRAIN + "train_ground_truth.tsv", "train_ground_truth", GT_COLS),
    (config.TSV_TEST + "test_source1.tsv", "test_source1", SRC_COLS),
    (config.TSV_TEST + "test_source2.tsv", "test_source2", SRC_COLS),
    (config.TSV_TEST + "test_source3.tsv", "test_source3", SRC_COLS),
]


def convert(src, name, cols):
    t = time.time()
    out = f"{config.RAW}{name}.parquet"
    ropts = pv.ReadOptions(block_size=32 << 20)          # 32 MB read blocks
    popts = pv.ParseOptions(delimiter="\t")
    copts = pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                              strings_can_be_null=True)
    reader = pv.open_csv(src, read_options=ropts, parse_options=popts,
                         convert_options=copts)
    writer = None
    n = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(out, batch.schema, compression="snappy")
            writer.write_batch(batch)
            n += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
        reader.close()
    print(f"{name:22} {n:>10,} rows -> {out}  ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    for src, name, cols in JOBS:
        convert(src, name, cols)
    print("all conversions done", flush=True)
