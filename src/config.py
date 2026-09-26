"""Central path config, resolved relative to the repo root.

Replaces the old hard-coded /home/claude/work/... paths so the pipeline runs
from wherever the repo is checked out. Every module imports its dirs from here.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "pq") + os.sep          # raw parquet (converted from the TSVs)
NORM = os.path.join(DATA, "norm") + os.sep       # normalized parquet (prep.py output)
TRAINPAIRS = os.path.join(DATA, "trainpairs") + os.sep
MODELDIR = os.path.join(ROOT, "model") + os.sep
OUTDIR = os.path.join(ROOT, "output") + os.sep

# where the original TSVs live (source for the parquet conversion)
TSV_TRAIN = os.path.join(DATA, "trainpairs", "pq", "train") + os.sep
TSV_TEST = os.path.join(DATA, "trainpairs", "pq", "test") + os.sep

for _d in (RAW, NORM, TRAINPAIRS, MODELDIR, OUTDIR):
    os.makedirs(_d, exist_ok=True)
