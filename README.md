# Business Entity Resolution — Amazon ML Challenge 2026

Match each Source-1 business to its records in Source-2 / Source-3 across US, India,
and (test-only, unseen) France. Metric: macro-averaged **F0.5** per S1 entity, so
precision is weighted 2x recall — wrong merges cost more than misses, and correctly
predicting an empty list for a singleton scores 1.0.

## Status (as of handoff)

Built and validated end-to-end **except** the full-scale test run, which needs more
compute than the 2-core / 7 GB cloud box could deliver in time (moving to an 8-core Mac).

| Stage | State |
|---|---|
| Normalization (`normalize.py`, `prep.py`) | Done — all 7 files normalized to `data/norm/*.parquet` |
| Blocking (`blocking.py`) | Done — sparse 5-signal TF-IDF + top-k. US recall 0.97, India 0.87 @k=20 |
| Features (`features.py`) | Done — 17 pairwise similarity features, cross-script robust |
| Matcher (`train_model.py`, `model/`) | Trained LightGBM. Val macro-F0.5 ≈ 0.94 (optimistic; see notes) |
| Inference (`infer.py`) | Written — produces both required TSVs; not yet run on full test |
| Packaging | Pending |

## Pipeline

1. **Normalize** — lowercase; `unidecode` folds accents & transliterates Devanagari→ASCII;
   expand street/legal abbreviations (Rd→road, Pvt→private, SARL kept); strip PO BOX / PMB / N/A;
   detect domain-style names; build a vowel-free **consonant skeleton** (script-robust key).
2. **Block** — per country, one sparse vector per record = weighted concat of:
   address words · address char 3-4grams · name words · name char 3grams · **skeleton char 3-4grams**.
   Top-k neighbours via `sparse_dot_topn`, pool sharded to bound memory. The skeleton block is
   what recovers transliterated Indian names ("private limited" and "praaivett limittedd" → same `prvtlmtd`).
3. **Score** — LightGBM on 17 features (rapidfuzz token/char ratios, Jaccard, Jaro-Winkler on skeleton,
   address overlap, numeric-token overlap, is-domain, non-latin flag, block score).
4. **Decide** — keep candidates with prob ≥ threshold (0.9, tuned for F0.5); empty list allowed → singletons.

## Key findings (what works / what doesn't)

- **Sparse TF-IDF matmul is the only high-recall blocker** tested (US 0.97 / India 0.87). But ~15-20h
  per country on 2 cores — the reason for moving to the Mac.
- **Dense embeddings (SVD→FAISS) fail**: even *exact* search on 200-dim SVD gave only 0.51 recall.
  The sparse char-n-gram signal doesn't survive dimensionality reduction. FAISS search is fast (min)
  but useless at this recall. IVFPQ compression was worse (0.34).
- **Exact / q-gram keys** are fast but cap at 0.64-0.74 recall — they miss typos in the key token itself.
- **India is the hard country** (47% of test): Devanagari names transliterate to phonetic garble, so raw
  n-grams fail; addresses are heavily reordered/abbreviated. The skeleton block + address matching lift it.
- **France (15%, unseen in train)**: pipeline is deliberately country-agnostic — vocab/IDF fit per country,
  no country-specific features — so the model transfers. `unidecode` handles French accents; SARL/SAS/SCI
  survive as name tokens.

## Caveats / TODO

- The **val macro-F0.5 ≈ 0.94 is optimistic**: it scores only candidates blocking surfaced, so it does not
  penalize matches blocking missed. True score is bounded by blocking recall (esp. India 0.87).
- Training pairs were generated with the earlier 3-block blocker; **regenerate with the 5-block blocker**
  so train/test candidate distributions match, then retrain.
- Not yet produced: `output/candidate_pairs.tsv`, `output/matching_results.tsv`, validator run, methodology doc.

## Files

```
src/normalize.py       text normalization (name + address)
src/prep.py            batch-normalize raw parquet → data/norm/
src/blocking.py        sparse 5-signal blocker + sharded top-k search   ← USE THIS
src/features.py        17 pairwise features
src/gen_train_pairs.py run blocking on train sample → labeled pairs
src/train_model.py     train LightGBM + tune F0.5 threshold
src/infer.py           full test inference → both TSVs
src/keyblock.py        (rejected) exact-key blocking — kept for reference
src/svdblock.py        (rejected) SVD+FAISS — kept for reference
model/matcher.txt      trained LightGBM booster
model/meta.json        threshold + feature list
data/trainpairs/*.parquet   labeled US/India candidate pairs
```

Data files (`data/norm/`, raw TSVs) are not included — they're regenerated from the dataset via `prep.py`.
