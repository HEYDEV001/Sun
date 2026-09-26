"""Train a LightGBM ranker/classifier on labelled candidate pairs and tune a
decision threshold that maximises macro-averaged F0.5 (the challenge metric)."""
import sys, time, json
import numpy as np, pandas as pd
import lightgbm as lgb
from features import FEATURE_NAMES
import config

D = config.TRAINPAIRS
MODELDIR = config.MODELDIR

def load():
    dfs = [pd.read_parquet(D + f"{c}.parquet") for c in ("US", "India")]
    return pd.concat(dfs, ignore_index=True)

def f_beta(prec, rec, beta=0.5):
    b2 = beta * beta
    d = b2 * prec + rec
    return (1 + b2) * prec * rec / d if d else 0.0

def macro_f05(df, keep):
    """df has columns qeid, y, pred(prob). keep = boolean mask of predicted matches.
    Score per S1 entity, then average. Entities with no true match score 1 iff we
    predict nothing for them."""
    g = df.assign(keep=keep)
    per = g.groupby("qeid").apply(
        lambda x: _entity_f05(x.y.values, x.keep.values), include_groups=False)
    return per.mean()

def _entity_f05(y, keep):
    tp = int((keep & (y == 1)).sum()); fp = int((keep & (y == 0)).sum())
    fn = int((~keep & (y == 1)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0                      # true singleton, predicted nothing
    if tp + fp == 0:                    # predicted nothing but there were matches
        return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn) if (tp + fn) else 0.0
    return f_beta(prec, rec)

def main():
    import os; os.makedirs(MODELDIR, exist_ok=True)
    df = load()
    qids = df.qeid.unique()
    rng = np.random.RandomState(0); rng.shuffle(qids)
    cut = int(0.8 * len(qids)); train_q = set(qids[:cut])
    tr = df[df.qeid.isin(train_q)]; va = df[~df.qeid.isin(train_q)]
    print(f"pairs train={len(tr):,} val={len(va):,} pos_rate={df.y.mean():.3f}", flush=True)

    params = dict(objective="binary", n_estimators=600, learning_rate=0.05,
                  num_leaves=63, min_child_samples=100, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2, verbose=-1,
                  scale_pos_weight=(tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    m = lgb.LGBMClassifier(**params)
    m.fit(tr[FEATURE_NAMES], tr.y,
          eval_set=[(va[FEATURE_NAMES], va.y)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    va = va.assign(pred=m.predict_proba(va[FEATURE_NAMES])[:, 1])

    # tune global threshold on val for macro-F0.5
    best = (0, 0.5)
    for thr in np.arange(0.10, 0.95, 0.02):
        s = macro_f05(va, va.pred.values >= thr)
        if s > best[0]:
            best = (s, round(float(thr), 3))
    print(f"BEST val macro-F0.5={best[0]:.4f} at thr={best[1]}", flush=True)
    # also report top-1-guaranteed variant (keep argmax if >= lower thr)
    imp = dict(zip(FEATURE_NAMES, m.feature_importances_.tolist()))
    print("importances:", json.dumps({k: v for k, v in sorted(imp.items(), key=lambda x: -x[1])}), flush=True)
    m.booster_.save_model(MODELDIR + "matcher.txt")
    json.dump({"threshold": best[1], "val_f05": best[0], "features": FEATURE_NAMES},
              open(MODELDIR + "meta.json", "w"), indent=2)
    print("saved model", flush=True)

if __name__ == "__main__":
    main()
