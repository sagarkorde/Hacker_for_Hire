"""HaaS listing detection: rule-based detector versus supervised baselines.

Primary setting: digital goods and services subset (hard negatives), label =
marketplace category "Services/Hacking". Evaluation uses vendor-disjoint
5-fold cross-validation (StratifiedGroupKFold, groups = vendor) so that no vendor
appears in both training and test folds. Hyperparameters are library defaults
(no tuning). Uncertainty: per-fold SD and a vendor-cluster bootstrap 95% CI on
pooled out-of-fold predictions. Paired comparison: exact McNemar test.

Outputs results/logs/classify.json and results/logs/oof_digital.parquet.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import rules

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "results" / "logs"
SEED = 42
N_BOOT = 1000


def models():
    word = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    return {
        "NB (word 1-2)": lambda: make_pipeline(TfidfVectorizer(**word), ComplementNB()),
        "LR (word 1-2)": lambda: make_pipeline(TfidfVectorizer(**word), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM (word 1-2)": lambda: make_pipeline(TfidfVectorizer(**word), LinearSVC(class_weight="balanced")),
        "LR (char 2-5)": lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True),
                                               LogisticRegression(max_iter=2000, class_weight="balanced")),
    }


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def prf(y, p):
    return dict(precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0))


def cluster_boot(y, pred, score, groups, rng):
    """Vendor-cluster bootstrap of P, R, F1 and average precision."""
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    out = {k: [] for k in ["precision", "recall", "f1", "ap"]}
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        yy, pp = y[idx], pred[idx]
        if yy.sum() == 0:
            continue
        m = prf(yy, pp)
        for k in ["precision", "recall", "f1"]:
            out[k].append(m[k])
        out["ap"].append(average_precision_score(yy, score[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}


def run_cv(df, splitter, groups, label="haas", with_rule=True, model_names=None):
    y = df[label].to_numpy().astype(int)
    X = df["text"].to_numpy()
    res, oof_pred, oof_score, per_fold = {}, {}, {}, {}
    names = model_names or list(models())
    if with_rule:
        rp = np.array([rules.predict(t) for t in X])
        oof_pred["Rule-based"], oof_score["Rule-based"] = rp[:, 0], rp[:, 1].astype(float)
        per_fold["Rule-based"] = []
    for n in names:
        oof_pred[n], oof_score[n], per_fold[n] = np.zeros(len(y), int), np.zeros(len(y)), []
    folds = list(splitter.split(X, y, groups))
    for tr, te in folds:
        if with_rule:
            per_fold["Rule-based"].append(prf(y[te], oof_pred["Rule-based"][te]))
        for n in names:
            m = models()[n]()
            m.fit(X[tr], y[tr])
            oof_pred[n][te] = m.predict(X[te])
            oof_score[n][te] = scores_of(m, X[te])
            per_fold[n].append(prf(y[te], oof_pred[n][te]))
    for n in oof_pred:
        pf = pd.DataFrame(per_fold[n])
        res[n] = {"pooled": prf(y, oof_pred[n]), "ap": float(average_precision_score(y, oof_score[n])),
                  "fold_mean": pf.mean().to_dict(), "fold_sd": pf.std(ddof=1).to_dict()}
    return res, oof_pred, oof_score, folds


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].reset_index(drop=True)
    groups = dig["Vendor"].to_numpy()
    y = dig["haas"].to_numpy().astype(int)
    out = {"n_digital": int(len(dig)), "n_pos": int(y.sum()), "n_vendors": int(len(np.unique(groups)))}

    # ---- E1: vendor-disjoint CV on the digital subset (primary result)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res, oof_pred, oof_score, folds = run_cv(dig, sgkf, groups)
    out["fold_positive_counts"] = [int(y[te].sum()) for _, te in folds]
    out["fold_sizes"] = [int(len(te)) for _, te in folds]
    for n in res:
        res[n]["ci95"] = cluster_boot(y, oof_pred[n], oof_score[n], groups, rng)
    out["E1_vendor_disjoint"] = res
    best = max([n for n in res if n != "Rule-based"], key=lambda n: res[n]["pooled"]["f1"])
    out["best_ml"] = best
    out["mcnemar_rule_vs_best"] = mcnemar(y, oof_pred["Rule-based"], oof_pred[best])
    curves = {}
    for n in res:
        p, r, _ = precision_recall_curve(y, oof_score[n])
        curves[n] = {"precision": p[::max(1, len(p) // 400)].tolist(), "recall": r[::max(1, len(r) // 400)].tolist()}
    (LOGDIR / "pr_curves.json").write_text(json.dumps(curves))

    # Error analysis: false positives by true category
    fp = {}
    for n in ["Rule-based", best]:
        mask = (oof_pred[n] == 1) & (y == 0)
        fp[n] = {k: int(v) for k, v in dig.loc[mask, "Category"].value_counts().head(8).items()}
        fp[n]["_total"] = int(mask.sum())
    out["false_positives_by_category"] = fp
    fn = {}
    for n in ["Rule-based", best]:
        fn[n] = int(((oof_pred[n] == 0) & (y == 1)).sum())
    out["false_negatives"] = fn
    pd.DataFrame({"Vendor": dig["Vendor"], "Category": dig["Category"], "Item": dig["Item"], "y": y,
                  **{f"pred::{k}": v for k, v in oof_pred.items()}, **{f"score::{k}": v for k, v in oof_score.items()}}
                 ).to_parquet(LOGDIR / "oof_digital.parquet", index=False)

    # ---- E2 (ablation): random, vendor-overlapping split to show leakage inflation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res_r, _, _, _ = run_cv(dig, skf, groups, with_rule=False, model_names=["LR (word 1-2)", "SVM (word 1-2)"])
    out["E2_random_split"] = res_r

    # ---- E3 (ablation): rule components and threshold
    X = dig["text"].to_numpy()
    abl = {}
    for name, kw in {"primary terms only": dict(use_secondary=False, use_entities=False),
                     "+ secondary terms": dict(use_secondary=True, use_entities=False),
                     "+ entities (full rule)": dict(use_secondary=True, use_entities=True)}.items():
        pr = np.array([rules.predict(t, **kw)[0] for t in X])
        abl[name] = prf(y, pr)
    thr = {}
    for t in range(2, 8):
        pr = np.array([rules.predict(x, threshold=t)[0] for x in X])
        thr[str(t)] = prf(y, pr)
    out["E3_rule_components"] = abl
    out["E3_rule_threshold"] = thr

    # ---- E4 (robustness): full market including drug listings, vendor-disjoint
    full = df.reset_index(drop=True)
    res_f, _, _, _ = run_cv(full, StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
                            full["Vendor"].to_numpy(), model_names=["LR (word 1-2)"])
    out["E4_full_market"] = {k: {"pooled": v["pooled"], "ap": v["ap"]} for k, v in res_f.items()}
    out["E4_n"] = int(len(full))

    # ---- E5: inference cost (single process)
    m = models()[best]().fit(X, y)
    t0 = time.perf_counter()
    for _ in range(3):
        m.predict(X)
    ml_rate = 3 * len(X) / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for x in X:
        rules.predict(x)
    rule_rate = len(X) / (time.perf_counter() - t0)
    out["E5_docs_per_s"] = {"Rule-based": rule_rate, best: ml_rate}
    out["runtime_s"] = time.time() - t_start
    (LOGDIR / "classify.json").write_text(json.dumps(out, indent=2))

    rows = []
    for n, v in res.items():
        rows.append([n, v["pooled"]["precision"], v["pooled"]["recall"], v["pooled"]["f1"], v["ci95"]["f1"][0], v["ci95"]["f1"][1],
                     v["fold_mean"]["f1"], v["fold_sd"]["f1"], v["ap"]])
    print(pd.DataFrame(rows, columns=["model", "P", "R", "F1", "F1_lo", "F1_hi", "F1_fold_mean", "F1_fold_sd", "AP"]).round(3).to_string())
    print("best", best, out["mcnemar_rule_vs_best"])
    print("random split", {k: round(v["pooled"]["f1"], 3) for k, v in res_r.items()})
    print("components", {k: round(v["f1"], 3) for k, v in abl.items()})
    print("threshold", {k: round(v["f1"], 3) for k, v in thr.items()})
    print("full market", {k: round(v["pooled"]["f1"], 3) for k, v in res_f.items()})
    print("fp", fp)
    print("rates", out["E5_docs_per_s"], "runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
