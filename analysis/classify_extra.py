"""Additional analyses requested in review.

  1. Paired vendor-cluster bootstrap of F1 differences (each model minus the rule).
  2. Tuned baselines: nested vendor-grouped CV (inner 3-fold grid over C).
  3. Leakage decomposition: within-vendor near-duplicates, near-duplicate-disjoint
     splits, and vendor names inside listing text.
  4. Dictionary-independent label audit: share of Services/Hacking listings in
     NMF themes that describe accounts, pay-TV or social-media engagement.
  5. Exported fold assignments for the transformer notebook.
Writes results/logs/classify_extra.json and results/logs/folds_vendor_disjoint.csv.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
SEED = 42
N_BOOT = 2000
NEAR_DUP = 0.8


def paired_boot(y, a, b, groups, rng):
    u = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in u}
    d = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        if y[i].sum() == 0:
            continue
        d.append(f1_score(y[i], b[i]) - f1_score(y[i], a[i]))
    d = np.array(d)
    return {"diff": float(f1_score(y, b) - f1_score(y, a)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_le_0": float((d <= 0).mean())}


def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].reset_index(drop=True)
    y = dig["haas"].to_numpy().astype(int)
    X = dig["text"].to_numpy()
    groups = dig["Vendor"].to_numpy()
    oof = pd.read_parquet(LOG / "oof_digital.parquet")
    assert (oof["y"].to_numpy() == y).all()

    # 1. paired cluster bootstrap: model minus rule
    rule = oof["pred::Rule-based"].to_numpy()
    out["paired_f1_diff_vs_rule"] = {m: paired_boot(y, rule, oof[f"pred::{m}"].to_numpy(), groups, rng)
                                    for m in ["NB (word 1-2)", "LR (word 1-2)", "SVM (word 1-2)", "LR (char 2-5)"]}

    # 5. export the exact vendor-disjoint folds used in classify.py
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(sgkf.split(X, y, groups)):
        fold[te] = k
    pd.DataFrame({"row": np.arange(len(y)), "vendor": groups, "fold": fold, "label": y, "text": X,
                  "rule_pred": oof["pred::Rule-based"].to_numpy(), "lr_char_pred": oof["pred::LR (char 2-5)"].to_numpy()}).to_csv(
        LOG / "folds_vendor_disjoint.csv", index=False)

    # 2. tuned baselines with nested vendor-grouped CV
    tuned = {}
    specs = {"LR tuned (word 1-2)": (make_pipeline(word_vec(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                                     {"logisticregression__C": [0.25, 1, 4, 16]}),
             "SVM tuned (word 1-2)": (make_pipeline(word_vec(), LinearSVC(class_weight="balanced")),
                                      {"linearsvc__C": [0.03, 0.1, 0.3, 1]})}
    for name, (pipe, grid) in specs.items():
        pred = np.zeros(len(y), int)
        score = np.zeros(len(y))
        chosen = []
        for k in range(5):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipe, grid, scoring="f1", cv=list(inner.split(X[tr], y[tr], groups[tr])), n_jobs=1)
            gs.fit(X[tr], y[tr])
            chosen.append(list(gs.best_params_.values())[0])
            pred[te] = gs.predict(X[te])
            m = gs.best_estimator_
            score[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
        tuned[name] = {"precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
                       "f1": float(f1_score(y, pred)), "ap": float(average_precision_score(y, score)), "chosen_C": chosen,
                       "paired_vs_rule": paired_boot(y, rule, pred, groups, rng)}
    out["tuned"] = tuned

    # 3. leakage decomposition
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    T = vec.fit_transform(X)
    S = (T @ T.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    A = S.multiply(S >= NEAR_DUP)
    n_comp, comp = connected_components(A, directed=False)
    pos = np.flatnonzero(y == 1)
    same_vendor_dup = []
    for r in pos:
        row = S.getrow(r)
        cols, vals = row.indices, row.data
        sv = (groups[cols] == groups[r]) & (vals >= NEAR_DUP)
        same_vendor_dup.append(bool(sv.any()))
    out["near_duplicates"] = {"threshold": NEAR_DUP, "components": int(n_comp), "listings": int(len(y)),
                              "haas_with_same_vendor_near_duplicate": float(np.mean(same_vendor_dup)),
                              "all_with_any_near_duplicate": float(np.mean(np.asarray(A.sum(axis=1)).ravel() > 0))}
    # near-duplicate-disjoint (but vendor-overlapping) split
    nd = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    for name, mk in {"LR (word 1-2)": lambda: make_pipeline(word_vec(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                     "SVM (word 1-2)": lambda: make_pipeline(word_vec(), LinearSVC(class_weight="balanced"))}.items():
        pred = np.zeros(len(y), int)
        for tr, te in nd.split(X, y, comp):
            pred[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        res[name] = float(f1_score(y, pred))
    out["near_duplicate_disjoint_f1"] = res
    # vendor names inside listing text
    def vname_in(v, t):
        v = re.sub(r"[^a-z0-9]", "", v.lower())
        return len(v) >= 4 and v in re.sub(r"[^a-z0-9]", "", t.lower())
    has = np.array([vname_in(v, t) for v, t in zip(groups, X)])
    out["vendor_name_in_text"] = {"all": float(has.mean()), "haas": float(has[y == 1].mean())}

    # 4. dictionary-independent label audit from NMF themes (labels in nmf_topics.json order)
    topics = json.loads((LOG / "nmf_topics.json").read_text())
    non_intrusion = [1, 3, 4]  # resold accounts; pay-TV/IPTV; social-media likes
    n = sum(t["n_listings"] for t in topics)
    out["nmf_non_intrusion_share"] = {"themes": non_intrusion, "listings": int(sum(topics[i]["n_listings"] for i in non_intrusion)),
                                      "share": float(sum(topics[i]["n_listings"] for i in non_intrusion) / n)}
    out["rule_recall_cap"] = float(json.loads((LOG / "label_audit.json").read_text())["haas_share_with_primary"])

    (LOG / "classify_extra.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
