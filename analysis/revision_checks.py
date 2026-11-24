"""Checks added in the second review round. Writes results/logs/revision_checks.json.

  1. Thresholds chosen inside the training folds: for every classifier, an inner three-fold
     vendor-grouped split of each training fold gives out-of-fold scores; the threshold that maximises
     F1 there is applied to the test fold. The rule's threshold is chosen the same way (scores 2-7),
     next to its fixed default. Paired vendor-cluster bootstrap against the fixed rule.
  2. Fold-level F1 for the tuned baselines (nested C search, as in classify_extra.py).
  3. Splits that keep vendors and near-duplicate clusters together at the same time, and
     sensitivity of the near-duplicate analysis to the cosine threshold (0.6-0.9).
  4. Price and vendor statistics for the custom-intrusion theme only.
  5. End-to-end detection on crawled pages: rule decisions on full page text, on page text without
     the breadcrumb, and on the listing body only (title and description), scored against the
     category labels over all digital listings, with listings the crawl did not reach counted as
     missed.
Run from analysis/.
"""
import html as htmlmod
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import rules
from classify import models, scores_of
from classify_extra import paired_boot, word_vec

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
SEED = 42
N_BOOT = 2000


def prf(y, p):
    return {"precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p)),
            "f1": float(f1_score(y, p))}


def cluster_ci(y, pred, groups, rng):
    u = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in u}
    f = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        if y[i].sum():
            f.append(f1_score(y[i], pred[i]))
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


def best_threshold(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 300)))
    f = [f1_score(y, s >= t) for t in cand]
    return float(cand[int(np.argmax(f))])


def inner_threshold_eval(X, y, groups, rule_scores, rng):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res, preds = {}, {}
    mk = models()
    for name in list(mk) + ["Rule-based (threshold chosen in training folds)"]:
        pred = np.zeros(len(y), int)
        thr_used, fold_f1 = [], []
        for tr, te in outer.split(X, y, groups):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            if name.startswith("Rule"):
                ts = [2, 3, 4, 5, 6, 7]
                f = [f1_score(y[tr], rule_scores[tr] >= t) for t in ts]
                thr = ts[int(np.argmax(f))]
                pred[te] = ((rule_scores[te] >= thr) & (rule_primary[te] >= 1)).astype(int)
            else:
                s_in = np.zeros(len(tr))
                for a, b in inner.split(X[tr], y[tr], groups[tr]):
                    m = mk[name]().fit(X[tr][a], y[tr][a])
                    s_in[b] = scores_of(m, X[tr][b])
                thr = best_threshold(y[tr], s_in)
                m = mk[name]().fit(X[tr], y[tr])
                pred[te] = (scores_of(m, X[te]) >= thr).astype(int)
            thr_used.append(float(thr))
            fold_f1.append(float(f1_score(y[te], pred[te])))
        r = prf(y, pred)
        r.update(ci95=cluster_ci(y, pred, groups, rng), thresholds=thr_used, fold_f1=fold_f1,
                 fold_sd=float(np.std(fold_f1, ddof=1)))
        res[name] = r
        preds[name] = pred
        print(name, round(r["f1"], 3), flush=True)
    return res, preds


def tuned_fold_f1(X, y, groups):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    grids = {"LR tuned (word 1-2)": (lambda: make_pipeline(word_vec(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                                     {"logisticregression__C": [0.25, 1, 4, 16]}),
             "SVM tuned (word 1-2)": (lambda: make_pipeline(word_vec(), LinearSVC(class_weight="balanced")),
                                      {"linearsvc__C": [0.03, 0.1, 0.3, 1]})}
    out = {}
    for name, (mk, grid) in grids.items():
        pred = np.zeros(len(y), int)
        ff = []
        for tr, te in outer.split(X, y, groups):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            gs = GridSearchCV(mk(), grid, scoring="f1", cv=list(inner.split(X[tr], y[tr], groups[tr])), n_jobs=1)
            gs.fit(X[tr], y[tr])
            pred[te] = gs.predict(X[te])
            ff.append(float(f1_score(y[te], pred[te])))
        out[name] = {"f1": float(f1_score(y, pred)), "fold_f1": ff, "fold_sd": float(np.std(ff, ddof=1))}
    return out


def dup_components(X, thr):
    T = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(X)
    S = (T @ T.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    A = S.multiply(S >= thr)
    n, comp = connected_components(A, directed=False)
    has = np.asarray(A.sum(axis=1)).ravel() > 0
    return comp, has


def joint_groups(vendors, comp):
    v_id = pd.factorize(vendors)[0]
    nv = v_id.max() + 1
    rows = v_id
    cols = nv + comp
    n = nv + comp.max() + 1
    G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    return lab[v_id]


def split_f1(X, y, groups, names):
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    mk = models()
    out = {}
    for name in names:
        pred = np.zeros(len(y), int)
        for tr, te in sg.split(X, y, groups):
            pred[te] = mk[name]().fit(X[tr], y[tr]).predict(X[te])
        out[name] = float(f1_score(y, pred))
    return out


def extract_body(page_html):
    h1 = re.search(r"<h1>(.*?)</h1>", page_html, re.S)
    desc = re.search(r"<div class='desc'>(.*?)</div>", page_html, re.S)
    txt = " . ".join(x.group(1) for x in (h1, desc) if x)
    txt = re.sub(r"<[^>]*>", " ", txt)
    return re.sub(r"\s+", " ", htmlmod.unescape(txt)).strip()


rule_primary = None


def main():
    global rule_primary
    rng = np.random.default_rng(SEED)
    out = {}
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].reset_index(drop=True)
    y = dig["haas"].to_numpy().astype(int)
    X = dig["text"].to_numpy()
    groups = dig["Vendor"].to_numpy()
    oof = pd.read_parquet(LOG / "oof_digital.parquet")
    assert (oof["y"].to_numpy() == y).all()
    rule = oof["pred::Rule-based"].to_numpy()
    sc = [rules.score(t) for t in X]
    rule_scores = np.array([s[3] for s in sc])
    rule_primary = np.array([s[0] for s in sc])
    assert (((rule_scores >= rules.THRESHOLD) & (rule_primary >= 1)).astype(int) == rule).all(), "rule scores do not reproduce"

    # 1. thresholds chosen inside training folds
    res, preds = inner_threshold_eval(X, y, groups, rule_scores, rng)
    for name, pred in preds.items():
        res[name]["paired_vs_fixed_rule"] = paired_boot(y, rule, pred, groups, rng)
    out["inner_threshold"] = res
    # 2. tuned fold SDs
    out["tuned_folds"] = tuned_fold_f1(X, y, groups)
    # 3. joint vendor + near-duplicate splits and threshold sensitivity
    sens = {}
    names = ["LR (word 1-2)", "SVM (word 1-2)", "LR (char 2-5)"]
    for thr in [0.6, 0.7, 0.8, 0.9]:
        comp, has = dup_components(X, thr)
        jg = joint_groups(groups, comp)
        sizes = pd.Series(jg).value_counts()
        sens[str(thr)] = {"share_with_near_duplicate": float(has.mean()),
                          "dup_disjoint_f1": split_f1(X, y, comp, names[:2]),
                          "joint_vendor_dup_f1": split_f1(X, y, jg, names),
                          "joint_groups": int(len(sizes)), "largest_joint_group_share": float(sizes.iloc[0] / len(y))}
        print(thr, sens[str(thr)], flush=True)
    out["near_duplicate_sensitivity"] = sens
    # 4. custom-intrusion theme
    top = pd.read_parquet(LOG / "haas_topics.parquet")
    t0 = top[top["topic"] == 0]
    vend = set(t0["Vendor"])
    full = df[df["Vendor"].isin(vend)]
    other = full[full["Category"] != "Services/Hacking"].groupby("Vendor").size()
    q = t0["price_btc"].quantile([0.25, 0.5, 0.75])
    out["custom_intrusion_theme"] = {"listings": int(len(t0)), "vendors": int(len(vend)),
                                     "median_btc": float(q[0.5]), "q1_btc": float(q[0.25]), "q3_btc": float(q[0.75]),
                                     "vendors_selling_elsewhere": int((other > 0).sum()),
                                     "theme_share_of_their_listings": float(len(t0) / len(full))}
    # 5. end-to-end detection on crawled pages
    truth = json.loads((ROOT / "data" / "site" / "truth.json").read_text())
    site = json.loads((ROOT / "data" / "site" / "site.json").read_text())["pages"]
    pages = pd.read_json(ROOT / "results" / "crawl_full" / "pages.jsonl", lines=True)
    pages["path"] = pages["url"].str.replace(r"^https?://[^/]+", "", regex=True)
    got = pages[pages["path"].isin(truth.keys()) & (pages["status"] == 200)].drop_duplicates("path").set_index("path")
    yy, p_src, p_full, p_nocat, p_body, reached = [], [], [], [], [], []
    agree_body = []
    for path, t in truth.items():
        src = t["item"] + " . " + t["desc"]
        yy.append(int(t["haas"]))
        ps = rules.predict(src)[0]
        p_src.append(ps)
        if path in got.index:
            text = got.loc[path, "text"]
            body = extract_body(site[path])
            reached.append(1)
            p_full.append(rules.predict(text)[0])
            p_nocat.append(rules.predict(re.sub(r"Category:\s*\S+", " ", text))[0])
            pb = rules.predict(body)[0]
            p_body.append(pb)
            agree_body.append(pb == ps)
        else:
            reached.append(0)
            p_full.append(0)
            p_nocat.append(0)
            p_body.append(0)
    yy, reached = np.array(yy), np.array(reached)
    e2e = {"listings": int(len(yy)), "positives": int(yy.sum()), "reached": int(reached.sum()),
           "positives_unreached": int(((reached == 0) & (yy == 1)).sum()),
           "body_only_rule_agreement_with_source": float(np.mean(agree_body))}
    for name, p in [("source_text_all_listings", p_src), ("crawled_full_page", p_full),
                    ("crawled_without_breadcrumb", p_nocat), ("crawled_body_only", p_body)]:
        e2e[name] = prf(yy, np.array(p))
    out["end_to_end_detection"] = e2e
    (LOG / "revision_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("inner_threshold",)}, indent=1))


if __name__ == "__main__":
    main()
