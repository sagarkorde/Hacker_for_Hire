"""Score the human-annotation study once results/annotation_sample.csv has been labelled.

Inputs
  results/annotation_sample.csv  columns annotator_A, annotator_B, adjudicated filled with the
                                 labels of docs/annotation_guideline.md
  results/logs/oof_digital.parquet  out-of-fold predictions from classify.py
Output
  results/logs/annotation_eval.json  (read by paper/build_doc.py when present)

What it reports
  * agreement: raw agreement and Cohen's kappa for the six labels and for the binary HaaS label
  * label quality: precision and recall of the market category (Services/Hacking) against the
    adjudicated binary label
  * detector quality: precision, recall and F1 of the rule and of the best TF-IDF model against the
    adjudicated binary label
Estimates are design-weighted: each stratum is weighted by (stratum size in the digital subset) /
(stratum size in the sample), so they refer to the whole digital subset, not to the sample.
95% intervals come from a vendor-cluster bootstrap within the sample (2,000 replicates).

Run:  python analysis/score_annotation.py [--csv path]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import rules

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
LABELS = ["HAAS_SERVICE", "HAAS_TOOL", "HAAS_GUIDE", "OTHER_FRAUD", "ACCOUNTS_MEDIA", "OTHER"]
HAAS = {"HAAS_SERVICE", "HAAS_TOOL", "HAAS_GUIDE"}
BEST = "LR (char 2-5)"
N_BOOT = 2000
SEED = 7


def stratum_sizes():
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].reset_index(drop=True)
    oof = pd.read_parquet(LOG / "oof_digital.parquet")
    flagged = (oof["pred::Rule-based"] == 1) | (oof[f"pred::{BEST}"] == 1)
    haas = dig["haas"].to_numpy()
    sizes = {"market_label_hacking": int(haas.sum()),
             "flagged_other_category": int((~haas & flagged).sum()),
             "random_other": int((~haas & ~flagged).sum())}
    key = dig[["Vendor", "Item", "Item Description"]].astype(str).agg("\x1f".join, axis=1)
    best = pd.Series(oof[f"pred::{BEST}"].to_numpy(), index=key)
    best = best[~best.index.duplicated()]
    return sizes, best


def weighted_prf(y, pred, w):
    tp = float(np.sum(w[(y == 1) & (pred == 1)]))
    fp = float(np.sum(w[(y == 0) & (pred == 1)]))
    fn = float(np.sum(w[(y == 1) & (pred == 0)]))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def boot(y, pred, w, groups, rng):
    u = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in u}
    res = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        if y[i].sum() == 0:
            continue
        res.append(weighted_prf(y[i], pred[i], w[i]))
    res = np.array(res)
    return [[float(np.percentile(res[:, k], 2.5)), float(np.percentile(res[:, k], 97.5))] for k in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "annotation_sample.csv"))
    ap.add_argument("--out", default=str(LOG / "annotation_eval.json"))
    a = ap.parse_args()
    s = pd.read_csv(a.csv, keep_default_na=False)
    for c in ["annotator_A", "annotator_B", "adjudicated"]:
        s[c] = s[c].str.strip().str.upper()
        bad = set(s[c]) - set(LABELS)
        if bad:
            raise SystemExit(f"column {c}: unknown or empty labels {sorted(bad)[:5]}; label every row first")
    sizes, best = stratum_sizes()
    n_s = s["stratum"].value_counts().to_dict()
    w = s["stratum"].map(lambda k: sizes[k] / n_s[k]).to_numpy()
    text = (s["Item"].astype(str) + " . " + s["Item Description"].astype(str)).str.strip()
    rule = np.array([rules.predict(t)[0] for t in text])
    key = s[["Vendor", "Item", "Item Description"]].astype(str).agg("\x1f".join, axis=1)
    ml = key.map(best)
    if ml.isna().any():
        raise SystemExit(f"{int(ml.isna().sum())} sampled rows not found in oof_digital.parquet")
    ml = ml.to_numpy().astype(int)
    market = (s["stratum"] == "market_label_hacking").to_numpy().astype(int)
    yA = s["annotator_A"].isin(HAAS).to_numpy().astype(int)
    yB = s["annotator_B"].isin(HAAS).to_numpy().astype(int)
    y = s["adjudicated"].isin(HAAS).to_numpy().astype(int)
    y_strict = (s["adjudicated"] == "HAAS_SERVICE").to_numpy().astype(int)
    groups = s["Vendor"].to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"n": int(len(s)), "strata_sample": n_s, "strata_population": sizes,
           "agreement": {"raw_6": float((s["annotator_A"] == s["annotator_B"]).mean()),
                         "kappa_6": float(cohen_kappa_score(s["annotator_A"], s["annotator_B"])),
                         "raw_binary": float((yA == yB).mean()),
                         "kappa_binary": float(cohen_kappa_score(yA, yB))},
           "adjudicated_counts": s["adjudicated"].value_counts().to_dict(),
           "weighted_haas_prevalence": float(np.sum(w * y) / np.sum(w)),
           "weighted_haas_service_prevalence": float(np.sum(w * y_strict) / np.sum(w))}
    for name, pred in [("market_label", market), ("rule", rule), (BEST, ml)]:
        for lab, yy in [("haas", y), ("haas_service_only", y_strict)]:
            p, r, f = weighted_prf(yy, pred, w)
            ci = boot(yy, pred, w, groups, rng)
            out.setdefault(name, {})[lab] = {"precision": p, "recall": r, "f1": f,
                                             "ci95": {"precision": ci[0], "recall": ci[1], "f1": ci[2]}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["agreement"], indent=1))
    print({k: round(v["haas"]["f1"], 3) for k, v in out.items() if isinstance(v, dict) and "haas" in v})


if __name__ == "__main__":
    main()
