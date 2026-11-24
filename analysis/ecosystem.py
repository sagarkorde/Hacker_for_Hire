"""Characterise the hacking-services segment of the Agora market.

Outputs results/logs/ecosystem.json, results/logs/label_audit.json and
results/logs/nmf_topics.json. All statistics are computed from
data/listings_clean.parquet (deduplicated listings).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

import rules

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
SEED = 42


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def concentration(counts):
    c = np.sort(np.asarray(counts))[::-1]
    s = c / c.sum()
    return {"n_vendors": int(len(c)), "top1_share": float(s[:1].sum()), "top5_share": float(s[:5].sum()),
            "top10_share": float(s[:10].sum()), "gini": gini(c), "hhi": float((s ** 2).sum())}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def main():
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].copy()
    haas = dig[dig["haas"]].copy()
    out = {}

    # ---- label audit (why category labels are a noisy proxy)
    sc = np.array([rules.score(t) for t in dig["text"]])
    dig["primary_hits"] = sc[:, 0]
    h = dig[dig["haas"]]
    audit = {"haas_with_any_primary_term": int((h["primary_hits"] >= 1).sum()), "haas_total": int(len(h)),
             "haas_share_with_primary": float((h["primary_hits"] >= 1).mean()),
             "nonhaas_with_2plus_primary": int(((~dig["haas"]) & (dig["primary_hits"] >= 2)).sum()),
             "nonhaas_2plus_by_category": {k: int(v) for k, v in dig[(~dig["haas"]) & (dig["primary_hits"] >= 2)]["Category"].value_counts().head(8).items()}}
    (LOG / "label_audit.json").write_text(json.dumps(audit, indent=2))

    # ---- vendor concentration
    out["concentration_haas"] = concentration(haas["Vendor"].value_counts().values)
    out["concentration_digital"] = concentration(dig["Vendor"].value_counts().values)
    out["concentration_full_market"] = concentration(df["Vendor"].value_counts().values)

    # ---- prices (BTC); indicative USD at 2014-2015 daily closes
    btc = pd.read_csv(ROOT / "data" / "btc_usd_daily.csv", thousands=",")
    btc.columns = [c.strip().strip('﻿').strip('"') for c in btc.columns]
    btc["Date"] = pd.to_datetime(btc["Date"], format="%b %d, %Y")
    win = btc[(btc["Date"] >= "2014-01-01") & (btc["Date"] <= "2015-12-31")]["Price"].astype(float)
    rate = {"median": float(win.median()), "p10": float(win.quantile(0.1)), "p90": float(win.quantile(0.9)), "days": int(len(win))}
    out["btc_usd_2014_2015"] = rate
    focus = ["Services/Hacking", "Data/Accounts", "Data/Software", "Data/Pirated", "Services/Money", "Information/Guides", "Services/Other"]
    price_tab = {}
    for c in focus:
        p = dig.loc[(dig["Category"] == c) & (dig["price_btc"] > 0), "price_btc"]
        price_tab[c] = {"n": int(len(p)), "median_btc": float(p.median()), "q1_btc": float(p.quantile(.25)),
                        "q3_btc": float(p.quantile(.75)), "median_usd_indicative": float(p.median() * rate["median"])}
    out["prices_by_category"] = price_tab
    groups = [dig.loc[(dig["Category"] == c) & (dig["price_btc"] > 0), "price_btc"].values for c in focus]
    kw = kruskal(*groups)
    n_kw = int(sum(len(g) for g in groups))
    out["kruskal_prices"] = {"H": float(kw.statistic), "p": float(kw.pvalue), "categories": focus, "n": n_kw,
                             "epsilon_squared": float(kw.statistic / (n_kw - 1))}
    a = haas.loc[haas["price_btc"] > 0, "price_btc"].values
    b = dig.loc[(~dig["haas"]) & (dig["price_btc"] > 0), "price_btc"].values
    mw = mannwhitneyu(a, b, alternative="two-sided")
    out["haas_vs_other_digital_price"] = {"U": float(mw.statistic), "p": float(mw.pvalue), "cliffs_delta": cliffs_delta(a, b),
                                          "median_haas": float(np.median(a)), "median_other": float(np.median(b))}
    out["haas_share_under_usd"] = {str(t): float((haas["price_btc"] * rate["median"] < t).mean()) for t in [10, 50, 100, 1000]}

    # ---- price anomalies within the HaaS segment (log10 BTC)
    hp = haas[haas["price_btc"] > 0].copy()
    lx = np.log10(hp["price_btc"].values)
    q1, q3 = np.percentile(lx, [25, 75])
    iqr_flag = (lx < q1 - 1.5 * (q3 - q1)) | (lx > q3 + 1.5 * (q3 - q1))
    med = np.median(lx)
    mad = np.median(np.abs(lx - med))
    rz_flag = np.abs((lx - med) / (1.4826 * mad)) > 2.5  # Leys et al. (2013) moderately conservative cut-off
    feats = np.column_stack([lx, np.log1p(hp["text"].str.len().values), hp["rating"].fillna(hp["rating"].median()).values])
    if_flag = IsolationForest(random_state=SEED, contamination="auto").fit_predict(feats) == -1

    def jac(x, y):
        return float((x & y).sum() / max(1, (x | y).sum()))
    out["price_anomalies_haas"] = {"n": int(len(hp)), "iqr": int(iqr_flag.sum()), "robust_z": int(rz_flag.sum()),
                                   "isolation_forest": int(if_flag.sum()), "all_three": int((iqr_flag & rz_flag & if_flag).sum()),
                                   "jaccard_iqr_rz": jac(iqr_flag, rz_flag), "jaccard_iqr_if": jac(iqr_flag, if_flag),
                                   "jaccard_rz_if": jac(rz_flag, if_flag),
                                   "iqr_subset_of_if": bool((iqr_flag & ~if_flag).sum() == 0),
                                   "rz_subset_of_if": bool((rz_flag & ~if_flag).sum() == 0),
                                   "iqr_high": int((lx > q3 + 1.5 * (q3 - q1)).sum()), "iqr_low": int((lx < q1 - 1.5 * (q3 - q1)).sum())}

    # ---- contact and payment entities (descriptions are truncated in the source data)
    def ent_share(frame):
        e = pd.DataFrame([rules.entities(t) for t in frame["text"]])
        return {k: float((e[k] > 0).mean()) for k in e.columns} | {"any": float((e.sum(axis=1) > 0).mean())}
    out["entity_share_haas"] = ent_share(haas)
    out["entity_share_other_digital"] = ent_share(dig[~dig["haas"]])

    # ---- vendor-category network: cross-listing of HaaS vendors
    hv = set(haas["Vendor"])
    vend = df[df["Vendor"].isin(hv)]
    tops = vend.groupby("Vendor")["top"].apply(set)
    cats = vend.groupby("Vendor")["Category"].apply(set)
    out["haas_vendors_cross_listing"] = {
        "n_vendors": len(hv),
        "only_haas_category": int(sum(1 for v in hv if cats[v] == {"Services/Hacking"})),
        "also_drugs": int(sum(1 for v in hv if "Drugs" in tops[v])),
        "also_data": int(sum(1 for v in hv if "Data" in tops[v])),
        "also_services_money": int(sum(1 for v in hv if "Services/Money" in cats[v])),
        "also_information": int(sum(1 for v in hv if tops[v] & {"Information", "Info"})),
        "also_counterfeits_forgeries": int(sum(1 for v in hv if tops[v] & {"Counterfeits", "Forgeries"})),
        "mean_categories_per_vendor": float(np.mean([len(cats[v]) for v in hv])),
        "haas_listing_share_of_their_listings": float(len(haas) / len(vend)),
    }
    # category co-occurrence through shared vendors (Jaccard of vendor sets)
    vsets = dig.groupby("Category")["Vendor"].apply(set)
    co = {c: len(vsets[c] & vsets["Services/Hacking"]) / len(vsets[c] | vsets["Services/Hacking"]) for c in vsets.index if c != "Services/Hacking"}
    out["category_vendor_jaccard_with_haas"] = dict(sorted(co.items(), key=lambda kv: -kv[1])[:6])
    out["bipartite_digital"] = {"vendors": int(dig["Vendor"].nunique()), "categories": int(dig["Category"].nunique()),
                                "edges": int(dig.groupby(["Vendor", "Category"]).ngroups),
                                "vendors_in_2plus_categories": int((dig.groupby("Vendor")["Category"].nunique() >= 2).sum())}

    # ---- NMF themes within Services/Hacking
    stop = list(ENGLISH_STOP_WORDS | {"btc", "usd", "price", "item", "will", "just", "best", "new", "100", "000"})
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b", sublinear_tf=True)
    X = vec.fit_transform(haas["text"])
    k = 6
    nmf = NMF(n_components=k, random_state=SEED, init="nndsvda", max_iter=1000)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    dom = W.argmax(axis=1)
    topics = []
    for i in range(k):
        top = terms[np.argsort(nmf.components_[i])[::-1][:8]].tolist()
        sel = haas.iloc[np.flatnonzero(dom == i)]
        topics.append({"topic": i, "top_terms": top, "n_listings": int((dom == i).sum()), "n_vendors": int(sel["Vendor"].nunique()),
                       "median_btc": float(sel["price_btc"].median()), "examples": sel["Item"].str[:60].head(4).tolist()})
    out["nmf"] = {"k": k, "reconstruction_err": float(nmf.reconstruction_err_), "vocab": int(len(terms))}
    (LOG / "nmf_topics.json").write_text(json.dumps(topics, indent=2))
    haas.assign(topic=dom)[["Vendor", "Item", "price_btc", "topic"]].to_parquet(LOG / "haas_topics.parquet", index=False)
    (LOG / "ecosystem.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(audit, indent=1))
    print(json.dumps(out, indent=1)[:6000])
    for t in topics:
        print(t["topic"], t["n_listings"], t["n_vendors"], round(t["median_btc"], 4), t["top_terms"], t["examples"][:2])


if __name__ == "__main__":
    main()
