"""End-to-end check on the full crawl of the emulated market (results/crawl_full).

Measures (i) coverage of listing pages, (ii) extraction fidelity: share of source
title/description tokens recovered and share of extracted tokens that are page
boilerplate, and (iii) agreement between rule-based decisions on crawled page text
and on the source listing text, with and without the category breadcrumb that the
page template shows (a label leak).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import rules

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return TOK.findall(s.lower())


def main():
    truth = json.loads((ROOT / "data" / "site" / "truth.json").read_text())
    pages = pd.read_json(ROOT / "results" / "crawl_full" / "pages.jsonl", lines=True)
    pages["path"] = pages["url"].str.replace(r"^https?://[^/]+", "", regex=True)
    lst = pages[pages["path"].isin(truth.keys()) & (pages["status"] == 200)].drop_duplicates("path")
    out = {"records": int(len(pages)), "status_counts": {str(k): int(v) for k, v in pages["status"].value_counts().items()},
           "listing_pages_total": len(truth), "listing_pages_crawled": int(len(lst)),
           "listing_coverage": len(lst) / len(truth)}
    rec, boiler, agree_full, agree_nocat = [], [], [], []
    dis_nocat, dis_fixed_by_footer = 0, 0
    for r in lst.itertuples():
        t = truth[r.path]
        src = t["item"] + " . " + t["desc"]
        st, et = set(toks(src)), toks(r.text)
        if st:
            rec.append(len(st & set(et)) / len(st))
        if et:
            boiler.append(sum(1 for w in et if w not in st) / len(et))
        nocat = re.sub(r"Category:\s*\S+", " ", r.text)
        y_src = rules.predict(src)[0]
        agree_full.append(rules.predict(r.text)[0] == y_src)
        agree_nocat.append(rules.predict(nocat)[0] == y_src)
        if rules.predict(nocat)[0] != y_src:
            dis_nocat += 1
            nofoot = nocat.replace("Escrow protected. Always use PGP.", "").replace("Enable JS", "")
            dis_fixed_by_footer += int(rules.predict(nofoot)[0] == y_src)
    out["token_recall_mean"] = float(np.mean(rec))
    out["token_recall_min"] = float(np.min(rec))
    out["boilerplate_share_mean"] = float(np.mean(boiler))
    out["rule_agreement_full_page"] = float(np.mean(agree_full))
    out["rule_agreement_without_breadcrumb"] = float(np.mean(agree_nocat))
    out["disagreements_without_breadcrumb"] = dis_nocat
    out["disagreements_removed_by_dropping_footer"] = dis_fixed_by_footer
    site = json.loads((ROOT / "data" / "site" / "site.json").read_text())["pages"]
    pos = {}
    for k, v in site.items():
        if k.startswith("/cat/"):
            for m in re.findall(r'href="(/listing/[0-9a-f]{12})"', v):
                pos[m] = int(k.rsplit("/", 1)[1])
    miss = [p for p in truth if p not in set(lst["path"])]
    out["unreached_listings"] = len(miss)
    out["unreached_min_index_page"] = int(min(pos[m] for m in miss)) if miss else None
    out["max_depth_crawled"] = int(pages["depth"].max())
    # REST service smoke test on 200 crawled pages
    import sys
    sys.path.insert(0, str(ROOT))
    from service.app import app
    client = TestClient(app)
    sample = lst.sample(n=min(200, len(lst)), random_state=1)
    ok = 0
    for r in sample.itertuples():
        res = client.post("/analyze", json={"text": r.text, "url": r.url})
        ok += int(res.status_code == 200 and res.json()["haas"] == bool(rules.predict(r.text)[0]))
    out["service_checks_passed"] = f"{ok}/{len(sample)}"
    summ = json.loads((LOG / "fullcrawl_summary.json").read_text())
    out["crawl_summary"] = summ
    (LOG / "e2e.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
