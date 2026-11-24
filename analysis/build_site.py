"""Render the digital-goods subset of Agora as a static market website.

The site is served by testbed/server.go with Tor-like latency so that crawlers can
be benchmarked reproducibly without contacting any live onion service.

Structure: home -> category index pages (paginated) -> listing pages -> vendor pages.
Each listing page links to its vendor, its category and up to three related listings.
A fixed share of listing links points to pages that do not exist (dead links).
"""
import hashlib
import html
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "site" / "site.json"
LOG = ROOT / "results" / "logs" / "build_site.json"
PER_PAGE = 50
DEAD_LINK_SHARE = 0.04
SEED = 7


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "x"


def lid(row) -> str:
    return hashlib.sha1((row.Vendor + "||" + row.Item).encode()).hexdigest()[:12]


HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif}} .nav a{{margin:4px}}</style>
<script>var session='{sid}'; function track(){{return session;}}</script></head><body>
<div class="nav"><a href="/">Home</a> <a href="/rules">Rules</a> <a href="/support">Support</a></div>
"""
FOOT = """<div class="footer">Escrow protected. Always use PGP.<noscript>Enable JS</noscript></div></body></html>"""


def main() -> None:
    rng = random.Random(SEED)
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    df = df[df["digital"]].reset_index(drop=True).rename(columns={"Item Description": "desc"})
    df["lid"] = [lid(r) for r in df.itertuples()]
    pages, truth = {}, {}

    cats = sorted(df["Category"].unique())
    vendors = sorted(df["Vendor"].unique())
    by_cat = {c: df[df["Category"] == c] for c in cats}

    # Home page
    links = "".join(f'<li><a href="/cat/{slug(c)}/1">{html.escape(c)}</a></li>' for c in cats)
    pages["/"] = HEAD.format(title="Market", sid="0") + f"<h1>Categories</h1><ul>{links}</ul>" + FOOT
    pages["/rules"] = HEAD.format(title="Rules", sid="0") + "<p>No scams. Finalize early only for trusted vendors.</p>" + FOOT
    pages["/support"] = HEAD.format(title="Support", sid="0") + "<p>Open a ticket.</p>" + FOOT

    dead_links = 0
    for c in cats:
        sub = by_cat[c]
        n_pages = (len(sub) + PER_PAGE - 1) // PER_PAGE
        for p in range(n_pages):
            chunk = sub.iloc[p * PER_PAGE:(p + 1) * PER_PAGE]
            items = []
            for r in chunk.itertuples():
                items.append(f'<li><a href="/listing/{r.lid}">{html.escape(r.Item[:90])}</a> {r.price_btc:.4f} BTC</li>')
                if rng.random() < DEAD_LINK_SHARE:
                    items.append(f'<li><a href="/listing/gone-{rng.getrandbits(40):010x}">[removed]</a></li>')
                    dead_links += 1
            nav = ""
            if p + 1 < n_pages:
                nav = f'<a href="/cat/{slug(c)}/{p + 2}">next</a>'
            pages[f"/cat/{slug(c)}/{p + 1}"] = (HEAD.format(title=c, sid=p) + f"<h2>{html.escape(c)}</h2><ul>{''.join(items)}</ul>{nav}" + FOOT)

    for r in df.itertuples():
        rel = by_cat[r.Category].sample(n=min(3, len(by_cat[r.Category])), random_state=int(r.lid[:6], 16) % 10_000)
        rel_links = "".join(f'<a href="/listing/{x}">related</a> ' for x in rel["lid"] if x != r.lid)
        body = (f"<h1>{html.escape(r.Item)}</h1><div class='price'>{r.price_btc:.6f} BTC</div>"
                f"<div class='desc'><p>{html.escape(r.desc)}</p></div>"
                f"<div>Vendor: <a href='/vendor/{slug(r.Vendor)}'>{html.escape(r.Vendor)}</a> "
                f"Rating {html.escape(r.Rating)} Ships from {html.escape(r.Origin)}</div>"
                f"<div>Category: <a href='/cat/{slug(r.Category)}/1'>{html.escape(r.Category)}</a></div>"
                f"<div class='rel'>{rel_links}</div>")
        pages[f"/listing/{r.lid}"] = HEAD.format(title=html.escape(r.Item[:60]), sid=r.lid) + body + FOOT
        truth[f"/listing/{r.lid}"] = {"category": r.Category, "haas": bool(r.haas), "vendor": r.Vendor,
                                      "item": r.Item, "desc": r.desc}

    for v in vendors:
        sub = df[df["Vendor"] == v].head(40)
        items = "".join(f'<li><a href="/listing/{x}">{html.escape(t[:80])}</a></li>' for x, t in zip(sub["lid"], sub["Item"]))
        pages[f"/vendor/{slug(v)}"] = HEAD.format(title=html.escape(v), sid=slug(v)) + f"<h1>{html.escape(v)}</h1><ul>{items}</ul>" + FOOT

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pages": pages}))
    (OUT.parent / "truth.json").write_text(json.dumps(truth))
    log = {"pages_total": len(pages), "listing_pages": int(len(df)), "vendor_pages": len(vendors),
           "category_index_pages": sum(1 for k in pages if k.startswith("/cat/")),
           "dead_links_injected": dead_links, "dead_link_share": DEAD_LINK_SHARE, "seed": SEED,
           "mean_page_bytes": sum(len(v) for v in pages.values()) / len(pages)}
    LOG.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
