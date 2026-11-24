"""Summarise bench.jsonl, the full crawl and the end-to-end check into numbers for the paper.

Writes paper/crawler_numbers.json (scalar values used in the text) and
results/logs/crawler_table.json (rows of Table 3).
Reads results/logs/bench_pinned.jsonl (CPU-pinned runs) when present, else bench.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
SIG = 0.5
NAMES = {"go": "Go", "asyncio": "Python asyncio", "threads": "Python threads"}


def bound(c, median):
    return 60 * c / (median * np.exp(SIG ** 2 / 2))


def main():
    ap = argparse.ArgumentParser()
    default = LOG / "bench_pinned.jsonl" if (LOG / "bench_pinned.jsonl").exists() else LOG / "bench.jsonl"
    ap.add_argument("--log", default=str(default))
    a = ap.parse_args()
    b = pd.read_json(a.log, lines=True)
    b["cpu_per_1k"] = b["cpu_s"] / b["attempted"] * 1000
    st = b[b["tag"] == "startup"]
    startup = st.groupby("impl")["cpu_s"].mean().to_dict() if len(st) else {}
    b["cpu_marg_1k"] = (b["cpu_s"] - b["impl"].map(startup).fillna(0)) / b["attempted"] * 1000
    m = b[b["tag"] == "main"].copy()
    m["eff"] = m["pages_per_min"] / bound(m["concurrency"], m["median_s"])
    g = m.groupby(["impl", "concurrency"]).agg(n=("rep", "count"), ppm=("pages_per_min", "mean"), ppm_sd=("pages_per_min", "std"),
                                               eff=("eff", "mean"), cpu=("cpu_per_1k", "mean"), cpu_sd=("cpu_per_1k", "std"),
                                               rss=("max_rss_mb", "mean"), rss_sd=("max_rss_mb", "std"),
                                               lq=("link_quality_ratio", "mean"), eff_sd=("eff", "std"),
                                               cpu_m=("cpu_marg_1k", "mean")).reset_index()
    rows = []
    for c in sorted(g["concurrency"].unique()):
        cells = [str(c), f"{bound(c, 1.0):,.0f}"]
        sub = {i: g[(g["impl"] == i) & (g["concurrency"] == c)].iloc[0] for i in ["go", "asyncio", "threads"]}
        cells += [f"{100 * sub[i].eff:.0f} ± {100 * sub[i].eff_sd:.0f}" for i in sub]
        cells += [f"{sub[i].cpu:.2f} ± {sub[i].cpu_sd:.2f}" for i in sub]
        cells += [f"{sub[i].rss:.1f} ± {sub[i].rss_sd:.1f}" for i in sub]
        rows.append(cells)
    (LOG / "crawler_table.json").write_text(json.dumps(rows, indent=1))
    g.to_csv(LOG / "crawler_grouped.csv", index=False)

    def val(impl, c, col):
        return float(g[(g["impl"] == impl) & (g["concurrency"] == c)][col].iloc[0])

    n = {}
    cs = sorted(g["concurrency"].unique())
    cmax = max(cs)
    # throughput parity: max relative gap between implementations at each c
    gaps = []
    for c in cs:
        v = [val(i, c, "ppm") for i in ["go", "asyncio", "threads"]]
        gaps.append((max(v) - min(v)) / np.mean(v))
    n["max_gap_pct"] = f"{100 * max(gaps):.1f}"
    n["eff_c1"] = f"{100 * np.mean([val(i, 1, 'eff') for i in NAMES]):.0f}"
    n["eff_c50"] = f"{100 * np.mean([val(i, 50, 'eff') for i in NAMES]):.0f}"
    n["eff_cmax_go"] = f"{100 * val('go', cmax, 'eff'):.0f}"
    n["eff_cmax_async"] = f"{100 * val('asyncio', cmax, 'eff'):.0f}"
    n["eff_cmax_thr"] = f"{100 * val('threads', cmax, 'eff'):.0f}"
    n["ppm_c1"] = f"{np.mean([val(i, 1, 'ppm') for i in NAMES]):.1f}"
    n["ppm_go_cmax"] = f"{val('go', cmax, 'ppm'):,.0f}"
    n["ppm_async_cmax"] = f"{val('asyncio', cmax, 'ppm'):,.0f}"
    n["ppm_thr_cmax"] = f"{val('threads', cmax, 'ppm'):,.0f}"
    n["speedup_go"] = f"{val('go', cmax, 'ppm') / val('go', 1, 'ppm'):.0f}"
    n["cmax"] = str(cmax)
    # CPU and memory ratios (Python / Go), averaged over concurrency levels
    for impl, key in [("asyncio", "async"), ("threads", "thr")]:
        cpu_r = [val(impl, c, "cpu") / val("go", c, "cpu") for c in cs]
        rss_r = [val(impl, c, "rss") / val("go", c, "rss") for c in cs]
        n[f"cpu_ratio_{key}_min"] = f"{min(cpu_r):.1f}"
        n[f"cpu_ratio_{key}_max"] = f"{max(cpu_r):.1f}"
        n[f"rss_ratio_{key}_min"] = f"{min(rss_r):.1f}"
        n[f"rss_ratio_{key}_max"] = f"{max(rss_r):.1f}"
    if startup:
        for impl, key in [("asyncio", "async"), ("threads", "thr")]:
            mr = [val(impl, c, "cpu_m") / val("go", c, "cpu_m") for c in cs]
            n[f"cpum_ratio_{key}_min"] = f"{min(mr):.1f}"
            n[f"cpum_ratio_{key}_max"] = f"{max(mr):.1f}"
        n["startup_go"] = f"{startup['go']:.2f}"
        n["startup_async"] = f"{startup['asyncio']:.2f}"
        n["startup_thr"] = f"{startup['threads']:.2f}"
        n["cpum_lo"] = f"{min(float(n['cpum_ratio_async_min']), float(n['cpum_ratio_thr_min'])):.1f}"
        n["cpum_hi"] = f"{max(float(n['cpum_ratio_async_max']), float(n['cpum_ratio_thr_max'])):.1f}"
    n["reps_main"] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(int(m["rep"].max()), str(int(m["rep"].max())))
    n["eff_cmax_lo"] = f"{100 * min(val(i, cmax, 'eff') for i in NAMES):.0f}"
    n["eff_cmax_hi"] = f"{100 * max(val(i, cmax, 'eff') for i in NAMES):.0f}"
    n["eff_cmax_txt"] = (f"{n['eff_cmax_lo']}% of the bound for each crawler" if n["eff_cmax_lo"] == n["eff_cmax_hi"]
                         else f"{n['eff_cmax_lo']}-{n['eff_cmax_hi']}% of the bound")
    n["cpu_lo"] = f"{min(float(n['cpu_ratio_async_min']), float(n['cpu_ratio_thr_min'])):.1f}"
    n["cpu_hi"] = f"{max(float(n['cpu_ratio_async_max']), float(n['cpu_ratio_thr_max'])):.1f}"
    n["rss_lo"] = f"{min(float(n['rss_ratio_async_min']), float(n['rss_ratio_thr_min'])):.1f}"
    n["rss_hi"] = f"{max(float(n['rss_ratio_async_max']), float(n['rss_ratio_thr_max'])):.1f}"
    n["go_cpu_c1"] = f"{val('go', 1, 'cpu'):.2f}"
    n["go_cpu_cmax"] = f"{val('go', cmax, 'cpu'):.2f}"
    n["async_cpu_cmax"] = f"{val('asyncio', cmax, 'cpu'):.2f}"
    n["thr_cpu_cmax"] = f"{val('threads', cmax, 'cpu'):.2f}"
    n["go_rss_c1"] = f"{val('go', 1, 'rss'):.1f}"
    n["go_rss_cmax"] = f"{val('go', cmax, 'rss'):.1f}"
    n["async_rss_cmax"] = f"{val('asyncio', cmax, 'rss'):.1f}"
    n["thr_rss_cmax"] = f"{val('threads', cmax, 'rss'):.1f}"
    n["lq_mean"] = f"{m['link_quality_ratio'].mean():.3f}"
    n["lq_min"] = f"{m['link_quality_ratio'].min():.3f}"
    n["lq_max"] = f"{m['link_quality_ratio'].max():.3f}"
    n["n_runs_main"] = int(len(m))
    n["max_ppm_cv"] = f"{100 * (g['ppm_sd'] / g['ppm']).max():.1f}"
    # sensitivity
    s = b[b["tag"] == "sensitivity"].copy()
    if len(s):
        s["eff"] = s["pages_per_min"] / bound(s["concurrency"], s["median_s"])
        gs = s.groupby(["impl", "median_s"]).agg(ppm=("pages_per_min", "mean"), eff=("eff", "mean"), cpu=("cpu_per_1k", "mean")).reset_index()
        n["reps_sens"] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(int(s["rep"].max()), str(int(s["rep"].max())))
        gs.to_csv(LOG / "crawler_sensitivity.csv", index=False)
        n["n_runs_sens"] = int(len(s))
        n["sens_eff_min"] = f"{100 * gs['eff'].min():.0f}"
        n["sens_eff_max"] = f"{100 * gs['eff'].max():.0f}"
        lo, hi = gs["median_s"].min(), gs["median_s"].max()
        n["sens_go_lo"] = f"{float(gs[(gs.impl == 'go') & (gs.median_s == lo)].ppm.iloc[0]):,.0f}"
        n["sens_go_hi"] = f"{float(gs[(gs.impl == 'go') & (gs.median_s == hi)].ppm.iloc[0]):,.0f}"
        n["sens_eff_go_lo"] = f"{100 * float(gs[(gs.impl == 'go') & (gs.median_s == lo)].eff.iloc[0]):.0f}"
        n["sens_eff_thr_lo"] = f"{100 * float(gs[(gs.impl == 'threads') & (gs.median_s == lo)].eff.iloc[0]):.0f}"
        n["sens_eff_async_lo"] = f"{100 * float(gs[(gs.impl == 'asyncio') & (gs.median_s == lo)].eff.iloc[0]):.0f}"
        for impl, key in [("go", "go"), ("asyncio", "async"), ("threads", "thr")]:
            n[f"sens_cpu_{key}_lo"] = f"{float(gs[(gs.impl == impl) & (gs.median_s == lo)].cpu.iloc[0]):.2f}"
    e2e = LOG / "e2e.json"
    if e2e.exists():
        e = json.loads(e2e.read_text())
        cs_ = e["crawl_summary"]
        n.update(fc_attempted=f"{cs_['attempted']:,}", fc_ok=f"{cs_['ok']:,}", fc_404=f"{cs_['not_found']:,}", fc_failed=cs_["failed_final"],
                 fc_ppm=f"{cs_['pages_per_min']:,.0f}", fc_min=f"{cs_['elapsed_s'] / 60:.1f}", fc_lq=f"{cs_['link_quality_ratio']:.3f}",
                 fc_retries=f"{cs_['retries']:,}",
                 e2e_cov=f"{100 * e['listing_coverage']:.1f}", e2e_listings=f"{e['listing_pages_crawled']:,}",
                 e2e_recall=f"{100 * e['token_recall_mean']:.1f}", e2e_recall_min=f"{100 * e['token_recall_min']:.1f}",
                 e2e_boiler=f"{100 * e['boilerplate_share_mean']:.0f}",
                 e2e_agree_full=f"{100 * e['rule_agreement_full_page']:.1f}", e2e_agree_nocat=f"{100 * e['rule_agreement_without_breadcrumb']:.1f}",
                 e2e_service=e["service_checks_passed"], e2e_unreached=e["unreached_listings"],
                 e2e_unreached_page=e["unreached_min_index_page"], e2e_dis=e["disagreements_without_breadcrumb"],
                 e2e_footer=e["disagreements_removed_by_dropping_footer"])
    out = ROOT / "paper" / "crawler_numbers.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(n, indent=1))
    print(json.dumps(n, indent=1))
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
