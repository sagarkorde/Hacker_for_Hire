"""Greyscale figures for the paper (JJCIT requires black/white reproducibility).

  fig1_architecture.png  pipeline and testbed
  fig2_crawler.png       throughput, CPU cost, memory, latency sensitivity (needs bench.jsonl)
  fig3_pr_curves.png     precision-recall curves, vendor-disjoint CV
  fig4_market.png        prices by category and vendor concentration
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
STY = {"go": dict(color="0.0", marker="o", ls="-", label="Go (goroutines)"),
       "asyncio": dict(color="0.45", marker="s", ls="--", label="Python asyncio"),
       "threads": dict(color="0.65", marker="^", ls=":", label="Python threads")}


def box(ax, x, y, w, h, text, fill="0.92", lw=0.8, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fill, ec="0.2", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, text=None, dy=0.025):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.8))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=7, color="0.25")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.axis("off")
    # collection layer
    ax.text(0.01, 0.95, "Collection (Go)", fontsize=8, weight="bold")
    box(ax, 0.01, 0.70, 0.16, 0.18, "Seed list\n(.onion URLs)", fill="1.0")
    box(ax, 0.22, 0.70, 0.22, 0.18, "Crawler\ngoroutine per URL,\nchannel semaphore (c)", fs=7.5)
    box(ax, 0.49, 0.70, 0.17, 0.18, "Extractor\nvisible text, links")
    box(ax, 0.72, 0.70, 0.27, 0.18, "Store\npages.jsonl (URL, status,\ntitle, text, timing)", fill="1.0", fs=7.5)
    arrow(ax, 0.17, 0.79, 0.22, 0.79)
    arrow(ax, 0.44, 0.79, 0.49, 0.79)
    arrow(ax, 0.66, 0.79, 0.72, 0.79)
    ax.annotate("", xy=(0.33, 0.88), xytext=(0.57, 0.88), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.8,
                connectionstyle="arc3,rad=0.45"))
    ax.text(0.585, 0.925, "same-host links (depth limit)", ha="left", fontsize=7, color="0.25")
    # transport
    box(ax, 0.23, 0.40, 0.20, 0.18, "Transport\nSOCKS5 (Tor) or\nlatency testbed", fill="1.0")
    ax.plot([0.33, 0.33], [0.58, 0.70], color="0.2", lw=0.8)
    # analysis layer (data flows right to left)
    ax.text(0.01, 0.31, "Analysis (Python)", fontsize=8, weight="bold")
    box(ax, 0.75, 0.06, 0.24, 0.22, "REST service\nPOST /analyze\n(FastAPI)", fill="1.0", fs=7.5)
    box(ax, 0.49, 0.06, 0.20, 0.22, "Detector\nrule dictionary +\nentity cues, or\nTF-IDF classifier", fs=7.5)
    box(ax, 0.22, 0.06, 0.22, 0.22, "Market analysis\nprices, vendors,\nNMF themes,\nprice anomalies", fill="1.0", fs=7.5)
    arrow(ax, 0.855, 0.70, 0.855, 0.28)
    arrow(ax, 0.75, 0.17, 0.69, 0.17)
    arrow(ax, 0.49, 0.17, 0.44, 0.17)
    fig.savefig(FIG / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_pr():
    curves = json.loads((LOG / "pr_curves.json").read_text())
    res = json.loads((LOG / "classify.json").read_text())["E1_vendor_disjoint"]
    show = {"Rule-based": ("0.0", "-"), "LR (char 2-5)": ("0.35", "--"), "LR (word 1-2)": ("0.55", "-."),
            "SVM (word 1-2)": ("0.7", ":")}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for n, (c, ls) in show.items():
        ax.plot(curves[n]["recall"], curves[n]["precision"], color=c, ls=ls, lw=1.2,
                label=f"{n} (AP={res[n]['ap']:.2f})")
    prev = json.loads((LOG / "classify.json").read_text())
    base = prev["n_pos"] / prev["n_digital"]
    ax.axhline(base, color="0.5", lw=0.6)
    ax.text(0.01, base + 0.015, f"prevalence {base:.3f}", ha="left", fontsize=7, color="0.3")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(FIG / "fig3_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_market():
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"] & (df["price_btc"] > 0)]
    order = ["Data/Pirated", "Data/Software", "Information/Guides", "Data/Accounts", "Services/Hacking", "Services/Other", "Services/Money"]
    data = [np.log10(dig.loc[dig["Category"] == c, "price_btc"]) for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bp = ax.boxplot(data, orientation='horizontal', showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="0.0", lw=1.2))
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor="0.55" if order[i] == "Services/Hacking" else "0.9", edgecolor="0.2")
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{c} (n={len(d)})" for c, d in zip(order, data)], fontsize=7)
    ax.set_xlabel("Listed price, log10 BTC (whiskers 1.5 IQR)")
    ax.set_title("(a) Prices by category", fontsize=8, loc="left")
    ax = axes[1]
    for name, frame, sty in [("Services/Hacking", dig[dig["haas"]], dict(color="0.0", ls="-")),
                             ("All digital", dig, dict(color="0.5", ls="--")),
                             ("Full market", df, dict(color="0.7", ls=":"))]:
        c = np.sort(frame["Vendor"].value_counts().values)
        x = np.concatenate([[0], np.arange(1, len(c) + 1) / len(c)])
        y = np.concatenate([[0], np.cumsum(c) / c.sum()])
        ax.plot(x, y, lw=1.2, label=name, **sty)
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.6)
    ax.set_xlabel("Share of vendors (ascending)")
    ax.set_ylabel("Share of listings")
    ax.set_title("(b) Vendor concentration (Lorenz)", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_market.png", bbox_inches="tight")
    plt.close(fig)


def fig_crawler():
    b = pd.read_json(LOG / ("bench_pinned.jsonl" if (LOG / "bench_pinned.jsonl").exists() else "bench.jsonl"), lines=True)
    b = b[b["tag"] != "startup"]
    m = b[b["tag"] == "main"].copy()
    m["cpu_per_1k"] = m["cpu_s"] / m["attempted"] * 1000
    g = m.groupby(["impl", "concurrency"]).agg(ppm=("pages_per_min", "mean"), ppm_sd=("pages_per_min", "std"),
                                               cpu=("cpu_per_1k", "mean"), cpu_sd=("cpu_per_1k", "std"),
                                               rss=("max_rss_mb", "mean"), rss_sd=("max_rss_mb", "std")).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    cs = sorted(m["concurrency"].unique())
    # theoretical bound: c / E[latency]
    med, sig = 1.0, 0.5
    mean_lat = med * np.exp(sig ** 2 / 2)
    axes[0].plot(cs, [60 * c / mean_lat for c in cs], color="0.8", lw=2.5, label="c / E[latency]")
    for impl, sty in STY.items():
        s = g[g["impl"] == impl]
        axes[0].errorbar(s["concurrency"], s["ppm"], yerr=s["ppm_sd"], ms=3.5, lw=1, capsize=2, **sty)
        axes[1].errorbar(s["concurrency"], s["cpu"], yerr=s["cpu_sd"], ms=3.5, lw=1, capsize=2, **sty)
        axes[2].errorbar(s["concurrency"], s["rss"], yerr=s["rss_sd"], ms=3.5, lw=1, capsize=2, **sty)
    for ax in axes[:3]:
        ax.set_xscale("log")
        ax.set_xlabel("Concurrency c")
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs], fontsize=7)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_yscale("log")
    from matplotlib.ticker import NullFormatter as _NF
    axes[0].yaxis.set_minor_formatter(_NF())
    axes[0].set_ylabel("Pages per minute")
    axes[0].set_title("(a) Throughput", fontsize=8, loc="left")
    axes[1].set_ylabel("CPU s per 1,000 pages")
    axes[1].set_title("(b) CPU cost", fontsize=8, loc="left")
    axes[2].set_ylabel("Peak RSS (MB)")
    axes[2].set_title("(c) Memory", fontsize=8, loc="left")
    s = b[b["tag"] == "sensitivity"]
    if s.empty:
        fig.tight_layout()
        fig.savefig(FIG / "fig2_crawler.png", bbox_inches="tight")
        plt.close(fig)
        return g
    gs = s.groupby(["impl", "median_s"]).agg(ppm=("pages_per_min", "mean"), sd=("pages_per_min", "std")).reset_index()
    for impl, sty in STY.items():
        q = gs[gs["impl"] == impl]
        axes[3].errorbar(q["median_s"], q["ppm"], yerr=q["sd"], ms=3.5, lw=1, capsize=2, **sty)
    meds = sorted(s["median_s"].unique())
    axes[3].plot(meds, [60 * 20 / (x * np.exp(sig ** 2 / 2)) for x in meds], color="0.8", lw=2.5, zorder=0)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xticks(meds)
    axes[3].set_xticklabels([str(x) for x in meds], fontsize=7)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    axes[3].xaxis.set_minor_formatter(NullFormatter())
    axes[3].yaxis.set_minor_formatter(NullFormatter())
    yt = [500, 1000, 2000, 4000]
    axes[3].yaxis.set_major_locator(FixedLocator(yt))
    axes[3].yaxis.set_major_formatter(FixedFormatter([f"{v:,}" for v in yt]))
    axes[3].tick_params(axis="y", labelsize=7)
    axes[3].set_xlabel("Median latency (s), c = 20")
    axes[3].set_ylabel("Pages per minute")
    axes[3].set_title("(d) Latency", fontsize=8, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_crawler.png", bbox_inches="tight")
    plt.close(fig)
    return g


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "pr", "market"]
    if "arch" in which:
        fig_architecture()
    if "pr" in which:
        fig_pr()
    if "market" in which:
        fig_market()
    if "crawler" in which:
        print(fig_crawler().round(2).to_string())
