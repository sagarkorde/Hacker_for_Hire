# Hacker_for_Hire: Go crawler, emulated-Tor testbed and HaaS detection

Code, logs and figures for the manuscript *Crawling and detecting hacker-for-hire offers on darknet markets: an
emulated Tor benchmark and a vendor-disjoint evaluation*.

No live onion service is contacted by any script in this repository. All crawling targets a local testbed.

## Layout

| Path | What it is |
|---|---|
| `crawler/` | Go crawler (`hfhcrawl`): goroutine per URL, channel semaphore, SOCKS5 proxy support (`-proxy socks5://127.0.0.1:9050` for Tor), JSONL output. `go test ./...` runs the extractor tests. |
| `testbed/` | Go HTTP server that replays the emulated market with log-normal latency, transient 503s and dead links. |
| `baselines/pycrawl.py` | Python asyncio (aiohttp) and thread-pool (requests) crawlers with identical behaviour. |
| `service/app.py` | FastAPI service, `POST /analyze`. |
| `analysis/` | Data preparation, site generation, benchmark runner, classification, market analysis, figures, end-to-end check, annotation sample. |
| `data/` | `fetch_data.sh` downloads the public Agora 2014-2015 listings (`Agora.csv`, SHA-256 `743c05fa682d3c940e80ed03fe52c6bc3035236c2f3fddc09c59baa797a745a2`) and `btc_usd_daily.csv` from their GitHub mirrors and checks both hashes. `onionperf_onion_latencies.csv`: OnionPerf onion-service latencies from Tor Metrics (22 Aug-15 Nov 2026), used to check the testbed latency. |
| `results/logs/` | Every number reported in the paper (JSON/JSONL/CSV). |
| `results/figures/` | Figures 1-4. |
| `docs/annotation_guideline.md` | Guideline for the pending human-annotation study (`analysis/score_annotation.py` scores it). |
| `docs/traceability.md` | Which script and log file produce each number in the paper. |
| `notebooks/transformer_baseline.ipynb`, `analysis/transformer_baseline.py` | Fine-tuned transformer on the same vendor-disjoint folds (Colab notebook, or the script, which `.github/workflows/transformer.yml` runs on GitHub Actions for three seeds). |
| `requirements.txt` | Python package versions used for the reported results. |

## Reproduce (Linux, Go >= 1.22, Python >= 3.10)

```bash
sh data/fetch_data.sh
pip install -r requirements.txt
(cd crawler && go test ./... && go build -o ../bin/hfhcrawl .)
(cd testbed && go build -o ../bin/testbed .)

python3 analysis/prepare.py          # results/logs/prepare.json
python3 analysis/build_site.py       # data/site/, results/logs/build_site.json
python3 analysis/classify.py         # results/logs/classify.json, oof_digital.parquet, pr_curves.json  (from analysis/)
python3 analysis/classify_extra.py   # paired bootstrap, tuned baselines, leakage decomposition, fold file (from analysis/)
python3 analysis/ecosystem.py        # results/logs/ecosystem.json, label_audit.json, nmf_topics.json   (from analysis/)
L=results/logs/bench_pinned.jsonl     # server pinned to CPU 0, crawler to CPU 1 (needs taskset and 2+ CPUs)
python3 analysis/bench.py --plan main --reps 5 --pin --log $L         # about 100 min
python3 analysis/bench.py --plan sensitivity --reps 3 --pin --log $L  # about 25 min
python3 analysis/bench.py --plan startup --reps 5 --pin --log $L      # interpreter/runtime start-up CPU, 1 min
./bin/testbed -median 1.0 -sigma 0.5 -p503 0.02 -seed 11 &
./bin/hfhcrawl -c 50 -max 20000 -depth 8 -out results/crawl_full > results/logs/fullcrawl_summary.json
python3 analysis/e2e.py              # results/logs/e2e.json  (from analysis/)
python3 analysis/crawler_summary.py  # Table 3 and crawler numbers
python3 analysis/figures.py arch pr market crawler
python3 analysis/sample_annotation.py
```

Transformer baseline: run the `transformer-baseline` workflow (Actions tab, "Run workflow"), or
`python3 analysis/transformer_baseline.py --seed 1` locally, then `python3 analysis/transformer_merge.py`.

`classify.py`, `ecosystem.py` and `e2e.py` import `analysis/rules.py`; run them from `analysis/` or add it to `PYTHONPATH`.

## Using the crawler on Tor

Run a local Tor client, then:

```bash
./bin/hfhcrawl -seed http://<address>.onion/ -proxy socks5://127.0.0.1:9050 -c 4 -max 500 -depth 3 -out out/
```

Keep per-host concurrency low, store text only, and obtain ethics approval before any live crawl.
