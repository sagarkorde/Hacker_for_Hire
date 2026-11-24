# Traceability: where every reported result comes from

All numbers in the manuscripts are inserted by `paper/build_doc.py` from the log files below; no number is typed by
hand. Section numbers refer to the manuscript (both versions share the same numbering).

| Manuscript item | Produced by | Log / output |
|---|---|---|
| Section 4.1, Table 2 (dataset) | `analysis/prepare.py`, `analysis/build_site.py` | `results/logs/prepare.json`, `results/logs/build_site.json` |
| Section 4.2 (latency check against Tor measurements) | OnionPerf latencies from Tor Metrics, summarised in the build | `data/onionperf_onion_latencies.csv`, `results/logs/onionperf_calibration.json` |
| Section 4.4 (environment) | environment capture | `results/logs/environment.json` |
| Section 5.1, Table 3, Figure 2a-c | `analysis/bench.py --plan main --reps 5 --pin`, `--plan startup`, `analysis/crawler_summary.py`, `analysis/figures.py crawler` | `results/logs/bench_pinned.jsonl` (tags `main`, `startup`), `crawler_grouped.csv`, `crawler_table.json`, `paper/crawler_numbers.json` |
| Section 5.1 equivalence test | `analysis/crawler_summary.py` | `results/logs/crawler_equivalence.json` |
| Section 5.1 end-to-end paragraph | `bin/hfhcrawl -c 50 ...`, `analysis/e2e.py`, `analysis/revision_checks.py` | `results/logs/fullcrawl_summary.json`, `e2e.json`, `revision_checks.json` (`end_to_end_detection`) |
| Section 5.2, Table 4, Figure 3 | `analysis/classify.py`, `analysis/classify_extra.py` | `results/logs/classify.json` (`E1_vendor_disjoint`, `false_positives_by_category`, `E5_docs_per_s`), `classify_extra.json` (`paired_f1_diff_vs_rule`, `tuned`), `revision_checks.json` (`inner_threshold`, `tuned_folds`), `pr_curves.json`, `oof_digital.parquet`; transformer: `transformer.json` (`analysis/transformer_baseline.py`, `transformer_merge.py`, GitHub Actions workflow) |
| Section 4.3, 5.3 (label audits, annotation) | `analysis/ecosystem.py`, `analysis/classify_extra.py`, `analysis/annotation_kit.py`, `analysis/score_annotation.py` | `results/logs/label_audit.json`, `classify_extra.json` (`rule_recall_cap`, `nmf_non_intrusion_share`), `results/annotation/`, `annotation_eval_provisional.json` (provisional model labels), `annotation_eval.json` (human labels, when available) |
| Section 5.4, Figure 4, Table 5 | `analysis/ecosystem.py`, `analysis/figures.py market` | `results/logs/ecosystem.json` (incl. `kruskal_prices.epsilon_squared`, `price_anomalies_haas.iqr_subset_of_if`), `nmf_topics.json`, `haas_topics.parquet`, `revision_checks.json` (`custom_intrusion_theme`) |
| Section 6, Table 6 (classification ablations, leakage decomposition) | `analysis/classify.py`, `analysis/classify_extra.py` | `results/logs/classify.json` (`E2_random_split`, `E3_rule_components`, `E3_rule_threshold`, `E4_full_market`), `classify_extra.json` (`near_duplicates`, `near_duplicate_disjoint_f1`, `vendor_name_in_text`), `revision_checks.json` (`near_duplicate_sensitivity`) |
| Section 6 (latency sensitivity), Figure 2d | `analysis/bench.py --plan sensitivity --reps 3 --pin` | `results/logs/bench_pinned.jsonl` (tag `sensitivity`), `crawler_sensitivity.csv` |
| Figure 1 | `analysis/figures.py arch` | `results/figures/fig1_architecture.png` |
| References | manual verification against Crossref | `results/references_verified.md`; Harvard versions checked by `paper/references_harvard.py` |

## Pending validation (reported as pending, not as results)

1. Human-annotated test set (400 listings, two blinded annotators): `results/annotation/annotate.html`, guideline
   `docs/annotation_guideline.md`, scorer `analysis/score_annotation.py` (writes `results/logs/annotation_eval.json`).
   The provisional labels in `results/annotation/provisional_labels_claude.csv` come from a language model and are
   reported as provisional only.
3. Live Tor crawl under ethics approval: the crawler supports `-proxy socks5://127.0.0.1:9050`; no live crawl was run.
