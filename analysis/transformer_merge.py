"""Merge transformer seeds into results/logs/transformer.json and compare with the rule.

Paired vendor-cluster bootstrap (2,000 replicates) of F1(transformer) - F1(rule), as for the
other models in classify_extra.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"


def main():
    d = pd.read_csv(LOG / "folds_vendor_disjoint.csv", keep_default_na=False)
    y, rule, g = d["label"].to_numpy(), d["rule_pred"].to_numpy(), d["vendor"].astype(str).to_numpy()
    u = np.unique(g)
    idx = {v: np.flatnonzero(g == v) for v in u}
    seeds = {}
    for f in sorted(LOG.glob("transformer_seed*.json")):
        r = json.loads(f.read_text())
        oof = pd.read_csv(LOG / f"transformer_oof_seed{r['seed']}.csv")
        assert (oof["row"].to_numpy() == d["row"].to_numpy()).all()
        pred = oof["pred"].to_numpy()
        rng = np.random.default_rng(r["seed"])
        f1s, diffs = [], []
        for _ in range(2000):
            i = np.concatenate([idx[v] for v in rng.choice(u, len(u))])
            if y[i].sum() == 0:
                continue
            a, b = f1_score(y[i], pred[i]), f1_score(y[i], rule[i])
            f1s.append(a)
            diffs.append(a - b)
        r["f1_ci95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
        r["diff_vs_rule"] = float(f1_score(y, pred) - f1_score(y, rule))
        r["diff_vs_rule_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        seeds[str(r["seed"])] = r
    assert seeds, "no transformer_seed*.json found"
    f1 = [v["f1"] for v in seeds.values()]
    out = {"model": next(iter(seeds.values()))["model"], "n_seeds": len(seeds), "seeds": seeds,
           "f1_mean": float(np.mean(f1)), "f1_sd_over_seeds": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0}
    for k in ["precision", "recall", "ap", "inference_listings_per_s"]:
        out[f"{k}_mean"] = float(np.mean([v[k] for v in seeds.values()]))
    (LOG / "transformer.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1))


if __name__ == "__main__":
    main()
