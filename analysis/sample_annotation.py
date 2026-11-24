"""Draw the sample for the pending human-annotation study (Section 8 of the paper).

Stratified sample of 400 listings from the digital subset:
  100 Services/Hacking listings, 100 non-hacking listings flagged by the rule or by
  the best classifier (likely false positives), 200 random other listings.
Annotators label each listing independently using docs/annotation_guideline.md;
agreement is then measured with a chance-corrected statistic before adjudication.
Writes results/annotation_sample.csv (labels left empty).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def main():
    oof = pd.read_parquet(ROOT / "results" / "logs" / "oof_digital.parquet")
    df = pd.read_parquet(ROOT / "data" / "listings_clean.parquet")
    dig = df[df["digital"]].reset_index(drop=True)
    dig["flagged"] = (oof["pred::Rule-based"] == 1) | (oof["pred::LR (char 2-5)"] == 1)
    pos = dig[dig["haas"]].sample(100, random_state=SEED)
    fpos = dig[(~dig["haas"]) & dig["flagged"]].sample(100, random_state=SEED)
    rest = dig[(~dig["haas"]) & (~dig["flagged"])].sample(200, random_state=SEED)
    s = pd.concat([pos.assign(stratum="market_label_hacking"), fpos.assign(stratum="flagged_other_category"),
                   rest.assign(stratum="random_other")])
    s = s.sample(frac=1, random_state=SEED)  # hide strata order from annotators
    out = s[["Vendor", "Category", "Item", "Item Description", "price_btc", "stratum"]].copy()
    out["annotator_A"] = ""
    out["annotator_B"] = ""
    out["adjudicated"] = ""
    out.to_csv(ROOT / "results" / "annotation_sample.csv", index=False)
    print(out["stratum"].value_counts())


if __name__ == "__main__":
    main()
