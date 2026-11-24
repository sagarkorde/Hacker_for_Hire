"""Clean and deduplicate the public Agora listings dataset.

Outputs
  data/listings_clean.parquet   one row per unique (vendor, normalised title)
  results/logs/prepare.json     every count reported in the paper's data section
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "Agora.csv"
OUT = ROOT / "data" / "listings_clean.parquet"
LOG = ROOT / "results" / "logs" / "prepare.json"

# Top-level categories treated as "digital goods and services" (the primary
# evaluation population: hard negatives for hacking-service detection).
DIGITAL_TOP = {"Services", "Data", "Info", "Information", "Other"}
HAAS_LABEL = "Services/Hacking"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main() -> None:
    log = {}
    df = pd.read_csv(RAW, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    log["raw_rows"] = int(len(df))
    log["raw_vendors"] = int(df["Vendor"].nunique())

    # 1. Drop malformed rows (shifted columns produce a non-BTC price field).
    price = df["Price"].astype(str).str.strip()
    ok = price.str.fullmatch(r"[0-9.eE+-]+ BTC")
    log["malformed_rows_dropped"] = int((~ok).sum())
    df = df[ok].copy()
    df["price_btc"] = price[ok].str.replace(" BTC", "", regex=False).astype(float)

    for c in ["Vendor", "Category", "Item", "Item Description", "Origin", "Destination", "Rating", "Remarks"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["top"] = df["Category"].str.split("/").str[0]
    df["text"] = (df["Item"] + " . " + df["Item Description"]).str.strip()
    log["rows_after_malformed"] = int(len(df))

    # 2. Remove exact re-listings: same vendor and same normalised title.
    df["key"] = df["Vendor"].map(norm) + "||" + df["Item"].map(norm)
    before = len(df)
    df = df.drop_duplicates("key", keep="first").copy()
    log["duplicate_relistings_removed"] = int(before - len(df))
    log["unique_listings"] = int(len(df))
    log["unique_vendors"] = int(df["Vendor"].nunique())
    log["categories"] = int(df["Category"].nunique())

    # 3. Rating parsed to a number where available ("4.96/5").
    r = df["Rating"].str.extract(r"([0-9.]+)/5")[0]
    df["rating"] = pd.to_numeric(r, errors="coerce")

    df["digital"] = df["top"].isin(DIGITAL_TOP)
    df["haas"] = df["Category"].eq(HAAS_LABEL)

    log["top_level_counts"] = {k: int(v) for k, v in df["top"].value_counts().head(15).items()}
    dig = df[df["digital"]]
    log["digital_listings"] = int(len(dig))
    log["digital_vendors"] = int(dig["Vendor"].nunique())
    log["digital_category_counts"] = {k: int(v) for k, v in dig["Category"].value_counts().items()}
    log["haas_listings"] = int(df["haas"].sum())
    log["haas_vendors"] = int(df.loc[df["haas"], "Vendor"].nunique())
    log["haas_prevalence_in_digital"] = float(dig["haas"].mean())
    log["haas_prevalence_in_full"] = float(df["haas"].mean())
    log["mean_words_digital"] = float(dig["text"].str.split().str.len().mean())
    log["median_words_digital"] = float(dig["text"].str.split().str.len().median())

    df.drop(columns=["key"]).to_parquet(OUT, index=False)
    LOG.write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
