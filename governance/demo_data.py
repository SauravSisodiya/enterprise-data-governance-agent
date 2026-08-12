"""
Downloads and samples the demonstration dataset.

    python -m governance.demo_data

Source: Online Retail II, UCI Machine Learning Repository (dataset 502).
Roughly a million real UK online-retail transactions from 2009-2011.

Why this dataset rather than a bigger synthetic one:

  It is messy in ways WE DID NOT CHOOSE. Around a quarter of rows have no
  customer id, quantities go negative for returns, descriptions are
  inconsistently cased, and rows repeat. Finding defects nobody planted is a
  far stronger demonstration than finding the ones we did.

  Its identifiers are already pseudonymised - customers appear as surrogate
  numbers, not names. That keeps the demonstration clear of real personal data,
  and it sets up the most interesting finding available: a pseudonymous
  customer id is still personal data under GDPR Art. 4(5), which is very
  commonly assumed to be false.

The evaluation set is the synthetic one. This dataset has no answer key, so it
is assessed by manual spot-check rather than precision and recall.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from governance import config

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SAMPLE_ROWS = 5000
SEED = 20260812


def download() -> pd.DataFrame:
    print(f"  downloading {URL}")
    request = urllib.request.Request(
        URL, headers={"User-Agent": "Mozilla/5.0 (capstone; research use)"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    print(f"  {len(payload) / 1_048_576:.1f} MB")

    archive = zipfile.ZipFile(io.BytesIO(payload))
    name = next(n for n in archive.namelist() if n.lower().endswith((".xlsx", ".csv")))
    print(f"  reading {name}")
    with archive.open(name) as handle:
        data = handle.read()

    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data), sheet_name=0)


def main() -> None:
    config.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    df = download()
    print(f"  full dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"  columns: {list(df.columns)}")

    # A contiguous block rather than a random sample: keeping whole invoices
    # together preserves the duplicate and return patterns that make the data
    # worth demonstrating on. Random sampling would quietly destroy them.
    start = 0
    sample = df.iloc[start:start + SAMPLE_ROWS].copy()

    out = config.DEMO_DIR / "online_retail.csv"
    sample.to_csv(out, index=False)

    print(f"\n  wrote {out}  ({len(sample)} rows)")
    print("\n  defects already present, none of them planted by us:")
    for column in sample.columns:
        nulls = sample[column].isna().mean()
        if nulls:
            print(f"    {column:<14} {nulls:.1%} missing")
    dupes = sample.duplicated().sum()
    print(f"    {'duplicate rows':<14} {dupes}")
    if "Quantity" in sample.columns:
        print(f"    {'negative qty':<14} {(sample.Quantity < 0).sum()} (returns)")

    print("\n  next:  python -m governance.run --dataset online_retail "
          "--path data/demo/online_retail.csv")


if __name__ == "__main__":
    main()
