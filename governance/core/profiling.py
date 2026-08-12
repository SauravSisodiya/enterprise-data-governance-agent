"""
Per-column statistics.

Deliberately boring: counts, rates and samples. No interpretation happens here -
deciding what a column *is* belongs to types.py, and deciding whether that is a
problem belongs to quality.py.

The one subtlety is what counts as missing. A cell holding "" or "N/A" is empty
in every sense that matters to a data steward, but pandas counts it as present.
Normalising sentinels first is what stops the completeness score being quietly
and confidently wrong.
"""
from __future__ import annotations

import pandas as pd

from governance import config
from governance.state import ColumnProfile


def normalise(s: pd.Series) -> pd.Series:
    """Replace sentinel placeholders with real nulls."""
    if s.dtype == object:
        return s.replace(config.SENTINEL_NULLS, None)
    return s


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(normalise)


def _python(value):
    """numpy scalars are not JSON-serialisable; unwrap them."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


def profile_column(name: str, raw: pd.Series) -> ColumnProfile:
    s = normalise(raw)
    non_null = s.dropna()
    n = len(s)

    top_share = 0.0
    if len(non_null):
        top_share = float(non_null.value_counts(normalize=True).iloc[0])

    mean_length = None
    if s.dtype == object and len(non_null):
        mean_length = float(non_null.astype(str).str.len().mean())

    numeric_min = numeric_max = numeric_mean = None
    if pd.api.types.is_numeric_dtype(s) and len(non_null):
        numeric_min = float(non_null.min())
        numeric_max = float(non_null.max())
        numeric_mean = float(non_null.mean())

    return ColumnProfile(
        name=name,
        dtype=str(raw.dtype),
        rows=n,
        null_count=int(s.isna().sum()),
        null_pct=round(float(s.isna().mean()) * 100, 2),
        distinct_count=int(non_null.nunique()),
        distinct_pct=round(non_null.nunique() / max(len(non_null), 1) * 100, 2),
        samples=tuple(_python(v) for v in non_null.drop_duplicates().head(5)),
        is_constant=bool(non_null.nunique() <= 1),
        near_constant=bool(top_share > 0.95),
        mean_length=mean_length,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        numeric_mean=numeric_mean,
    )


def profile(df: pd.DataFrame) -> list[ColumnProfile]:
    return [profile_column(c, df[c]) for c in df.columns]
