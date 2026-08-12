"""
Semantic type inference - deciding what a column actually holds.

This is the linchpin of the whole system. Validity cannot be checked until we
know what "valid" means for a column, and compliance cannot shortlist personal
data without knowing which columns plausibly contain it.

Evidence is considered in this order, and the order matters:

    1. the column NAME
    2. the VALUES
    3. the dtype
    4. length heuristics

Name evidence comes first on purpose. If the type were inferred purely from
values and validity were then scored against that inference, the reasoning would
be circular - the type would simply become "whatever 80% of the values look
like", and validity would be 100% by construction. Taking the type from the name
where possible breaks the loop: a column called `email` is *expected* to hold
email addresses, so the 20% that do not are a genuine defect rather than a
redefinition of the type.
"""
from __future__ import annotations

import re

import pandas as pd

from governance import config
from governance.core.profiling import normalise
from governance.state import CatalogEntry, ColumnProfile

# Compiled once. Kept module-level so both validity and PII detection use the
# exact same expressions - one library, two ways of applying it.
COMPILED: dict[str, re.Pattern] = {
    name: re.compile(pattern) for name, pattern in config.PATTERNS.items()
}

# Longest hints first, so a specific hint always beats a generic one that
# happens to be a substring of the same column name.
_HINTS: list[tuple[str, str]] = sorted(
    ((semantic_type, hint)
     for semantic_type, hints in config.NAME_HINTS.items()
     for hint in hints),
    key=lambda pair: -len(pair[1]),
)


def type_from_name(column: str) -> str | None:
    key = config.normalise_name(column)
    for semantic_type, hint in _HINTS:
        if hint in key:
            return semantic_type
    return None


def match_rate(s: pd.Series, semantic_type: str) -> float:
    """Share of non-null values that are ENTIRELY a well-formed instance."""
    pattern = COMPILED.get(semantic_type)
    if pattern is None:
        return 0.0
    values = normalise(s).dropna().astype(str)
    if values.empty:
        return 0.0
    return float(values.str.fullmatch(pattern).mean())


def _looks_like_dates(s: pd.Series) -> bool:
    values = normalise(s).dropna()
    if values.empty:
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= config.TYPE_INFERENCE_THRESHOLD


def infer_type(column: str, s: pd.Series,
               prof: ColumnProfile) -> tuple[str, str]:
    """Returns (semantic_type, evidence)."""
    named = type_from_name(column)
    if named is not None:
        return named, "name"

    # Types with a generic pattern are excluded from value-only inference. See
    # config.NAME_ONLY_TYPES - a five-digit product code is indistinguishable
    # from a postcode by value, and guessing wrong turns a product catalogue
    # into a table of personal data.
    rates = {t: match_rate(s, t) for t in COMPILED
             if t not in config.NAME_ONLY_TYPES}
    if rates:
        best, rate = max(rates.items(), key=lambda kv: kv[1])
        if rate >= config.TYPE_INFERENCE_THRESHOLD:
            return best, "value"

    if pd.api.types.is_numeric_dtype(s):
        return "numeric", "dtype"
    if pd.api.types.is_datetime64_any_dtype(s) or _looks_like_dates(s):
        return "date", "value"
    if prof.mean_length is not None and prof.mean_length > config.FREETEXT_MIN_MEAN_LENGTH:
        return "free_text", "heuristic"
    return "categorical", "heuristic"


def build_catalog(df: pd.DataFrame,
                  profiles: list[ColumnProfile]) -> list[CatalogEntry]:
    catalog: list[CatalogEntry] = []
    for prof in profiles:
        semantic_type, evidence = infer_type(prof.name, df[prof.name], prof)
        catalog.append(CatalogEntry(
            profile=prof,
            semantic_type=semantic_type,
            type_evidence=evidence,
            data_class=config.DATA_CLASS.get(semantic_type,
                                             config.DEFAULT_DATA_CLASS),
            is_freetext=(semantic_type == "free_text"),
        ))
    return catalog


def entry_for(catalog: list[CatalogEntry], column: str) -> CatalogEntry | None:
    return next((e for e in catalog if e.name == column), None)
