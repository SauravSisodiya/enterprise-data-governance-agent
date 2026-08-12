"""
Masking utilities.

Masking is deterministic on purpose: the same input always produces the same
token. That means distinct customers can still be counted, joined and analysed
without any identity ever being visible - which is why we mask rather than
simply deleting.

Nothing leaves this system unmasked. Values are masked before they are written
to a report, a log, or a model prompt.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from governance import config


def mask_value(value, kind: str = "generic") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)

    if kind == "email" and "@" in text:
        local, _, domain = text.partition("@")
        head = local[0] if local else "*"
        return f"{head}***@{domain}"

    if kind == "phone":
        digits = [c for c in text if c.isdigit()]
        if len(digits) >= 4:
            return "*" * (len(digits) - 4) + "".join(digits[-4:])

    if kind == "person_name":
        return f"{text[0]}." if text else ""

    digest = hashlib.sha256((text + config.MASK_SALT).encode()).hexdigest()
    return f"{kind[:3]}_{digest[:10]}"


def mask_series(s: pd.Series, kind: str = "generic") -> pd.Series:
    return s.map(lambda v: mask_value(v, kind))


def preview(df: pd.DataFrame, column: str, kind: str,
            n: int = 3) -> list[str]:
    """A few masked examples, safe to show in a report or dashboard."""
    values = df[column].dropna().head(n)
    return [mask_value(v, kind) for v in values]
