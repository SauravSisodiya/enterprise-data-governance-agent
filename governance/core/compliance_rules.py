"""
Personal-data classification, masking, and clause citation.

Three detectors, deliberately kept apart because they fail differently:

  1. classify()             column level, by NAME and by VALUE
  2. scan_freetext()        cell level, inside prose
  3. scan_high_confidence() cell level, anywhere, for unmistakable formats

Why (3) exists separately from (1): a column-level detector needs a match RATE
before it will flag anything, otherwise a single coincidental match turns every
column into personal data. But eight national ID numbers hidden among 500 rows
is a rate of 1.6% - below any sane threshold, and exactly the case that matters
most. National IDs and card numbers are specific enough that one occurrence is
worth reporting, so they bypass the rate test and are flagged cell by cell.

A note on language. This module reports CONTROL GAPS, not violations. Whether
processing is lawful depends on consent, purpose and retention policy, none of
which are in a CSV. What can be demonstrated from the data alone is that
personal data is present and stored in the clear - so that is what we say.
"""
from __future__ import annotations

import pandas as pd

from governance import config
from governance.core import masking
from governance.core.profiling import normalise
from governance.core.types import COMPILED, match_rate, type_from_name
from governance.state import CatalogEntry, ComplianceReport, Finding, PIIColumn


def _articles(data_class: str, issue_type: str | None = None) -> tuple[str, ...]:
    """
    Baseline citation, from a static map. Semantic retrieval over the real
    regulation text enriches these with the actual clause wording later - it
    does not replace them, so citations still work with no policy corpus loaded.
    """
    refs = list(config.ARTICLE_MAP.get(data_class, []))
    if issue_type:
        refs += config.ISSUE_ARTICLE_MAP.get(issue_type, [])
    return tuple(dict.fromkeys(refs))


# --------------------------------------------------------------------------
# 1. column-level classification
# --------------------------------------------------------------------------
def classify(df: pd.DataFrame,
             catalog: list[CatalogEntry]) -> tuple[list[PIIColumn], list[Finding]]:
    columns: list[PIIColumn] = []
    findings: list[Finding] = []

    for entry in catalog:
        if entry.data_class == config.DEFAULT_DATA_CLASS:
            continue

        by_name = type_from_name(entry.name) is not None
        rate = match_rate(df[entry.name], entry.semantic_type)
        by_value = rate >= config.VALUE_MATCH_THRESHOLD

        if by_name and by_value:
            evidence = "name+value"
        elif by_value:
            evidence = "value"          # the name told us nothing
        elif by_name:
            evidence = "name"
        else:
            continue

        kind = entry.semantic_type
        columns.append(PIIColumn(
            column=entry.name,
            semantic_type=kind,
            data_class=entry.data_class,
            evidence=evidence,
            match_rate=round(rate * 100, 2),
            masked=True,
            articles=_articles(entry.data_class),
        ))

        findings.append(Finding(
            source="compliance",
            issue_type="unmasked_pii_column",
            column=entry.name,
            description=(f"'{entry.name}' holds {entry.data_class.replace('_', ' ')} "
                         f"data ({kind}) and is stored in the clear."),
            data_class=entry.data_class,
            total_rows=len(df),
            scope="column",
            evidence={"classified_by": evidence, "match_rate_pct": round(rate * 100, 2),
                      "semantic_type": kind},
            articles=_articles(entry.data_class, "unmasked_pii_column"),
        ))

    return columns, findings


# --------------------------------------------------------------------------
# 2. free-text scan
# --------------------------------------------------------------------------
def scan_freetext(df: pd.DataFrame,
                  catalog: list[CatalogEntry]) -> list[Finding]:
    """
    Personal data written into prose, where there is no whole-cell pattern to
    match. This is the case a regular expression over the full value can never
    reach, because the email address is inside the sentence rather than being
    the sentence.
    """
    findings: list[Finding] = []

    for entry in catalog:
        if not entry.is_freetext:
            continue
        values = normalise(df[entry.name]).dropna().astype(str)
        if values.empty:
            continue

        for kind in config.FREETEXT_SCAN_TYPES:
            pattern = COMPILED.get(kind)
            if pattern is None:
                continue
            hit = values.str.contains(pattern)
            rows = [int(i) for i in values.index[hit]]
            if not rows:
                continue

            data_class = config.DATA_CLASS.get(kind, config.DEFAULT_DATA_CLASS)
            findings.append(Finding(
                source="compliance",
                issue_type="pii_in_freetext",
                column=entry.name,
                description=(f"{len(rows)} free-text values in '{entry.name}' "
                             f"contain an embedded {kind.replace('_', ' ')}."),
                data_class=data_class,
                total_rows=len(df),
                rows=tuple(sorted(rows)),
                evidence={"pii_type": kind, "scanned_values": len(values)},
                articles=_articles(data_class, "pii_in_freetext"),
            ))
    return findings


# --------------------------------------------------------------------------
# 3. high-confidence scan
# --------------------------------------------------------------------------
def scan_high_confidence(df: pd.DataFrame,
                         catalog: list[CatalogEntry]) -> list[Finding]:
    """
    Unmistakable formats, anywhere, at any rate. This is what finds personal
    data in a column whose name gives nothing away.
    """
    findings: list[Finding] = []

    for entry in catalog:
        if entry.is_freetext:
            continue                      # already covered by scan_freetext
        values = normalise(df[entry.name]).dropna().astype(str)
        if values.empty:
            continue

        for kind in config.HIGH_CONFIDENCE_PII_TYPES:
            if entry.semantic_type == kind:
                continue                  # reported as a PII column instead
            pattern = COMPILED.get(kind)
            if pattern is None:
                continue
            hit = values.str.contains(pattern)
            rows = [int(i) for i in values.index[hit]]
            if not rows:
                continue

            data_class = config.DATA_CLASS.get(kind, config.DEFAULT_DATA_CLASS)
            findings.append(Finding(
                source="compliance",
                issue_type="pii_in_mislabeled_column",
                column=entry.name,
                description=(f"{len(rows)} values in '{entry.name}' match the "
                             f"{kind.replace('_', ' ')} format. The column name "
                             f"does not indicate that it holds personal data."),
                data_class=data_class,
                total_rows=len(df),
                rows=tuple(sorted(rows)),
                evidence={"pii_type": kind, "column_classified_as": entry.semantic_type,
                          "scanned_values": len(values)},
                articles=_articles(data_class, "pii_in_freetext"),
            ))
    return findings


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def assess(df: pd.DataFrame,
           catalog: list[CatalogEntry]) -> tuple[ComplianceReport, list[Finding]]:
    pii_columns, findings = classify(df, catalog)
    findings += scan_freetext(df, catalog)
    findings += scan_high_confidence(df, catalog)

    masked_preview = {
        c.column: masking.preview(df, c.column, c.semantic_type)
        for c in pii_columns
    }

    report = ComplianceReport(
        pii_columns=tuple(pii_columns),
        freetext_columns=tuple(e.name for e in catalog if e.is_freetext),
        masked_preview=masked_preview,
    )
    return report, findings
