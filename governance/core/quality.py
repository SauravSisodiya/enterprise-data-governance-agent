"""
Data quality scoring - four dimensions, each 0-100.

    completeness   populated cells over required cells
    uniqueness     one minus the duplicate rate on the business key
    validity       share of values conforming to their column's expected format
    consistency    one agreed surface form per value, plus declared cross-field rules

Accuracy and timeliness are NOT computed. Accuracy needs a reference dataset to
compare against and timeliness needs an agreed freshness SLA; neither exists
here. They are reported as NOT ASSESSED rather than given an invented number -
an unmeasured dimension is a known gap, a fabricated score is a defect.

Two things every dimension does:

  * it returns an aggregate SCORE, and
  * it returns cell-level FINDINGS.

These are independent. A dimension can pass its threshold while still producing
findings, because 25 bad values out of 1,300 is genuinely 98% valid and also
genuinely 25 things somebody has to fix.
"""
from __future__ import annotations

import pandas as pd

from governance import config
from governance.core.profiling import normalise, normalise_frame
from governance.core.types import COMPILED, entry_for
from governance.state import CatalogEntry, DimensionScore, Finding, QualityReport

EXPOSURE_RANK = {"non_personal": 0, "pseudonymous_identifier": 1,
                 "quasi_identifier": 2, "direct_identifier": 3}


def _worst_data_class(catalog: list[CatalogEntry],
                      columns: list[str] | None = None) -> str:
    """
    The most exposing classification among the given columns.

    A duplicated row duplicates everything in it, so the exposure of a duplicate
    finding is driven by the most sensitive column in the record - not by the
    fact that duplication itself is a mundane defect.
    """
    entries = [e for e in catalog if columns is None or e.name in columns]
    if not entries:
        return config.DEFAULT_DATA_CLASS
    return max((e.data_class for e in entries),
               key=lambda c: EXPOSURE_RANK.get(c, 0))


# --------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------
def completeness(df: pd.DataFrame, catalog: list[CatalogEntry],
                 required: list[str] | None) -> tuple[DimensionScore, list[Finding]]:
    columns = [c for c in (required or list(df.columns)) if c in df.columns]
    if not columns:
        return DimensionScore("completeness", None,
                              config.THRESHOLDS["completeness"],
                              not_assessed_reason="no required columns present"), []

    sub = normalise_frame(df[columns])
    score = 100.0 * (1 - sub.isna().sum().sum() / sub.size)

    findings: list[Finding] = []
    failing: list[str] = []
    per_column: dict[str, float] = {}
    for column in columns:
        null_pct = round(float(sub[column].isna().mean()) * 100, 2)
        per_column[column] = null_pct
        if null_pct > config.NULL_HEAVY_COLUMN_THRESHOLD:
            failing.append(column)
            entry = entry_for(catalog, column)
            findings.append(Finding(
                source="quality",
                issue_type="null_heavy_column",
                column=column,
                description=f"{null_pct:.1f}% of values in '{column}' are missing.",
                data_class=entry.data_class if entry else config.DEFAULT_DATA_CLASS,
                total_rows=len(df),
                scope="column",
                evidence={"null_pct": null_pct,
                          "threshold": config.NULL_HEAVY_COLUMN_THRESHOLD},
                articles=tuple(config.ISSUE_ARTICLE_MAP.get("null_heavy_column", [])),
            ))

    return DimensionScore(
        dimension="completeness",
        score=round(score, 2),
        threshold=config.THRESHOLDS["completeness"],
        failing_columns=tuple(failing),
        evidence={"null_pct_by_column": per_column,
                  "columns_assessed": len(columns)},
    ), findings


# --------------------------------------------------------------------------
# uniqueness
# --------------------------------------------------------------------------
def uniqueness(df: pd.DataFrame, catalog: list[CatalogEntry],
               business_key: list[str] | None) -> tuple[DimensionScore, list[Finding]]:
    key = [c for c in (business_key or list(df.columns)) if c in df.columns]
    if not key:
        return DimensionScore("uniqueness", None, config.THRESHOLDS["uniqueness"],
                              not_assessed_reason="business key not present"), []

    dup_mask = df.duplicated(subset=key, keep="first")
    dup_rows = [int(i) for i in df.index[dup_mask]]
    score = 100.0 * (1 - len(dup_rows) / max(len(df), 1))

    findings: list[Finding] = []
    if dup_rows:
        label = " + ".join(key)
        findings.append(Finding(
            source="quality",
            issue_type="duplicate_record",
            column=key[0] if len(key) == 1 else label,
            description=(f"{len(dup_rows)} records repeat an earlier row with the "
                         f"same {label}."),
            data_class=_worst_data_class(catalog),
            total_rows=len(df),
            rows=tuple(dup_rows),
            evidence={"business_key": key, "duplicate_count": len(dup_rows)},
            articles=tuple(config.ISSUE_ARTICLE_MAP.get("duplicate_record", [])),
        ))

    return DimensionScore(
        dimension="uniqueness",
        score=round(score, 2),
        threshold=config.THRESHOLDS["uniqueness"],
        failing_columns=tuple(key) if dup_rows else (),
        evidence={"business_key": key, "duplicate_count": len(dup_rows)},
    ), findings


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------
def validity(df: pd.DataFrame,
             catalog: list[CatalogEntry]) -> tuple[DimensionScore, list[Finding]]:
    checked = valid = 0
    per_column: dict[str, float] = {}
    failing: list[str] = []
    findings: list[Finding] = []

    for entry in catalog:
        pattern = COMPILED.get(entry.semantic_type)
        if pattern is None:
            # No rule for this type, so it is excluded from the denominator
            # entirely. Counting unrules-able columns as valid would inflate
            # the score with columns we never actually checked.
            continue

        values = normalise(df[entry.name]).dropna().astype(str)
        if values.empty:
            continue

        ok = values.str.fullmatch(pattern)
        checked += len(values)
        valid += int(ok.sum())
        rate = round(float(ok.mean()) * 100, 2)
        per_column[entry.name] = rate

        bad_rows = [int(i) for i in values.index[~ok]]
        if bad_rows:
            issue_type = f"invalid_{entry.semantic_type}"
            if issue_type not in config.SEVERITY:
                issue_type = "invalid_format"
            findings.append(Finding(
                source="quality",
                issue_type=issue_type,
                column=entry.name,
                description=(f"{len(bad_rows)} values in '{entry.name}' are not "
                             f"well-formed {entry.semantic_type.replace('_', ' ')}."),
                data_class=entry.data_class,
                total_rows=len(df),
                rows=tuple(bad_rows),
                evidence={"expected_type": entry.semantic_type,
                          "conformance_pct": rate},
            ))
        if rate < config.THRESHOLDS["validity"]:
            failing.append(entry.name)

    if checked == 0:
        return DimensionScore("validity", None, config.THRESHOLDS["validity"],
                              not_assessed_reason="no column has an applicable "
                                                  "format rule"), []

    return DimensionScore(
        dimension="validity",
        score=round(100.0 * valid / checked, 2),
        threshold=config.THRESHOLDS["validity"],
        failing_columns=tuple(failing),
        evidence={"values_checked": checked, "values_valid": valid,
                  "conformance_by_column": per_column},
    ), findings


# --------------------------------------------------------------------------
# consistency
# --------------------------------------------------------------------------
def _canonical_checks(df: pd.DataFrame, catalog: list[CatalogEntry]
                      ) -> tuple[list[float], list[Finding], list[str]]:
    """Each real-world value should have exactly one surface form."""
    scores: list[float] = []
    findings: list[Finding] = []
    failing: list[str] = []

    for column, mapping in config.CANONICAL.items():
        if column not in df.columns:
            continue
        raw = normalise(df[column]).dropna().astype(str)
        if raw.empty:
            continue

        canon = raw.map(mapping)
        known = canon.notna()
        if not known.any():
            continue

        odd_rows: list[int] = []
        for value, group in raw[known].groupby(canon[known]):
            majority = group.value_counts().idxmax()
            odd_rows += [int(i) for i in group.index[group != majority]]

        agreement = 1 - len(odd_rows) / int(known.sum())
        scores.append(agreement)

        if odd_rows:
            failing.append(column)
            entry = entry_for(catalog, column)
            findings.append(Finding(
                source="quality",
                issue_type="inconsistent_value",
                column=column,
                description=(f"{len(odd_rows)} values in '{column}' use a "
                             f"non-standard spelling of a value that appears "
                             f"elsewhere in its agreed form."),
                data_class=entry.data_class if entry else config.DEFAULT_DATA_CLASS,
                total_rows=len(df),
                rows=tuple(sorted(odd_rows)),
                evidence={"agreement_pct": round(agreement * 100, 2)},
            ))
    return scores, findings, failing


def _cross_field_checks(df: pd.DataFrame, catalog: list[CatalogEntry]
                        ) -> tuple[list[float], list[Finding], list[str]]:
    scores: list[float] = []
    findings: list[Finding] = []
    failing: list[str] = []

    # Date columns must be real datetimes before a comparison means anything.
    frame = normalise_frame(df)
    for entry in catalog:
        if entry.semantic_type == "date" and entry.name in frame.columns:
            frame[entry.name] = pd.to_datetime(frame[entry.name],
                                               errors="coerce", format="mixed")

    for name, columns, expression in config.CROSS_FIELD_RULES:
        if not all(c in frame.columns for c in columns):
            continue
        # A null is an incompleteness problem. Failing the row here as well
        # would penalise one defect under two dimensions.
        evaluable = frame[columns].notna().all(axis=1)
        if not evaluable.any():
            continue
        try:
            ok = frame[evaluable].eval(expression)
        except Exception:
            continue

        scores.append(float(ok.mean()))
        bad_rows = [int(i) for i in ok.index[~ok.astype(bool)]]
        if bad_rows:
            failing.append(columns[0])
            entry = entry_for(catalog, columns[0])
            findings.append(Finding(
                source="quality",
                issue_type="inconsistent_value",
                column=columns[0],
                description=(f"{len(bad_rows)} rows break the rule "
                             f"'{expression}'."),
                data_class=entry.data_class if entry else config.DEFAULT_DATA_CLASS,
                total_rows=len(df),
                rows=tuple(sorted(bad_rows)),
                evidence={"rule": name, "expression": expression,
                          "rows_evaluated": int(evaluable.sum())},
            ))
    return scores, findings, failing


def consistency(df: pd.DataFrame,
                catalog: list[CatalogEntry]) -> tuple[DimensionScore, list[Finding]]:
    a_scores, a_findings, a_failing = _canonical_checks(df, catalog)
    b_scores, b_findings, b_failing = _cross_field_checks(df, catalog)

    scores = a_scores + b_scores
    if not scores:
        return DimensionScore("consistency", None,
                              config.THRESHOLDS["consistency"],
                              not_assessed_reason="no canonical vocabulary or "
                                                  "cross-field rule applies"), []

    return DimensionScore(
        dimension="consistency",
        score=round(100.0 * sum(scores) / len(scores), 2),
        threshold=config.THRESHOLDS["consistency"],
        failing_columns=tuple(a_failing + b_failing),
        evidence={"checks_run": len(scores),
                  "canonical_columns": list(config.CANONICAL),
                  "cross_field_rules": [r[0] for r in config.CROSS_FIELD_RULES]},
    ), a_findings + b_findings


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _all_unassessed(reason: str) -> QualityReport:
    return QualityReport(
        dimensions=tuple(
            DimensionScore(name, None, config.THRESHOLDS.get(name, 0.0),
                           not_assessed_reason=reason)
            for name in list(config.THRESHOLDS) + config.NOT_ASSESSED),
        overall_score=None)


def assess(df: pd.DataFrame, catalog: list[CatalogEntry],
           profile: config.DatasetProfile) -> tuple[QualityReport, list[Finding]]:
    # An empty dataset has no quality, and dividing by zero rows produces NaN -
    # which json.dumps writes as a bare NaN token that is not valid JSON and
    # breaks every downstream consumer. Declare it unassessed instead.
    if len(df) == 0:
        return _all_unassessed("dataset contains no rows"), []

    dimensions: list[DimensionScore] = []
    findings: list[Finding] = []

    for score, found in (
        completeness(df, catalog, profile.required_columns),
        uniqueness(df, catalog, profile.business_key),
        validity(df, catalog),
        consistency(df, catalog),
    ):
        dimensions.append(score)
        findings += found

    for name in config.NOT_ASSESSED:
        reason = ("requires a reference dataset to compare against"
                  if name == "accuracy"
                  else "requires an agreed data freshness SLA")
        dimensions.append(DimensionScore(
            dimension=name, score=None,
            threshold=config.THRESHOLDS.get(name, 0.0),
            not_assessed_reason=reason))

    assessed = [d for d in dimensions if d.assessed]
    overall = None
    if assessed:
        weights = [config.DIMENSION_WEIGHTS.get(d.dimension, 1.0) for d in assessed]
        overall = round(
            sum(w * d.score for w, d in zip(weights, assessed)) / sum(weights), 2)

    return QualityReport(dimensions=tuple(dimensions), overall_score=overall), findings
