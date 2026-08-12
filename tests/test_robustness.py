"""
Inputs that should not crash the pipeline.

Every case here is one a real CSV can produce. Two of them did crash or corrupt
output when first tried:

  * a duplicated column header made `df[name]` return a DataFrame instead of a
    Series, and profiling died on it;
  * a frame with zero rows divided by zero and produced a NaN score, which
    json.dumps writes as a bare `NaN` token - not valid JSON, and enough to
    break every downstream consumer.

The strict-JSON assertion is the important part of each test. A report that
cannot be parsed is worse than a report that was never written.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from governance import report, run

CASES = {
    "empty_frame": pd.DataFrame(),
    "no_rows": pd.DataFrame({"email": pd.Series(dtype=object),
                             "n": pd.Series(dtype=float)}),
    "single_row": pd.DataFrame({"customer_email": ["a@b.com"]}),
    "all_null_column": pd.DataFrame({"a": [None] * 5, "b": [1, 2, 3, 4, 5]}),
    "sentinel_nulls": pd.DataFrame({"a": ["", "N/A", "null", "-", "?"]}),
    "no_personal_data": pd.DataFrame({"widget_count": [1, 2, 3],
                                      "price": [1.5, 2.5, 3.5]}),
    "unicode_columns": pd.DataFrame({"客户邮箱": ["a@b.com"], "café": ["x"]}),
    "spaces_in_names": pd.DataFrame({"Customer E-Mail": ["a@b.com"], "Order #": [1]}),
    "duplicate_columns": pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]),
    "constant_column": pd.DataFrame({"same": ["v"] * 20,
                                     "email": ["a@b.com"] * 20}),
    "mixed_types": pd.DataFrame({"m": [1, "two", 3.0, None, True]}),
    "long_strings": pd.DataFrame({"notes": ["word " * 400] * 3}),
    "five_digit_numbers": pd.DataFrame({"sku": [12345, 67890, 11111]}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_pipeline_survives(name):
    ctx = run.run(CASES[name].copy(), f"probe_{name}", llm_enabled=False)
    assert "quality_report" in ctx and "findings" in ctx


@pytest.mark.parametrize("name", list(CASES))
def test_report_is_strict_json(name):
    """allow_nan=False is the whole point: NaN and Infinity are not JSON."""
    ctx = run.run(CASES[name].copy(), f"probe_{name}", llm_enabled=False)
    blob = json.dumps(report.to_dict(ctx), allow_nan=False)
    assert json.loads(blob)["dataset"] == f"probe_{name}"


def test_empty_dataset_is_unassessed_not_zero():
    """
    Nothing about an empty dataset is 0% complete - it is unmeasurable. Scoring
    it zero would put a frightening number in a report that means nothing.
    """
    ctx = run.run(pd.DataFrame({"a": pd.Series(dtype=object)}), "empty")
    q = ctx["quality_report"]
    assert q.overall_score is None
    assert all(d.score is None for d in q.dimensions)
    assert all("no rows" in (d.not_assessed_reason or "") for d in q.dimensions)


def test_duplicate_columns_are_renamed_and_audited():
    ctx = run.run(pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]), "dupes")
    assert [e.name for e in ctx["catalog"]] == ["x", "x.1"]
    audited = [e for e in ctx["audit_log"]
               if e["action"] == "renamed_duplicate_columns"]
    assert audited, "a silently renamed column is a silently altered dataset"
    assert "x -> x.1" in audited[0]["detail"]["renames"]


def test_a_column_of_five_digit_numbers_is_not_personal_data():
    """
    Product codes look exactly like US postcodes. Caught on the real dataset:
    every five-digit StockCode was classified as a postcode, turning a product
    catalogue into a table of personal data.
    """
    ctx = run.run(pd.DataFrame({"sku": [12345, 67890, 11111]}), "skus")
    assert not ctx["compliance_report"].pii_columns
