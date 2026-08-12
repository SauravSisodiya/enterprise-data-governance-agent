"""
The three properties the design review claims, as tests.

  1. Scores are reproducible - same input, same numbers, every run.
  2. The report is complete with the language model switched off.
  3. The pipeline survives a dataset whose columns we did not choose.

(3) is the one that protects the demo. The system is built against synthetic
data whose columns we named ourselves; the demo runs on real data whose columns
we did not. Overfitting the name taxonomy to our own schema is the easiest way
to build something that finds nothing in front of an audience.
"""
from __future__ import annotations

import pandas as pd
import pytest

from governance import config, run
from governance.core import risk


@pytest.fixture(scope="module")
def synthetic_ctx():
    df, name, profile = run.load("synthetic")
    return run.run(df, name, profile, llm_enabled=False)


def test_scores_are_reproducible():
    df, name, profile = run.load("synthetic")
    a = run.run(df, name, profile)
    b = run.run(df, name, profile)

    assert a["quality_report"] == b["quality_report"]
    assert a["issues"] == b["issues"]
    assert [f.risk for f in a["issues"]] == [f.risk for f in b["issues"]]


def test_report_is_complete_without_a_model(synthetic_ctx):
    ctx = synthetic_ctx
    assert ctx["llm_enabled"] is False
    assert ctx["catalog"], "catalog is empty"
    assert ctx["quality_report"] is not None
    assert ctx["compliance_report"] is not None
    assert ctx["issues"], "no findings raised"

    # `issues` is the inbox the agents append to; `findings` is the settled
    # record after scoring, citation and gating. Same content, different stage.
    assert len(ctx["findings"]) == len(ctx["issues"])
    assert all(f.risk is None for f in ctx["issues"]), "the inbox is unscored"

    # every assessed dimension has a real number ...
    assessed = [d for d in ctx["quality_report"].dimensions if d.assessed]
    assert assessed and all(d.score is not None for d in assessed)

    # ... every settled finding is scored and banded ...
    assert all(f.risk is not None and f.band for f in ctx["findings"])

    # ... and nothing carries narrative text, because nothing generated any.
    assert all(f.narrative is None for f in ctx["findings"])
    assert all(d.narrative is None for d in ctx["quality_report"].dimensions)
    assert all(e.description is None for e in ctx["catalog"])


def test_not_assessed_dimensions_are_declared_not_scored(synthetic_ctx):
    for name in config.NOT_ASSESSED:
        dim = synthetic_ctx["quality_report"].by_name(name)
        assert dim is not None, f"{name} missing from the report entirely"
        assert dim.score is None
        assert dim.not_assessed_reason


def test_findings_are_immutable(synthetic_ctx):
    """Scoring returns new objects; it never mutates what the core produced."""
    finding = synthetic_ctx["issues"][0]
    with pytest.raises(Exception):
        finding.risk = 1                      # frozen dataclass
    rescored = finding.scored(42, "Medium")
    assert rescored is not finding and finding.risk != 42


def test_survives_an_unfamiliar_schema():
    """A dataset with columns we have never seen must not crash the pipeline."""
    df = pd.DataFrame({
        "Invoice":      ["536365", "536365", "536366", "536366"],
        "StockCode":    ["85123A", "71053", "84406B", "84406B"],
        "Description":  ["WHITE HANGING HEART", "WHITE METAL LANTERN", None, "CREAM CUPID"],
        "Quantity":     [6, 6, -2, 8],
        "InvoiceDate":  ["2010-12-01", "2010-12-01", "2010-12-01", "2010-12-01"],
        "Price":        [2.55, 3.39, 2.75, 2.75],
        "Customer ID":  ["17850", "17850", None, "13047"],
        "Country":      ["United Kingdom", "United Kingdom", "France", "France"],
    })
    ctx = run.run(df, "online_retail", config.DATASET_PROFILES["online_retail"])

    assert len(ctx["catalog"]) == 8
    assert ctx["quality_report"].overall_score is not None
    # "Customer ID" is a pseudonymous identifier - still personal data under
    # GDPR Art. 4(5). Finding it on a schema we did not design is the point.
    pii = {c.column for c in ctx["compliance_report"].pii_columns}
    assert "Customer ID" in pii, f"expected Customer ID to be classified, got {pii}"


def test_risk_is_bounded_and_banded(synthetic_ctx):
    for f in synthetic_ctx["findings"]:
        assert 0 <= f.risk <= 100
        assert f.band == config.band_for(f.risk)
        assert f.status in {"pending_review", "auto_logged"}
        if f.risk >= config.REVIEW_THRESHOLD:
            assert f.status == "pending_review"
