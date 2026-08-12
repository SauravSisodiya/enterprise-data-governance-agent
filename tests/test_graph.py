"""
LangGraph orchestration.

Two things are worth asserting here:

  1. The graph and the sequential runner produce identical results. They call
     the same node functions, so this should hold by construction - the test
     exists to catch someone later "optimising" one of the two paths.

  2. The reducer on `issues` is genuinely load-bearing. The design review claims
     that without it the parallel branch raises InvalidUpdateError. That claim
     is asserted here against real LangGraph rather than taken on trust.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest

from governance import run
from governance.graph.build import build_graph, run_graph


@pytest.fixture(scope="module")
def both():
    df, name, _ = run.load("synthetic")
    return (run.run(df, name, llm_enabled=False),
            run_graph(df, name, llm_enabled=False))


def test_graph_matches_the_sequential_runner(both):
    sequential, graph = both

    assert sequential["quality_report"] == graph["quality_report"]
    assert sequential["compliance_report"] == graph["compliance_report"]
    assert sequential["catalog"] == graph["catalog"]

    a, b = sequential["findings"], graph["findings"]
    assert len(a) == len(b)
    assert [f.id for f in a] == [f.id for f in b]
    assert [f.risk for f in a] == [f.risk for f in b]
    assert [f.band for f in a] == [f.band for f in b]
    assert [f.status for f in a] == [f.status for f in b]
    assert [tuple(c.reference for c in f.citations) for f in a] == \
           [tuple(c.reference for c in f.citations) for f in b]

    assert [r.finding_id for r in sequential["recommendations"]] == \
           [r.finding_id for r in graph["recommendations"]]


def test_both_branches_reach_the_join(both):
    """Quality and compliance each contribute findings to the merged list."""
    _, graph = both
    sources = {f.source for f in graph["findings"]}
    assert sources == {"quality", "compliance"}


def test_every_node_is_recorded_in_the_audit_log(both):
    _, graph = both
    actors = {entry["actor"] for entry in graph["audit_log"]}
    assert {"metadata_agent", "quality_agent", "compliance_agent", "join_node",
            "risk_node", "review_gate", "recommendation_agent"} <= actors


def test_narrative_node_is_skipped_when_disabled(both):
    _, graph = build_graph(), None
    df, name, _ = run.load("synthetic")
    ctx = run_graph(df, name, llm_enabled=False)
    # The conditional edge routes past the narrative node entirely.
    assert ctx.get("executive_summary") is None
    assert not [e for e in ctx["audit_log"] if e["actor"] == "narrative_layer"]


def test_reducer_is_load_bearing():
    """
    Without operator.add on a key two parallel nodes both write, LangGraph
    refuses to guess how to merge them. This is the failure the design review
    warns about; here it is, reproduced.
    """
    from langgraph.graph import END, START, StateGraph

    class Unreduced(TypedDict, total=False):
        issues: list           # no reducer

    def fan(_state): return {}
    def left(_state): return {"issues": ["a"]}
    def right(_state): return {"issues": ["b"]}

    g = StateGraph(Unreduced)
    for name, fn in (("fan", fan), ("left", left), ("right", right)):
        g.add_node(name, fn)
    g.add_edge(START, "fan")
    g.add_edge("fan", "left")
    g.add_edge("fan", "right")
    g.add_edge("left", END)
    g.add_edge("right", END)

    with pytest.raises(Exception) as caught:
        g.compile().invoke({"issues": []})
    assert "InvalidUpdate" in type(caught.value).__name__ or \
           "one value per step" in str(caught.value).lower()

    # The same graph with a reducer merges both branches cleanly.
    class Reduced(TypedDict, total=False):
        issues: Annotated[list, operator.add]

    g2 = StateGraph(Reduced)
    for name, fn in (("fan", fan), ("left", left), ("right", right)):
        g2.add_node(name, fn)
    g2.add_edge(START, "fan")
    g2.add_edge("fan", "left")
    g2.add_edge("fan", "right")
    g2.add_edge("left", END)
    g2.add_edge("right", END)

    assert sorted(g2.compile().invoke({"issues": []})["issues"]) == ["a", "b"]
