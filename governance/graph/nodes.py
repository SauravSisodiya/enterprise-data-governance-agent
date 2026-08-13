"""
The pipeline steps, as pure functions.

Each takes the GovernanceContext and returns a PARTIAL update - only the keys it
changed. That is LangGraph's contract, and it is also a good discipline for the
sequential runner: a node that returns its whole world is a node that can
clobber someone else's work.

Both the sequential runner and the graph call these same functions, so there is
exactly one implementation of every step. The graph is a different wiring of the
same parts, not a second version of the pipeline.
"""
from __future__ import annotations

from dataclasses import replace

from governance import config, report
from governance.core import compliance_rules, gate, profiling, quality, risk, types
from governance.core import recommend as core_recommend
from governance.state import GovernanceContext


def _profile_for(state: GovernanceContext) -> config.DatasetProfile:
    return config.DATASET_PROFILES.get(state["dataset_name"], config.DEFAULT_PROFILE)


# --------------------------------------------------------------------------
def _disambiguate(df):
    """
    Give repeated column names a suffix.

    A duplicated header is common in exported spreadsheets, and with one in
    place `df[name]` returns a DataFrame rather than a Series - which crashed
    profiling outright. Renaming keeps the run going and the audit log records
    that it happened, so the duplication is visible rather than silently
    resolved.
    """
    counts, renamed, changes = {}, [], []
    for name in df.columns:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            renamed.append(name)
        else:
            new = f"{name}.{counts[name] - 1}"
            renamed.append(new)
            changes.append((str(name), new))
    if changes:
        df = df.copy()
        df.columns = renamed
    return df, changes


def metadata_node(state: GovernanceContext) -> dict:
    """Runs first, always. Everything downstream needs the column inventory."""
    df, renames = _disambiguate(state["dataframe"])
    catalog = types.build_catalog(df, profiling.profile(df))

    entries = [report.audit(state, "metadata_agent", "profiled_columns",
                            columns=len(catalog), rows=len(df))]
    if renames:
        entries.append(report.audit(
            state, "metadata_agent", "renamed_duplicate_columns",
            renames=[f"{old} -> {new}" for old, new in renames]))

    return {"dataframe": df, "catalog": catalog, "audit_log": entries}


def quality_node(state: GovernanceContext) -> dict:
    quality_report, findings = quality.assess(
        state["dataframe"], state["catalog"], _profile_for(state))
    return {
        "quality_report": quality_report,
        "issues": findings,                       # appended by the reducer
        "audit_log": [report.audit(state, "quality_agent", "scored_dimensions",
                                   overall=quality_report.overall_score,
                                   findings=len(findings))],
    }


def compliance_node(state: GovernanceContext) -> dict:
    compliance_report, findings = compliance_rules.assess(
        state["dataframe"], state["catalog"])
    return {
        "compliance_report": compliance_report,
        "issues": findings,                       # appended by the reducer
        "audit_log": [report.audit(state, "compliance_agent",
                                   "classified_personal_data",
                                   pii_columns=len(compliance_report.pii_columns),
                                   findings=len(findings))],
    }


def join_node(state: GovernanceContext) -> dict:
    """
    A barrier, not a step.

    It exists so the graph has a single point where both parallel branches have
    definitely finished. Everything after this can assume `issues` is complete.
    """
    return {"audit_log": [report.audit(state, "join_node", "merged_findings",
                                       issues=len(state.get("issues", [])))]}


def risk_node(state: GovernanceContext) -> dict:
    findings = risk.score_all(state.get("issues", []))
    return {
        "findings": findings,                     # settled record, replaced
        "audit_log": [report.audit(state, "risk_node", "scored_findings",
                                   findings=len(findings))],
    }


def cite_node(state: GovernanceContext) -> dict:
    """
    Attach the clause text behind each finding.

    A lookup, not a generation: the article reference travels with the chunk
    from the moment the corpus is loaded, so it cannot be invented. Skipped
    silently when no index exists - findings keep the static references from
    config.ARTICLE_MAP, so citations lose quality but never disappear.
    """
    from governance.policy.retrieve import PolicyIndex, query_for

    findings = state.get("findings", [])
    index = PolicyIndex.load()
    if index is None or not index.chunks:
        return {"audit_log": [report.audit(
            state, "policy_retrieval", "skipped",
            reason="no policy index; falling back to static article references")]}

    cited, updated = 0, []
    for finding in findings:
        query = query_for(finding.issue_type, finding.data_class)
        if query is None or finding.data_class == config.DEFAULT_DATA_CLASS:
            updated.append(finding)
            continue
        hits = index.search(query, k=2, min_score=config.CITATION_MIN_SCORE)
        updated.append(replace(finding, citations=tuple(hits)) if hits else finding)
        cited += bool(hits)

    return {
        "findings": updated,
        "audit_log": [report.audit(state, "policy_retrieval", "cited_findings",
                                   backend=index.backend, chunks=len(index.chunks),
                                   findings_cited=cited,
                                   placeholder_corpus=index.uses_placeholder_text,
                                   degraded_reason=index.degraded_reason)],
    }


def review_gate(state: GovernanceContext) -> dict:
    findings = gate.apply(state.get("findings", []))
    waiting = len(gate.pending(findings))
    return {
        "findings": findings,
        "audit_log": [report.audit(state, "review_gate", "routed_findings",
                                   pending_review=waiting,
                                   auto_logged=len(findings) - waiting,
                                   threshold=config.REVIEW_THRESHOLD)],
    }


def recommend_node(state: GovernanceContext) -> dict:
    recommendations = core_recommend.build(state.get("findings", []))
    return {
        "recommendations": recommendations,
        "audit_log": [report.audit(state, "recommendation_agent",
                                   "drafted_recommendations",
                                   count=len(recommendations))],
    }


def narrative_node(state: GovernanceContext) -> dict:
    """
    Optional, additive, never load-bearing.

    Every call inside returns None when no model is reachable, and every caller
    treats that as "no narrative". The report is already complete before this
    node runs.
    """
    if not state.get("llm_enabled"):
        return {}

    from governance.narrative import describe, explain, summarize
    from governance.narrative import recommend as narrate_recommend
    from governance.narrative.client import Client

    client = Client(backend=state.get("llm_backend", "auto"))
    if not client.available:
        if client.backend in ("groq", "grok"):
            key_env = {"groq": "GROQ_API_KEY", "xai": "XAI_API_KEY",
                      "grok": "XAI_API_KEY"}.get(client.backend, "API key")
            reason = (f"no {key_env} found (checked Streamlit secrets and "
                     "the environment) - deterministic report is unaffected")
        else:
            reason = (f"backend '{client.backend}': local Ollama server not "
                     "reachable at http://127.0.0.1:11434 - deterministic "
                     "report is unaffected")
        return {"audit_log": [report.audit(
            state, "narrative_layer", "skipped",
            backend=client.backend, reason=reason)]}

    findings = state.get("findings", [])
    # Descriptions are generated with the findings in hand so the catalog
    # cannot contradict the findings displayed beside it.
    catalog = describe.run(state["catalog"], client, findings)
    quality_report = explain.run(state["quality_report"],
                                 state["dataset_name"], client)
    recommendations = narrate_recommend.run(
        state.get("recommendations", []), findings, client)

    updated = dict(state)
    updated["catalog"] = catalog
    updated["quality_report"] = quality_report
    summary = summarize.run(updated, client)
    client.save()

    return {
        "catalog": catalog,
        "quality_report": quality_report,
        "recommendations": recommendations,
        "executive_summary": summary,
        "audit_log": [report.audit(state, "narrative_layer", "generated_prose",
                                   **client.stats())],
    }


# The order the sequential runner uses. The graph reproduces it, with
# quality and compliance fanned out in parallel.
SEQUENCE = [
    metadata_node,
    quality_node,
    compliance_node,
    join_node,
    risk_node,
    cite_node,
    review_gate,
    recommend_node,
    narrative_node,
]