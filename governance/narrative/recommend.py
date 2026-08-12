"""
Rationales for the recommendations the core has already assembled.

The action, the owner and the effort arrive already decided from the playbook in
config.py. This module writes only the RATIONALE - the paragraph that explains
to whoever picks up the ticket why it is worth their afternoon.

Nothing here can change what gets done, who does it, or how urgent it is.
"""
from __future__ import annotations

from dataclasses import replace

from governance import config
from governance.narrative.client import Client, build_prompt
from governance.state import Finding, Recommendation

ROLE = ("You are a data governance lead writing the justification on a "
        "remediation ticket for an engineering team.")


def run(recommendations: list[Recommendation], findings: list[Finding],
        client: Client, limit: int = 12) -> list[Recommendation]:
    """
    `limit` caps how many rationales are generated. Recommendations arrive
    ordered by risk, so the cap always spends the time budget on the findings
    that matter most. Anything past it keeps its rule-based rationale.
    """
    by_id = {f.id: f for f in findings}
    out: list[Recommendation] = []

    for index, rec in enumerate(recommendations):
        finding = by_id.get(rec.finding_id)
        if finding is None or index >= limit:
            out.append(rec)
            continue

        prompt = build_prompt(
            ROLE,
            {
                "column": finding.column,
                "problem": finding.description,
                "risk_score_out_of_100": rec.risk,
                "risk_band": rec.band,
                "rows_affected": finding.affected_rows,
                "rows_in_dataset": finding.total_rows,
                "regulation_references": list(finding.articles) or "none",
                "agreed_action": rec.action,
            },
            "In two or three sentences, explain why this matters and what the "
            "consequence of leaving it unfixed would be. Do not restate the "
            "action and do not propose a different one. Describe this as a "
            "control gap or an exposure - never as a violation, a breach, or "
            "non-compliance. Whether processing is lawful depends on consent, "
            "purpose and retention policy, none of which are visible in this "
            "dataset, so a violation cannot honestly be asserted.",
            forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))

        text = client.generate(prompt)
        out.append(replace(rec, rationale=text) if text else rec)

    return out
