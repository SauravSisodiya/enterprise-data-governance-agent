"""
The executive summary - the Enterprise Reporting Framework deliverable.

One call, at the end, over the finished report. It sees only aggregates: scores,
counts and bands. No column values, no findings detail, no personal data.
"""
from __future__ import annotations

from governance import config
from governance.narrative.client import Client, build_prompt
from governance.state import GovernanceContext

ROLE = ("You are writing the opening paragraph of a data governance report for "
        "an executive audience.")


def run(ctx: GovernanceContext, client: Client) -> str | None:
    quality = ctx.get("quality_report")
    issues = ctx.get("issues", [])
    if quality is None:
        return None

    bands: dict[str, int] = {}
    for finding in issues:
        bands[finding.band or "Unscored"] = bands.get(finding.band or "Unscored", 0) + 1

    failing = [d.dimension for d in quality.dimensions
               if d.assessed and not d.passed]
    unassessed = [d.dimension for d in quality.dimensions if not d.assessed]

    prompt = build_prompt(
        ROLE,
        {
            "dataset": ctx.get("dataset_name"),
            "rows": ctx.get("total_rows"),
            "columns": len(ctx.get("catalog", [])),
            "overall_quality_score_out_of_100": quality.overall_score,
            "dimensions_below_threshold": failing or "none",
            "dimensions_not_assessed": unassessed or "none",
            "total_findings": len(issues),
            "findings_by_severity": ", ".join(f"{k} {v}" for k, v in bands.items()),
            "awaiting_human_approval": sum(1 for f in issues
                                           if f.status == "pending_review"),
        },
        "Write a single paragraph of four or five sentences summarising the "
        "state of this dataset for a senior stakeholder. State plainly that the "
        "unassessed dimensions were not measured and why that is a known gap "
        "rather than a result.",
        forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))

    return client.generate(prompt)
