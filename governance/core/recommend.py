"""
The deterministic half of the Governance Recommendation agent.

It groups, orders, and assigns - and never recomputes a risk score. The score
arrives from risk.py already settled; this module only decides what to do about
it and in what order.

Ordering is by risk, then by how many rows are affected. Everything else comes
straight out of the playbook in config.py, so the same finding always produces
the same action and the same owner.

The language model's contribution to a recommendation is the rationale, added
later in narrative/recommend.py. Without it, every recommendation here is still
complete and actionable - it just reads more tersely.
"""
from __future__ import annotations

from governance import config
from governance.state import Finding, Recommendation


def _fallback_rationale(finding: Finding) -> str:
    return (f"{finding.band} risk ({finding.risk}/100): {finding.description} "
            f"Affects {finding.affected_rows} of {finding.total_rows} rows.")


def build(findings: list[Finding]) -> list[Recommendation]:
    ordered = sorted(findings,
                     key=lambda f: (-(f.risk or 0), -f.affected_rows, f.column))

    seen: set[tuple[str, str]] = set()
    recommendations: list[Recommendation] = []

    for finding in ordered:
        # Two findings of the same type on the same column are one problem.
        key = (finding.issue_type, finding.column)
        if key in seen:
            continue
        seen.add(key)

        action, owner, effort = config.REMEDIATION_PLAYBOOK.get(
            finding.issue_type, config.DEFAULT_REMEDIATION)

        recommendations.append(Recommendation(
            finding_id=finding.id,
            column=finding.column,
            issue_type=finding.issue_type,
            action=action,
            suggested_owner=owner,
            effort=effort,
            rationale=_fallback_rationale(finding),
            risk=finding.risk or 0,
            band=finding.band or "Unscored",
            status=finding.status,
        ))
    return recommendations
