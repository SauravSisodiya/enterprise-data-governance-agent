"""
The human-in-the-loop gate.

Separated from risk.py because scoring and gating answer different questions.
Scoring asks "how bad is this?"; gating asks "who is allowed to decide?". The
threshold between them is a policy choice an organisation makes, and keeping it
in its own module makes that visible.

Nothing is ever applied automatically. The gate decides only whether a finding
waits for a named human or is recorded and allowed past.
"""
from __future__ import annotations

from governance import config
from governance.state import Finding


def status_for(finding: Finding) -> str:
    # Anything the language model located but the rules could not confirm goes
    # to a person regardless of score. An unconfirmed finding is a question,
    # not a number to act on.
    if finding.detection == "llm_unconfirmed":
        return "pending_review"
    risk = finding.risk or 0
    return "pending_review" if risk >= config.REVIEW_THRESHOLD else "auto_logged"


def apply(findings: list[Finding]) -> list[Finding]:
    """Returns new findings. Nothing is mutated."""
    return [f.with_status(status_for(f)) for f in findings]


def pending(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.status == "pending_review"]
