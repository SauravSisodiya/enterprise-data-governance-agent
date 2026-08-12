"""
The human-in-the-loop review queue.

Decisions live in their own file rather than inside the report, for one reason:
the report is regenerated on every run, and a human decision must outlive the
run that produced the finding. Decisions are keyed on the finding id, which is
a hash of (source, issue type, column, scope) - stable across runs, so a
decision made today still attaches to the same finding tomorrow.

Every decision is appended to the audit log with a named actor and a timestamp.
The log is opened in append mode and nothing here ever rewrites it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from governance import config

DECISIONS_PATH = config.OUT_DIR / "review_decisions.json"
AUDIT_PATH = config.OUT_DIR / "audit_log.jsonl"

APPROVED = "approved"
REJECTED = "rejected"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict[str, dict[str, Any]]:
    if not DECISIONS_PATH.exists():
        return {}
    try:
        return json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(decisions: dict[str, dict[str, Any]]) -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2), encoding="utf-8")


def _audit(entry: dict) -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def record(finding_id: str, decision: str, actor: str,
           note: str = "", finding_summary: str = "") -> dict:
    """Record one approve/reject. Returns the stored decision."""
    if decision not in (APPROVED, REJECTED):
        raise ValueError(f"decision must be {APPROVED!r} or {REJECTED!r}")
    if not actor.strip():
        raise ValueError("a decision needs a named actor - accountability is "
                         "the point of the gate")

    decisions = load()
    stored = {
        "finding_id": finding_id,
        "decision": decision,
        "actor": actor.strip(),
        "note": note.strip(),
        "decided_at": _now(),
    }
    decisions[finding_id] = stored
    _save(decisions)

    _audit({"timestamp": stored["decided_at"], "actor": actor.strip(),
            "action": f"review_{decision}",
            "detail": {"finding_id": finding_id, "note": note.strip(),
                       "finding": finding_summary}})
    return stored


def clear(finding_id: str, actor: str) -> None:
    """Undo a decision. Recorded in the log like any other action."""
    decisions = load()
    if finding_id in decisions:
        removed = decisions.pop(finding_id)
        _save(decisions)
        _audit({"timestamp": _now(), "actor": actor,
                "action": "review_reopened",
                "detail": {"finding_id": finding_id,
                           "previous_decision": removed.get("decision")}})


def decorate(findings: list[dict]) -> list[dict]:
    """Attach any stored decision to each finding dict from the report."""
    decisions = load()
    out = []
    for finding in findings:
        merged = dict(finding)
        decision = decisions.get(finding.get("id"))
        merged["decision"] = decision
        merged["effective_status"] = (
            decision["decision"] if decision else finding.get("status"))
        out.append(merged)
    return out


def summary(findings: list[dict]) -> dict[str, int]:
    decorated = decorate(findings)
    counts = {"pending_review": 0, APPROVED: 0, REJECTED: 0, "auto_logged": 0}
    for finding in decorated:
        key = finding["effective_status"]
        counts[key] = counts.get(key, 0) + 1
    return counts
