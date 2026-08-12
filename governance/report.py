"""
Turning the GovernanceContext into files on disk.

Two artefacts:
    out/governance_report.json   the full machine-readable result
    out/audit_log.jsonl          append-only, one line per agent action

The audit log is append-only by construction - it is opened in append mode and
nothing in the codebase ever rewrites it. That is what makes "any run can be
replayed from its own log" true rather than aspirational.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any

from governance import config
from governance.state import GovernanceContext


def _plain(value: Any) -> Any:
    """Make dataclasses, tuples and numpy scalars JSON-safe."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {f.name: _plain(getattr(value, f.name))
               for f in dataclasses.fields(value)}
        # Computed properties worth carrying into the JSON.
        for extra in ("id", "affected_rows", "passed", "assessed"):
            if hasattr(type(value), extra):
                out[extra] = _plain(getattr(value, extra))
        return out
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and (value != value or value in (
            float("inf"), float("-inf"))):
        # NaN and infinity are written by json.dumps as bare NaN / Infinity
        # tokens, which are not valid JSON and break strict parsers. Null is
        # the honest representation of "no value" anyway.
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit(ctx: GovernanceContext, actor: str, action: str, **detail) -> dict:
    """Build one audit entry. Callers append it to ctx['audit_log']."""
    return {"timestamp": now(), "actor": actor, "action": action,
            "dataset": ctx.get("dataset_name"), "detail": _plain(detail)}


def to_dict(ctx: GovernanceContext) -> dict:
    findings = ctx.get("findings", [])
    quality = ctx.get("quality_report")

    by_band: dict[str, int] = {}
    for f in findings:
        by_band[f.band or "Unscored"] = by_band.get(f.band or "Unscored", 0) + 1

    return {
        "dataset": ctx.get("dataset_name"),
        "generated_at": now(),
        "llm_enabled": ctx.get("llm_enabled", False),
        "summary": {
            "rows": ctx.get("total_rows"),
            "columns": len(ctx.get("catalog", [])),
            "overall_quality_score": quality.overall_score if quality else None,
            "findings_total": len(findings),
            "findings_by_band": by_band,
            "pending_review": sum(1 for f in findings if f.status == "pending_review"),
        },
        "catalog": _plain(ctx.get("catalog", [])),
        "quality_report": _plain(quality),
        "compliance_report": _plain(ctx.get("compliance_report")),
        "findings": _plain(findings),
        "recommendations": _plain(ctx.get("recommendations", [])),
        "executive_summary": ctx.get("executive_summary"),
    }


def write(ctx: GovernanceContext) -> tuple:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUT_DIR / "governance_report.json"
    audit_path = config.OUT_DIR / "audit_log.jsonl"

    report_path.write_text(json.dumps(to_dict(ctx), indent=2), encoding="utf-8")

    with audit_path.open("a", encoding="utf-8") as fh:
        for entry in ctx.get("audit_log", []):
            fh.write(json.dumps(entry) + "\n")

    return report_path, audit_path
