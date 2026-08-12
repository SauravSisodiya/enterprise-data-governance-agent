"""
The AI Governance Assistant.

A scoped question-answering surface over the findings this system produced -
not a general chatbot. Two constraints make that distinction real:

  1. Retrieval runs over governance_report.json ONLY. The assistant has no
     access to the dataset, the regulation corpus, or anything else. If a fact
     is not in the report, the assistant cannot reach it.

  2. Every answer names the finding ids it drew on, and those ids are attached
     by the retrieval step, not written by the model. An answer that cites
     nothing retrieved nothing, and says so.

With no model reachable it degrades to showing the matching findings directly.
That is a worse experience and still a truthful one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from governance import config

STOPWORDS = {"the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
             "on", "with", "that", "this", "it", "as", "be", "by", "from", "at",
             "which", "what", "why", "how", "any", "all", "me", "my", "do",
             "does", "show", "tell", "list", "there", "has", "have", "was"}
# Underscore is a separator, not part of a word. Keeping it made
# "duplicate_record" a single token that the question "duplicates" could never
# match, and "customer_email" unreachable from either "customer" or "email".
TOKEN = re.compile(r"[a-z0-9]+")


# A question asks about "duplicates"; the finding says "duplicate record".
# Exact token matching misses that, and misses masked/masking, record/records,
# missing/missed. Light suffix stripping closes the gap without pulling in a
# stemming library for four rules.
def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in TOKEN.findall(str(text).lower())
            if t not in STOPWORDS and len(t) > 1}


@dataclass
class Answer:
    text: str
    finding_ids: list[str]
    grounded: bool          # False when nothing in the report matched
    used_model: bool


def load_report(path=None) -> dict[str, Any] | None:
    path = path or (config.OUT_DIR / "governance_report.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _searchable(finding: dict) -> str:
    parts = [finding.get("column", ""), finding.get("issue_type", ""),
             finding.get("description", ""), finding.get("data_class", ""),
             finding.get("band", ""), finding.get("status", "")]
    parts += [c.get("reference", "") for c in finding.get("citations", [])]
    return " ".join(str(p) for p in parts)


def retrieve(report: dict, question: str, k: int = 4) -> list[dict]:
    """Rank findings by token overlap with the question."""
    wanted = _tokens(question)
    if not wanted:
        return []

    scored = []
    for finding in report.get("findings", []):
        have = _tokens(_searchable(finding))
        overlap = len(wanted & have)
        if overlap:
            # Ties broken by risk, so the more serious finding surfaces first.
            scored.append((overlap, finding.get("risk") or 0, finding))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [finding for _, _, finding in scored[:k]]


def _fallback_text(matches: list[dict]) -> str:
    lines = []
    for f in matches:
        lines.append(
            f"[{f['id']}] {f['column']} - {f['issue_type'].replace('_', ' ')}: "
            f"{f['description']} Risk {f.get('risk')} ({f.get('band')}), "
            f"status {f.get('status')}.")
    return "\n".join(lines)


def ask(question: str, report: dict | None = None,
        client=None) -> Answer:
    report = report if report is not None else load_report()
    if report is None:
        return Answer("No governance report has been generated yet. Run the "
                      "pipeline first.", [], grounded=False, used_model=False)

    matches = retrieve(report, question)
    if not matches:
        return Answer(
            "Nothing in this report matches that question. The assistant can "
            "only answer from findings the pipeline produced - it has no access "
            "to the dataset or to any outside source.",
            [], grounded=False, used_model=False)

    ids = [f["id"] for f in matches]

    if client is None or not getattr(client, "available", False):
        return Answer(_fallback_text(matches), ids,
                      grounded=True, used_model=False)

    from governance.narrative.client import build_prompt

    facts = {"question": question}
    for i, f in enumerate(matches, start=1):
        facts[f"finding_{i}"] = (
            f"id {f['id']}; column {f['column']}; {f['issue_type']}; "
            f"{f['description']} risk {f.get('risk')} ({f.get('band')}); "
            f"status {f.get('status')}; "
            f"references {', '.join(c['reference'] for c in f.get('citations', [])) or 'none'}")

    prompt = build_prompt(
        "You are a data governance assistant answering a question about a "
        "governance report that has already been produced.",
        facts,
        "Answer the question using only the findings listed above. Refer to "
        "each finding by its id in square brackets. If the findings do not "
        "answer the question, say so plainly rather than speculating.",
        forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))

    text = client.generate(prompt)
    if not text:
        return Answer(_fallback_text(matches), ids,
                      grounded=True, used_model=False)
    return Answer(text, ids, grounded=True, used_model=True)
