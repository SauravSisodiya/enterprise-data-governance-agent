"""
Column descriptions and glossary terms.

Turns a profiled column into a sentence a business user can read. This is the
difference between a JSON dump of statistics and something worth calling a data
catalog.

Personal data never reaches the prompt. The model is given the column name, its
inferred type and its statistics - never the values themselves.
"""
from __future__ import annotations

from dataclasses import replace

from governance.narrative.client import Client, build_prompt
from governance.state import CatalogEntry

ROLE = ("You are a data steward writing one-line descriptions for an "
        "enterprise data catalog.")


def _facts(entry: CatalogEntry, findings: list) -> dict:
    p = entry.profile
    facts = {
        "column_name": entry.name,
        "inferred_type": entry.semantic_type,
        "storage_type": p.dtype,
        "rows": p.rows,
        "empty_percent": p.null_pct,
        "distinct_percent": p.distinct_pct,
    }
    # Sample values are withheld for anything holding personal data. A catalog
    # description is not worth leaking an email address into a prompt for.
    if entry.data_class == "non_personal":
        facts["example_values"] = list(p.samples[:3])
    else:
        facts["example_values"] = "withheld - column holds personal data"

    # What the Compliance agent concluded about this column.
    #
    # Without this, the catalog can contradict the findings sitting next to it:
    # a column called cust_ref profiles as ordinary reference codes, so the
    # model confidently calls it "unique customer references" while a Critical
    # finding reports national IDs inside it. One shared context is supposed to
    # prevent exactly that, so the conclusion has to reach this prompt.
    relevant = [f for f in findings if f.column == entry.name
                and f.source == "compliance"]
    if relevant:
        facts["compliance_findings"] = [
            f"{f.issue_type.replace('_', ' ')} ({f.description})"
            for f in relevant]
    return facts


def run(catalog: list[CatalogEntry], client: Client,
        findings: list | None = None) -> list[CatalogEntry]:
    findings = findings or []
    out: list[CatalogEntry] = []
    for entry in catalog:
        prompt = build_prompt(
            ROLE, _facts(entry, findings),
            "In one sentence, say what this column most likely holds and why a "
            "business user would care about it. If compliance findings are "
            "listed, the description must be consistent with them and must not "
            "claim the column is harmless. Then, on a new line beginning "
            "'TERM:', give a short business-glossary term for it.")
        text = client.generate(prompt)

        if not text:
            out.append(entry)
            continue

        description, term = text, None
        if "TERM:" in text:
            description, _, term = text.partition("TERM:")
            term = term.strip().strip(".") or None
        out.append(replace(entry, description=description.strip(),
                           glossary_term=term))
    return out
