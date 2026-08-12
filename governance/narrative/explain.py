"""
Business narratives for failing quality dimensions.

A score of 61 tells an engineer something. It tells a commercial stakeholder
nothing. This turns the former into the latter.

Only FAILING dimensions are explained - there is no value in generating prose
about a check that passed, and every skipped call is several seconds saved on a
laptop.
"""
from __future__ import annotations

from dataclasses import replace

from governance import config
from governance.narrative.client import Client, build_prompt
from governance.state import QualityReport

ROLE = ("You are a data governance analyst explaining a quality failure to a "
        "commercial stakeholder who does not work with data day to day.")

MEANING = {
    "completeness": "how much of the expected data is actually present",
    "uniqueness":   "whether the same real-world record appears more than once",
    "validity":     "whether values are in the format the field expects",
    "consistency":  "whether the same thing is always written the same way",
}


def run(report: QualityReport, dataset: str, client: Client) -> QualityReport:
    dimensions = []
    for dimension in report.dimensions:
        if not dimension.assessed or dimension.passed:
            dimensions.append(dimension)
            continue

        prompt = build_prompt(
            ROLE,
            {
                "dataset": dataset,
                "dimension": dimension.dimension,
                "dimension_measures": MEANING.get(dimension.dimension, ""),
                "score_out_of_100": dimension.score,
                "threshold_required": dimension.threshold,
                "columns_at_fault": list(dimension.failing_columns) or "none identified",
            },
            "In two or three sentences, explain what has gone wrong and what it "
            "means commercially for the business. Do not suggest a fix.",
            forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))

        text = client.generate(prompt)
        dimensions.append(replace(dimension, narrative=text) if text else dimension)

    return replace(report, dimensions=tuple(dimensions))
