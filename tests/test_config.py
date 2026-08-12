"""
Internal consistency of config.py.

Every tunable value in the system lives in one file, which makes it easy to
tune and equally easy to leave an entry dangling. A severity with no
remediation, a data class with no exposure weight, or a retrieval query for an
issue type that no longer exists all fail silently at runtime - the finding
simply gets a default and nobody notices.

These checks are cheap and they caught a real one: `missing_required` sat in
SEVERITY with no playbook entry and no code path that ever emitted it.
"""
from __future__ import annotations

import re

from governance import config

DATA_CLASSES = set(config.DATA_CLASS.values()) | {config.DEFAULT_DATA_CLASS}


def test_every_severity_has_a_remediation():
    missing = set(config.SEVERITY) - set(config.REMEDIATION_PLAYBOOK)
    assert not missing, f"no remediation defined for: {missing}"


def test_no_orphan_remediations():
    orphans = set(config.REMEDIATION_PLAYBOOK) - set(config.SEVERITY)
    assert not orphans, f"remediation for unknown issue types: {orphans}"


def test_every_data_class_is_scorable_and_citable():
    assert DATA_CLASSES <= set(config.EXPOSURE)
    assert DATA_CLASSES <= set(config.ARTICLE_MAP)


def test_retrieval_queries_reference_things_that_exist():
    assert set(config.RETRIEVAL_QUERY) <= set(config.SEVERITY)
    assert set(config.DATA_CLASS_QUERY) <= DATA_CLASSES


def test_pattern_subsets_reference_real_patterns():
    for name in ("NAME_ONLY_TYPES", "FREETEXT_SCAN_TYPES",
                 "HIGH_CONFIDENCE_PII_TYPES"):
        assert set(getattr(config, name)) <= set(config.PATTERNS), name


def test_name_hints_lead_somewhere():
    dangling = [h for h in config.NAME_HINTS
                if h not in config.PATTERNS and h not in config.DATA_CLASS]
    assert not dangling, f"name hints with no pattern or classification: {dangling}"


def test_every_regex_compiles():
    for name, pattern in config.PATTERNS.items():
        re.compile(pattern)                       # raises on a bad pattern
    for pattern in config.FORBIDDEN_NARRATIVE_TERMS:
        re.compile(pattern)


def test_risk_bands_are_well_formed():
    ceilings = [c for c, _ in config.RISK_BANDS]
    assert ceilings == sorted(ceilings)
    assert ceilings[-1] == 100
    assert 0 < config.REVIEW_THRESHOLD <= 100
    # Every score in range lands in exactly one band.
    assert all(config.band_for(score) for score in range(0, 101))


def test_thresholds_cover_the_measured_dimensions():
    assert {"completeness", "uniqueness", "validity",
            "consistency"} <= set(config.THRESHOLDS)
    # And the two we decline to measure are declared, not silently absent.
    assert set(config.NOT_ASSESSED) == {"accuracy", "timeliness"}


def test_dimension_weights_exist_for_every_threshold():
    assert set(config.THRESHOLDS) <= set(config.DIMENSION_WEIGHTS)
