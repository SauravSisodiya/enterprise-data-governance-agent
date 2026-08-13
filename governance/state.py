"""
The shared GovernanceContext and the objects that live inside it.

This is the "single ground truth" pattern: every agent reads from one object
and writes back to it. No agent keeps private state, which is what stops two
agents reaching contradictory conclusions about the same column.

Two conventions worth knowing before reading further:

1. Anything the deterministic core produces is a FROZEN dataclass. It cannot be
   mutated after construction. To attach a risk score to a Finding you build a
   new Finding with dataclasses.replace(). That is what makes "the LLM can
   never alter a computed value" a structural guarantee rather than a promise.

2. Narrative text produced by the language model lives in its own field and is
   never parsed. No code reads a number back out of a string the model wrote.
"""
from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict

import pandas as pd


# ==========================================================================
# Catalog - written by the Metadata Discovery agent
# ==========================================================================
@dataclass(frozen=True)
class ColumnProfile:
    """Raw statistics for one column. Pure arithmetic, no interpretation."""
    name: str
    dtype: str
    rows: int
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_pct: float
    samples: tuple[Any, ...]
    is_constant: bool
    near_constant: bool
    mean_length: float | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None


@dataclass(frozen=True)
class CatalogEntry:
    """A profiled column plus what we concluded it is."""
    profile: ColumnProfile
    semantic_type: str            # "email", "free_text", "numeric", ...
    type_evidence: str            # "name" | "value" | "dtype" | "heuristic"
    data_class: str               # key into config.EXPOSURE
    is_freetext: bool

    # Written by the LLM layer. Never read back as data.
    description: str | None = None
    glossary_term: str | None = None

    @property
    def name(self) -> str:
        return self.profile.name


# ==========================================================================
# Findings - appended by every agent
# ==========================================================================
@dataclass(frozen=True)
class Finding:
    """
    One defect, at one place, detected by one method.

    `rows` carries the actual row positions so evaluation can be done at cell
    level rather than "did it notice something was wrong with this column".
    Column-scope findings (an entire column being unmasked, say) leave it empty.
    """
    source: str                   # "quality" | "compliance"
    issue_type: str               # key into config.SEVERITY
    column: str
    description: str
    data_class: str               # key into config.EXPOSURE
    total_rows: int
    rows: tuple[int, ...] = ()
    scope: str = "cell"           # "cell" | "column"
    detection: str = "rule"       # "rule" | "llm_located_rule_confirmed"
                                  # | "llm_unconfirmed"
    evidence: dict[str, Any] = field(default_factory=dict)
    articles: tuple[str, ...] = ()
    citations: tuple["Citation", ...] = ()

    # Assigned by risk.py, which returns new Findings rather than mutating.
    risk: int | None = None
    band: str | None = None
    status: str = "open"          # "open" | "pending_review" | "approved"
                                  # | "rejected" | "auto_logged"

    # Written by the LLM layer.
    narrative: str | None = None

    @property
    def affected_rows(self) -> int:
        """Column-scope findings affect every row by definition."""
        return self.total_rows if self.scope == "column" else len(self.rows)

    @property
    def id(self) -> str:
        """
        Stable across runs, so the assistant can cite a finding by id and the
        audit log stays meaningful between executions.
        """
        key = f"{self.source}|{self.issue_type}|{self.column}|{self.scope}"
        return hashlib.sha1(key.encode()).hexdigest()[:10]

    def scored(self, risk: int, band: str) -> "Finding":
        return replace(self, risk=risk, band=band)

    def with_status(self, status: str) -> "Finding":
        return replace(self, status=status)

    def with_narrative(self, text: str) -> "Finding":
        return replace(self, narrative=text)


# ==========================================================================
# Reports
# ==========================================================================
@dataclass(frozen=True)
class DimensionScore:
    """
    One data-quality dimension. `score` is None when the dimension could not be
    assessed - which is reported as NOT ASSESSED rather than given a number.
    """
    dimension: str
    score: float | None
    threshold: float
    failing_columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    not_assessed_reason: str | None = None

    # Written by the LLM layer.
    narrative: str | None = None

    @property
    def assessed(self) -> bool:
        return self.score is not None

    @property
    def passed(self) -> bool | None:
        return None if not self.assessed else self.score >= self.threshold


@dataclass(frozen=True)
class QualityReport:
    dimensions: tuple[DimensionScore, ...]
    overall_score: float | None

    def by_name(self, name: str) -> DimensionScore | None:
        return next((d for d in self.dimensions if d.dimension == name), None)


@dataclass(frozen=True)
class PIIColumn:
    column: str
    semantic_type: str
    data_class: str
    evidence: str                 # "name" | "value" | "name+value"
    match_rate: float
    masked: bool
    articles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComplianceReport:
    pii_columns: tuple[PIIColumn, ...]
    freetext_columns: tuple[str, ...]
    masked_preview: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyChunk:
    """
    One passage of regulation, carrying its own reference.

    The reference is attached HERE, at load time, not produced later by a model.
    That is what makes a citation a lookup rather than a generation - the clause
    number cannot be hallucinated because nothing generates it.
    """
    reference: str                # "GDPR Art. 4(1)"
    title: str                    # "Definition of personal data"
    text: str
    source: str                   # file it came from
    is_placeholder: bool = False  # true while paraphrased stand-in text is loaded


@dataclass(frozen=True)
class Citation:
    reference: str
    title: str
    text: str
    score: float
    is_placeholder: bool = False


@dataclass(frozen=True)
class Recommendation:
    """
    A remediation item. Assembled by rules from the playbook; the rationale is
    the only part the language model contributes. Applied by nobody - every one
    of these waits for a human decision.
    """
    finding_id: str
    column: str
    issue_type: str
    action: str
    suggested_owner: str
    effort: str
    rationale: str
    risk: int
    band: str
    status: str = "pending_review"

    @property
    def title(self) -> str:
        """A recommendation that does not name its target is not actionable."""
        return f"{self.column} - {self.issue_type.replace('_', ' ')}"


# ==========================================================================
# The shared context
# ==========================================================================
class GovernanceContext(TypedDict, total=False):
    """
    Shaped as a TypedDict so it drops straight into LangGraph later.

    The two Annotated fields carry reducers. Without them, two agents writing to
    `issues` in the same parallel step raise InvalidUpdateError - LangGraph
    refuses to guess how to merge concurrent writes to one key. `operator.add`
    tells it to concatenate.
    """
    dataset_name: str
    dataframe: pd.DataFrame
    total_rows: int

    catalog: list[CatalogEntry]
    quality_report: QualityReport | None
    compliance_report: ComplianceReport | None

    # Two collections, deliberately separate.
    #
    # `issues` is the INBOX. Quality and Compliance run in parallel and both
    # append to it, which is what the operator.add reducer is for - without it
    # LangGraph raises InvalidUpdateError rather than guess how to merge two
    # concurrent writes to one key.
    #
    # `findings` is the SETTLED RECORD, written once by risk scoring and then
    # replaced as citations and review status are applied. It cannot share a
    # key with the inbox: an append reducer would double the list every time a
    # later node touched it.
    issues: Annotated[list[Finding], operator.add]
    findings: list[Finding]
    audit_log: Annotated[list[dict], operator.add]

    recommendations: list[Recommendation]
    executive_summary: str | None

    llm_enabled: bool
    # Declared as a real field (not just assigned post-construction) because
    # LangGraph's StateGraph only tracks keys present in this schema as
    # channels - an undeclared key set directly on the dict before .invoke()
    # is silently dropped when the graph runs, even though it survives fine
    # in the sequential runner (which never goes through .invoke() at all).
    # That mismatch previously made the LangGraph path always fall back to
    # "auto" regardless of what backend was actually requested.
    llm_backend: str


def new_context(dataset_name: str, df: pd.DataFrame,
                llm_enabled: bool = False,
                llm_backend: str = "auto") -> GovernanceContext:
    return GovernanceContext(
        dataset_name=dataset_name,
        dataframe=df,
        total_rows=len(df),
        catalog=[],
        quality_report=None,
        compliance_report=None,
        issues=[],
        findings=[],
        audit_log=[],
        recommendations=[],
        executive_summary=None,
        llm_enabled=llm_enabled,
        llm_backend=llm_backend,
    )