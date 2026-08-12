"""
The assistant and the review queue - the two pieces of the dashboard that hold
real logic, tested without starting Streamlit.
"""
from __future__ import annotations

import pytest

from governance import assistant, review


@pytest.fixture(scope="module")
def report():
    """
    Build the report here rather than reading out/governance_report.json.

    Reading it made these tests depend on whichever dataset was last run: they
    passed against the synthetic report and failed against the real one, which
    has no email column and no Critical findings. A test that depends on
    ambient state is not a test.
    """
    from governance import report as reporting
    from governance import run

    df, name, _ = run.load("synthetic")
    return reporting.to_dict(run.run(df, name, llm_enabled=False))


# ------------------------------------------------------------------ assistant
def test_question_words_reach_compound_field_names(report):
    """
    "duplicates" has to find a finding whose type is `duplicate_record`. That
    needs the underscore treated as a separator and the plural stemmed - it
    failed on both counts before.
    """
    answer = assistant.ask("tell me about duplicates", report)
    assert answer.grounded
    matched = [f for f in report["findings"] if f["id"] in answer.finding_ids]
    assert any(f["issue_type"] == "duplicate_record" for f in matched)


def test_column_names_are_reachable_by_their_parts(report):
    answer = assistant.ask("show me the email column", report)
    assert answer.grounded
    assert any("email" in f["column"]
               for f in report["findings"] if f["id"] in answer.finding_ids)


def test_unanswerable_questions_say_so(report):
    """
    The assistant must not reach outside the report. An unrelated question gets
    a plain refusal, not a guess.
    """
    answer = assistant.ask("what is the weather in paris", report)
    assert not answer.grounded
    assert answer.finding_ids == []
    assert "only answer from findings" in answer.text


def test_every_grounded_answer_names_its_findings(report):
    answer = assistant.ask("which columns hold personal data", report)
    assert answer.grounded and answer.finding_ids
    known = {f["id"] for f in report["findings"]}
    # Ids come from retrieval, not from the model, so they always exist.
    assert set(answer.finding_ids) <= known


def test_answers_without_a_model_still_carry_the_findings(report):
    answer = assistant.ask("what is critical", report, client=None)
    assert answer.grounded and not answer.used_model
    for finding_id in answer.finding_ids:
        assert finding_id in answer.text


# --------------------------------------------------------------------- review
def test_decision_round_trip(report, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DECISIONS_PATH", tmp_path / "decisions.json")
    monkeypatch.setattr(review, "AUDIT_PATH", tmp_path / "audit.jsonl")

    pending = [f for f in report["findings"] if f["status"] == "pending_review"]
    assert pending, "no findings are awaiting review"
    finding_id = pending[0]["id"]

    review.record(finding_id, review.APPROVED, "A. Reviewer", "scheduled")
    decorated = {f["id"]: f for f in review.decorate(report["findings"])}
    assert decorated[finding_id]["effective_status"] == review.APPROVED
    assert decorated[finding_id]["decision"]["actor"] == "A. Reviewer"

    review.clear(finding_id, "A. Reviewer")
    decorated = {f["id"]: f for f in review.decorate(report["findings"])}
    assert decorated[finding_id]["effective_status"] == "pending_review"


def test_a_decision_requires_a_named_actor(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DECISIONS_PATH", tmp_path / "decisions.json")
    monkeypatch.setattr(review, "AUDIT_PATH", tmp_path / "audit.jsonl")
    # Accountability is the entire point of the gate: an anonymous approval
    # records nothing useful.
    with pytest.raises(ValueError, match="named actor"):
        review.record("abc123", review.APPROVED, "   ")


def test_decisions_are_appended_to_the_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DECISIONS_PATH", tmp_path / "decisions.json")
    monkeypatch.setattr(review, "AUDIT_PATH", tmp_path / "audit.jsonl")

    review.record("abc123", review.APPROVED, "A. Reviewer", "ok")
    review.record("def456", review.REJECTED, "B. Reviewer", "false positive")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    import json
    entries = [json.loads(line) for line in lines]
    assert entries[0]["action"] == "review_approved"
    assert entries[1]["action"] == "review_rejected"
    assert entries[1]["actor"] == "B. Reviewer"


def test_invalid_decision_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "DECISIONS_PATH", tmp_path / "decisions.json")
    with pytest.raises(ValueError):
        review.record("abc123", "maybe", "A. Reviewer")
