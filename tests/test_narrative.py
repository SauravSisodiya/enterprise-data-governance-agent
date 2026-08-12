"""
The three guarantees the narrative layer makes.

  fail-soft      no model reachable -> every field stays None, nothing breaks
  cache-first    a repeated prompt never reaches the model twice
  fact-bounded   a response containing an invented number is rejected
"""
from __future__ import annotations

import json

from governance import config, run
from governance.narrative.client import Client, build_prompt


def test_prompt_collects_allowed_numbers():
    prompt = build_prompt("role", {"score": 61, "threshold": 80,
                                   "columns": ["a", "b"]}, "explain it")
    # Values as floats, plus the COUNT of any list - "both failing columns" is
    # a faithful sentence, so 2 has to be admissible.
    assert prompt.allowed_numbers == {61.0, 80.0, 2.0}
    assert "61" in prompt.text and "explain it" in prompt.text


def test_off_backend_leaves_the_report_untouched():
    df, name, profile = run.load("synthetic")
    plain = run.run(df, name, profile, llm_enabled=False)
    with_llm = run.run(df, name, profile, llm_enabled=True, backend="off")

    # Identical numbers, and no prose anywhere.
    assert plain["quality_report"] == with_llm["quality_report"]
    assert plain["issues"] == with_llm["issues"]
    assert all(e.description is None for e in with_llm["catalog"])
    assert with_llm["executive_summary"] is None
    assert with_llm["recommendations"], "recommendations must survive without a model"

    skipped = [e for e in with_llm["audit_log"] if e["action"] == "skipped"]
    assert skipped, "the audit log must record that the narrative layer was skipped"


def test_echo_backend_fills_prose_without_touching_numbers():
    df, name, profile = run.load("synthetic")
    plain = run.run(df, name, profile, llm_enabled=False)
    echoed = run.run(df, name, profile, llm_enabled=True, backend="echo")

    assert any(e.description for e in echoed["catalog"])
    assert echoed["executive_summary"]
    # Prose was added; not one computed value moved.
    assert [d.score for d in plain["quality_report"].dimensions] == \
           [d.score for d in echoed["quality_report"].dimensions]
    assert [f.risk for f in plain["issues"]] == [f.risk for f in echoed["issues"]]


def test_cache_prevents_a_second_call(tmp_path):
    client = Client(backend="echo", cache_path=tmp_path / "cache.json")
    prompt = build_prompt("role", {"rows": 500}, "say something")

    first = client.generate(prompt)
    assert client.calls == 1 and client.cache_hits == 0

    second = client.generate(prompt)
    assert second == first
    assert client.calls == 1, "a cached prompt must not reach the model again"
    assert client.cache_hits == 1

    client.save()
    assert json.loads((tmp_path / "cache.json").read_text())


def test_invented_numbers_are_rejected(tmp_path):
    """A response quoting a figure that was never in the facts is thrown away."""
    class Liar(Client):
        def _call_model(self, prompt: str):
            return "Completeness came in at 47 percent, which is below target."

    client = Liar(backend="echo", cache_path=tmp_path / "cache.json")
    prompt = build_prompt("role", {"score": 92, "threshold": 95}, "explain")

    assert client.generate(prompt) is None
    assert client.rejections == 2          # both attempts refused
    assert not client._cache               # nothing invented reaches the cache


def test_rounding_and_trailing_zeros_are_accepted(tmp_path):
    """
    An earlier version compared numbers as STRINGS and rejected the model for
    writing "95" when the facts said "95.0" - a faithful sentence thrown away
    over a trailing zero. Comparison is numeric, with half-a-unit tolerance for
    the rounding that prose does naturally.
    """
    class Rounder(Client):
        def _call_model(self, prompt: str):
            return "The score of 93 falls short of the required 95."

    client = Rounder(backend="echo", cache_path=tmp_path / "c.json")
    prompt = build_prompt("role", {"score_out_of_100": 92.5,
                                   "threshold": 95.0}, "explain")
    assert client.generate(prompt) is not None
    assert client.rejections == 0


def test_numbers_in_fact_labels_count_as_supplied(tmp_path):
    """A key like `score_out_of_100` puts 100 in front of the model."""
    prompt = build_prompt("role", {"score_out_of_100": 92.5}, "explain")
    assert 100.0 in prompt.allowed_numbers
    assert 92.5 in prompt.allowed_numbers


def test_forbidden_terms_are_rejected(tmp_path):
    """
    This system reports control gaps, not violations. Instructing the model is
    not enough - it will write both in one sentence - so a rule checks the
    output instead.
    """
    from governance import config

    class Overreacher(Client):
        def _call_model(self, prompt: str):
            return "This is a clear violation of the regulation."

    client = Overreacher(backend="echo", cache_path=tmp_path / "c.json")
    prompt = build_prompt("role", {"rows": 500}, "explain",
                          forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))

    assert client.generate(prompt) is None
    assert "violation" in client.rejected_terms
    assert not client._cache


def test_permitted_language_passes(tmp_path):
    from governance import config

    class Careful(Client):
        def _call_model(self, prompt: str):
            return ("This is a control gap that exposes personal data and "
                    "carries a risk of a data breach if left unaddressed.")

    client = Careful(backend="echo", cache_path=tmp_path / "c.json")
    prompt = build_prompt("role", {"rows": 500}, "explain",
                          forbidden=tuple(config.FORBIDDEN_NARRATIVE_TERMS))
    # "risk of a data breach" is a statement about consequence, not an
    # assertion that this dataset is in breach - it must survive.
    assert client.generate(prompt) is not None


def test_numbers_present_in_the_facts_are_allowed(tmp_path):
    class Honest(Client):
        def _call_model(self, prompt: str):
            return "The score of 92 sits below the required 95."

    client = Honest(backend="echo", cache_path=tmp_path / "cache.json")
    prompt = build_prompt("role", {"score": 92, "threshold": 95}, "explain")

    assert client.generate(prompt) is not None
    assert client.rejections == 0
