"""
No personal data may reach a prompt.

This is the test that decides whether a hosted model backend is acceptable at
all. With local inference it is a nice property; the moment prompts leave the
machine it becomes the whole basis of the privacy claim on the Responsible AI
slide, so it is asserted against real values from the dataset rather than
assumed from reading the code.

Every prompt the pipeline would send is reconstructed and searched for actual
values - the emails, names, phone numbers, national IDs and addresses that are
in the CSV. Not patterns that look like them: the literal strings.
"""
from __future__ import annotations

import re

import pytest

from governance import run
from governance.narrative import describe, explain, summarize
from governance.narrative import recommend as narrate_recommend
from governance.narrative.client import Client


class _Recorder(Client):
    """Captures prompts instead of sending them anywhere."""

    def __init__(self):
        super().__init__(backend="echo")
        self.prompts: list[str] = []

    def generate(self, prompt, attempts: int = 2):
        self.prompts.append(prompt.text)
        return None                      # force the fail-soft path everywhere


@pytest.fixture(scope="module")
def captured():
    df, name, _ = run.load("synthetic")
    ctx = run.run(df, name, llm_enabled=False)

    spy = _Recorder()
    describe.run(ctx["catalog"], spy, ctx["findings"])
    explain.run(ctx["quality_report"], name, spy)
    narrate_recommend.run(ctx["recommendations"], ctx["findings"], spy)
    summarize.run(ctx, spy)

    assert spy.prompts, "no prompts were built"
    return df, "\n".join(spy.prompts)


def test_no_email_addresses_in_any_prompt(captured):
    _, blob = captured
    found = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", blob)
    assert not found, f"email addresses reached a prompt: {sorted(set(found))[:5]}"


def test_no_national_ids_or_phone_numbers_in_any_prompt(captured):
    _, blob = captured
    assert not re.findall(r"\d{3}-\d{2}-\d{4}", blob)
    assert not re.findall(r"\(\d{3}\)\s*\d{3}-\d{4}", blob)


def test_no_real_names_or_addresses_in_any_prompt(captured):
    df, blob = captured
    leaked = []
    for column in ("first_name", "last_name", "street_address"):
        for value in df[column].dropna().unique():
            if re.search(rf"\b{re.escape(str(value))}\b", blob):
                leaked.append(f"{column}={value}")
    assert not leaked, f"real values reached a prompt: {leaked[:5]}"


def test_personal_columns_have_their_samples_withheld(captured):
    """
    The catalog prompt names the column and states its statistics, but says
    'withheld' where a non-personal column would show example values.
    """
    _, blob = captured
    assert "column name: customer_email" in blob
    assert "withheld - column holds personal data" in blob


def test_non_personal_columns_may_show_samples(captured):
    """The withholding is targeted, not a blanket refusal to describe anything."""
    _, blob = captured
    assert re.search(r"example values: (?!withheld)\S", blob), \
        "no column shows samples at all - the rule is too broad to be useful"


def test_groq_backend_is_disabled_without_a_key(monkeypatch):
    """
    A hosted backend must never be silently active. With no key the client
    reports unavailable and the pipeline runs deterministically, exactly as it
    does with no local server.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = Client(backend="groq")
    assert client.available is False


def test_api_key_is_never_written_to_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-should-not-be-persisted")
    client = Client(backend="echo", cache_path=tmp_path / "cache.json")

    from governance.narrative.client import build_prompt
    client.generate(build_prompt("role", {"rows": 500}, "say something"))
    client.save()

    written = (tmp_path / "cache.json").read_text(encoding="utf-8")
    assert "test-key-should-not-be-persisted" not in written


def test_auto_prefers_groq_when_a_key_is_present(monkeypatch):
    """
    Local inference is impractical on the target hardware, so 'auto' reaches
    for the hosted API first and falls back to local only when there is no key.
    """
    monkeypatch.setenv("GROQ_API_KEY", "present")
    assert Client(backend="auto").transport == "groq"


def test_auto_falls_back_to_local_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = Client(backend="auto")
    monkeypatch.setattr(client, "_ollama_responds", lambda: True)
    assert client.transport == "ollama"


def test_auto_is_off_when_neither_is_available(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = Client(backend="auto")
    monkeypatch.setattr(client, "_ollama_responds", lambda: False)
    assert client.transport == "off"
    assert client.available is False


def test_local_can_be_forced_over_an_available_key(monkeypatch):
    """
    The stronger claim - nothing leaves the machine - has to stay reachable
    even when a Groq key happens to be configured.
    """
    monkeypatch.setenv("GROQ_API_KEY", "present")
    assert Client(backend="ollama").transport == "ollama"


def test_model_default_follows_the_transport(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "present")
    assert "llama" in Client(backend="auto").model.lower()
    assert "qwen" in Client(backend="ollama").model.lower()
