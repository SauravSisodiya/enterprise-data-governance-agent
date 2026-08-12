"""
Policy chunking and retrieval.

The property that matters most here is the last one: an index that has silently
fallen back to a weaker method must SAY so. A governance tool that degrades
quietly and then reports it did not is worse than one that fails outright.
"""
from __future__ import annotations

import numpy as np
import pytest

from governance.policy.chunk import from_markdown, load_corpus
from governance.policy.retrieve import PolicyIndex, query_for
from governance.state import PolicyChunk

SAMPLE = """<!-- PLACEHOLDER header -->

## GDPR Art. 4(1) | Definition of personal data

Personal data means any information relating to an identified or identifiable
natural person.

## GDPR Art. 32 | Security of processing

The controller must implement appropriate technical measures including
pseudonymisation and encryption of personal data.
"""


def test_markdown_chunks_carry_their_reference(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE, encoding="utf-8")

    chunks = from_markdown(path)
    assert len(chunks) == 2
    assert chunks[0].reference == "GDPR Art. 4(1)"
    assert chunks[0].title == "Definition of personal data"
    assert "identifiable" in chunks[0].text
    # The reference travels with the text from load time. That is what makes a
    # citation a lookup rather than something a model could invent.
    assert all(c.reference for c in chunks)


def test_placeholder_text_is_flagged(tmp_path):
    path = tmp_path / "placeholder.md"
    path.write_text(SAMPLE, encoding="utf-8")
    assert all(c.is_placeholder for c in from_markdown(path))

    clean = tmp_path / "official.md"
    clean.write_text(SAMPLE.replace("PLACEHOLDER header", "official text"),
                     encoding="utf-8")
    assert not any(c.is_placeholder for c in from_markdown(clean))


def test_shipped_corpus_is_the_official_text():
    chunks = load_corpus()
    assert chunks, "no policy corpus found in policy/source/"
    assert any(c.reference.startswith("GDPR") for c in chunks)
    assert any(c.reference.startswith("CCPA") for c in chunks)

    # The corpus must be verbatim regulation, never paraphrase. If this fails,
    # placeholder text has come back - do not weaken the assertion, re-run
    # `python -m governance.policy.fetch`.
    placeholders = [c.reference for c in chunks if c.is_placeholder]
    assert not placeholders, f"paraphrased text in the corpus: {set(placeholders)}"

    # Spot-check a phrase that only appears in the real Article 4.
    art4 = " ".join(c.text for c in chunks if c.reference == "GDPR Art. 4")
    assert "identified or identifiable natural person" in art4


def test_keyword_fallback_still_ranks_sensibly():
    """No vectors at all: the index must work, just less well."""
    chunks = [
        PolicyChunk("GDPR Art. 32", "Security of processing",
                    "pseudonymisation and encryption of personal data", "x.md"),
        PolicyChunk("GDPR Art. 35", "Impact assessment",
                    "assessment of the impact of envisaged processing", "x.md"),
    ]
    index = PolicyIndex(chunks)
    assert index.backend == "keyword"

    hits = index.search("encryption of personal data", k=2)
    assert hits[0].reference == "GDPR Art. 32"
    # Art. 35 shares no tokens with the query, so it is dropped rather than
    # returned as a weak match. A citation nobody asked for is worse than none.
    assert len(hits) == 1
    assert hits[0].score > 0


def test_index_round_trips_through_disk(tmp_path):
    chunks = [PolicyChunk("GDPR Art. 32", "Security", "encryption", "x.md")]
    vectors = np.array([[0.6, 0.8]], dtype=np.float32)

    PolicyIndex(chunks, vectors).save(tmp_path)
    loaded = PolicyIndex.load(tmp_path)

    assert loaded is not None
    assert loaded.chunks == chunks
    assert np.allclose(loaded.vectors, vectors)


def test_missing_index_returns_none(tmp_path):
    assert PolicyIndex.load(tmp_path / "nothing") is None


def test_degraded_index_does_not_claim_to_use_embeddings(tmp_path, monkeypatch):
    """
    Vectors on disk are not enough - querying them needs the model too. If the
    model cannot load, the index must report 'keyword', not 'embeddings'.
    """
    chunks = [PolicyChunk("GDPR Art. 32", "Security",
                          "encryption of personal data", "x.md")]
    index = PolicyIndex(chunks, np.array([[0.6, 0.8]], dtype=np.float32))
    assert index.backend == "embeddings"          # optimistic before any query

    def explode(*_args, **_kwargs):
        raise RuntimeError("model cache is corrupt")
    monkeypatch.setattr(index, "_embed_query",
                        lambda q: (setattr(index, "degraded_reason",
                                           "RuntimeError: model cache is corrupt")
                                   or None))

    hits = index.search("encryption", k=1)
    assert hits, "must still return results via the fallback"
    assert index.backend == "keyword", "the index lied about what it did"
    assert index.degraded_reason


def test_query_uses_regulatory_vocabulary_not_database_vocabulary():
    """
    Measured: a query built from column names and issue types scores ~0.40 and
    returns the wrong clauses. The same finding phrased in the regulation's own
    language scores ~0.69 and returns GDPR Art. 25 and Art. 32.
    """
    query = query_for("unmasked_pii_column", "direct_identifier")
    assert query and "security of processing" in query
    assert "pseudonymisation" in query
    # The classification contributes its own regulatory phrasing.
    assert "identification number" in query
    # No database jargon leaks in - it is not language any regulation uses.
    assert "column" not in query and "unmasked_pii_column" not in query


def test_non_regulatory_issues_are_not_cited():
    """A malformed email is a quality defect, not a regulatory question."""
    assert query_for("invalid_email", "direct_identifier") is None
    assert query_for("inconsistent_value", "non_personal") is None
    assert query_for("unmasked_pii_column", "direct_identifier") is not None
