"""
Regulation ingestion and clause retrieval.

This package sits OUTSIDE governance/core/ on purpose, even though what it does
is deterministic. Retrieval is a lookup, not a generation - but computing the
embeddings needs a model, and core's guarantee is that it has no model
dependency of any kind. Rather than weaken that guarantee, retrieval lives here
and is applied as an enrichment step after the core has finished.

Retrieval has exactly one job: supplying the clause text behind a finding that
has ALREADY been made. It is not how personal data is detected. Detection is
pattern matching and value inspection in core/compliance_rules.py. Keeping the
two apart is what stops the design becoming muddled - and it is why the system
still produces cited findings when no policy corpus is loaded at all, falling
back to the static article map in config.py.
"""
from governance.policy.chunk import load_corpus
from governance.policy.retrieve import PolicyIndex

__all__ = ["load_corpus", "PolicyIndex"]
