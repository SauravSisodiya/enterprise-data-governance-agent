"""
The language-model layer.

Everything here produces PROSE. Nothing here produces a number that any other
part of the system reads back. Model output is written to dedicated `narrative`
and `description` fields and is never parsed.

Three properties this package guarantees:

  Cache-first     every response is written to disk on first generation and
                  read from disk thereafter. A demo never waits on a model.

  Fail-soft       when no model is reachable, every function returns None and
                  the pipeline continues. The governance report is complete
                  without this package doing anything at all.

  Fact-bounded    prompts state the computed findings as settled facts, and
                  responses containing a number that was not in those facts are
                  rejected and regenerated.
"""
from governance.narrative.client import Client, Prompt, build_prompt

__all__ = ["Client", "Prompt", "build_prompt"]
