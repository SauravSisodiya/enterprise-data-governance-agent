"""
Verify the model backend before you rely on it.

    python -m governance.narrative.check
    python -m governance.narrative.check --backend ollama

Reports which transport `auto` resolves to, lists the models the account can
actually reach, makes one real call, and checks the guardrails fire. Every
failure prints the thing to do about it.

This exists because a model backend fails in ways that are silent from inside
the pipeline: the narrative layer is designed to degrade quietly, so a bad key
or a retired model id looks exactly like "no model configured". That is correct
behaviour for a run and useless for setting one up.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from governance.narrative.client import (GROQ_URL, OLLAMA_URL, Client,
                                         build_prompt)

MODELS_URL = "https://api.groq.com/openai/v1/models"


def _tick(ok: bool) -> str:
    return "  OK  " if ok else "  --  "


def list_groq_models(api_key: str) -> list[str]:
    request = urllib.request.Request(
        MODELS_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return sorted(m["id"] for m in body.get("data", []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the model backend.")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "groq", "ollama", "off", "echo"])
    ap.add_argument("--model", help="override the model id")
    args = ap.parse_args()

    client = Client(backend=args.backend, model=args.model)
    print(f"\n  requested backend : {client.requested}")
    print(f"  resolves to       : {client.transport}")
    print(f"  model             : {client.model}")

    if client.transport == "off":
        print("\n  Nothing is configured. The pipeline will still produce a "
              "complete, scored,\n  cited report - only the prose is skipped.\n"
              "\n  To use Groq:   setx GROQ_API_KEY \"your-key\"   (then reopen "
              "the terminal)\n  To use local:  ollama serve\n")
        return

    # ---------------------------------------------------------------- groq
    if client.transport == "groq":
        print(f"\n{_tick(bool(client.api_key))}API key present "
              f"({len(client.api_key)} chars)")
        try:
            models = list_groq_models(client.api_key)
        except urllib.error.HTTPError as exc:
            print(f"{_tick(False)}Could not list models: HTTP {exc.code}")
            if exc.code in (401, 403):
                print("\n      The key was rejected. Check it at "
                      "https://console.groq.com/keys\n")
            return
        except Exception as exc:
            print(f"{_tick(False)}Could not reach Groq: {type(exc).__name__}: {exc}")
            return

        print(f"{_tick(True)}Reached the API - {len(models)} models available")
        known = client.model in models
        print(f"{_tick(known)}Model '{client.model}' "
              f"{'is available' if known else 'is NOT in the list'}")
        if not known:
            print("\n      Available ids:")
            for m in models[:15]:
                print(f"        {m}")
            print(f"\n      Set one with:  setx GROQ_MODEL \"{models[0]}\"\n")
            return

    # -------------------------------------------------------------- ollama
    if client.transport == "ollama":
        print(f"\n{_tick(client.available)}Local server at {OLLAMA_URL}")
        if not client.available:
            print("\n      Start it with:  ollama serve\n")
            return

    # ------------------------------------------------------- one real call
    prompt = build_prompt(
        "You are a data governance analyst.",
        {"dimension": "completeness", "score_out_of_100": 92.5,
         "threshold_required": 95.0},
        "In one sentence, say what this means for the business.")

    # Bypass the cache so this is a genuine round trip.
    client._cache = {}
    started = time.perf_counter()
    text = client.generate(prompt)
    elapsed = time.perf_counter() - started

    print(f"\n{_tick(bool(text))}Live call: {elapsed:.2f}s"
          + (f", {len(text.split())} words" if text else ""))
    if not text:
        print(f"      rejections: {client.rejections}   "
              f"rate-limited: {client.rate_limited}")
        if client.last_error:
            print(f"      last error: {client.last_error}")
        print("\n      A response was refused or never arrived. The pipeline "
              "would run\n      without prose rather than fail.\n")
        return

    print(f"\n      {text[:220]}")

    # ------------------------------------------------------- the guardrails
    from governance.narrative.client import forbidden_terms, invented_numbers
    from governance import config

    bad_number = invented_numbers("Completeness sat at 47 percent.", {92.5})
    bad_term = forbidden_terms("This is a clear violation.",
                               tuple(config.FORBIDDEN_NARRATIVE_TERMS))
    print(f"\n{_tick(bool(bad_number))}Invented-number guard would reject "
          f"{sorted(bad_number)}")
    print(f"{_tick(bool(bad_term))}Forbidden-term guard would reject {bad_term}")

    estimate = elapsed * 30
    print(f"\n  A full run makes about 30 calls -> roughly {estimate:.0f}s cold, "
          f"~3.5s cached.\n")


if __name__ == "__main__":
    main()
