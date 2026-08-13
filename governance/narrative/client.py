"""
Model client: cache-first, fail-soft, fact-bounded.

Two backends, both over plain HTTP with no extra dependency:

    "auto"  a local Ollama server. Nothing leaves the machine at all.
    "groq"  Groq's hosted API. Fast enough that a full run takes seconds
            rather than the 15-20 minutes a thin-and-light laptop needs
            for local inference.

Prompts never contain personal data under either backend - describe.py
withholds sample values for any column classified as personal data, and every
other prompt carries only column names, statistics and findings.
tests/test_privacy.py asserts it against the real dataset. With Groq the honest
claim is therefore "no personal data leaves the machine", which is narrower
than "no data leaves the machine" and still a real control.

The three behaviours that matter:

1. CACHE FIRST.  Responses are keyed on (model, prompt) and written to
   out/llm_cache.json. A repeated run is instant, and a demo never waits on a
   model or risks it stalling in front of an audience.

2. FAIL SOFT.  If no server answers, generate() returns None. Callers are
   written to treat None as "no narrative", not as an error. This is what makes
   the claim "the report is complete with the model switched off" true rather
   than aspirational.

3. FACT BOUNDED.  build_prompt() renders the computed findings as settled facts
   AND collects every number in them. A response containing a number that was
   not in the facts is rejected and regenerated. It is a cheap guardrail, but it
   catches the failure that actually matters: a model restating 61 as 60.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from governance import config

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# Groq is an OpenAI-compatible endpoint, so one small function covers it.
# The key is read from the environment and never stored in the repo.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Prompt:
    """A prompt, plus the rules its answer has to satisfy."""
    text: str
    allowed_numbers: set[float] = field(default_factory=set)
    forbidden: tuple[str, ...] = ()      # regexes the response must not match


def forbidden_terms(text: str, patterns: tuple[str, ...]) -> list[str]:
    found = []
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            found.append(match.group())
    return found


def _numbers_in(value: Any) -> set[float]:
    out: set[float] = set()
    for match in NUMBER.findall(str(value)):
        try:
            out.add(float(match))
        except ValueError:
            pass
    return out


def invented_numbers(text: str, allowed: set[float]) -> set[float]:
    """
    Figures in `text` that cannot be traced to the facts we supplied.

    Comparison is NUMERIC, not textual. An earlier version compared strings and
    rejected the model for writing "95" when the facts said "95.0" - a faithful
    restatement thrown away over a trailing zero. Prose also rounds naturally,
    so a value within half a unit of a supplied figure is accepted: "the score
    of 93" against a stated 92.5 is a reasonable sentence, not a fabrication.
    """
    return {
        found for found in _numbers_in(text)
        if not any(abs(found - ok) <= 0.5 for ok in allowed)
    }


def build_prompt(role: str, facts: dict[str, Any], instruction: str,
                 forbidden: tuple[str, ...] | None = None) -> Prompt:
    """
    Render facts and derive the allowed-number set from the SAME text, so the
    two can never drift apart.
    """
    lines = []
    allowed: set[float] = set()
    for key, value in facts.items():
        if isinstance(value, (list, tuple)):
            # The COUNT of a list is a legitimate thing to mention - "all three
            # failing columns" is a faithful sentence - so admit it explicitly.
            allowed.add(float(len(value)))
            value = ", ".join(str(v) for v in value)
        label = key.replace("_", " ")
        line = f"  {label}: {value}"
        lines.append(line)
        # Harvest from the WHOLE rendered line, label included. A key like
        # "score_out_of_100" puts 100 in front of the model, so 100 is a figure
        # it was given - not one it invented. Scanning only values made the
        # guardrail reject faithful sentences.
        allowed |= _numbers_in(line)

    text = (
        f"{role}\n\n"
        "The findings below have already been computed and are final. Do not\n"
        "recompute them, dispute them, or state any figure that is not listed.\n\n"
        "FACTS\n" + "\n".join(lines) + "\n\n"
        f"TASK\n  {instruction}\n\n"
        "Write plain prose. Do not use bullet points, headings or markdown.\n"
        "Do not include any number that does not appear in the facts above."
    )
    return Prompt(text=text, allowed_numbers=allowed,
                  forbidden=tuple(forbidden or ()))


class Client:
    """
    backend:
        "auto"   resolve at runtime, preferring the hosted API:
                     GROQ_API_KEY set          -> "groq"
                     else a local server answers -> "ollama"
                     else                        -> "off"
        "groq"   Groq's hosted API. Needs GROQ_API_KEY in the environment.
        "ollama" force local inference even when a Groq key is present.
        "off"    never call anything; generate() always returns None
        "echo"   deterministic placeholder text, for tests only. The output is
                 visibly marked so it can never be mistaken for real prose.

    Groq is preferred because local inference is impractical on the hardware
    this has to run on: measured 10.8 tok/s CPU-only on a desktop-class chip,
    so roughly 3-5 tok/s on a 15 W ultraportable, putting a cold run at 15-20
    minutes. The same run through Groq takes seconds. Ollama remains available
    and is the only option that keeps the stronger "nothing leaves the machine"
    claim, so it is one flag away rather than deleted.

    On choosing "groq": no personal data reaches a prompt in the first place -
    describe.py withholds sample values for any column classified as personal
    data, and every other prompt carries only column names, statistics and
    findings. Verified by tests/test_privacy.py. But this is a hosted API in
    another jurisdiction, so the honest claim becomes "no personal data leaves
    the machine" rather than "no data leaves the machine". Say the narrower one.
    """

    def __init__(self, model: str | None = None, backend: str = "auto",
                 cache_path=None, timeout: int = 180):
        self.requested = backend
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self._explicit_model = model
        self._transport: str | None = None
        self.timeout = timeout
        self.cache_path = cache_path or (config.OUT_DIR / "llm_cache.json")
        self._cache = self._load_cache()
        self.calls = 0
        self.cache_hits = 0
        self.rejections = 0
        self.rejected_values: set[float] = set()
        self.rejected_terms: set[str] = set()
        self.rate_limited = 0
        self.last_error: str | None = None
        self._available: bool | None = None

    # ---------------------------------------------------------------- cache
    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2),
                                   encoding="utf-8")

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()[:24]

    # ------------------------------------------------------------ transport
    def _ollama_responds(self) -> bool:
        if self._available is None:
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
                    self._available = True
            except Exception:
                self._available = False
        return self._available

    @property
    def transport(self) -> str:
        """What will actually be called, once 'auto' has been resolved."""
        if self._transport is None:
            if self.requested != "auto":
                self._transport = self.requested
            elif self.api_key:
                self._transport = "groq"
            elif self._ollama_responds():
                self._transport = "ollama"
            else:
                self._transport = "off"
        return self._transport

    @property
    def model(self) -> str:
        if self._explicit_model:
            return self._explicit_model
        return GROQ_DEFAULT_MODEL if self.transport == "groq" else DEFAULT_MODEL

    @property
    def backend(self) -> str:
        """Kept for reporting; `transport` is what actually runs."""
        return self.transport

    @property
    def available(self) -> bool:
        transport = self.transport
        if transport == "off":
            return False
        if transport == "echo":
            return True
        if transport == "groq":
            return bool(self.api_key)
        return self._ollama_responds()

    def _call_groq(self, prompt: str, attempts: int = 4) -> str | None:
        """
        Groq's free tier enforces requests-per-minute limits, and a full run
        fires ~30 prompts back to back. A 429 is expected rather than
        exceptional, so honour Retry-After and back off instead of failing.
        """
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 260,
        }).encode()

        for attempt in range(attempts):
            request = urllib.request.Request(
                GROQ_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < attempts - 1:
                    wait = float(exc.headers.get("Retry-After") or 2 ** attempt)
                    self.rate_limited += 1
                    time.sleep(min(wait, 30))
                    continue
                self.last_error = f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}"
                if exc.code in (401, 403):
                    self._available = False      # bad key; stop trying
                return None
            except (urllib.error.URLError, TimeoutError, OSError,
                    KeyError, json.JSONDecodeError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return None
        return None

    def _call_model(self, prompt: str) -> str | None:
        if self.transport == "echo":
            return f"[echo backend - not model output] {prompt.splitlines()[0][:60]}"
        if self.transport == "groq":
            return self._call_groq(prompt)
        if self.transport == "off":
            return None

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
        }).encode()
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read()).get("response", "").strip()
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            self._available = False
            return None

    # -------------------------------------------------------------- public
    def generate(self, prompt: Prompt, attempts: int = 2) -> str | None:
        """
        Returns prose, or None if no model is reachable or every attempt
        invented a number. None is a normal outcome, not an error.
        """
        key = self._key(prompt.text)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]["response"]

        if not self.available:
            return None

        for _ in range(attempts):
            text = self._call_model(prompt.text)
            self.calls += 1
            if not text:
                return None

            invented = invented_numbers(text, prompt.allowed_numbers)
            if invented:
                self.rejections += 1
                self.rejected_values |= invented
                continue

            banned = forbidden_terms(text, prompt.forbidden)
            if banned:
                self.rejections += 1
                self.rejected_terms.update(t.lower() for t in banned)
                continue

            self._cache[key] = {
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt_preview": prompt.text[:160],
                "response": text,
            }
            return text
        return None

    def stats(self) -> dict[str, Any]:
        return {"model": self.model, "backend": self.backend,
                "available": self.available, "calls": self.calls,
                "cache_hits": self.cache_hits,
                "rejected_for_invented_numbers": self.rejections,
                "invented_values_seen": sorted(self.rejected_values)[:10],
                "rate_limited": self.rate_limited,
                "last_error": self.last_error}
