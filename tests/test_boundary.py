"""
The slide-5 claim, enforced.

"Rules decide, the LLM explains" is only credible if the separation is
structural. This test asserts that nothing in governance/core/ can reach a
language model, an HTTP client, or the narrative layer - so the boundary holds
by construction rather than by anyone remembering to respect it.
"""
from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "governance" / "core"

BANNED = ("ollama", "openai", "anthropic", "transformers", "sentence_transformers",
          "fastembed", "httpx", "requests", "urllib", "socket",
          "governance.narrative")


def imported_names(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_core_cannot_reach_a_model():
    offences = []
    for path in sorted(CORE.rglob("*.py")):
        for name in imported_names(path):
            for banned in BANNED:
                if name == banned or name.startswith(banned + "."):
                    offences.append(f"{path.name} imports {name}")
    assert not offences, (
        "the deterministic core must not depend on a model or the network:\n  "
        + "\n  ".join(offences))


def test_core_modules_exist():
    expected = {"profiling.py", "types.py", "quality.py",
                "compliance_rules.py", "masking.py", "risk.py"}
    found = {p.name for p in CORE.glob("*.py")}
    assert expected <= found, f"missing from core: {expected - found}"
