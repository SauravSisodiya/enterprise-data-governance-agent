"""
Turning regulation documents into clause-anchored chunks.

The important output is not the text - it is the REFERENCE attached to each
piece of text. A chunk that knows it came from GDPR Art. 4(1) lets the
Compliance agent cite a specific clause. A chunk that is just a paragraph lets
it gesture at "GDPR", which is worth very little in a governance report.

Two input formats:

  .md    headings of the form   ## <reference> | <title>
         Used for curated corpora, and for the placeholder text that ships with
         the repo so the pipeline is testable before the official text arrives.

  .pdf   split on article headings found in the text. Real regulation PDFs vary
         enormously in structure, so this is best-effort and prints what it
         found - check the output before trusting it.
"""
from __future__ import annotations

import re
from pathlib import Path

from governance import config
from governance.state import PolicyChunk

MD_HEADING = re.compile(r"^##\s+(?P<ref>[^|]+?)\s*\|\s*(?P<title>.+?)\s*$", re.M)

# Matches "Article 32", "Art. 5(1)(c)", "Section 1798.140", "1798.100"
PDF_HEADING = re.compile(
    r"^\s*(?P<ref>(?:Article|Art\.?|Section|Sec\.?|§)\s*\d+[\w().\-]*"
    r"|\d{4}\.\d{2,3})\s*[.\-–]?\s*(?P<title>[A-Z][^\n]{0,80})?\s*$",
    re.M)

PLACEHOLDER_MARKER = "PLACEHOLDER"


def from_markdown(path: Path) -> list[PolicyChunk]:
    raw = path.read_text(encoding="utf-8")
    is_placeholder = PLACEHOLDER_MARKER in raw[:2000]

    matches = list(MD_HEADING.finditer(raw))
    chunks: list[PolicyChunk] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[match.end():end].strip()
        if not body:
            continue
        chunks.append(PolicyChunk(
            reference=match.group("ref").strip(),
            title=match.group("title").strip(),
            text=" ".join(body.split()),
            source=path.name,
            is_placeholder=is_placeholder,
        ))
    return chunks


def from_pdf(path: Path) -> list[PolicyChunk]:
    try:
        import pdfplumber
    except ImportError:
        print(f"  ! pdfplumber not installed; skipping {path.name}")
        return []

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    raw = "\n".join(pages)

    matches = list(PDF_HEADING.finditer(raw))
    if not matches:
        print(f"  ! no article headings found in {path.name}; "
              f"the splitter needs tuning for this document")
        return []

    chunks: list[PolicyChunk] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[match.end():end].strip()
        if len(body) < 80:                 # a heading with no substance under it
            continue
        chunks.append(PolicyChunk(
            reference=" ".join(match.group("ref").split()),
            title=(match.group("title") or "").strip(),
            text=" ".join(body.split())[:2000],
            source=path.name,
        ))
    return chunks


def load_corpus(directory: Path | None = None) -> list[PolicyChunk]:
    directory = directory or (config.POLICY_DIR / "source")
    if not directory.exists():
        return []

    chunks: list[PolicyChunk] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".md":
            chunks += from_markdown(path)
        elif path.suffix.lower() == ".pdf":
            chunks += from_pdf(path)
    return chunks
