"""
Downloads the official regulation text and writes it into policy/source/.

    python -m governance.policy.fetch

Sources:
    GDPR   gdpr-info.eu, which reproduces the official text of Regulation
           (EU) 2016/679. EU legislation is free to reuse.
    CCPA   leginfo.legislature.ca.gov, the California Legislative Information
           service. California codes are in the public domain.

The text is stored VERBATIM. It is never summarised, paraphrased or passed
through a language model on the way in - the entire value of clause citation is
that it quotes what the regulation actually says.

Long articles are split into parts of roughly 900 characters at sentence
boundaries, because retrieval works better over passages than over whole
articles. Every part keeps its exact article reference, so a citation stays
correct regardless of which part matched.
"""
from __future__ import annotations

import html
import re
import urllib.request
from pathlib import Path

from governance import config

UA = {"User-Agent": "Mozilla/5.0 (data-governance capstone; research use)"}
MAX_CHARS = 900

GDPR_ARTICLES = ["4", "5", "6", "9", "25", "30", "32", "35"]
GDPR_RECITALS = ["26"]
CCPA_SECTIONS = ["1798.100", "1798.105", "1798.140", "1798.150"]


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def _text(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<li[^>]*>", " \n- ", fragment, flags=re.I)
    fragment = re.sub(r"</(p|div|li|tr|h\d)>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment)
    lines = [" ".join(line.split()) for line in fragment.splitlines()]
    return "\n".join(line for line in lines if line)


def _split(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Break into passages at sentence boundaries, never mid-sentence."""
    sentences = re.split(r"(?<=[.;:])\s+", " ".join(text.split()))
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts or [text.strip()]


# --------------------------------------------------------------------------
def fetch_gdpr() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    for article in GDPR_ARTICLES:
        page = _get(f"https://gdpr-info.eu/art-{article}-gdpr/")
        title_tag = re.search(r"<title>(.*?)</title>", page, re.S)
        title = "Article"
        if title_tag:
            raw = html.unescape(" ".join(title_tag.group(1).split()))
            match = re.search(r"GDPR\s*[–—-]\s*(.+?)\s*-\s*General Data", raw)
            title = match.group(1) if match else raw[:60]

        body = re.search(r'<div class="entry-content">(.*?)</div>\s*</div>', page, re.S)
        if not body:
            print(f"  ! could not extract Art. {article}")
            continue
        text = _text(body.group(1))
        text = re.sub(r"Suitable Recitals.*$", "", text, flags=re.S).strip()
        for i, part in enumerate(_split(text), start=1):
            suffix = f" (part {i})" if i > 1 else ""
            out.append((f"GDPR Art. {article}", f"{title}{suffix}", part))
        print(f"  GDPR Art. {article:<4} {title[:46]}")

    for recital in GDPR_RECITALS:
        page = _get(f"https://gdpr-info.eu/recitals/no-{recital}/")
        body = re.search(r'<div class="entry-content">(.*?)</div>\s*</div>', page, re.S)
        if not body:
            continue
        text = _text(body.group(1))
        for i, part in enumerate(_split(text), start=1):
            suffix = f" (part {i})" if i > 1 else ""
            out.append((f"GDPR Recital {recital}",
                        f"Recital {recital}{suffix}", part))
        print(f"  GDPR Recital {recital}")
    return out


def fetch_ccpa() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    base = ("https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"
            "?lawCode=CIV&sectionNum=")

    for section in CCPA_SECTIONS:
        page = _get(base + section)
        body = re.search(r'<div id="codeLawSectionNoHead">(.*?)</div>\s*</div>',
                         page, re.S)
        if not body:
            print(f"  ! could not extract CCPA {section}")
            continue
        text = _text(body.group(1))
        # Drop the division / part / title breadcrumb that precedes the section.
        cut = text.find(f"{section}.")
        if cut > 0:
            text = text[cut:]
        text = re.sub(r"\(Amended by Stats.*$", "", text, flags=re.S).strip()
        for i, part in enumerate(_split(text), start=1):
            suffix = f" (part {i})" if i > 1 else ""
            out.append((f"CCPA {section}", f"Section {section}{suffix}", part))
        print(f"  CCPA {section:<10} {len(text)} chars")
    return out


def write(path: Path, header: str, chunks: list[tuple[str, str, str]]) -> None:
    lines = [f"<!--\n{header}\n-->\n"]
    for reference, title, text in chunks:
        lines.append(f"## {reference} | {title}\n\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source = config.POLICY_DIR / "source"
    source.mkdir(parents=True, exist_ok=True)

    print("fetching GDPR ...")
    gdpr = fetch_gdpr()
    write(source / "gdpr.md",
          "Official text of Regulation (EU) 2016/679 (GDPR).\n"
          "Retrieved verbatim from gdpr-info.eu. Not paraphrased.",
          gdpr)

    print("\nfetching CCPA ...")
    ccpa = fetch_ccpa()
    write(source / "ccpa.md",
          "Official text of the California Consumer Privacy Act,\n"
          "California Civil Code sections 1798.100 et seq.\n"
          "Retrieved verbatim from leginfo.legislature.ca.gov. Not paraphrased.",
          ccpa)

    removed = []
    for stale in source.glob("*_placeholder.md"):
        stale.unlink()
        removed.append(stale.name)

    print(f"\n  wrote {source/'gdpr.md'}  ({len(gdpr)} passages)")
    print(f"  wrote {source/'ccpa.md'}  ({len(ccpa)} passages)")
    if removed:
        print(f"  removed placeholder(s): {', '.join(removed)}")
    print("\n  next:  python -m governance.policy.build")


if __name__ == "__main__":
    main()
