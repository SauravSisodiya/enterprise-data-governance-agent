"""
Build the policy index.

    python -m governance.policy.build
    python -m governance.policy.build --query "customer email address"

Reads everything in policy/source/, splits it into clause-anchored chunks,
embeds them, and writes policy/index/. Run this once after dropping in the
official regulation text.
"""
from __future__ import annotations

import argparse

from governance.policy.chunk import load_corpus
from governance.policy.retrieve import PolicyIndex


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the policy retrieval index.")
    ap.add_argument("--query", help="run a test search after building")
    args = ap.parse_args()

    chunks = load_corpus()
    if not chunks:
        raise SystemExit("no policy documents found in policy/source/")

    index = PolicyIndex.build(chunks)
    directory = index.save()

    print(f"  {len(chunks)} chunks from "
          f"{len({c.source for c in chunks})} document(s)")
    print(f"  backend: {index.backend}")
    print(f"  wrote {directory}")

    if index.uses_placeholder_text:
        print("\n  WARNING: the corpus contains PLACEHOLDER text, which is "
              "paraphrased and\n           not the official regulation. "
              "Citations built from it must not be\n           used in a report "
              "or a demonstration. Replace the files in\n           policy/source/ "
              "with the official text and re-run this command.")

    print("\n  by document:")
    for source in sorted({c.source for c in chunks}):
        refs = [c.reference for c in chunks if c.source == source]
        print(f"    {source:<28} {len(refs):>3} chunks   {', '.join(refs[:4])}...")

    if args.query:
        print(f"\n  search: {args.query!r}")
        for hit in index.search(args.query, k=3):
            print(f"    {hit.score:.3f}  {hit.reference:<22} {hit.title}")


if __name__ == "__main__":
    main()
