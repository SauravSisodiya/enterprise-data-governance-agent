"""
Pipeline entry point.

    python -m governance.run --dataset synthetic
    python -m governance.run --path data/demo/online_retail.csv --llm

Runs the node functions from governance/graph/nodes.py in order, applying each
partial update to the context as it goes. The LangGraph version in
governance/graph/build.py runs the SAME functions with quality and compliance
fanned out in parallel.

Keeping one implementation of every step is deliberate. A sequential runner that
duplicated the logic would drift from the graph the first time either was
touched, and then two things claiming to be the same pipeline would quietly
disagree.

This runner has no orchestration dependency at all, which makes it the fallback
that still works if LangGraph is unavailable, and the reference the graph has to
reproduce.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from governance import config, report
from governance.graph import nodes
from governance.state import GovernanceContext, new_context


def load(dataset: str | None = None,
         path: str | None = None) -> tuple[pd.DataFrame, str, config.DatasetProfile]:
    if path:
        csv = Path(path)
        name = dataset or csv.stem
    elif dataset == "synthetic":
        csv = config.SYNTHETIC_DIR / "customers.csv"
        name = "synthetic"
    else:
        raise SystemExit("give --dataset synthetic or --path <file.csv>")

    if not csv.exists():
        raise SystemExit(f"dataset not found: {csv}\n"
                         f"run `python -m governance.synthetic` first")

    df = pd.read_csv(csv)
    profile = config.DATASET_PROFILES.get(name, config.DEFAULT_PROFILE)
    return df, name, profile


def _apply(state: GovernanceContext, update: dict) -> None:
    """
    Merge a node's partial update, honouring the same reducer semantics
    LangGraph applies: the two accumulating keys append, everything else is
    replaced. Written out explicitly so the behaviour is visible rather than
    inherited from a framework.
    """
    for key, value in update.items():
        if key in ("issues", "audit_log"):
            state[key] = state.get(key, []) + value
        else:
            state[key] = value


def run(df: pd.DataFrame, name: str, profile: config.DatasetProfile | None = None,
        llm_enabled: bool = False, backend: str = "auto") -> GovernanceContext:
    state = new_context(name, df, llm_enabled=llm_enabled)
    state["llm_backend"] = backend
    for node in nodes.SEQUENCE:
        _apply(state, node(state))
    return state


# --------------------------------------------------------------------------
# console summary
# --------------------------------------------------------------------------
def summarise(ctx: GovernanceContext) -> None:
    q = ctx["quality_report"]
    c = ctx["compliance_report"]
    findings = ctx.get("findings", [])

    print(f"\n  dataset            {ctx['dataset_name']}  "
          f"({ctx['total_rows']} rows x {len(ctx['catalog'])} columns)")
    print(f"  language model     {'on' if ctx.get('llm_enabled') else 'off'}")

    print("\n  DATA QUALITY")
    for d in q.dimensions:
        if not d.assessed:
            print(f"    {d.dimension:<14} {'NOT ASSESSED':>8}   {d.not_assessed_reason}")
            continue
        verdict = "pass" if d.passed else "FAIL"
        failing = f"   failing: {', '.join(d.failing_columns)}" if d.failing_columns else ""
        print(f"    {d.dimension:<14} {d.score:>8.2f}   {verdict}"
              f"  (threshold {d.threshold:g}){failing}")
    print(f"    {'overall':<14} {q.overall_score:>8.2f}")

    print("\n  PERSONAL DATA")
    for p in c.pii_columns:
        print(f"    {p.column:<18} {p.semantic_type:<14} {p.data_class:<24}"
              f" by {p.evidence}")
    if c.freetext_columns:
        print(f"    free-text scanned: {', '.join(c.freetext_columns)}")

    print("\n  FINDINGS")
    bands: dict[str, int] = {}
    for f in findings:
        bands[f.band] = bands.get(f.band, 0) + 1
    for _, band in reversed(config.RISK_BANDS):
        if band in bands:
            print(f"    {band:<10} {bands[band]}")
    pending = sum(1 for f in findings if f.status == "pending_review")
    print(f"    {'total':<10} {len(findings)}   ({pending} pending review)")

    print("\n  TOP FINDINGS")
    print(f"    {'risk':>4}  {'band':<9} {'column':<18} {'rows':>5}  issue")
    for f in findings[:10]:
        print(f"    {f.risk:>4}  {f.band:<9} {f.column:<18} {f.affected_rows:>5}"
              f"  {f.issue_type}")

    recs = ctx.get("recommendations", [])
    if recs:
        print("\n  TOP RECOMMENDATIONS")
        for r in recs[:5]:
            print(f"    [{r.risk:>3} {r.band:<8}] {r.title}")
            print(f"          {r.action}")
            print(f"          owner: {r.suggested_owner}   effort: {r.effort}"
                  f"   status: {r.status}")

    if ctx.get("executive_summary"):
        print("\n  EXECUTIVE SUMMARY")
        print(f"    {ctx['executive_summary']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the governance pipeline.")
    ap.add_argument("--dataset", default="synthetic",
                    help="named dataset profile (default: synthetic)")
    ap.add_argument("--path", help="path to a CSV, for datasets without a profile")
    ap.add_argument("--llm", action="store_true",
                    help="enable the narrative layer (default: off)")
    ap.add_argument("--backend", default="auto", choices=["auto", "groq", "off", "echo"],
                    help="model backend; 'echo' emits marked placeholder text "
                         "for testing without a model")
    ap.add_argument("--graph", action="store_true",
                    help="run through LangGraph instead of sequentially")
    args = ap.parse_args()

    df, name, _ = load(args.dataset, args.path)

    if args.graph:
        from governance.graph.build import run_graph
        ctx = run_graph(df, name, llm_enabled=args.llm, backend=args.backend)
    else:
        ctx = run(df, name, llm_enabled=args.llm, backend=args.backend)

    summarise(ctx)
    report_path, audit_path = report.write(ctx)
    print(f"\n  wrote {report_path}")
    print(f"  wrote {audit_path}\n")


if __name__ == "__main__":
    main()
