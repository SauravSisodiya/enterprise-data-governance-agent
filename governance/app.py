"""
The governance dashboard.

    streamlit run governance/app.py

Five tabs, one per required deliverable, plus the review queue. The dashboard is
not a sixth deliverable - it is the surface the other five are delivered through.

It reads out/governance_report.json. It does not run the pipeline unless asked,
and it never calls a language model to render a page: all prose was generated
and cached when the report was produced. That is deliberate. A dashboard that
generates text on page load is a dashboard that stalls in front of an audience.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from governance import assistant, config, review

BAND_COLOUR = {"Critical": "#C21B66", "High": "#E0502A",
               "Medium": "#E88B00", "Low": "#008F82"}

st.set_page_config(page_title="Enterprise Data Governance Agent",
                   page_icon="🛡", layout="wide")


# --------------------------------------------------------------------------
def load_report():
    path = config.OUT_DIR / "governance_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def band_chip(band: str) -> str:
    colour = BAND_COLOUR.get(band, "#5F5F70")
    return (f"<span style='background:{colour};color:#fff;padding:2px 9px;"
            f"border-radius:10px;font-size:0.78rem;font-weight:600'>{band}</span>")


def run_pipeline(dataset: str, path: str | None, use_llm: bool,
                 use_graph: bool, backend: str = "auto"):
    from governance import report as reporting
    from governance import run as runner

    df, name, _ = runner.load(dataset, path)
    if use_graph:
        from governance.graph.build import run_graph
        ctx = run_graph(df, name, llm_enabled=use_llm, backend=backend)
    else:
        ctx = runner.run(df, name, llm_enabled=use_llm, backend=backend)
    reporting.write(ctx)
    return ctx


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Enterprise Data Governance Agent")
    report = load_report()

    if report:
        st.caption(f"**{report['dataset']}** · {report['summary']['rows']:,} rows "
                   f"× {report['summary']['columns']} columns")
        st.caption(f"generated {report['generated_at']}")
        st.caption("language model: "
                   f"{'on' if report.get('llm_enabled') else 'off'}")

    st.divider()
    st.markdown("**Run the pipeline**")
    dataset = st.selectbox("dataset", ["synthetic", "online_retail"])
    path = ("data/demo/online_retail.csv" if dataset == "online_retail" else None)
    use_llm = st.checkbox("narrative layer", value=False,
                          help="Adds prose. Every number is produced without it.")
    import os
    backend = st.radio("model backend", ["auto", "groq"], horizontal=True,
                       help="auto = local Ollama, nothing leaves the machine. "
                            "groq = hosted API; no personal data is ever put in "
                            "a prompt, but prompts do leave the machine.",
                       disabled=not use_llm)
    if use_llm and backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        st.caption(":orange[GROQ_API_KEY is not set - the narrative layer will "
                   "be skipped and the report produced without it.]")
    use_graph = st.checkbox("via LangGraph", value=True,
                            help="Same results as the sequential runner.")

    if st.button("Run", type="primary", width="stretch"):
        with st.spinner("running..."):
            run_pipeline(dataset, path, use_llm, use_graph, backend)
        st.rerun()

    st.divider()
    reviewer = st.text_input("Reviewer name", value="",
                             placeholder="required to approve or reject")
    st.caption("Every decision is written to the audit log with this name.")

if report is None:
    st.title("No report yet")
    st.write("Run the pipeline from the sidebar, or:")
    st.code("python -m governance.run --dataset synthetic", language="bash")
    st.stop()

findings = review.decorate(report.get("findings", []))
summary = report["summary"]

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.markdown(f"## {report['dataset']}")
counts = review.summary(report.get("findings", []))
cols = st.columns(5)
cols[0].metric("Quality score", f"{summary.get('overall_quality_score', 0):.1f}")
cols[1].metric("Findings", summary.get("findings_total", 0))
cols[2].metric("Awaiting review", counts.get("pending_review", 0))
cols[3].metric("Approved", counts.get(review.APPROVED, 0))
cols[4].metric("Rejected", counts.get(review.REJECTED, 0))

tabs = st.tabs(["Catalog", "Quality", "Compliance", "Assistant", "Report",
                "Review queue"])

# --------------------------------------------------------------------------
# 1. Data Catalog
# --------------------------------------------------------------------------
with tabs[0]:
    st.caption("Deliverable 1 — Data Catalog")
    rows = []
    for entry in report["catalog"]:
        p = entry["profile"]
        rows.append({
            "column": p["name"],
            "type": entry["semantic_type"],
            "by": entry["type_evidence"],
            "classification": entry["data_class"],
            "empty %": p["null_pct"],
            "distinct %": p["distinct_pct"],
            "description": entry.get("description") or "—",
            "glossary term": entry.get("glossary_term") or "—",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if not report.get("llm_enabled"):
        st.info("Descriptions and glossary terms are written by the language "
                "layer. The catalog itself — every column, type, classification "
                "and statistic — is produced without it.")

# --------------------------------------------------------------------------
# 2. Data Quality
# --------------------------------------------------------------------------
with tabs[1]:
    st.caption("Deliverable 2 — Data Quality Dashboard")
    dimensions = report["quality_report"]["dimensions"]

    scored = [d for d in dimensions if d["score"] is not None]
    cols = st.columns(len(scored))
    for col, d in zip(cols, scored):
        delta = d["score"] - d["threshold"]
        col.metric(d["dimension"].title(), f"{d['score']:.1f}",
                   f"{delta:+.1f} vs threshold",
                   delta_color="normal" if delta >= 0 else "inverse")

    for d in dimensions:
        if d["score"] is None:
            st.warning(f"**{d['dimension'].title()} — NOT ASSESSED.** "
                       f"{d['not_assessed_reason'].capitalize()}. Reported as "
                       f"unmeasured rather than given an invented number.")
            continue
        if d["passed"]:
            continue
        with st.container(border=True):
            st.markdown(f"**{d['dimension'].title()} · {d['score']:.2f} / "
                        f"{d['threshold']:.0f}** — below threshold")
            if d["failing_columns"]:
                st.markdown("Columns at fault: `"
                            + "`, `".join(d["failing_columns"]) + "`")
            if d.get("narrative"):
                st.markdown(f"> {d['narrative']}")

# --------------------------------------------------------------------------
# 3. Compliance
# --------------------------------------------------------------------------
with tabs[2]:
    st.caption("Deliverable 3 — Compliance Monitoring System")
    compliance = report["compliance_report"]

    st.markdown("#### Personal data inventory")
    st.dataframe(pd.DataFrame([{
        "column": c["column"],
        "type": c["semantic_type"],
        "classification": c["data_class"],
        "detected by": c["evidence"],
        "value match %": c["match_rate"],
        "masked in outputs": "yes" if c["masked"] else "no",
    } for c in compliance["pii_columns"]]),
        width="stretch", hide_index=True)

    if compliance.get("masked_preview"):
        with st.expander("Masking preview — values as they leave this system"):
            st.dataframe(pd.DataFrame([
                {"column": k, "masked examples": ", ".join(v)}
                for k, v in compliance["masked_preview"].items()],
            ), width="stretch", hide_index=True)

    st.markdown("#### Control gaps")
    st.caption("Reported as control gaps rather than violations: lawful basis, "
               "consent and retention policy are not present in the data, so a "
               "violation cannot honestly be asserted from it.")
    for f in findings:
        if f["source"] != "compliance":
            continue
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{f['column']}** — {f['issue_type'].replace('_',' ')}"
                          f"  \n{f['description']}")
            right.markdown(band_chip(f["band"]) + f"<br><small>risk {f['risk']}"
                           f"</small>", unsafe_allow_html=True)
            for c in f.get("citations", []):
                flag = " ⚠ placeholder" if c.get("is_placeholder") else ""
                with st.expander(f"{c['reference']} — {c['title']} "
                                 f"({c['score']:.2f}){flag}"):
                    st.write(c["text"])

# --------------------------------------------------------------------------
# 4. Assistant
# --------------------------------------------------------------------------
with tabs[3]:
    st.caption("Deliverable 4 — AI Governance Assistant")
    st.markdown("Answers come only from findings in this report. The assistant "
                "has no access to the dataset or to any outside source, and "
                "every answer names the finding ids it drew on.")

    question = st.text_input("Ask about these findings",
                             placeholder="which columns hold personal data?")
    if question:
        client = None
        if report.get("llm_enabled"):
            from governance.narrative.client import Client
            client = Client()
        answer = assistant.ask(question, report, client)

        if not answer.grounded:
            st.warning(answer.text)
        else:
            st.markdown(answer.text)
            st.caption(
                ("answered by the language model · " if answer.used_model
                 else "no model reachable, showing matched findings directly · ")
                + "grounded in " + ", ".join(f"`{i}`" for i in answer.finding_ids))

    with st.expander("Try one of these"):
        for example in ["which columns hold personal data",
                        "what is critical",
                        "tell me about duplicates",
                        "what was not assessed"]:
            st.code(example, language=None)

# --------------------------------------------------------------------------
# 5. Report
# --------------------------------------------------------------------------
with tabs[4]:
    st.caption("Deliverable 5 — Enterprise Reporting Framework")

    if report.get("executive_summary"):
        st.markdown("#### Executive summary")
        st.markdown(f"> {report['executive_summary']}")
    else:
        st.info("The executive summary is written by the language layer. "
                "Enable it in the sidebar and re-run.")

    st.markdown("#### Prioritised remediation")
    for r in report.get("recommendations", [])[:20]:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{r['column']}** — {r['issue_type'].replace('_',' ')}"
                          f"  \n{r['action']}")
            right.markdown(band_chip(r["band"]) + f"<br><small>risk {r['risk']}"
                           f"</small>", unsafe_allow_html=True)
            left.caption(f"owner: {r['suggested_owner']} · effort: {r['effort']} "
                         f"· status: {r['status']}")
            if r.get("rationale"):
                left.markdown(f"> {r['rationale']}")

    def as_markdown() -> str:
        lines = [f"# Data Governance Report — {report['dataset']}", "",
                 f"Generated {report['generated_at']}", ""]
        if report.get("executive_summary"):
            lines += ["## Executive summary", "", report["executive_summary"], ""]
        lines += ["## Quality", ""]
        for d in report["quality_report"]["dimensions"]:
            if d["score"] is None:
                lines.append(f"- **{d['dimension']}** — NOT ASSESSED "
                             f"({d['not_assessed_reason']})")
            else:
                verdict = "pass" if d["passed"] else "FAIL"
                lines.append(f"- **{d['dimension']}** — {d['score']:.2f} / "
                             f"{d['threshold']:.0f} · {verdict}")
        lines += ["", "## Findings", ""]
        for f in report["findings"]:
            refs = ", ".join(c["reference"] for c in f.get("citations", []))
            lines.append(f"- `{f['id']}` **{f['column']}** — "
                         f"{f['issue_type'].replace('_',' ')} · risk {f['risk']} "
                         f"({f['band']}) · {f['status']}"
                         + (f" · {refs}" if refs else ""))
        lines += ["", "## Remediation", ""]
        for r in report.get("recommendations", []):
            lines.append(f"- **{r['column']}** — {r['action']} "
                         f"_(owner {r['suggested_owner']}, effort {r['effort']})_")
        return "\n".join(lines)

    st.download_button("Download report (Markdown)", as_markdown(),
                       file_name=f"governance_report_{report['dataset']}.md",
                       mime="text/markdown")

# --------------------------------------------------------------------------
# 6. Review queue
# --------------------------------------------------------------------------
with tabs[5]:
    st.caption("Human-in-the-loop control gate")
    st.markdown(f"Findings scoring **{config.REVIEW_THRESHOLD} or above** are "
                "held here. Nothing is applied automatically; low and medium "
                "risk findings are recorded without blocking.")

    queue = [f for f in findings if f["status"] == "pending_review"]
    if not queue:
        st.success("Nothing awaiting review.")

    for f in queue:
        decided = f.get("decision")
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{f['column']}** — {f['issue_type'].replace('_',' ')}"
                          f"  \n{f['description']}")
            left.caption(f"`{f['id']}` · {f['affected_rows']} of "
                         f"{f['total_rows']} rows · {f['data_class']}")
            right.markdown(band_chip(f["band"]) + f"<br><small>risk {f['risk']}"
                           f"</small>", unsafe_allow_html=True)

            if decided:
                verb = ("Approved" if decided["decision"] == review.APPROVED
                        else "Rejected")
                left.success(f"{verb} by **{decided['actor']}** "
                             f"at {decided['decided_at']}"
                             + (f" — {decided['note']}" if decided["note"] else ""))
                if right.button("Reopen", key=f"reopen-{f['id']}"):
                    review.clear(f["id"], reviewer or "unknown")
                    st.rerun()
                continue

            note = left.text_input("Note (optional)", key=f"note-{f['id']}",
                                   label_visibility="collapsed",
                                   placeholder="note, optional")
            approve, reject = left.columns(2)
            disabled = not reviewer.strip()
            if approve.button("Approve", key=f"ok-{f['id']}", disabled=disabled,
                              width="stretch"):
                review.record(f["id"], review.APPROVED, reviewer, note,
                              f"{f['column']} / {f['issue_type']}")
                st.rerun()
            if reject.button("Reject", key=f"no-{f['id']}", disabled=disabled,
                             width="stretch"):
                review.record(f["id"], review.REJECTED, reviewer, note,
                              f"{f['column']} / {f['issue_type']}")
                st.rerun()
            if disabled:
                left.caption("Enter a reviewer name in the sidebar to decide. "
                             "A decision without a named actor is not auditable.")

    audit_path = config.OUT_DIR / "audit_log.jsonl"
    if audit_path.exists():
        with st.expander("Audit log — append-only"):
            entries = [json.loads(line) for line in
                       audit_path.read_text(encoding="utf-8").strip().splitlines()]
            st.dataframe(pd.DataFrame([
                {"time": e["timestamp"], "actor": e["actor"],
                 "action": e["action"], "detail": json.dumps(e["detail"])[:120]}
                for e in entries[-40:]][::-1],
            ), width="stretch", hide_index=True)
