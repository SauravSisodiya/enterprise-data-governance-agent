# Enterprise Data Governance Agent

A multi-agent system that profiles a dataset, scores its quality, classifies and
masks personal data, ranks findings by risk, and routes the serious ones to a
human before anything is acted on.

## The one design rule

**Rules decide. The language model explains.**

Everything that produces a *number* — profiling statistics, quality scores, PII
matches, risk scores — is deterministic code in `governance/core/`. Everything
that produces a *sentence* — column descriptions, quality narratives, drafted
remediation — is the language model, in `governance/narrative/`.

The separation is structural, not a convention: `core/` imports pandas and numpy
and nothing else, so it cannot call a model even by accident. Model output is
only ever stored as text and is never parsed back into a value.

Consequence: the system produces a complete, correct governance report with the
model switched off entirely.

## Layout

```
governance/
├── config.py          every tunable value in the system, in one file
├── state.py           GovernanceContext + the objects inside it
├── synthetic.py       generates the evaluation dataset and its answer key
├── core/              deterministic. no model imports, ever.
├── policy/            regulation chunking + semantic retrieval
├── narrative/         language model layer, cache-first
├── graph/             LangGraph wiring
├── evaluate.py        precision / recall against the answer key
├── run.py             pipeline entry point
└── app.py             Streamlit dashboard
data/
├── synthetic/         evaluation set — defects planted by us, so measurable
└── demo/              real dataset — messy in ways we did not anticipate
policy/source/         official GDPR + CCPA text
out/                   governance_report.json · audit_log.jsonl · llm_cache.json
```

## Two datasets, two jobs

| | Evaluation set | Demonstration set |
|---|---|---|
| Data | synthetic, 500 rows × 14 cols | real, sampled |
| Answer key | 125 labelled defects | none — manual spot-check |
| Purpose | measure precision and recall | show it works on data we did not design |

You cannot measure a system without knowing the right answer in advance, and you
cannot claim it works in practice without running it on something you did not
build. Hence both.

## Setup

```bash
pip install -r requirements.txt
```

Then pick a model backend. **Either is optional** — the pipeline produces a
complete, scored, cited report with no model at all.

**Hosted (default, and the practical choice on a thin-and-light laptop):**

```bash
setx GROQ_API_KEY "your-key-from-console.groq.com"
```

Reopen the terminal, then verify before relying on it:

```bash
python -m governance.narrative.check
```

That resolves the backend, lists the models your account can actually reach,
makes one real call, and proves the guardrails fire. A model backend fails
silently from inside the pipeline — the narrative layer degrades quietly by
design, so a bad key looks exactly like "no model configured". Correct for a
run, useless for setting one up.

**Local (keeps the stronger privacy claim):**

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M && ollama serve
```

```bash
python -m governance.run --dataset synthetic --llm --backend ollama
```

`--backend auto` (the default) resolves to Groq when `GROQ_API_KEY` is set,
falls back to a local server if one answers, and to no prose at all otherwise.

### Why hosted is the default

Measured on the same model and prompt:

| | tok/s | full cold run |
|---|---|---|
| RTX 4070 laptop GPU | 32.3 | ~2 min |
| CPU only, 8 P-cores at 55 W | 10.8 | ~6 min |
| 15 W ultraportable (estimated 3–5) | — | 15–20 min |
| Groq | — | seconds |

Every run is cached regardless, so a second run is ~3.5s on any machine.

### What leaves the machine

**No personal data reaches a prompt under either backend.** `describe.py`
withholds sample values for any column classified as personal data; every other
prompt carries only column names, statistics and findings.
`tests/test_privacy.py` asserts this against the real values in the dataset —
literal strings, not patterns that resemble them.

So the honest claim depends on the backend:

| Backend | Claim |
|---|---|
| `ollama` | Nothing leaves the machine. |
| `groq` | No **personal data** leaves the machine. Prompts do. |

The second is narrower and still a real control. Say the narrower one when
demonstrating with Groq.

## The policy corpus

`policy/source/` holds the **verbatim official text** — GDPR Articles 4, 5, 6, 9,
25, 30, 32, 35 and Recital 26, plus CCPA §§1798.100, .105, .140 and .150. 105
passages in total. Refresh it with:

```bash
python -m governance.policy.fetch
```

Nothing is summarised on the way in. Chunks carry an `is_placeholder` flag that
propagates to every citation, and a test asserts the corpus is free of it.

**Queries are phrased in the regulation's vocabulary, not the database's.** This
is the single biggest factor in retrieval quality. Measured on this corpus:

| Query style | Top result | Score |
|---|---|---|
| `"a column named customer_email; the concern is unmasked pii column"` | CCPA breach liability ✗ | 0.405 |
| `"security of processing; encryption and pseudonymisation; appropriate technical measures"` | GDPR Art. 25, Art. 32 ✓ | 0.690 |

No regulation talks about columns, so searching with column names returns
whatever happens to be nearest. The mapping from finding to regulatory phrasing
lives in `config.RETRIEVAL_QUERY`. An issue type with no entry there is not
cited at all — a malformed email address is a quality defect, not a regulatory
matter, and attaching a statute to it would pad the report rather than inform it.

## Reading the evaluation

The synthetic set currently scores 1.000 precision and recall. **That is a
correctness check, not a performance claim.** We planted the defects and we
wrote the detectors, so anything less than a near-perfect score would mean the
plumbing is broken. What it demonstrates is that every detector is wired to the
right column and reports the right rows — nothing more.

Two things make the number meaningful, and both are outstanding:

- **Decoys** in the synthetic set — values that look like defects but are not,
  so precision has something to measure against.
- **The demonstration dataset**, where the defects were not chosen by us.

## Notes

`governance/synthetic.py` writes the answer key *as it plants each defect*, then
re-derives every defect from the finished file and refuses to write anything if
the two disagree. An answer key that silently contradicts its own dataset is
worse than none at all — every metric computed against it would be wrong, and
nothing about the output would look broken.
