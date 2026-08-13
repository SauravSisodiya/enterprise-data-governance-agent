"""
Measures the system against the synthetic answer key.

    python -m governance.evaluate

Comparison is at CELL level, not column level. It is not enough to notice that
something is wrong with a column - the system has to identify the same rows the
generator planted. A detector that flags all 500 rows would "find" every defect
while being useless, and cell-level matching is what exposes that.

    true positive   a (defect, column, row) we found and the key lists
    false positive  a (defect, column, row) we found and the key does not
    false negative  a (defect, column, row) the key lists and we missed

Recall matters more than precision for personal data: a missed PII column is a
compliance exposure, while a false alarm costs a reviewer thirty seconds.

Only the synthetic dataset can be evaluated this way, because only it has a
known answer. Real data is assessed by manual spot-check instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from governance import config, run
from governance.state import Finding

# defect_type -> which agent is expected to catch it
AGENT_FOR = {
    "null_heavy_column":        "quality",
    "duplicate_record":         "quality",
    "invalid_email":            "quality",
    "inconsistent_value":       "quality",
    "unmasked_pii_column":      "compliance",
    "pii_in_freetext":          "compliance",
    "pii_in_mislabeled_column": "compliance",
}

Key = tuple[str, str, int | None]      # (defect_type, column, row or None)


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def expected_keys() -> set[Key]:
    path = config.SYNTHETIC_DIR / "ground_truth.json"
    if not path.exists():
        raise SystemExit("no answer key; run `python -m governance.synthetic` first")
    data = json.loads(path.read_text(encoding="utf-8"))

    keys: set[Key] = set()
    for entry in data["entries"]:
        if entry["scope"] == "column":
            keys.add((entry["defect_type"], entry["column"], None))
        else:
            for row in entry["rows"]:
                keys.add((entry["defect_type"], entry["column"], row))
    return keys


def observed_keys(findings: list[Finding]) -> set[Key]:
    keys: set[Key] = set()
    for f in findings:
        if f.scope == "column":
            keys.add((f.issue_type, f.column, None))
        else:
            for row in f.rows:
                keys.add((f.issue_type, f.column, row))
    return keys


def score(expected: set[Key], observed: set[Key]) -> Metrics:
    return Metrics(tp=len(expected & observed),
                   fp=len(observed - expected),
                   fn=len(expected - observed))


def by_group(expected: set[Key], observed: set[Key],
             group_of) -> dict[str, Metrics]:
    groups = {group_of(k) for k in expected | observed}
    return {
        g: score({k for k in expected if group_of(k) == g},
                 {k for k in observed if group_of(k) == g})
        for g in sorted(groups)
    }


def _row(label: str, m: Metrics) -> str:
    return (f"    {label:<28} {m.tp:>5} {m.fp:>5} {m.fn:>5}   "
            f"{m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f}")


def main() -> None:
    df, name, profile = run.load("synthetic")
    ctx = run.run(df, name, llm_enabled=False)

    expected = expected_keys()
    findings = ctx.get("findings", [])
    observed = observed_keys(findings)
    overall = score(expected, observed)

    print(f"\n  evaluation set: {name}  "
          f"({len(expected)} labelled defects, {len(findings)} findings raised)")

    header = (f"    {'':<28} {'TP':>5} {'FP':>5} {'FN':>5}   "
              f"{'prec':>6} {'recall':>6} {'F1':>6}")

    print("\n  BY DEFECT TYPE")
    print(header)
    for defect_type, m in by_group(expected, observed, lambda k: k[0]).items():
        print(_row(defect_type, m))

    print("\n  BY AGENT")
    print(header)
    for agent, m in by_group(expected, observed,
                             lambda k: AGENT_FOR.get(k[0], "unknown")).items():
        print(_row(agent, m))

    print("\n  OVERALL")
    print(header)
    print(_row("all defects", overall))

    # Personal data is called out separately: recall here is the number that
    # actually matters, because a missed PII column is a compliance exposure.
    pii = score({k for k in expected if k[0].startswith(("pii_", "unmasked_"))},
                {k for k in observed if k[0].startswith(("pii_", "unmasked_"))})
    print(_row("personal data only", pii))

    targets = [("PII recall", pii.recall, 0.95),
               ("overall precision", overall.precision, 0.80)]
    print()
    for label, actual, target in targets:
        verdict = "MET" if actual >= target else "MISSED"
        print(f"    target {label:<20} >= {target:.2f}   actual {actual:.3f}   {verdict}")

    if overall.fn:
        print(f"\n  MISSED ({overall.fn})")
        for k in sorted(expected - observed)[:15]:
            print(f"    {k[0]:<28} {k[1]:<18} row {k[2]}")
    if overall.fp:
        print(f"\n  FALSE ALARMS ({overall.fp})")
        for k in sorted(observed - expected)[:15]:
            print(f"    {k[0]:<28} {k[1]:<18} row {k[2]}")
    print()


if __name__ == "__main__":
    main()