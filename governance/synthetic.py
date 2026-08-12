"""
Generates the synthetic evaluation dataset and its answer key.

The important property of this module is that it writes the ground truth AS IT
PLANTS each defect, not afterwards. Trying to label defects after the fact means
re-deriving them with the same logic you are trying to test, which is circular
and quietly useless.

Run:
    python -m governance.synthetic

Writes:
    data/synthetic/customers.csv
    data/synthetic/ground_truth.json

Every defect is placed with a seeded RNG, so the file is byte-identical on every
machine and every run.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from governance import config

SEED = 20260812
N_UNIQUE = 460
N_DUPLICATES = 40
TOTAL_ROWS = N_UNIQUE + N_DUPLICATES        # 500

# How many of each defect to plant.
N_INVALID_EMAIL = 25
N_INCONSISTENT_STATE = 30
N_PII_IN_NOTES = 12
N_ID_IN_CUST_REF = 8

NULL_HEAVY = {                              # column -> share of rows nulled
    "phone_no": 0.35,
    "street_address": 0.32,
    "last_order_date": 0.38,
}

# Columns that hold personal data and are stored in the clear. Each is one
# column-scope defect. These are exactly the columns a correct classifier
# should flag from the name alone.
UNMASKED_PII_COLUMNS = [
    "customer_id",      # pseudonymous, but still personal data - GDPR Art. 4(5)
    "first_name",
    "last_name",
    "customer_email",
    "phone_no",
    "street_address",
    "postcode",
]


# --------------------------------------------------------------------------
# Value pools - hand-rolled rather than Faker, to keep the dependency list
# short and the output fully deterministic.
# --------------------------------------------------------------------------
FIRST_NAMES = [
    "Sarah", "Michael", "Priya", "James", "Aisha", "David", "Elena", "Omar",
    "Grace", "Daniel", "Mei", "Thomas", "Fatima", "Lucas", "Anna", "Rajesh",
    "Chloe", "Marcus", "Yuki", "Sofia", "Ethan", "Nadia", "Oliver", "Leila",
]
LAST_NAMES = [
    "Chen", "Okafor", "Nakamura", "Silva", "Patel", "Novak", "Rossi", "Haddad",
    "Kowalski", "Dubois", "Andersson", "Mbeki", "Fischer", "Tanaka", "Moreau",
    "Costa", "Ivanov", "Weber", "Larsen", "Reyes", "Khan", "Murphy",
]
CITIES = {
    "NY": ["New York", "Buffalo", "Rochester"],
    "CA": ["Los Angeles", "San Diego", "Sacramento"],
    "TX": ["Houston", "Austin", "Dallas"],
    "FL": ["Miami", "Orlando", "Tampa"],
    "IL": ["Chicago", "Springfield", "Peoria"],
    "WA": ["Seattle", "Spokane", "Tacoma"],
}
STATES = list(CITIES)
STREETS = ["Oakwood Drive", "Maple Avenue", "Cedar Lane", "Birch Street",
           "Elm Court", "Willow Road", "Pine Crescent", "Aspen Way"]

# Long-form spellings used to create representation inconsistency.
STATE_LONG_FORM = {"NY": "New York", "CA": "California", "TX": "Texas",
                   "FL": "Florida", "IL": "Illinois", "WA": "Washington"}

NOTE_TEMPLATES = [
    "Customer called about a delayed shipment, resolved on the same day.",
    "Requested a refund for the second item in the order, approved.",
    "Asked to change the delivery window to weekday mornings only.",
    "Reported a damaged package, replacement dispatched.",
    "Enquired about loyalty points balance, no further action needed.",
    "Wants marketing preferences updated to email only.",
    "Called to confirm the billing address after a failed payment.",
    "Asked whether the item is available in a larger size.",
]
# Notes that embed personal data in prose. A regular expression scanning the
# whole cell would never see these; only a scan inside the text will.
NOTE_TEMPLATES_WITH_PII = [
    "Customer asked us to follow up at {email} rather than this account.",
    "Duplicate account query - the other one is registered to {email}.",
    "Forwarded the invoice to {email} at the customer's request.",
    "Contact preference updated, best address is {email}.",
]

# Malformed addresses, one per defect row.
#
# Two constraints, both learned the hard way:
#   - Each must be UNIQUE. Reusing a fixed pool makes corrupted rows collide
#     with each other and silently manufacture duplicate records.
#   - Each must genuinely fail config.PATTERNS["email"]. A string our own rule
#     accepts is not a defect by our own definition, and labelling it as one
#     measures the system against a standard we never implemented.
#     ("grace@example..com" was in this list until the check caught it.)
INVALID_EMAIL_TEMPLATES = [
    "{n}{i}@",                  # no domain
    "@example{i}.com",          # no local part
    "{n}{i}@example",           # no top-level domain
    "{n}{i}.example.com",       # no @ at all
    "{n} {i}@example.com",      # space in the local part
    "{n}{i}@@example.com",      # doubled @
    "{n}{i}@.com",              # empty domain label
    "{n}{i}@ example.com",      # space after the @
]


# --------------------------------------------------------------------------
# Ground truth accumulator
# --------------------------------------------------------------------------
@dataclass
class GroundTruthEntry:
    defect_type: str
    scope: str                       # "cell" | "column"
    column: str
    expected_agent: str              # "quality" | "compliance"
    expected_dimension: str | None = None
    rows: list[int] = field(default_factory=list)
    note: str | None = None

    @property
    def count(self) -> int:
        return 1 if self.scope == "column" else len(self.rows)


class GroundTruth:
    """Collects the answer key as defects are planted."""

    def __init__(self) -> None:
        self.entries: list[GroundTruthEntry] = []

    def cell(self, defect_type: str, column: str, rows: list[int],
             agent: str, dimension: str | None = None,
             note: str | None = None) -> None:
        self.entries.append(GroundTruthEntry(
            defect_type=defect_type, scope="cell", column=column,
            expected_agent=agent, expected_dimension=dimension,
            rows=sorted(int(r) for r in rows), note=note))

    def column(self, defect_type: str, column: str, agent: str,
               dimension: str | None = None, note: str | None = None) -> None:
        self.entries.append(GroundTruthEntry(
            defect_type=defect_type, scope="column", column=column,
            expected_agent=agent, expected_dimension=dimension, note=note))

    @property
    def total(self) -> int:
        return sum(e.count for e in self.entries)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.defect_type] = out.get(e.defect_type, 0) + e.count
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "synthetic/customers.csv",
            "seed": SEED,
            "total_rows": TOTAL_ROWS,
            "total_defects": self.total,
            "defects_by_type": self.summary(),
            "entries": [asdict(e) | {"count": e.count} for e in self.entries],
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _clean_rows(rng: random.Random) -> list[dict[str, Any]]:
    """460 well-formed, unique customer records."""
    rows = []
    for i in range(N_UNIQUE):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        state = rng.choice(STATES)
        signup = pd.Timestamp("2022-01-01") + pd.Timedelta(days=rng.randint(0, 700))
        last_order = signup + pd.Timedelta(days=rng.randint(1, 400))
        rows.append({
            # Emails are built from the row index so two customers can never
            # collide by chance - that would corrupt the duplicate count.
            "customer_id":     f"CUST{100000 + i}",
            "first_name":      first,
            "last_name":       last,
            "customer_email":  f"{first.lower()}.{last.lower()}{i}@example.com",
            "phone_no":        f"({rng.randint(200, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            "street_address":  f"{rng.randint(1, 400)} {rng.choice(STREETS)}",
            "city":            rng.choice(CITIES[state]),
            "state":           state,
            "postcode":        f"{rng.randint(10000, 99999)}",
            "signup_date":     signup.strftime("%Y-%m-%d"),
            "last_order_date": last_order.strftime("%Y-%m-%d"),
            "lifetime_value":  round(rng.uniform(15, 4800), 2),
            "cust_ref":        f"CR-{rng.randint(10000, 99999)}",
            "notes":           rng.choice(NOTE_TEMPLATES),
        })
    return rows


def _add_duplicates(rows: list[dict], rng: random.Random) -> list[dict]:
    """Copy 40 records wholesale, the way a botched re-import would."""
    sources = rng.sample(range(len(rows)), N_DUPLICATES)
    return rows + [dict(rows[i]) for i in sources]


def _pick(rng: random.Random, candidates: list[int], n: int) -> list[int]:
    return sorted(rng.sample(candidates, n))


def generate() -> tuple[pd.DataFrame, GroundTruth]:
    rng = random.Random(SEED)
    gt = GroundTruth()

    rows = _add_duplicates(_clean_rows(rng), rng)
    rng.shuffle(rows)
    df = pd.DataFrame(rows).reset_index(drop=True)

    # ---- duplicate records -------------------------------------------------
    # Labelled after the shuffle, using the same "keep the first occurrence"
    # convention the detector uses, so the label identifies the redundant copy
    # rather than an arbitrary member of the pair.
    dup_mask = df.duplicated(subset=["customer_email"], keep="first")
    dup_rows = df.index[dup_mask].tolist()
    assert len(dup_rows) == N_DUPLICATES, f"expected {N_DUPLICATES} duplicates, got {len(dup_rows)}"
    gt.cell("duplicate_record", "customer_email", dup_rows,
            agent="quality", dimension="uniqueness",
            note="Row is a redundant copy of an earlier record with the same email.")

    # Rows involved in a duplicate pair are off-limits for email corruption -
    # changing one copy's address would dissolve the pair we just labelled.
    dup_involved = set(df.index[df.duplicated(subset=["customer_email"], keep=False)])
    safe_rows = [i for i in df.index if i not in dup_involved]

    # ---- malformed email addresses ----------------------------------------
    bad_email_rows = _pick(rng, safe_rows, N_INVALID_EMAIL)
    for r in bad_email_rows:
        template = rng.choice(INVALID_EMAIL_TEMPLATES)
        df.at[r, "customer_email"] = template.format(
            n=rng.choice(FIRST_NAMES).lower(), i=r)
    gt.cell("invalid_email", "customer_email", bad_email_rows,
            agent="quality", dimension="validity",
            note="Value does not conform to the email format.")

    # ---- mixed state encodings --------------------------------------------
    state_rows = _pick(rng, list(df.index), N_INCONSISTENT_STATE)
    for r in state_rows:
        df.at[r, "state"] = STATE_LONG_FORM[df.at[r, "state"]]
    gt.cell("inconsistent_value", "state", state_rows,
            agent="quality", dimension="consistency",
            note="Long-form spelling of a state that appears elsewhere as a code.")

    # ---- personal data buried in free text --------------------------------
    note_rows = _pick(rng, list(df.index), N_PII_IN_NOTES)
    for r in note_rows:
        template = rng.choice(NOTE_TEMPLATES_WITH_PII)
        addr = f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}@example.com"
        df.at[r, "notes"] = template.format(email=addr)
    gt.cell("pii_in_freetext", "notes", note_rows,
            agent="compliance",
            note="Email address embedded in prose; invisible to a whole-cell match.")

    # ---- national IDs in a column whose name reveals nothing ---------------
    ref_rows = _pick(rng, list(df.index), N_ID_IN_CUST_REF)
    for r in ref_rows:
        df.at[r, "cust_ref"] = f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    gt.cell("pii_in_mislabeled_column", "cust_ref", ref_rows,
            agent="compliance",
            note="National ID format in a column named cust_ref; name-based "
                 "classification cannot find these.")

    # ---- null-heavy columns ------------------------------------------------
    for column, share in NULL_HEAVY.items():
        null_rows = _pick(rng, list(df.index), int(TOTAL_ROWS * share))
        for r in null_rows:
            df.at[r, column] = None
        gt.column("null_heavy_column", column,
                  agent="quality", dimension="completeness",
                  note=f"{share:.0%} of values are missing.")

    # ---- personal data stored unmasked -------------------------------------
    for column in UNMASKED_PII_COLUMNS:
        gt.column("unmasked_pii_column", column, agent="compliance",
                  note="Column holds personal data and is stored in the clear.")

    return df, gt


def verify(df: pd.DataFrame, gt: GroundTruth) -> None:
    """
    Re-derive every planted defect straight from the written data and confirm
    the answer key agrees - not just on counts, but on exact row positions.

    This exists because an answer key that disagrees with its own dataset is
    worse than no answer key: every metric computed against it is wrong, and
    nothing about the output looks broken.
    """
    import re

    email = re.compile(config.PATTERNS["email"])
    national_id = re.compile(config.PATTERNS["national_id"])

    def rows_for(defect_type: str) -> set[int]:
        return {r for e in gt.entries if e.defect_type == defect_type
                for r in e.rows}

    observed: dict[str, set[int]] = {
        "duplicate_record": set(
            df.index[df.duplicated(subset=["customer_email"], keep="first")]),
        "invalid_email": set(
            df.index[~df.customer_email.astype(str).str.fullmatch(email)]),
        "inconsistent_value": set(
            df.index[df.state.astype(str).str.len() > 2]),
        "pii_in_freetext": set(
            df.index[df.notes.astype(str).str.contains(email)]),
        "pii_in_mislabeled_column": set(
            df.index[df.cust_ref.astype(str).str.fullmatch(national_id)]),
    }

    problems = []
    for defect_type, found in observed.items():
        claimed = rows_for(defect_type)
        if found != claimed:
            problems.append(
                f"{defect_type}: answer key lists {len(claimed)} rows, data "
                f"contains {len(found)}  "
                f"(+{len(found - claimed)} unlabelled, "
                f"-{len(claimed - found)} labelled but absent)")

    for column, share in NULL_HEAVY.items():
        actual = df[column].isna().mean()
        if abs(actual - share) > 0.005:
            problems.append(f"{column}: expected {share:.0%} null, got {actual:.1%}")

    for column in UNMASKED_PII_COLUMNS:
        if column not in df.columns:
            problems.append(f"{column}: labelled as PII but not in the dataset")

    if problems:
        raise AssertionError(
            "ground truth does not match the generated data:\n  - "
            + "\n  - ".join(problems))


def main() -> None:
    df, gt = generate()
    verify(df, gt)

    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.SYNTHETIC_DIR / "customers.csv"
    gt_path = config.SYNTHETIC_DIR / "ground_truth.json"

    df.to_csv(csv_path, index=False)
    gt_path.write_text(json.dumps(gt.to_dict(), indent=2), encoding="utf-8")

    print(f"wrote {csv_path}  ({len(df)} rows x {len(df.columns)} columns)")
    print(f"wrote {gt_path}")
    print(f"\n{gt.total} labelled defects:")
    for defect_type, n in sorted(gt.summary().items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {defect_type}")


if __name__ == "__main__":
    main()
