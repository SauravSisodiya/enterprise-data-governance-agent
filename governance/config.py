"""
Every tunable value in the system lives in this file.

Nothing that affects a score, a classification, or a risk band is buried in
logic anywhere else. That is a deliberate design decision: when someone asks
"why did this column score 61?", the answer has to be a threshold and a
calculation they can read, not behaviour spread across five modules.

Sections:
    1. Missing-value handling
    2. Pattern library         (used for both validity and PII detection)
    3. Column-name taxonomy
    4. Data classification     (what kind of personal data a column holds)
    5. Data-quality thresholds and weights
    6. Consistency rules
    7. Risk scoring
    8. Regulation mapping
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------
def _load_dotenv() -> None:
    """
    Read KEY=value pairs from a .env file at the project root.

    Twelve lines instead of a dependency. A real environment variable always
    wins - setdefault, not assignment - so an explicitly exported key overrides
    the file rather than the other way round.

    The point of supporting this at all: `setx GROQ_API_KEY "gsk_..."` writes
    the key into PowerShell's history file, where it stays. A gitignored .env
    keeps it in exactly one place you control.
    """
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv()
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
DEMO_DIR = DATA_DIR / "demo"
POLICY_DIR = ROOT / "policy"
OUT_DIR = ROOT / "out"


# --------------------------------------------------------------------------
# 1. Missing-value handling
# --------------------------------------------------------------------------
# A cell holding "" or "N/A" is missing in every sense that matters, but pandas
# counts it as present. Normalising these first is what stops the completeness
# score being quietly wrong.
SENTINEL_NULLS = ["", " ", "  ", "N/A", "n/a", "NA", "NULL", "null", "None",
                  "-", "--", "unknown", "Unknown", "?"]


# --------------------------------------------------------------------------
# 2. Pattern library
# --------------------------------------------------------------------------
# One library, two uses:
#   - VALIDITY  uses re.fullmatch  -> "is this whole cell a well-formed email?"
#   - PII SCAN  uses re.search     -> "is there an email hiding in this sentence?"
# Patterns are deliberately unanchored so the same string serves both.
PATTERNS: dict[str, str] = {
    "email":       r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    "phone":       r"(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}",
    "national_id": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"(?:\d{4}[\s\-]?){3}\d{4}",
    "postcode_us": r"\d{5}(?:-\d{4})?",
    "ip_address":  r"(?:\d{1,3}\.){3}\d{1,3}",
    "url":         r"https?://\S+",
}

# Which patterns are safe to hunt for inside free text.
#
# postcode_us is excluded on purpose: "\d{5}" matches any five-digit number, so
# scanning prose with it would flag order numbers, prices and dates as personal
# data. It stays useful for validity (where the whole cell must match) but is a
# false-positive machine when used loosely.
FREETEXT_SCAN_TYPES = ["email", "phone", "national_id", "credit_card"]

# A column is treated as PII by value evidence when more than this share of its
# non-null values match a personal-data pattern.
VALUE_MATCH_THRESHOLD = 0.05

# ...except for these. A national ID or card number is specific enough that a
# single occurrence is worth reporting, so they bypass the rate threshold and
# are flagged cell by cell. This is what catches a handful of ID numbers hiding
# in a column whose name gives nothing away.
HIGH_CONFIDENCE_PII_TYPES = ["national_id", "credit_card"]

# Types whose pattern is too generic to establish from values alone. They may
# only be assigned when the COLUMN NAME also supports it.
#
# "\d{5}" is a US postcode and it is also a product code, an order number, a
# part number and a year range. Running against a real retail dataset, every
# five-digit StockCode was classified as a postcode, which made a product
# catalogue look like a table of personal data and produced 996 spurious
# "invalid format" findings. The name is the only thing that can disambiguate.
NAME_ONLY_TYPES = ["postcode_us"]

# Minimum share of values that must match a pattern before we accept it as the
# column's semantic type on value evidence alone.
TYPE_INFERENCE_THRESHOLD = 0.80

# Mean character length above which an unclassified text column is treated as
# free text (and therefore scanned for embedded personal data).
FREETEXT_MIN_MEAN_LENGTH = 25


# --------------------------------------------------------------------------
# 3. Column-name taxonomy
# --------------------------------------------------------------------------
# Matched against the column name normalised to lowercase alphanumerics, so
# "Customer ID", "customer_id" and "CustomerID" all collapse to "customerid".
#
# Keep these generic. Tuning them to the column names in our own synthetic data
# is the single easiest way to build something that finds nothing on a real
# dataset.
NAME_HINTS: dict[str, list[str]] = {
    "email":         ["email", "emailaddress", "mail"],
    "phone":         ["phone", "mobile", "telephone", "tel", "contactno",
                      "contactnumber"],
    "national_id":   ["ssn", "socialsecurity", "nationalid", "nid", "taxid",
                      "aadhaar"],
    "person_name":   ["name", "firstname", "lastname", "fname", "lname",
                      "surname", "fullname", "givenname"],
    # "city" and "town" are deliberately absent: a city on its own is weak
    # evidence, and including it flags almost every customer table.
    "address":       ["address", "addr", "street"],
    "postcode_us":   ["postcode", "postalcode", "zip", "zipcode"],
    "date_of_birth": ["dob", "dateofbirth", "birthdate", "birthday"],
    "credit_card":   ["card", "cardnumber", "creditcard", "pan"],
    "customer_id":   ["customerid", "custid", "clientid", "userid", "accountid",
                      "memberid"],
}


# --------------------------------------------------------------------------
# 4. Data classification
# --------------------------------------------------------------------------
# Maps a semantic type to how exposing it is. This drives the `exposure` term
# in the risk formula.
#
# "pseudonymous_identifier" exists because a hashed or surrogate customer key
# is still personal data under GDPR Art. 4(5) - a point that is very commonly
# assumed to be false.
DATA_CLASS: dict[str, str] = {
    "email":         "direct_identifier",
    "phone":         "direct_identifier",
    "national_id":   "direct_identifier",
    "person_name":   "direct_identifier",
    "credit_card":   "direct_identifier",
    "date_of_birth": "quasi_identifier",
    "address":       "quasi_identifier",
    "postcode_us":   "quasi_identifier",
    "ip_address":    "quasi_identifier",
    "customer_id":   "pseudonymous_identifier",
}
DEFAULT_DATA_CLASS = "non_personal"


# --------------------------------------------------------------------------
# 5. Data-quality thresholds and weights
# --------------------------------------------------------------------------
# A dimension scores 0-100. It PASSES when it meets or beats its threshold.
THRESHOLDS: dict[str, float] = {
    "completeness": 95.0,
    "uniqueness":   99.0,
    "validity":     98.0,
    "consistency":  95.0,
}

# Used for the single headline score. Only assessed dimensions are averaged -
# accuracy and timeliness never enter this calculation, they are reported as
# NOT ASSESSED. See docs: measuring accuracy needs a reference dataset, and
# timeliness needs an agreed freshness SLA. Neither exists here.
DIMENSION_WEIGHTS: dict[str, float] = {
    "completeness": 1.0,
    "uniqueness":   1.0,
    "validity":     1.0,
    "consistency":  0.5,   # rule-dependent, so weighted lower than the rest
}

NOT_ASSESSED = ["accuracy", "timeliness"]

# A column this empty is a finding in its own right, over and above its
# contribution to the completeness score.
NULL_HEAVY_COLUMN_THRESHOLD = 30.0


# --------------------------------------------------------------------------
# 5b. Per-dataset profiles
# --------------------------------------------------------------------------
# Two things cannot be inferred from the data and must be declared:
#
#   required_columns - an optional column being empty is not a defect, and
#                      treating it as one produces a frightening number that
#                      means nothing.
#   business_key     - what makes two rows "the same record". Full-row
#                      deduplication would miss a repeated customer whose
#                      last_login happens to differ.
#
# Datasets we have not seen before fall back to DEFAULT_PROFILE. Keeping this
# separate from the detection logic is what stops the system being tuned to
# column names we chose ourselves.
class DatasetProfile:
    def __init__(self, name: str, required_columns: list[str] | None = None,
                 business_key: list[str] | None = None):
        self.name = name
        self.required_columns = required_columns   # None = every column
        self.business_key = business_key           # None = every column

DEFAULT_PROFILE = DatasetProfile("default")

DATASET_PROFILES: dict[str, DatasetProfile] = {
    "synthetic": DatasetProfile(
        "synthetic",
        required_columns=None,
        business_key=["customer_email"],
    ),
    "online_retail": DatasetProfile(
        "online_retail",
        required_columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"],
        business_key=["Invoice", "StockCode"],
    ),
}


# --------------------------------------------------------------------------
# 6. Consistency rules
# --------------------------------------------------------------------------
# (a) Representation consistency: each real-world value should have exactly one
#     surface form. "NY" and "New York" in the same column is a defect.
CANONICAL: dict[str, dict[str, str]] = {
    "state": {
        "NY": "NY", "New York": "NY", "N.Y.": "NY", "new york": "NY",
        "CA": "CA", "California": "CA", "Calif.": "CA", "california": "CA",
        "TX": "TX", "Texas": "TX", "texas": "TX",
        "FL": "FL", "Florida": "FL", "florida": "FL",
        "IL": "IL", "Illinois": "IL", "illinois": "IL",
        "WA": "WA", "Washington": "WA", "washington": "WA",
    },
}

# (b) Cross-field rules, written as pandas expressions so they stay readable to
#     someone who does not write Python. Each is (name, columns, expression);
#     the expression must evaluate to True when the row is CONSISTENT.
#
#     Rules whose columns are absent from the dataset are skipped, not failed.
#     Rows where any referenced column is null are excluded from the rule
#     entirely - a missing value is an incompleteness problem, and counting it
#     twice would penalise the same defect under two dimensions.
CROSS_FIELD_RULES: list[tuple[str, list[str], str]] = [
    ("signup_before_last_order", ["signup_date", "last_order_date"],
     "signup_date <= last_order_date"),
    ("non_negative_value", ["lifetime_value"],
     "lifetime_value >= 0"),
]

# --------------------------------------------------------------------------
# 6b. Remediation playbook
# --------------------------------------------------------------------------
# The standard remedial action for each kind of finding, plus who owns it and
# roughly what it costs.
#
# These are rules, not model output, on purpose. "Who should fix this" and "how
# long will it take" are decisions an organisation makes once and applies
# consistently - not something to re-derive, differently, on every run. The
# language model writes the RATIONALE that goes with the action; it does not
# choose the action, the owner, or the effort.
#
#   issue_type: (action, owner, effort)
REMEDIATION_PLAYBOOK: dict[str, tuple[str, str, str]] = {
    "unmasked_pii_column": (
        "Apply masking or tokenisation before this column is exported, shared "
        "or loaded into a downstream system.",
        "Data Engineering", "Medium"),
    "pii_in_freetext": (
        "Redact the embedded personal data and add a free-text scan to the "
        "ingestion pipeline.",
        "Data Engineering", "Medium"),
    "pii_in_mislabeled_column": (
        "Reclassify the column, mask its contents, and correct the schema "
        "documentation so its purpose is not misleading.",
        "Data Governance", "Low"),
    "null_heavy_column": (
        "Trace the source system for this field and decide whether it is "
        "genuinely optional or the feed is broken.",
        "Data Engineering", "Medium"),
    "duplicate_record": (
        "Deduplicate on the business key and add a uniqueness constraint at "
        "the point of ingestion.",
        "Data Engineering", "Low"),
    "invalid_email": (
        "Add format validation at capture, and quarantine the existing "
        "malformed values for correction.",
        "Application Team", "Low"),
    "invalid_format": (
        "Add format validation at capture and quarantine existing values that "
        "do not conform.",
        "Application Team", "Low"),
    "inconsistent_value": (
        "Introduce a controlled vocabulary for this field and normalise the "
        "existing values to it.",
        "Data Governance", "Low"),
}
DEFAULT_REMEDIATION = ("Review this finding and decide on a remedial action.",
                       "Data Governance", "Unknown")


# --------------------------------------------------------------------------
# 6c. Language the narrative layer may not use
# --------------------------------------------------------------------------
# This system reports control gaps, not violations. Whether processing is
# lawful depends on consent, purpose and retention policy - none of which are
# visible in a dataset - so asserting a violation is a claim we cannot support.
#
# Instructing the model not to say it is NOT sufficient: a small model will
# happily write "this represents a control gap ... which would classify this as
# a violation" in the same sentence. So the rule is enforced by a check on the
# output, exactly like the invented-number guardrail. The model does not police
# itself; a rule polices the model.
#
# "breach" is deliberately absent - "risk of a data breach" is a legitimate
# statement about consequence, not an assertion about this dataset's status.
FORBIDDEN_NARRATIVE_TERMS: list[str] = [
    r"\bviolat\w*",
    r"\bnon-?complian\w*",
    r"\bunlawful\w*",
    r"\billegal\w*",
    r"\bin breach of\b",
]


# Salt for deterministic masking. The same input always masks to the same token,
# so distinct customers can still be counted without any identity being visible.
# A production deployment would source this from a secret store.
MASK_SALT = "governance-agent-demo-salt"


# --------------------------------------------------------------------------
# 7. Risk scoring
# --------------------------------------------------------------------------
#   risk = severity x exposure x volume_factor,  normalised to 0-100
#
# severity: how bad this kind of finding is, 1-5
SEVERITY: dict[str, int] = {
    "unmasked_pii_column":      5,
    "pii_in_freetext":          5,
    "pii_in_mislabeled_column": 5,
    "null_heavy_column":        3,
    "duplicate_record":         3,
    "invalid_email":            2,
    "invalid_format":           2,
    "inconsistent_value":       2,
}

# exposure: how exposing the data touched by the finding is, 1-3.
# Note this is a property of the DATA, not of the finding type - 40 duplicate
# rows in a product table and 40 duplicate customer records are not the same
# problem.
EXPOSURE: dict[str, int] = {
    "direct_identifier":       3,
    "quasi_identifier":        2,
    "pseudonymous_identifier": 2,
    "non_personal":            1,
}

# volume_factor = VOLUME_FLOOR + (1 - VOLUME_FLOOR) * share_of_rows_affected
#
# The floor exists because without it a single leaked national ID in 500 rows
# scores near zero, which is plainly wrong. It keeps severity dominant and lets
# volume act as an amplifier rather than a veto.
VOLUME_FLOOR = 0.5

# max of severity x exposure, used to normalise the raw score onto 0-100
_MAX_RAW = max(SEVERITY.values()) * max(EXPOSURE.values())

RISK_BANDS: list[tuple[int, str]] = [
    (25,  "Low"),
    (50,  "Medium"),
    (75,  "High"),
    (100, "Critical"),
]

# Findings at or above this score are blocked pending human approval.
REVIEW_THRESHOLD = 51


# --------------------------------------------------------------------------
# 8. Regulation mapping
# --------------------------------------------------------------------------
# The deterministic baseline citation for a finding, used when no policy corpus
# is loaded. Semantic retrieval over the real regulation text enriches these
# with the actual clause wording - it does not replace them, so citations still
# work with the policy index absent.
ARTICLE_MAP: dict[str, list[str]] = {
    "direct_identifier":       ["GDPR Art. 4(1)", "GDPR Art. 32(1)(a)"],
    "quasi_identifier":        ["GDPR Art. 4(1)", "GDPR Recital 26"],
    "pseudonymous_identifier": ["GDPR Art. 4(5)", "GDPR Recital 26"],
    "non_personal":            [],
}

# Retrieval queries, written in the REGULATION'S vocabulary rather than the
# database's.
#
# This matters more than anything else about retrieval quality. A query built
# from column names and issue types - "a column named customer_email; the
# concern is unmasked pii column" - scores around 0.40 and returns the wrong
# clauses, because no regulation talks that way. The same finding expressed as
# "security of processing; encryption and pseudonymisation; appropriate
# technical measures" scores around 0.69 and returns GDPR Art. 25 and Art. 32.
#
# An issue type with no entry here is NOT cited at all. A malformed email
# address is a data quality defect, not a regulatory matter, and attaching a
# statute to it would be padding.
RETRIEVAL_QUERY: dict[str, str] = {
    "unmasked_pii_column":
        "security of processing personal data; encryption and pseudonymisation; "
        "appropriate technical and organisational measures to protect personal "
        "data; data protection by design and by default",
    "pii_in_freetext":
        "data minimisation; personal data adequate, relevant and limited to what "
        "is necessary for the purposes of processing; data protection by design",
    "pii_in_mislabeled_column":
        "definition of personal data; information relating to an identified or "
        "identifiable natural person; records of processing activities and "
        "categories of personal data held",
    "null_heavy_column":
        "accuracy principle; personal data shall be accurate and where necessary "
        "kept up to date; inaccurate data erased or rectified without delay",
    "duplicate_record":
        "accuracy principle; every reasonable step to ensure personal data that "
        "is inaccurate is erased or rectified without delay",
}

# Added to the query when the column carries this classification.
DATA_CLASS_QUERY: dict[str, str] = {
    "direct_identifier":
        "an identifier such as a name, an identification number, location data "
        "or an online identifier",
    "pseudonymous_identifier":
        "pseudonymisation; personal data which could be attributed to a natural "
        "person by the use of additional information kept separately",
    "quasi_identifier":
        "factors specific to the physical, economic or social identity of a "
        "natural person; indirect identification",
}

# Below this cosine score a passage is not a citation, it is noise. Good matches
# on a well-phrased query score 0.6-0.8; unrelated passages sit around 0.3.
CITATION_MIN_SCORE = 0.45


ISSUE_ARTICLE_MAP: dict[str, list[str]] = {
    "null_heavy_column":  ["GDPR Art. 5(1)(d)"],   # accuracy principle
    "duplicate_record":   ["GDPR Art. 5(1)(d)"],
    "pii_in_freetext":    ["GDPR Art. 5(1)(c)", "GDPR Art. 25"],
    "unmasked_pii_column": ["GDPR Art. 32(1)(a)"],
}


# --------------------------------------------------------------------------
# Derived helpers (no tunable values below this line)
# --------------------------------------------------------------------------
def band_for(score: int) -> str:
    """Map a 0-100 risk score onto its band name."""
    for ceiling, name in RISK_BANDS:
        if score <= ceiling:
            return name
    return RISK_BANDS[-1][1]


def normalise_name(column: str) -> str:
    """'Customer ID' -> 'customerid'. Used for name-based classification."""
    return "".join(ch for ch in str(column).lower() if ch.isalnum())
