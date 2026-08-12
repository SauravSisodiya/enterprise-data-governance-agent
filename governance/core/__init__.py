"""
The deterministic core.

Every number the system produces originates here: profiling statistics, quality
scores, PII matches, risk scores. All of it is plain arithmetic and pattern
matching over a DataFrame.

Nothing in this package may import a language model client, an HTTP library, or
`governance.narrative`. That is not a style preference - it is what makes the
boundary a structural guarantee instead of a promise. `tests/test_boundary.py`
enforces it.
"""
