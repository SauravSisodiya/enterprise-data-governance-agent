"""
Risk scoring.

    risk = severity x exposure x volume_factor,  normalised onto 0-100

severity   how bad this kind of finding is                          (1-5)
exposure   how exposing the data it touches is                      (1-3)
volume     share of rows affected, floored so it amplifies rather
           than vetoes                                          (0.5-1.0)

Two design choices worth being able to defend:

  Why exposure is a property of the DATA, not of the finding type.
      Forty duplicate rows in a product-reference table and forty duplicate
      rows holding customer emails are not the same problem. The finding type
      sets severity; what the finding touches sets exposure.

  Why volume has a floor.
      Without it, one leaked national ID among 500 rows scores close to zero,
      which is plainly wrong. The floor keeps severity dominant and lets volume
      act as an amplifier rather than a veto.

No language model is involved at any point in this file, and none ever will be.
The same input produces the same score on every run and on every machine, which
is what makes the number defensible in a review.
"""
from __future__ import annotations

from governance import config
from governance.state import Finding


def volume_factor(finding: Finding) -> float:
    if finding.total_rows <= 0:
        return config.VOLUME_FLOOR
    share = finding.affected_rows / finding.total_rows
    return config.VOLUME_FLOOR + (1 - config.VOLUME_FLOOR) * min(share, 1.0)


def score(finding: Finding) -> int:
    severity = config.SEVERITY.get(finding.issue_type, 1)
    exposure = config.EXPOSURE.get(finding.data_class, 1)
    raw = severity * exposure * volume_factor(finding)
    return int(round(raw / config._MAX_RAW * 100))


def score_all(findings: list[Finding]) -> list[Finding]:
    """
    Returns NEW findings, scored and ranked. Nothing is mutated in place.

    Deciding who reviews what is a separate concern and lives in core/gate.py -
    this module only answers "how bad is it?".
    """
    scored = [f.scored(v, config.band_for(v))
              for f, v in ((f, score(f)) for f in findings)]
    return sorted(scored, key=lambda f: (-(f.risk or 0), f.column))
