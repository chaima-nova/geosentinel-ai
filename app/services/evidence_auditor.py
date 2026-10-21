"""Deterministic evidence auditor.

This is a rule check, not an LLM agent. It re-tests the conclusion a pipeline
is about to publish against the evidence behind it, and returns every rule the
conclusion breaks. The coordinator uses it as a second line of defence after
:func:`app.services.risk_service.compute_heps`: even if a scoring path forgets
a check, a score backed by unverified demographics still cannot be published.

Rules (each returns one violation string when broken):

1. A HEPS score requires verified demographic evidence.
2. A ``passed`` verdict requires every component to be observed.
3. A ``passed`` verdict requires a score to exist.
4. Without a score there must be no stress level.
5. A ``passed`` verdict may not rest on satellite evidence that failed,
   was invalid, stale or missing, or on no satellite provenance at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.schemas import AuditStatus
from app.services.risk_service import EvidenceProvenance

__all__ = ["audit_conclusion"]


def audit_conclusion(
    *,
    heps_score: float | None,
    stress_level: str | None,
    audit_status: str,
    data_complete: bool,
    vulnerability_provenance: EvidenceProvenance,
    satellite_provenance: EvidenceProvenance | None,
) -> list[str]:
    """Return the rules that a proposed conclusion breaks.

    Args:
        heps_score: The proposed score, or ``None``.
        stress_level: The proposed stress class, or ``None``.
        audit_status: The proposed audit status value.
        data_complete: Whether every evidence component was observed.
        vulnerability_provenance: Provenance of the demographic evidence.
        satellite_provenance: Provenance of the satellite evidence, or
            ``None`` when the satellite stage did not produce a layer.

    Returns:
        A list of violation descriptions. Empty when the conclusion is
        supported by its evidence.
    """
    violations: list[str] = []
    passed = audit_status == AuditStatus.PASSED.value

    if heps_score is not None and vulnerability_provenance is not EvidenceProvenance.VERIFIED:
        violations.append(
            "score published without verified demographic evidence "
            f"(provenance={vulnerability_provenance.value})"
        )
    if passed and not data_complete:
        violations.append("'passed' verdict on incomplete evidence")
    if passed and heps_score is None:
        violations.append("'passed' verdict without a score")
    if heps_score is None and stress_level is not None:
        violations.append("stress level published without a score")
    # Fail closed: a passed verdict with no satellite provenance at all is an
    # inconsistent state, not a clean pass.
    if passed and (satellite_provenance is None or satellite_provenance in _UNUSABLE_SATELLITE):
        violations.append(
            "'passed' verdict on satellite evidence that is "
            f"{satellite_provenance.value if satellite_provenance else 'absent'}"
        )
    return violations


_UNUSABLE_SATELLITE: Sequence[EvidenceProvenance] = (
    EvidenceProvenance.FAILED,
    EvidenceProvenance.INVALID,
    EvidenceProvenance.STALE,
    EvidenceProvenance.MISSING,
)
