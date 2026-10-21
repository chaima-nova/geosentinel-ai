"""Unit tests for the deterministic evidence auditor."""

from __future__ import annotations

import pytest

from app.services.evidence_auditor import audit_conclusion
from app.services.risk_service import EvidenceProvenance as P


def _check(**overrides: object) -> list[str]:
    base: dict[str, object] = {
        "heps_score": 0.5,
        "stress_level": "moderate",
        "audit_status": "escalated",
        "data_complete": False,
        "vulnerability_provenance": P.VERIFIED,
        "satellite_provenance": P.VERIFIED,
    }
    base.update(overrides)
    return audit_conclusion(**base)  # type: ignore[arg-type]


def test_supported_partial_conclusion_has_no_violations() -> None:
    assert _check() == []


@pytest.mark.parametrize(
    "provenance", [P.PLACEHOLDER, P.SYNTHETIC, P.MISSING, P.FAILED, P.STALE, P.INVALID]
)
def test_score_without_verified_demographics_is_blocked(provenance: P) -> None:
    violations = _check(vulnerability_provenance=provenance)
    assert any("without verified demographic evidence" in v for v in violations)


def test_passed_on_incomplete_evidence_is_blocked() -> None:
    violations = _check(audit_status="passed", data_complete=False)
    assert "'passed' verdict on incomplete evidence" in violations


def test_passed_without_score_is_blocked() -> None:
    violations = _check(heps_score=None, stress_level=None, audit_status="passed")
    assert "'passed' verdict without a score" in violations


def test_stress_without_score_is_blocked() -> None:
    violations = _check(heps_score=None, stress_level="high", audit_status="insufficient_evidence")
    assert "stress level published without a score" in violations


@pytest.mark.parametrize("provenance", [P.FAILED, P.INVALID, P.STALE, P.MISSING, None])
def test_passed_on_unusable_satellite_evidence_is_blocked(provenance: P | None) -> None:
    violations = _check(
        audit_status="passed", data_complete=True, satellite_provenance=provenance
    )
    assert any("satellite evidence" in v for v in violations)


def test_clean_passed_verdict_is_allowed() -> None:
    assert (
        _check(audit_status="passed", data_complete=True, satellite_provenance=P.VERIFIED) == []
    )


def test_withheld_result_with_insufficient_status_is_allowed() -> None:
    assert (
        _check(
            heps_score=None,
            stress_level=None,
            audit_status="insufficient_evidence",
            vulnerability_provenance=P.MISSING,
        )
        == []
    )
