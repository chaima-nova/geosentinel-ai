"""Tests for the HEPS engine, :mod:`app.services.risk_service`."""

from __future__ import annotations

import pytest

from app.core.schemas import GridStressLevel
from app.services.risk_service import (
    DENSITY_SHARE,
    LST_COOL_BASELINE_C,
    LST_EXTREME_C,
    NDVI_BARE,
    NDVI_DENSE,
    POPULATION_DENSITY_MAX,
    RiskComputationError,
    RiskWeights,
    StaticVulnerabilityProvider,
    StressThresholds,
    VulnerabilityContext,
    VulnerabilityProvider,
    compute_heps,
)


def ctx(density: float = 5_000.0, svi: float = 0.5) -> VulnerabilityContext:
    """A demographic context with sensible mid-to-high exposure."""
    return VulnerabilityContext(density, svi)


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------


class TestRiskWeights:
    """Blend weights must form a convex combination."""

    def test_defaults_sum_to_one(self) -> None:
        weights = RiskWeights()
        assert weights.heat + weights.cooling + weights.vulnerability == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"heat": 0.5, "cooling": 0.4, "vulnerability": 0.2},  # 1.1
            {"heat": 0.2, "cooling": 0.2, "vulnerability": 0.2},  # 0.6
            {"heat": -0.1, "cooling": 0.8, "vulnerability": 0.3},  # negative
        ],
    )
    def test_invalid_weights_rejected(self, kwargs: dict) -> None:
        with pytest.raises(RiskComputationError):
            RiskWeights(**kwargs)

    def test_zero_weights_allowed_if_they_sum_to_one(self) -> None:
        weights = RiskWeights(heat=0.0, cooling=1.0, vulnerability=0.0)
        assert weights.as_dict == {"heat": 0.0, "cooling": 1.0, "vulnerability": 0.0}


class TestStressThresholds:
    """Classification boundaries."""

    def test_invalid_ordering_rejected(self) -> None:
        with pytest.raises(RiskComputationError):
            StressThresholds(moderate=0.6, high=0.5, extreme=0.8)

    def test_extreme_above_one_rejected(self) -> None:
        with pytest.raises(RiskComputationError):
            StressThresholds(moderate=0.3, high=0.5, extreme=1.5)

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.0, GridStressLevel.LOW),
            (0.29, GridStressLevel.LOW),
            (0.30, GridStressLevel.MODERATE),  # boundary is inclusive
            (0.54, GridStressLevel.MODERATE),
            (0.55, GridStressLevel.HIGH),
            (0.74, GridStressLevel.HIGH),
            (0.75, GridStressLevel.EXTREME),
            (1.0, GridStressLevel.EXTREME),
        ],
    )
    def test_classification(self, score: float, expected: GridStressLevel) -> None:
        assert StressThresholds().classify(score) is expected


class TestVulnerabilityContext:
    """Demographic input guards."""

    def test_negative_density_rejected(self) -> None:
        with pytest.raises(RiskComputationError):
            VulnerabilityContext(-1.0)

    @pytest.mark.parametrize("svi", [-0.1, 1.1])
    def test_out_of_range_svi_rejected(self, svi: float) -> None:
        with pytest.raises(RiskComputationError):
            VulnerabilityContext(1000.0, svi)

    def test_defaults(self) -> None:
        assert VulnerabilityContext(0.0).social_vulnerability == 0.5


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


class TestComputeHeps:
    """Score arithmetic, verified against hand computation."""

    def test_complete_case_matches_hand_computation(self) -> None:
        # heat  = (45 - 15) / (50 - 15) = 30/35 = 0.857142857
        # cool  = 1 - (0.15 - 0.1) / (0.8 - 0.1) = 1 - 0.071428 = 0.928571
        # vuln  = 0.5 * (8000/10000) + 0.5 * 0.8 = 0.4 + 0.4 = 0.8
        # HEPS  = 0.40*0.857142857 + 0.35*0.928571 + 0.25*0.8
        expected = 0.40 * (30 / 35) + 0.35 * (1 - (0.05 / 0.7)) + 0.25 * 0.8
        result = compute_heps(lst_celsius=45.0, ndvi=0.15, vulnerability=ctx(8_000.0, 0.8))
        assert result.heps_score == pytest.approx(expected, abs=1e-12)
        assert result.data_complete is True
        assert result.notes == ()

    def test_components_are_reported(self) -> None:
        result = compute_heps(lst_celsius=45.0, ndvi=0.15, vulnerability=ctx(8_000.0, 0.8))
        assert result.factors.heat_exposure == pytest.approx(30 / 35)
        assert result.factors.cooling_deficit == pytest.approx(1 - 0.05 / 0.7)
        assert result.factors.vulnerability == pytest.approx(0.8)

    def test_weights_are_unchanged_when_complete(self) -> None:
        result = compute_heps(lst_celsius=30.0, ndvi=0.4, vulnerability=ctx())
        assert result.weights_used == pytest.approx({"heat": 0.4, "cooling": 0.35, "vulnerability": 0.25})

    def test_custom_weights_are_honoured(self) -> None:
        weights = RiskWeights(heat=0.7, cooling=0.2, vulnerability=0.1)
        result = compute_heps(lst_celsius=45.0, ndvi=0.15, vulnerability=ctx(), weights=weights)
        assert result.weights_used == pytest.approx(weights.as_dict)
        assert result.heps_score > compute_heps(
            lst_celsius=45.0, ndvi=0.15, vulnerability=ctx()
        ).heps_score

    @pytest.mark.parametrize(
        "lst, expected_heat",
        [(LST_COOL_BASELINE_C, 0.0), (LST_EXTREME_C, 1.0), (0.0, 0.0), (80.0, 1.0)],
    )
    def test_heat_exposure_is_clamped(self, lst: float, expected_heat: float) -> None:
        result = compute_heps(lst_celsius=lst, ndvi=0.4, vulnerability=ctx())
        assert result.factors.heat_exposure == pytest.approx(expected_heat)

    @pytest.mark.parametrize(
        "ndvi, expected_deficit",
        [(NDVI_DENSE, 0.0), (NDVI_BARE, 1.0), (-1.0, 1.0), (1.0, 0.0)],
    )
    def test_cooling_deficit_is_clamped_and_inverted(self, ndvi: float, expected_deficit: float) -> None:
        result = compute_heps(lst_celsius=30.0, ndvi=ndvi, vulnerability=ctx())
        assert result.factors.cooling_deficit == pytest.approx(expected_deficit)

    def test_score_is_bounded(self) -> None:
        worst = compute_heps(lst_celsius=80.0, ndvi=-1.0, vulnerability=ctx(999_999.0, 1.0))
        best = compute_heps(lst_celsius=-20.0, ndvi=1.0, vulnerability=ctx(0.0, 0.0))
        assert worst.heps_score == pytest.approx(1.0)
        assert best.heps_score == pytest.approx(0.0)


class TestMonotonicity:
    """The score must move in the direction the domain expects."""

    def test_hotter_is_worse(self) -> None:
        cool = compute_heps(lst_celsius=20.0, ndvi=0.4, vulnerability=ctx())
        hot = compute_heps(lst_celsius=48.0, ndvi=0.4, vulnerability=ctx())
        assert hot.heps_score > cool.heps_score

    def test_less_vegetation_is_worse(self) -> None:
        green = compute_heps(lst_celsius=35.0, ndvi=0.75, vulnerability=ctx())
        bare = compute_heps(lst_celsius=35.0, ndvi=0.05, vulnerability=ctx())
        assert bare.heps_score > green.heps_score

    def test_more_exposure_is_worse(self) -> None:
        sparse = compute_heps(lst_celsius=35.0, ndvi=0.3, vulnerability=ctx(100.0, 0.2))
        dense = compute_heps(lst_celsius=35.0, ndvi=0.3, vulnerability=ctx(9_500.0, 0.9))
        assert dense.heps_score > sparse.heps_score

    def test_density_and_svi_both_contribute(self) -> None:
        density_only = compute_heps(lst_celsius=35.0, ndvi=0.3, vulnerability=ctx(10_000.0, 0.0))
        svi_only = compute_heps(lst_celsius=35.0, ndvi=0.3, vulnerability=ctx(0.0, 1.0))
        # Equal weights on each half of the equity component.
        assert density_only.factors.vulnerability == pytest.approx(
            svi_only.factors.vulnerability
        )
        assert DENSITY_SHARE == 0.5


# ---------------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------------


class TestMissingEvidence:
    """Unavailable components renormalise; they are never zero-filled."""

    def test_missing_lst_excludes_heat_and_renormalises(self) -> None:
        result = compute_heps(lst_celsius=None, ndvi=0.15, vulnerability=ctx(8_000.0, 0.8))
        assert result.factors.heat_exposure is None
        assert result.data_complete is False
        # Remaining weights 0.35 / 0.25 renormalised over 0.60.
        assert result.weights_used == pytest.approx(
            {"cooling": 0.35 / 0.60, "vulnerability": 0.25 / 0.60}
        )
        assert result.weights_used["cooling"] + result.weights_used["vulnerability"] == pytest.approx(1.0)
        assert len(result.notes) == 1 and "land-surface temperature" in result.notes[0]

    def test_missing_ndvi_excludes_cooling(self) -> None:
        result = compute_heps(lst_celsius=45.0, ndvi=None, vulnerability=ctx())
        assert result.factors.cooling_deficit is None
        assert result.weights_used == pytest.approx(
            {"heat": 0.40 / 0.65, "vulnerability": 0.25 / 0.65}
        )
        assert "NDVI unavailable" in result.notes[0]

    def test_missing_both_leaves_only_equity(self) -> None:
        result = compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(8_000.0, 0.8))
        assert result.weights_used == {"vulnerability": 1.0}
        assert result.heps_score == pytest.approx(0.8)
        assert len(result.notes) == 2

    def test_renormalisation_does_not_drag_the_score_toward_zero(self) -> None:
        """The core guarantee: a missing LST must not read as 'no heat'.

        With equity alone at 0.8 the score stays 0.8, rather than the 0.2 that
        zero-filling heat and cooling would have produced.
        """
        renormalised = compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(8_000.0, 0.8))
        zero_filled = 0.25 * 0.8  # what naive zero-filling would give
        assert renormalised.heps_score == pytest.approx(0.8)
        assert renormalised.heps_score > zero_filled * 3

    def test_incomplete_assessment_recommends_escalation(self) -> None:
        assert compute_heps(lst_celsius=None, ndvi=0.4, vulnerability=ctx()).recommended_audit_status == "escalated"
        assert compute_heps(lst_celsius=30.0, ndvi=0.4, vulnerability=ctx()).recommended_audit_status == "passed"

    def test_every_component_unavailable_raises(self) -> None:
        """With zero weight on equity and no observations there is nothing to score."""
        weights = RiskWeights(heat=0.5, cooling=0.5, vulnerability=0.0)
        with pytest.raises(RiskComputationError, match="zero weight"):
            compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(), weights=weights)

    def test_equity_only_is_still_computable_with_zero_satellite_weight(self) -> None:
        weights = RiskWeights(heat=0.5, cooling=0.5, vulnerability=0.0)
        result = compute_heps(lst_celsius=45.0, ndvi=0.15, vulnerability=ctx(), weights=weights)
        # vulnerability carries no weight, so heat and cooling split everything.
        assert "vulnerability" not in result.weights_used or result.weights_used.get(
            "vulnerability", 1.0
        ) == pytest.approx(0.0)
        assert result.heps_score == pytest.approx(0.5 * (30 / 35) + 0.5 * (1 - 0.05 / 0.7))


# ---------------------------------------------------------------------------
# Vulnerability provider
# ---------------------------------------------------------------------------


class TestVulnerabilityProvider:
    """The placeholder provider and its protocol."""

    def test_static_provider_satisfies_the_protocol(self) -> None:
        assert isinstance(StaticVulnerabilityProvider(), VulnerabilityProvider)

    async def test_static_provider_returns_neutral_defaults(self) -> None:
        context = await StaticVulnerabilityProvider().fetch([0.0, 0.0, 1.0, 1.0])
        assert context.population_density == 1_000.0
        assert context.social_vulnerability == 0.5

    async def test_static_provider_is_area_independent(self) -> None:
        provider = StaticVulnerabilityProvider()
        a = await provider.fetch([0.0, 0.0, 1.0, 1.0])
        b = await provider.fetch([-180.0, -90.0, 180.0, 90.0])
        assert a == b

    async def test_custom_values(self) -> None:
        provider = StaticVulnerabilityProvider(population_density=2_500.0, social_vulnerability=0.9)
        context = await provider.fetch([0.0, 0.0, 1.0, 1.0])
        assert context.population_density == 2_500.0
        assert context.social_vulnerability == 0.9

    def test_neutral_baseline_blends_to_a_midrange_score(self) -> None:
        result = compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(1_000.0, 0.5))
        assert result.heps_score == pytest.approx(0.5 * 0.1 + 0.5 * 0.5)

    def test_density_saturation_cap(self) -> None:
        at_cap = compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(POPULATION_DENSITY_MAX, 0.0))
        above_cap = compute_heps(lst_celsius=None, ndvi=None, vulnerability=ctx(1_000_000.0, 0.0))
        assert at_cap.heps_score == above_cap.heps_score == pytest.approx(0.5)

class TestNormalisationGuard:
    """The low-level helper's own guards."""

    def test_rejects_a_non_increasing_range(self) -> None:
        from app.services.risk_service import _normalise

        with pytest.raises(RiskComputationError, match="strictly increasing"):
            _normalise(0.5, 1.0, 1.0)
        with pytest.raises(RiskComputationError, match="strictly increasing"):
            _normalise(0.5, 1.0, 0.0)

    @pytest.mark.parametrize(
        "value, expected", [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (-5.0, 0.0), (15.0, 1.0)]
    )
    def test_maps_and_clamps(self, value: float, expected: float) -> None:
        from app.services.risk_service import _normalise

        assert _normalise(value, 0.0, 10.0) == pytest.approx(expected)
