"""Heat Equity Priority Score (HEPS) engine.

HEPS is a 0-1 composite that ranks where heat risk is most acute, combining
three independent lines of evidence:

1. **Heat exposure** — land-surface temperature, normalised against a cool
   baseline and an extreme ceiling.
2. **Cooling deficit** — the absence of vegetation cooling. NDVI is inverted,
   so bare ground scores high and dense canopy scores low.
3. **Equity weighting** — how many people are exposed and how socially
   vulnerable they are. A hot, treeless, empty field is a lower priority than
   a hot, treeless, densely populated neighbourhood.

The three are blended with configurable weights that must sum to 1.

Missing evidence is renormalised, never zero-filled
---------------------------------------------------
This matters more than it looks. ``SatelliteDataService`` cannot supply
land-surface temperature at all — Sentinel-2 has no thermal band — so its
``lst_celsius_max`` is always the ``LST_NOT_AVAILABLE`` placeholder of 0.0.
Feeding that straight into the formula would score heat exposure at 0 and
silently *understate* risk everywhere.

Instead each component is either observed or unknown. Unknown components drop
out and the remaining weights are renormalised to sum to 1, so the score stays
comparable instead of being dragged toward zero. The result reports
``data_complete=False`` and explains itself in ``notes``.

The formula is a documented, configurable heuristic — not a published
standard. :class:`RiskWeights` and :class:`StressThresholds` exist so the
constants can be tuned and tested without touching the arithmetic.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.core.schemas import GridStressLevel

logger = logging.getLogger(__name__)

__all__ = [
    "RiskAssessment",
    "RiskComputationError",
    "RiskFactors",
    "RiskWeights",
    "StaticVulnerabilityProvider",
    "StressThresholds",
    "VulnerabilityContext",
    "VulnerabilityProvider",
    "compute_heps",
]

# --- Normalisation anchors -------------------------------------------------
# Chosen so that a temperate vegetated suburb lands mid-range and an arid,
# treeless, densely built district lands near the top.

#: LST at or below which there is effectively no heat stress, in °C.
LST_COOL_BASELINE_C = 15.0
#: LST at or above which heat stress is maximal, in °C.
LST_EXTREME_C = 50.0

#: NDVI at or below which the surface provides no evaporative cooling.
NDVI_BARE = 0.1
#: NDVI at or above which canopy cooling is treated as complete.
NDVI_DENSE = 0.8

#: Population density (people per km²) at which exposure is maximal.
POPULATION_DENSITY_MAX = 10_000.0

#: Relative weight of density versus the social vulnerability index within the
#: single equity component.
DENSITY_SHARE = 0.5

#: Neutral values used when no demographic source is wired up. Deliberately
#: mid-range: a placeholder should not silently make an area look safe.
DEFAULT_POPULATION_DENSITY = 1_000.0
DEFAULT_SOCIAL_VULNERABILITY = 0.5


class RiskComputationError(RuntimeError):
    """Raised when HEPS cannot be computed from any available evidence."""


def _clamp01(value: float) -> float:
    """Clamp to the closed unit interval."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _normalise(value: float, low: float, high: float) -> float:
    """Linearly map ``value`` from ``[low, high]`` onto ``[0, 1]``.

    Args:
        value: The measurement.
        low: Input that maps to 0.
        high: Input that maps to 1.

    Returns:
        The clamped normalised value.

    Raises:
        RiskComputationError: If the range is not strictly increasing.
    """
    if high <= low:
        raise RiskComputationError(
            f"normalisation range must be strictly increasing, got [{low}, {high}]"
        )
    return _clamp01((value - low) / (high - low))


@dataclass(frozen=True)
class RiskWeights:
    """Blend weights for the three HEPS components.

    Attributes:
        heat: Weight on land-surface temperature exposure.
        cooling: Weight on the vegetation cooling deficit.
        vulnerability: Weight on population exposure and social vulnerability.
    """

    heat: float = 0.40
    cooling: float = 0.35
    vulnerability: float = 0.25

    def __post_init__(self) -> None:
        for name in ("heat", "cooling", "vulnerability"):
            value = getattr(self, name)
            if value < 0.0:
                raise RiskComputationError(f"weight {name!r} must be >= 0, got {value}")
        total = self.heat + self.cooling + self.vulnerability
        if abs(total - 1.0) > 1e-9:
            raise RiskComputationError(
                f"weights must sum to 1.0, got {total:.6f} "
                f"(heat={self.heat}, cooling={self.cooling}, vulnerability={self.vulnerability})"
            )

    @property
    def as_dict(self) -> dict[str, float]:
        """Weights keyed by component name."""
        return {
            "heat": self.heat,
            "cooling": self.cooling,
            "vulnerability": self.vulnerability,
        }


@dataclass(frozen=True)
class StressThresholds:
    """Cut-offs mapping a HEPS score onto :class:`GridStressLevel`.

    Attributes:
        moderate: Score at or above which stress becomes ``moderate``.
        high: Score at or above which stress becomes ``high``.
        extreme: Score at or above which stress becomes ``extreme``.
    """

    moderate: float = 0.30
    high: float = 0.55
    extreme: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 < self.moderate < self.high < self.extreme <= 1.0:
            raise RiskComputationError(
                "thresholds must satisfy 0 < moderate < high < extreme <= 1, got "
                f"moderate={self.moderate}, high={self.high}, extreme={self.extreme}"
            )

    def classify(self, score: float) -> GridStressLevel:
        """Map a HEPS score to a stress level.

        Args:
            score: A HEPS score in [0, 1].

        Returns:
            The matching :class:`GridStressLevel`.
        """
        if score >= self.extreme:
            return GridStressLevel.EXTREME
        if score >= self.high:
            return GridStressLevel.HIGH
        if score >= self.moderate:
            return GridStressLevel.MODERATE
        return GridStressLevel.LOW


@dataclass(frozen=True)
class VulnerabilityContext:
    """Demographic exposure for one area.

    Attributes:
        population_density: Residents per square kilometre.
        social_vulnerability: Composite social vulnerability index in [0, 1],
            where 1 is most vulnerable.
    """

    population_density: float
    social_vulnerability: float = DEFAULT_SOCIAL_VULNERABILITY

    def __post_init__(self) -> None:
        if self.population_density < 0.0:
            raise RiskComputationError(
                f"population_density must be >= 0, got {self.population_density}"
            )
        if not 0.0 <= self.social_vulnerability <= 1.0:
            raise RiskComputationError(
                f"social_vulnerability must be within [0, 1], got {self.social_vulnerability}"
            )


@runtime_checkable
class VulnerabilityProvider(Protocol):
    """Source of demographic exposure for a bounding box."""

    async def fetch(self, bbox: Sequence[float]) -> VulnerabilityContext:
        """Return the demographic context covering ``bbox``."""
        ...


class StaticVulnerabilityProvider:
    """Placeholder provider returning a neutral baseline.

    No demographic dataset is wired up yet, so this returns mid-range values
    rather than zeros: a missing equity input should not make an area look
    safe. Swap in a real source (WorldPop, GHSL, a national census) by
    implementing :class:`VulnerabilityProvider` and passing it to
    ``CoordinatorAgent``.

    Args:
        population_density: Residents per km² to report for every area.
        social_vulnerability: Index in [0, 1] to report for every area.
    """

    def __init__(
        self,
        population_density: float = DEFAULT_POPULATION_DENSITY,
        social_vulnerability: float = DEFAULT_SOCIAL_VULNERABILITY,
    ) -> None:
        self._context = VulnerabilityContext(population_density, social_vulnerability)

    async def fetch(self, bbox: Sequence[float]) -> VulnerabilityContext:
        """Return the same neutral context for any area."""
        return self._context


@dataclass(frozen=True)
class RiskFactors:
    """The normalised components behind a HEPS score.

    A component is ``None`` when its underlying evidence was unavailable, which
    is distinct from a genuine 0.0.

    Attributes:
        heat_exposure: Normalised land-surface temperature, or ``None``.
        cooling_deficit: Inverted normalised NDVI, or ``None``.
        vulnerability: Blended density and social vulnerability in [0, 1].
    """

    heat_exposure: float | None
    cooling_deficit: float | None
    vulnerability: float


@dataclass(frozen=True)
class RiskAssessment:
    """A computed HEPS score with everything needed to audit it.

    Attributes:
        heps_score: The composite score in [0, 1].
        stress_level: The score mapped through :class:`StressThresholds`.
        factors: The normalised components, with ``None`` for missing evidence.
        weights_used: The renormalised weights actually applied.
        data_complete: False when any component had to drop out.
        notes: Human-readable explanations of any adjustment.
    """

    heps_score: float
    stress_level: GridStressLevel
    factors: RiskFactors
    weights_used: dict[str, float] = field(default_factory=dict)
    data_complete: bool = True
    notes: tuple[str, ...] = ()

    @property
    def recommended_audit_status(self) -> str:
        """Audit status implied by how much evidence backed the score.

        Returns:
            ``"passed"`` when every component was observed, otherwise
            ``"escalated"`` so a human reviews the incomplete assessment.
        """
        return "passed" if self.data_complete else "escalated"


def compute_heps(
    *,
    lst_celsius: float | None,
    ndvi: float | None,
    vulnerability: VulnerabilityContext,
    weights: RiskWeights | None = None,
    thresholds: StressThresholds | None = None,
) -> RiskAssessment:
    """Compute the Heat Equity Priority Score for one area.

    Pass ``None`` for any measurement that is genuinely unavailable — the
    sentinel constants from :mod:`app.services.satellite_service` should be
    translated to ``None`` by the caller. Available components are blended with
    their weights renormalised to sum to 1.

    Args:
        lst_celsius: Maximum land-surface temperature in °C, or ``None``.
        ndvi: Mean NDVI in [-1, 1], or ``None``.
        vulnerability: Demographic exposure for the area.
        weights: Blend weights. Defaults to :class:`RiskWeights`.
        thresholds: Stress cut-offs. Defaults to :class:`StressThresholds`.

    Returns:
        A :class:`RiskAssessment` carrying the score and its provenance.

    Raises:
        RiskComputationError: If no component is available, or the weights or
            thresholds are invalid.
    """
    weights = weights or RiskWeights()
    thresholds = thresholds or StressThresholds()
    notes: list[str] = []

    heat_exposure: float | None = None
    if lst_celsius is not None:
        heat_exposure = _normalise(lst_celsius, LST_COOL_BASELINE_C, LST_EXTREME_C)
    else:
        notes.append(
            "land-surface temperature unavailable; heat exposure excluded and "
            "weights renormalised"
        )

    cooling_deficit: float | None = None
    if ndvi is not None:
        cooling_deficit = 1.0 - _normalise(ndvi, NDVI_BARE, NDVI_DENSE)
    else:
        notes.append(
            "NDVI unavailable; cooling deficit excluded and weights renormalised"
        )

    vulnerability_score = _blend_vulnerability(vulnerability)

    components: dict[str, tuple[float, float]] = {
        "heat": (weights.heat, heat_exposure if heat_exposure is not None else 0.0),
        "cooling": (weights.cooling, cooling_deficit if cooling_deficit is not None else 0.0),
        "vulnerability": (weights.vulnerability, vulnerability_score),
    }
    # The equity component is always present; the others only if observed.
    observed = {
        name: (weight, value)
        for name, (weight, value) in components.items()
        if name == "vulnerability"
        or (name == "heat" and heat_exposure is not None)
        or (name == "cooling" and cooling_deficit is not None)
    }

    active_total = sum(weight for weight, _ in observed.values())
    if active_total <= 0.0:
        raise RiskComputationError(
            "cannot compute HEPS: every available component carries zero weight"
        )

    renormalised = {name: weight / active_total for name, (weight, _) in observed.items()}
    score = _clamp01(
        sum(renormalised[name] * value for name, (_, value) in observed.items())
    )

    # Every component that drops out appends its own note above, so a partial
    # assessment always carries an explanation.
    data_complete = len(observed) == len(components)

    assessment = RiskAssessment(
        heps_score=score,
        stress_level=thresholds.classify(score),
        factors=RiskFactors(
            heat_exposure=heat_exposure,
            cooling_deficit=cooling_deficit,
            vulnerability=vulnerability_score,
        ),
        weights_used=renormalised,
        data_complete=data_complete,
        notes=tuple(notes),
    )
    logger.debug(
        "HEPS %.4f (%s) from heat=%s cooling=%s vulnerability=%.4f",
        assessment.heps_score,
        assessment.stress_level.value,
        _fmt(heat_exposure),
        _fmt(cooling_deficit),
        vulnerability_score,
    )
    return assessment


def _blend_vulnerability(context: VulnerabilityContext) -> float:
    """Combine population density and social vulnerability into one 0-1 score.

    Args:
        context: The demographic inputs.

    Returns:
        The blended equity component in [0, 1].
    """
    density = _clamp01(context.population_density / POPULATION_DENSITY_MAX)
    return _clamp01(
        DENSITY_SHARE * density + (1.0 - DENSITY_SHARE) * context.social_vulnerability
    )


def _fmt(value: float | None) -> str:
    """Render a possibly-missing factor for logging."""
    return "n/a" if value is None else f"{value:.4f}"
