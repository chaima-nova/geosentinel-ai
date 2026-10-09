"""Request orchestration for GeoSentinel-AI.

:class:`CoordinatorAgent` owns the request lifecycle: it takes a validated
:class:`~app.core.schemas.SpatialQueryRequest`, gathers satellite metrics and
demographic exposure, runs the HEPS engine, and consolidates everything into a
:class:`~app.core.schemas.RiskAnalysisResult`.

The pipeline is a :mod:`langgraph` ``StateGraph`` rather than a hand-rolled
``await`` chain, so each phase is an isolated node with its own error handling
and the graph can later gain parallel branches (e.g. fetching Sentinel-3 SLSTR
alongside Sentinel-2) without restructuring::

    START -> gather_satellite -> gather_vulnerability -> assess -> consolidate -> END
                       |                  |
                       +-- on error ------>+-- on error ----> consolidate

Failure semantics
-----------------
A tool raising an *unexpected* exception fails the run: ``audit_status``
becomes ``"failed"`` and the result carries no score. That is deliberately
distinct from a tool degrading. :class:`SatelliteDataService` is built never to
raise for operational failures — it returns sentinel metadata instead — and
when it does, the pipeline keeps going with whatever evidence exists and marks
the result ``"escalated"`` for human review.

An incomplete HEPS is still a useful HEPS, but a HEPS with no equity component
is not a *Heat Equity* score at all, so a vulnerability-provider failure fails
the run rather than degrading it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.schemas import (
    AuditStatus,
    GridStressLevel,
    RiskAnalysisResult,
    SatelliteLayerMetadata,
    SpatialQueryRequest,
)
from app.services.risk_service import (
    RiskComputationError,
    RiskWeights,
    StaticVulnerabilityProvider,
    StressThresholds,
    VulnerabilityContext,
    VulnerabilityProvider,
    compute_heps,
)
from app.services.satellite_service import (
    NDVI_NO_DATA,
    UNAVAILABLE_STAC_ID,
    SatelliteDataService,
)

logger = logging.getLogger(__name__)

__all__ = ["CoordinatorAgent", "PipelineState"]


class PipelineState(TypedDict, total=False):
    """State threaded through the coordinator graph.

    Attributes:
        request: The validated query being executed.
        layer: Satellite metrics, or ``None`` if retrieval produced nothing.
        vulnerability: Demographic exposure as a plain dict, for state
            serialisability.
        assessment: The computed HEPS payload, as a plain dict.
        error: Set when a phase failed hard; short-circuits to consolidation.
        result: The final result payload, written by the consolidation phase.
        phases: Names of the phases that ran, in order.
    """

    request: SpatialQueryRequest
    layer: dict[str, Any] | None
    vulnerability: dict[str, float]
    assessment: dict[str, Any]
    error: str | None
    result: dict[str, Any]
    phases: list[str]


class CoordinatorAgent:
    """Orchestrates satellite retrieval, risk scoring and consolidation.

    Args:
        satellite_service: Source of Sentinel-2 metrics. Defaults to a
            :class:`SatelliteDataService` pointed at the CDSE STAC API.
        vulnerability_provider: Source of demographic exposure. Defaults to
            :class:`StaticVulnerabilityProvider`, a documented neutral
            placeholder — wire a real dataset for meaningful equity weighting.
        weights: HEPS blend weights.
        thresholds: HEPS-to-stress-level cut-offs.
        provides_lst: Whether the wired satellite source supplies land-surface
            temperature. **Defaults to False**, because Sentinel-2 has no
            thermal band and ``SatelliteLayerMetadata.lst_celsius_max`` is
            always a placeholder. Flip this only once a thermal source such as
            Sentinel-3 SLSTR is wired in; leaving it False keeps a fake 0.0 °C
            from being scored as "no heat".

    Example::

        agent = CoordinatorAgent()
        result = await agent.execute_pipeline(
            SpatialQueryRequest(
                query="heat stress in dense housing",
                bbox=[5.2, 31.8, 5.5, 32.1],
                start_date="2024-06-01",
                end_date="2024-06-30",
            )
        )
        print(result.heps_score, result.grid_stress_level, result.audit_status)
    """

    def __init__(
        self,
        *,
        satellite_service: SatelliteDataService | None = None,
        vulnerability_provider: VulnerabilityProvider | None = None,
        weights: RiskWeights | None = None,
        thresholds: StressThresholds | None = None,
        provides_lst: bool = False,
    ) -> None:
        self._satellite = satellite_service or SatelliteDataService()
        self._vulnerability: VulnerabilityProvider = (
            vulnerability_provider or StaticVulnerabilityProvider()
        )
        self._weights = weights or RiskWeights()
        self._thresholds = thresholds or StressThresholds()
        self._provides_lst = provides_lst
        self._graph = self._build_graph()

    # --- Graph construction ------------------------------------------------

    def _build_graph(self) -> Any:
        """Compile the pipeline graph."""
        graph: StateGraph = StateGraph(PipelineState)
        graph.add_node("gather_satellite", self._gather_satellite)
        graph.add_node("gather_vulnerability", self._gather_vulnerability)
        graph.add_node("assess", self._assess)
        graph.add_node("consolidate", self._consolidate)

        graph.add_edge(START, "gather_satellite")
        graph.add_conditional_edges(
            "gather_satellite",
            self._route_after_satellite,
            {"gather_vulnerability": "gather_vulnerability", "consolidate": "consolidate"},
        )
        graph.add_conditional_edges(
            "gather_vulnerability",
            self._route_after_vulnerability,
            {"assess": "assess", "consolidate": "consolidate"},
        )
        graph.add_edge("assess", "consolidate")
        graph.add_edge("consolidate", END)
        return graph.compile()

    @staticmethod
    def _route_after_satellite(state: PipelineState) -> str:
        """Skip ahead to consolidation if retrieval failed hard."""
        return "consolidate" if state.get("error") else "gather_vulnerability"

    @staticmethod
    def _route_after_vulnerability(state: PipelineState) -> str:
        """Skip scoring if demographic exposure could not be obtained."""
        return "consolidate" if state.get("error") else "assess"

    # --- Public API --------------------------------------------------------

    async def execute_pipeline(self, request: SpatialQueryRequest) -> RiskAnalysisResult:
        """Run the full pipeline for one spatial query.

        Args:
            request: A validated :class:`SpatialQueryRequest`. Passing the model
                rather than raw arguments means bbox and date validation has
                already happened at the API boundary.

        Returns:
            A :class:`RiskAnalysisResult`. ``audit_status`` reports how much
            evidence backed it: ``"passed"`` when every component was observed,
            ``"escalated"`` when the score is usable but incomplete, and
            ``"failed"`` when a phase raised and no score could be produced.
        """
        logger.info(
            "pipeline start query=%r bbox=%s window=%s..%s",
            request.query,
            request.bbox,
            request.start_date,
            request.end_date,
        )
        initial: PipelineState = {
            "request": request,
            "layer": None,
            "error": None,
            "phases": [],
        }
        final = await self._graph.ainvoke(initial)

        result = RiskAnalysisResult.model_validate(final["result"])
        logger.info(
            "pipeline end audit=%s stress=%s heps=%.4f phases=%s",
            result.audit_status,
            result.grid_stress_level,
            result.heps_score,
            final.get("phases", []),
        )
        return result

    async def aclose(self) -> None:
        """Release resources held by the underlying tools.

        Closes the satellite service's HTTP client if this agent owns it. A
        caller-supplied service or provider is left alone, since the caller
        owns its lifecycle. Safe to call more than once.
        """
        close = getattr(self._satellite, "aclose", None)
        if close is not None:
            await close()

    # --- Phases ------------------------------------------------------------

    async def _gather_satellite(self, state: PipelineState) -> dict[str, Any]:
        """Phase 1: retrieve satellite metrics for the window.

        Args:
            state: The pipeline state.

        Returns:
            Partial state carrying either the layer or an error.
        """
        request = state["request"]
        logger.info("phase=gather_satellite start bbox=%s", request.bbox)
        try:
            layer = await self._satellite.fetch_sentinel_indices(
                bbox=request.bbox,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            # Validate what the tool returned; a malformed document is a hard
            # error rather than something to score.
            SatelliteLayerMetadata.model_validate(layer)
        except Exception as exc:  # noqa: BLE001 - a tool failure must not escape
            logger.error(
                "phase=gather_satellite failed %s: %s", type(exc).__name__, exc
            )
            return {
                "error": f"gather_satellite failed: {type(exc).__name__}: {exc}",
                "phases": [*state.get("phases", []), "gather_satellite"],
            }

        if layer["stac_id"] == UNAVAILABLE_STAC_ID:
            logger.warning(
                "phase=gather_satellite degraded: no usable scene; scoring from "
                "demographic exposure alone"
            )
        else:
            logger.info(
                "phase=gather_satellite done scene=%s cloud=%.1f%% ndvi=%.4f",
                layer["stac_id"],
                layer["cloud_cover"],
                layer["ndvi_mean"],
            )
        return {
            "layer": layer,
            "phases": [*state.get("phases", []), "gather_satellite"],
        }

    async def _gather_vulnerability(self, state: PipelineState) -> dict[str, Any]:
        """Phase 2: retrieve demographic exposure for the bbox.

        Args:
            state: The pipeline state.

        Returns:
            Partial state carrying either the exposure or an error.
        """
        request = state["request"]
        logger.info("phase=gather_vulnerability start bbox=%s", request.bbox)
        try:
            context = await self._vulnerability.fetch(request.bbox)
        except Exception as exc:  # noqa: BLE001 - a tool failure must not escape
            logger.error(
                "phase=gather_vulnerability failed %s: %s", type(exc).__name__, exc
            )
            return {
                "error": f"gather_vulnerability failed: {type(exc).__name__}: {exc}",
                "phases": [*state.get("phases", []), "gather_vulnerability"],
            }

        logger.info(
            "phase=gather_vulnerability done density=%.1f svi=%.3f",
            context.population_density,
            context.social_vulnerability,
        )
        return {
            "vulnerability": {
                "population_density": context.population_density,
                "social_vulnerability": context.social_vulnerability,
            },
            "phases": [*state.get("phases", []), "gather_vulnerability"],
        }

    async def _assess(self, state: PipelineState) -> dict[str, Any]:
        """Phase 3: compute HEPS from whatever evidence was gathered.

        Args:
            state: The pipeline state.

        Returns:
            Partial state carrying the assessment or an error.
        """
        logger.info("phase=assess start")
        layer = state.get("layer")
        context = VulnerabilityContext(**state["vulnerability"])

        # Translate tool sentinels into explicit "unavailable" so the risk
        # engine renormalises instead of scoring a placeholder as a real 0.
        ndvi = None
        lst_celsius = None
        if layer and layer["stac_id"] != UNAVAILABLE_STAC_ID:
            if layer["ndvi_mean"] != NDVI_NO_DATA:
                ndvi = layer["ndvi_mean"]
            if self._provides_lst:
                lst_celsius = layer["lst_celsius_max"]

        if not self._provides_lst:
            logger.warning(
                "phase=assess land-surface temperature not supplied by the wired "
                "source (Sentinel-2 has no thermal band); heat exposure excluded"
            )

        try:
            assessment = compute_heps(
                lst_celsius=lst_celsius,
                ndvi=ndvi,
                vulnerability=context,
                weights=self._weights,
                thresholds=self._thresholds,
            )
        except RiskComputationError as exc:
            logger.error("phase=assess failed RiskComputationError: %s", exc)
            return {
                "error": f"assess failed: {exc}",
                "phases": [*state.get("phases", []), "assess"],
            }

        for note in assessment.notes:
            logger.warning("phase=assess incomplete: %s", note)
        logger.info(
            "phase=assess done heps=%.4f stress=%s complete=%s",
            assessment.heps_score,
            assessment.stress_level.value,
            assessment.data_complete,
        )
        return {
            "assessment": {
                "heps_score": assessment.heps_score,
                "stress_level": assessment.stress_level.value,
                "audit_status": assessment.recommended_audit_status,
                "data_complete": assessment.data_complete,
                "notes": list(assessment.notes),
                "weights_used": assessment.weights_used,
                "heat_exposure": assessment.factors.heat_exposure,
                "cooling_deficit": assessment.factors.cooling_deficit,
                "vulnerability": assessment.factors.vulnerability,
            },
            "phases": [*state.get("phases", []), "assess"],
        }

    async def _consolidate(self, state: PipelineState) -> dict[str, Any]:
        """Phase 4: assemble the final :class:`RiskAnalysisResult`.

        Always runs, including after a hard failure, so callers get a schema
        valid result describing what went wrong rather than an exception.

        Args:
            state: The pipeline state.

        Returns:
            Partial state carrying the result under the ``result`` key.
        """
        request = state["request"]
        assessment = state.get("assessment")
        error = state.get("error")
        phases = [*state.get("phases", []), "consolidate"]

        if error:
            logger.error("phase=consolidate producing failed result: %s", error)
            result = RiskAnalysisResult(
                heps_score=0.0,
                grid_stress_level=GridStressLevel.LOW.value,
                geojson_features=_empty_collection(request.bbox, error),
                audit_status=AuditStatus.FAILED.value,
            )
            return {"result": result.model_dump(), "phases": phases}

        # _assess always populates this when there is no error.
        assert assessment is not None, "assess must run before consolidate without an error"
        layer = state.get("layer")
        result = RiskAnalysisResult(
            heps_score=assessment["heps_score"],
            grid_stress_level=assessment["stress_level"],
            geojson_features=_feature_collection(request, layer, assessment),
            audit_status=assessment["audit_status"],
        )
        logger.info("phase=consolidate done audit=%s", result.audit_status)
        return {"result": result.model_dump(), "phases": phases}


# ---------------------------------------------------------------------------
# GeoJSON assembly
# ---------------------------------------------------------------------------


def _bbox_polygon(bbox: Sequence[float]) -> dict[str, Any]:
    """Render a bbox as a closed GeoJSON Polygon.

    Args:
        bbox: ``[lon_min, lat_min, lon_max, lat_max]``.

    Returns:
        A GeoJSON Polygon geometry.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    ring = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def _feature_collection(
    request: SpatialQueryRequest,
    layer: dict[str, Any] | None,
    assessment: dict[str, Any],
) -> dict[str, Any]:
    """Build the GeoJSON FeatureCollection describing the verdict.

    Args:
        request: The query the verdict answers.
        layer: Satellite metrics, if any were retrieved.
        assessment: The HEPS payload.

    Returns:
        A GeoJSON FeatureCollection with a single feature covering the bbox.
    """
    properties: dict[str, Any] = {
        "query": request.query,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "heps_score": assessment["heps_score"],
        "grid_stress_level": assessment["stress_level"],
        "data_complete": assessment["data_complete"],
        "notes": assessment["notes"],
        "weights_used": assessment["weights_used"],
        "factors": {
            "heat_exposure": assessment["heat_exposure"],
            "cooling_deficit": assessment["cooling_deficit"],
            "vulnerability": assessment["vulnerability"],
        },
    }
    if layer:
        properties["stac_id"] = layer["stac_id"]
        properties["cloud_cover"] = layer["cloud_cover"]
        properties["ndvi_mean"] = layer["ndvi_mean"]
        properties["lst_celsius_max"] = layer["lst_celsius_max"]

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _bbox_polygon(request.bbox),
                "properties": properties,
            }
        ],
    }


def _empty_collection(bbox: Sequence[float], error: str) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection for a failed run.

    Args:
        bbox: The queried area.
        error: The failure description, recorded as a property.

    Returns:
        A GeoJSON FeatureCollection with one feature carrying the error.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _bbox_polygon(bbox),
                "properties": {"error": error},
            }
        ],
    }
