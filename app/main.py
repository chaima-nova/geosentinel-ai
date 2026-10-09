"""FastAPI application for GeoSentinel-AI.

Exposes the risk-analysis pipeline over HTTP:

* ``POST /api/v1/query`` — run a spatio-temporal query end to end.
* ``GET /health`` — process status and configuration summary.

Run locally::

    uvicorn app.main:app --reload --port 8000

Interactive docs are at ``/docs`` and the OpenAPI schema at ``/openapi.json``.

Notes
-----
CORS is open to the two conventional Vite/Next dev origins only. That is a
development convenience, not a security boundary: set ``CORS_ORIGINS`` before
deploying anywhere reachable.

``/health`` reports *configuration*, not the liveness of external systems. It
never probes the database or the Copernicus API, so a 200 here means "the
process is up and consistently configured" — nothing more. Credentials are
reported as configured-or-not, and the DSN is returned with its password
masked.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.agents.coordinator import CoordinatorAgent
from app.core.config import Settings, get_settings
from app.core.schemas import RiskAnalysisResult, SpatialQueryRequest
from app.services.satellite_service import DEFAULT_STAC_URL

logger = logging.getLogger(__name__)

__all__ = [
    "API_V1_PREFIX",
    "app",
    "create_app",
    "get_coordinator",
    "get_current_settings",
    "mask_password",
]

API_V1_PREFIX = "/api/v1"

#: Origins allowed for local frontend development.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",  # Next.js / CRA
    "http://localhost:5173",  # Vite
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_coordinator(request: Request) -> CoordinatorAgent:
    """Return the process-wide coordinator.

    Resolved from ``app.state`` so tests can replace it wholesale via
    ``app.dependency_overrides`` without touching global state.

    Args:
        request: The inbound request.

    Returns:
        The shared :class:`CoordinatorAgent`.
    """
    return request.app.state.coordinator


CoordinatorDep = Annotated[CoordinatorAgent, Depends(get_coordinator)]


def get_current_settings(request: Request) -> Settings:
    """Return the settings this application was built with.

    Read from ``app.state`` rather than the global ``get_settings()`` singleton
    so an app built by :func:`create_app` with custom settings reports those,
    consistently across CORS, ``/health`` and the lifespan log.

    Args:
        request: The inbound request.

    Returns:
        The :class:`Settings` bound to this app.
    """
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_current_settings)]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Exposed as a factory so tests can construct an isolated app with its own
    configuration. The settings passed here are the ones the routes actually
    use — they are bound into the request dependency, not read from the global
    singleton at call time, so a custom ``settings`` is honoured consistently
    across CORS, ``/health`` and the lifespan log.

    Args:
        settings: Configuration to use. Defaults to the process-wide settings.

    Returns:
        A configured :class:`FastAPI` instance.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Own the coordinator for the life of the process."""
        application.state.settings = resolved
        application.state.coordinator = CoordinatorAgent()
        logger.info(
            "%s v%s starting (environment=%s, cors=%s)",
            resolved.app_name,
            __version__,
            resolved.environment,
            ", ".join(resolved.cors_origins),
        )
        if missing := resolved.missing_secrets():
            logger.warning("credentials not configured: %s", ", ".join(missing))
        try:
            yield
        finally:
            await application.state.coordinator.aclose()
            logger.info("%s shut down", resolved.app_name)

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        description=(
            "Multi-agent climate risk platform. Submit a spatio-temporal query "
            "and receive a Heat Equity Priority Score for the area."
        ),
        lifespan=lifespan,
    )

    # Bound eagerly rather than only in the lifespan, so the settings
    # dependency resolves even before startup completes.
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.get(
        "/health",
        tags=["system"],
        summary="Process status and configuration summary",
        response_description="System status",
    )
    async def health(current: SettingsDep) -> dict[str, Any]:
        """Report process status and configuration.

        Credentials are reported as configured-or-not rather than by value, and
        ``database.url`` is returned with its password masked. No external
        system is probed, so a 200 does not imply the database or the
        Copernicus API is reachable.

        Args:
            current: The settings this app was built with.

        Returns:
            A status document.
        """
        return {
            "status": "ok",
            "app": {
                "name": current.app_name,
                "version": __version__,
                "environment": current.environment,
                "python": sys.version.split()[0],
            },
            "credentials": {
                "configured": not current.missing_secrets(),
                "missing": current.missing_secrets(),
                "openai": current.has_openai_credentials,
                "copernicus": current.has_copernicus_credentials,
            },
            "database": {
                "configured": bool(current.postgres_url),
                "url": mask_password(current.postgres_url),
                "driver": "psycopg2",
                "extensions": ["postgis", "vector"],
            },
            "external_services": {"copernicus_stac": DEFAULT_STAC_URL},
            "cors_origins": list(current.cors_origins),
        }

    @app.post(
        f"{API_V1_PREFIX}/query",
        response_model=RiskAnalysisResult,
        tags=["risk"],
        status_code=status.HTTP_200_OK,
        summary="Run the climate risk pipeline for a spatio-temporal query",
        responses={
            422: {"description": "The request body failed schema validation."},
            500: {"description": "The pipeline raised an unexpected error."},
        },
    )
    async def query(
        request: SpatialQueryRequest, coordinator: CoordinatorDep
    ) -> RiskAnalysisResult:
        """Execute the full pipeline for one query.

        The pipeline degrades rather than fails: an unreachable satellite API or
        an unreadable band still yields a schema-valid result carrying
        ``audit_status`` of ``escalated`` or ``failed``. Callers should read
        that field rather than assume a 200 means a complete score.

        Args:
            request: The validated spatio-temporal query.
            coordinator: The orchestrating agent.

        Returns:
            The consolidated risk result.

        Raises:
            HTTPException: 500 if the pipeline raises unexpectedly.
        """
        logger.info(
            "POST %s/query query=%r bbox=%s window=%s..%s",
            API_V1_PREFIX,
            request.query,
            request.bbox,
            request.start_date,
            request.end_date,
        )
        try:
            result = await coordinator.execute_pipeline(request)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to clients
            logger.exception("pipeline raised for query=%r", request.query)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"risk pipeline failed: {type(exc).__name__}",
            ) from exc

        logger.info(
            "query complete audit=%s stress=%s heps=%.4f",
            result.audit_status,
            result.grid_stress_level,
            result.heps_score,
        )
        return result

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a JSON body for any error that escapes a route.

        Without this, an unhandled exception produces a bare 500 with no JSON
        body, which is awkward for API clients.

        Args:
            request: The request being handled.
            exc: The escaping exception.

        Returns:
            A 500 JSON response with no internal detail.
        """
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    return app


def mask_password(url: str) -> str:
    """Return a DSN with its password replaced by ``***``.

    Args:
        url: The connection string.

    Returns:
        The same URL with the password masked, or the input unchanged when it
        carries no password.
    """
    parts = urlsplit(url)
    if not parts.password:
        return url
    netloc = f"{parts.username or ''}:***@{parts.hostname or ''}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


app = create_app()
