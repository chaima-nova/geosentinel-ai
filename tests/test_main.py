"""Tests for the FastAPI application, :mod:`app.main`.

Every test uses :class:`fastapi.testclient.TestClient`, so requests go through
the real ASGI stack — routing, validation, CORS middleware and the lifespan —
with only the :class:`CoordinatorAgent` replaced. No network is touched.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.core.schemas import RiskAnalysisResult
from app.main import API_V1_PREFIX, app as production_app, create_app, mask_password

QUERY_URL = f"{API_V1_PREFIX}/query"

VALID_BODY: dict[str, Any] = {
    "query": "heat stress in dense housing",
    "bbox": [5.2, 31.8, 5.5, 32.1],
    "start_date": "2024-06-01",
    "end_date": "2024-06-30",
}


def make_result(**overrides: Any) -> RiskAnalysisResult:
    """A valid pipeline result."""
    payload: dict[str, Any] = {
        "heps_score": 0.6667,
        "grid_stress_level": "high",
        "audit_status": "escalated",
        "geojson_features": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[5.2, 31.8], [5.5, 31.8], [5.5, 32.1], [5.2, 32.1], [5.2, 31.8]]
                        ],
                    },
                    "properties": {"heps_score": 0.6667},
                }
            ],
        },
    }
    payload.update(overrides)
    return RiskAnalysisResult(**payload)


class StubCoordinator:
    """CoordinatorAgent stand-in returning a canned result."""

    def __init__(self, result: RiskAnalysisResult | None = None, exc: Exception | None = None) -> None:
        self.result = result or make_result()
        self.exc = exc
        self.calls: list[Any] = []
        self.closed = False

    async def execute_pipeline(self, request: Any) -> RiskAnalysisResult:
        self.calls.append(request)
        if self.exc is not None:
            raise self.exc
        return self.result

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def stub() -> StubCoordinator:
    """A coordinator stub whose calls are recorded."""
    return StubCoordinator()


@pytest.fixture
def client(stub: StubCoordinator) -> Any:
    """A TestClient with the coordinator dependency overridden."""
    from app.main import get_coordinator

    application = create_app()
    application.dependency_overrides[get_coordinator] = lambda: stub
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/v1/query — success
# ---------------------------------------------------------------------------


class TestQuerySuccess:
    """The 200 path."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.post(QUERY_URL, json=VALID_BODY)
        assert response.status_code == 200

    def test_body_is_the_pipeline_result(self, client: TestClient) -> None:
        body = client.post(QUERY_URL, json=VALID_BODY).json()
        assert body["heps_score"] == pytest.approx(0.6667)
        assert body["grid_stress_level"] == "high"
        assert body["audit_status"] == "escalated"
        assert body["geojson_features"]["type"] == "FeatureCollection"

    def test_body_validates_as_risk_analysis_result(self, client: TestClient) -> None:
        body = client.post(QUERY_URL, json=VALID_BODY).json()
        assert isinstance(RiskAnalysisResult.model_validate(body), RiskAnalysisResult)

    def test_request_reaches_the_coordinator(self, client: TestClient, stub: StubCoordinator) -> None:
        client.post(QUERY_URL, json=VALID_BODY)
        assert len(stub.calls) == 1
        forwarded = stub.calls[0]
        assert forwarded.query == VALID_BODY["query"]
        assert forwarded.bbox == VALID_BODY["bbox"]
        assert forwarded.start_date == "2024-06-01"
        assert forwarded.end_date == "2024-06-30"

    def test_content_type_is_json(self, client: TestClient) -> None:
        response = client.post(QUERY_URL, json=VALID_BODY)
        assert response.headers["content-type"] == "application/json"

    @pytest.mark.parametrize("audit", ["passed", "escalated", "failed"])
    def test_every_audit_status_is_a_200(
        self, client: TestClient, stub: StubCoordinator, audit: str
    ) -> None:
        """A degraded pipeline is not an HTTP error; the status is in the body."""
        stub.result = make_result(audit_status=audit)
        response = client.post(QUERY_URL, json=VALID_BODY)
        assert response.status_code == 200
        assert response.json()["audit_status"] == audit

    def test_integer_bbox_is_accepted_and_coerced(self, client: TestClient, stub: StubCoordinator) -> None:
        response = client.post(QUERY_URL, json={**VALID_BODY, "bbox": [0, 0, 1, 1]})
        assert response.status_code == 200
        assert all(isinstance(v, float) for v in stub.calls[0].bbox)

    def test_iso_datetime_window_is_accepted(self, client: TestClient) -> None:
        response = client.post(
            QUERY_URL,
            json={**VALID_BODY, "start_date": "2024-06-01T00:00:00Z", "end_date": "2024-06-30T23:59:59Z"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/query — validation errors
# ---------------------------------------------------------------------------


class TestQueryValidationErrors:
    """The 422 path, produced by the schema rather than by the endpoint."""

    @pytest.mark.parametrize(
        "body, expected_loc",
        [
            ({**VALID_BODY, "bbox": [1.0, 2.0, 3.0]}, ["body", "bbox"]),
            ({**VALID_BODY, "bbox": [1.0, 2.0, 3.0, 4.0, 5.0]}, ["body", "bbox"]),
            ({**VALID_BODY, "bbox": [-200.0, 0.0, 10.0, 10.0]}, ["body", "bbox"]),
            ({**VALID_BODY, "bbox": [0.0, 0.0, 10.0, 95.0]}, ["body", "bbox"]),
            ({**VALID_BODY, "bbox": [10.0, 0.0, 5.0, 10.0]}, ["body", "bbox"]),
            ({**VALID_BODY, "query": ""}, ["body", "query"]),
            ({**VALID_BODY, "query": "   "}, ["body", "query"]),
            ({**VALID_BODY, "start_date": "01/06/2024"}, ["body", "start_date"]),
            ({**VALID_BODY, "end_date": "not a date"}, ["body", "end_date"]),
            ({**VALID_BODY, "region": "eu"}, ["body", "region"]),
            ({**VALID_BODY, "bbox": "not a list"}, ["body", "bbox"]),
        ],
    )
    def test_returns_422_with_the_offending_field(
        self, client: TestClient, body: dict, expected_loc: list[str]
    ) -> None:
        response = client.post(QUERY_URL, json=body)
        assert response.status_code == 422
        locations = [e["loc"] for e in response.json()["detail"]]
        assert expected_loc in locations

    def test_backwards_window_is_422_not_500(self, client: TestClient) -> None:
        """Cross-field validation must surface as a client error."""
        response = client.post(
            QUERY_URL, json={**VALID_BODY, "start_date": "2024-06-30", "end_date": "2024-06-01"}
        )
        assert response.status_code == 422
        assert "must not be after" in response.json()["detail"][0]["msg"]

    @pytest.mark.parametrize("missing", ["query", "bbox", "start_date", "end_date"])
    def test_missing_required_field_is_422(self, client: TestClient, missing: str) -> None:
        body = {k: v for k, v in VALID_BODY.items() if k != missing}
        response = client.post(QUERY_URL, json=body)
        assert response.status_code == 422
        assert ["body", missing] in [e["loc"] for e in response.json()["detail"]]

    def test_empty_body_is_422(self, client: TestClient) -> None:
        response = client.post(QUERY_URL, json={})
        assert response.status_code == 422
        assert len(response.json()["detail"]) == 4

    def test_non_json_body_is_422(self, client: TestClient) -> None:
        response = client.post(QUERY_URL, content="not json", headers={"content-type": "application/json"})
        assert response.status_code == 422

    def test_no_body_at_all_is_422(self, client: TestClient) -> None:
        response = client.post(QUERY_URL)
        assert response.status_code == 422

    def test_422_detail_is_a_structured_list(self, client: TestClient) -> None:
        detail = client.post(QUERY_URL, json={**VALID_BODY, "bbox": [1, 2, 3]}).json()["detail"]
        assert isinstance(detail, list)
        assert {"loc", "msg", "type"} <= set(detail[0])

    def test_422_does_not_invoke_the_coordinator(
        self, client: TestClient, stub: StubCoordinator
    ) -> None:
        client.post(QUERY_URL, json={**VALID_BODY, "bbox": [1.0, 2.0, 3.0]})
        client.post(QUERY_URL, json={})
        client.post(QUERY_URL, json={**VALID_BODY, "start_date": "2024-06-30", "end_date": "2024-06-01"})
        assert stub.calls == []


# ---------------------------------------------------------------------------
# POST /api/v1/query — failures
# ---------------------------------------------------------------------------


class TestQueryFailures:
    """What happens when the pipeline itself breaks."""

    def test_unexpected_pipeline_error_is_500(self, client: TestClient, stub: StubCoordinator) -> None:
        stub.exc = RuntimeError("tool exploded")
        response = client.post(QUERY_URL, json=VALID_BODY)
        assert response.status_code == 500

    def test_500_does_not_leak_the_exception_message(
        self, client: TestClient, stub: StubCoordinator
    ) -> None:
        stub.exc = RuntimeError("internal secret path /var/keys")
        body = client.post(QUERY_URL, json=VALID_BODY).text
        assert "tool exploded" not in body
        assert "/var/keys" not in body

    def test_500_body_is_json(self, client: TestClient, stub: StubCoordinator) -> None:
        stub.exc = ValueError("bad")
        response = client.post(QUERY_URL, json=VALID_BODY)
        assert response.headers["content-type"] == "application/json"
        assert "detail" in response.json()

    def test_get_on_the_query_path_is_405(self, client: TestClient) -> None:
        assert client.get(QUERY_URL).status_code == 405


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    """Status and configuration reporting."""

    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_reports_ok(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_reports_app_metadata(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["app"]["name"] == "GeoSentinel-AI"
        assert body["app"]["version"]
        assert body["app"]["environment"] == "development"

    def test_reports_cors_origins(self, client: TestClient) -> None:
        assert client.get("/health").json()["cors_origins"] == [
            "http://localhost:3000",
            "http://localhost:5173",
        ]

    def test_reports_database_configuration(self, client: TestClient) -> None:
        database = client.get("/health").json()["database"]
        assert database["configured"] is True
        assert database["extensions"] == ["postgis", "vector"]

    def test_reports_missing_credentials(self) -> None:
        settings = Settings(
            postgres_url="postgresql://u:p@localhost:5432/db",
            openai_api_key=SecretStr(""),
            copernicus_client_id="",
            copernicus_client_secret=SecretStr(""),
        )
        application = create_app(settings)
        with TestClient(application) as test_client:
            body = test_client.get("/health").json()
        assert body["credentials"]["configured"] is False
        assert body["credentials"]["missing"] == [
            "OPENAI_API_KEY",
            "COPERNICUS_CLIENT_ID",
            "COPERNICUS_CLIENT_SECRET",
        ]
        assert body["credentials"]["openai"] is False
        assert body["credentials"]["copernicus"] is False

    def test_credentials_reported_as_configured_when_present(self) -> None:
        settings = Settings(
            postgres_url="postgresql://u:p@localhost:5432/db",
            openai_api_key=SecretStr("sk-real"),
            copernicus_client_id="cid",
            copernicus_client_secret=SecretStr("csecret"),
        )
        with TestClient(create_app(settings)) as test_client:
            body = test_client.get("/health").json()
        assert body["credentials"]["configured"] is True
        assert body["credentials"]["missing"] == []
        assert body["credentials"]["openai"] is True
        assert body["credentials"]["copernicus"] is True

    def test_never_leaks_secret_values(self) -> None:
        settings = Settings(
            postgres_url="postgresql://geosentinel:supersecretpw@localhost:5432/geosentinel",
            openai_api_key=SecretStr("sk-live-abcdef123456"),
            copernicus_client_id="client-id-xyz",
            copernicus_client_secret=SecretStr("client-secret-xyz"),
        )
        with TestClient(create_app(settings)) as test_client:
            body = test_client.get("/health").text
        assert "sk-live-abcdef123456" not in body
        assert "client-secret-xyz" not in body
        assert "supersecretpw" not in body

    def test_database_password_is_masked(self) -> None:
        settings = Settings(postgres_url="postgresql://geosentinel:supersecretpw@dbhost:5432/geosentinel")
        with TestClient(create_app(settings)) as test_client:
            url = test_client.get("/health").json()["database"]["url"]
        assert url == "postgresql://geosentinel:***@dbhost:5432/geosentinel"

    def test_health_does_not_require_a_body(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"


class TestMaskPassword:
    """The DSN redaction helper."""

    def test_masks_the_password(self) -> None:
        assert mask_password("postgresql://u:pw@h:5432/db") == "postgresql://u:***@h:5432/db"

    def test_leaves_a_passwordless_dsn_alone(self) -> None:
        assert mask_password("postgresql://h:5432/db") == "postgresql://h:5432/db"

    def test_preserves_query_and_fragment(self) -> None:
        assert (
            mask_password("postgresql://u:pw@h/db?sslmode=require#frag")
            == "postgresql://u:***@h/db?sslmode=require#frag"
        )

    def test_handles_a_missing_port(self) -> None:
        assert mask_password("postgresql://u:pw@h/db") == "postgresql://u:***@h/db"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCors:
    """Cross-origin behaviour for local frontend development."""

    @pytest.mark.parametrize(
        "origin", ["http://localhost:3000", "http://localhost:5173"]
    )
    def test_preflight_is_allowed_for_dev_origins(self, client: TestClient, origin: str) -> None:
        response = client.options(
            QUERY_URL,
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    @pytest.mark.parametrize(
        "origin", ["http://localhost:3000", "http://localhost:5173"]
    )
    def test_simple_request_carries_the_origin_header(self, client: TestClient, origin: str) -> None:
        response = client.post(QUERY_URL, json=VALID_BODY, headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    def test_credentials_are_allowed(self, client: TestClient) -> None:
        response = client.post(
            QUERY_URL, json=VALID_BODY, headers={"Origin": "http://localhost:3000"}
        )
        assert response.headers["access-control-allow-credentials"] == "true"

    @pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:9999", "null"])
    def test_unknown_origin_gets_no_cors_header(self, client: TestClient, origin: str) -> None:
        response = client.post(QUERY_URL, json=VALID_BODY, headers={"Origin": origin})
        assert "access-control-allow-origin" not in response.headers

    def test_preflight_for_an_unknown_origin_is_rejected(self, client: TestClient) -> None:
        response = client.options(
            QUERY_URL,
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert response.status_code == 400

    def test_origins_are_configurable(self) -> None:
        settings = Settings(cors_origins="https://app.example.com")
        with TestClient(create_app(settings)) as test_client:
            allowed = test_client.post(
                QUERY_URL, json=VALID_BODY, headers={"Origin": "https://app.example.com"}
            )
            blocked = test_client.post(
                QUERY_URL, json=VALID_BODY, headers={"Origin": "http://localhost:3000"}
            )
        assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
        assert "access-control-allow-origin" not in blocked.headers

    def test_health_is_also_cors_enabled(self, client: TestClient) -> None:
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# ---------------------------------------------------------------------------
# Application wiring
# ---------------------------------------------------------------------------


class TestApplicationWiring:
    """App construction, docs and lifecycle."""

    def test_create_app_returns_a_configured_app(self) -> None:
        application = create_app()
        assert isinstance(application, FastAPI)
        assert application.title == "GeoSentinel-AI"

    def test_module_level_app_exists(self) -> None:
        assert isinstance(production_app, FastAPI)

    def test_required_routes_are_registered(self, client: TestClient) -> None:
        paths = {route.path for route in client.app.routes if hasattr(route, "path")}
        assert {"/health", QUERY_URL} <= paths

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert QUERY_URL in schema["paths"]
        assert "/health" in schema["paths"]
        assert "RiskAnalysisResult" in schema["components"]["schemas"]
        assert "SpatialQueryRequest" in schema["components"]["schemas"]

    def test_openapi_documents_the_422_response(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "422" in schema["paths"][QUERY_URL]["post"]["responses"]

    def test_docs_are_served(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_dependency_override_replaces_the_real_coordinator(self) -> None:
        """The override must serve requests while the app still owns its own."""
        from app.main import get_coordinator

        stub = StubCoordinator()
        application = create_app()
        application.dependency_overrides[get_coordinator] = lambda: stub
        with TestClient(application) as test_client:
            assert test_client.post(QUERY_URL, json=VALID_BODY).status_code == 200
            own = application.state.coordinator

        assert stub.calls, "the override should have served the request"
        assert own is not stub, "app.state must hold the app's own coordinator"
        assert stub.closed is False, "the app must not close an injected stub"

    def test_real_coordinator_is_closed_on_shutdown(self) -> None:
        """Without an override, the app builds and tears down its own agent."""
        application = create_app()
        with TestClient(application) as test_client:
            coordinator = application.state.coordinator
            assert test_client.get("/health").status_code == 200
        assert coordinator._satellite._client is None  # released by aclose()

    def test_factory_isolates_apps(self) -> None:
        first = create_app()
        second = create_app()
        assert first is not second

class TestUnhandledExceptionHandler:
    """The JSON safety net for errors that escape every route.

    ``TestClient`` re-raises server exceptions by default, which would bypass
    the handler entirely, so these tests opt out of that behaviour.
    """

    def test_returns_a_json_500(self) -> None:
        application = create_app()

        @application.get("/boom")
        async def boom() -> None:
            raise RuntimeError("explode")

        with TestClient(application, raise_server_exceptions=False) as test_client:
            response = test_client.get("/boom")

        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"detail": "internal server error"}

    def test_does_not_leak_the_exception(self) -> None:
        application = create_app()

        @application.get("/boom")
        async def boom() -> None:
            raise RuntimeError("secret detail /var/keys")

        with TestClient(application, raise_server_exceptions=False) as test_client:
            body = test_client.get("/boom").text
        assert "secret detail" not in body
        assert "/var/keys" not in body

    def test_logs_the_failure(self, caplog) -> None:
        import logging

        application = create_app()

        @application.get("/boom")
        async def boom() -> None:
            raise RuntimeError("explode")

        with caplog.at_level(logging.ERROR, logger="app.main"):
            with TestClient(application, raise_server_exceptions=False) as test_client:
                test_client.get("/boom")
        assert any("unhandled error on GET /boom" in r.message for r in caplog.records)
