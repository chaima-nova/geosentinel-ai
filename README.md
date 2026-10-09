# GeoSentinel-AI

Multi-agent climate risk platform. Agents ingest satellite and model data
(Copernicus / Sentinel), embed derived climate-risk context, and retrieve it by
**location + time + meaning** from a single PostgreSQL database.

## Repository layout

```
docker-compose.yml        PostgreSQL 16 + PostGIS + pgvector, init script wired in
infra/
  init.sql                Extensions + geo_context_embeddings schema and indexes
  postgres/Dockerfile     PostGIS base image with pgvector installed
app/
  main.py                 FastAPI app: /api/v1/query and /health
  core/config.py          Settings (Pydantic BaseSettings) for all credentials
  core/schemas.py         Pydantic v2 contracts shared by every agent
  services/
    satellite_service.py  Async Sentinel-2 STAC client + NDVI computation
    risk_service.py       Heat Equity Priority Score (HEPS) engine
  agents/
    coordinator.py        LangGraph pipeline orchestrating the above
tests/                    Offline unit suite + gated live-API integration tests
requirements.txt          Python dependencies
requirements-dev.txt      Test dependencies
.env.example              Copy to .env and fill in
```

## Quickstart

```bash
# 1. Python environment (Python >= 3.11 required; see note on rasterio below)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Credentials
cp .env.example .env      # then edit: OPENAI_API_KEY, COPERNICUS_CLIENT_*

# 3. Database — builds the PostGIS+pgvector image and runs infra/init.sql
docker compose up -d --build
docker compose logs -f postgres
```

Verify the schema came up:

```bash
docker compose exec postgres psql -U geosentinel -d geosentinel \
  -c '\d geo_context_embeddings' \
  -c 'SELECT extname, extversion FROM pg_extension;'
```

Read the configuration from Python:

```python
from app.core.config import get_settings

settings = get_settings()
print(settings.postgres_url)                        # DSN from env / .env
print(settings.openai_api_key.get_secret_value())   # SecretStr, never logged
print(settings.missing_secrets())                   # [] when fully configured
```

## Configuration

`app/core/config.py` defines a single `Settings` class (Pydantic
`BaseSettings`). Values come from the process environment first, then `.env`.

| Variable                  | Field                      | Default                                              |
| ------------------------- | -------------------------- | ---------------------------------------------------- |
| `POSTGRES_URL`            | `postgres_url`             | `postgresql://geosentinel:geosentinel@localhost:5432/geosentinel` |
| `OPENAI_API_KEY`          | `openai_api_key`           | empty (`SecretStr`)                                   |
| `COPERNICUS_CLIENT_ID`    | `copernicus_client_id`     | empty                                                 |
| `COPERNICUS_CLIENT_SECRET`| `copernicus_client_secret` | empty (`SecretStr`)                                   |
| `ENVIRONMENT`             | `environment`              | `development`                                         |

`POSTGRES_URL` is validated at construction time: a non-Postgres scheme, a
missing host, or an empty value raises a `ValidationError` naming the field.
Use `settings.postgres_url_psycopg2` when a driver must be explicit for
SQLAlchemy. Credentials are held in `SecretStr`, so they are redacted in
`repr()`, `str()`, `model_dump()` and `model_dump_json()`.

## Data contracts

`app/core/schemas.py` defines the Pydantic v2 models that cross every boundary —
HTTP request, agent-to-agent handoff, and the `metadata` column:

| Model                     | Role                                            |
| ------------------------- | ----------------------------------------------- |
| `SpatialQueryRequest`     | Spatio-temporal semantic search request          |
| `SatelliteLayerMetadata`  | Derived measurements for one acquisition         |
| `RiskAnalysisResult`      | Risk verdict produced by the analysis agent      |

All three use `extra="forbid"`, so a misspelled or unexpected field fails loudly
instead of being dropped. Values are checked against the physical ranges the
satellite products use, which catches unit errors at the boundary:

- `bbox` — exactly 4 values, lon in [-180, 180], lat in [-90, 90], and
  `min < max`. Antimeridian-crossing boxes must be split into two requests.
- `start_date` / `end_date` — ISO 8601 (date or datetime, `Z` or offset); the
  window must run forwards. Naive values are read as UTC so they compare
  safely against offset-aware ones.
- `cloud_cover` 0-100, `ndvi_mean` -1 to 1, `lst_celsius_max` -100 to 100 — the
  last bound exists so a temperature left in Kelvin (~300) is rejected.
- `grid_stress_level` and `audit_status` are validated against the
  `GridStressLevel` / `AuditStatus` enums (case-insensitive in, canonical
  lowercase out). Edit those enums if your vocabulary differs.
- `geojson_features` must be a GeoJSON `FeatureCollection` or `Feature` with the
  members RFC 7946 requires.

Helpers worth knowing: `SpatialQueryRequest.bbox_wkt` returns a closed WKT
`POLYGON` you can pass straight to `ST_GeomFromText`, and `.start_datetime` /
`.end_datetime` give timezone-aware `datetime`s.

Cross-field failures raise `ValidationError`, not a bare `ValueError`, so
FastAPI turns them into a `422` rather than a `500`.

## Satellite retrieval

`app/services/satellite_service.py` provides `SatelliteDataService`, an async
client for the Copernicus Data Space STAC API:

```python
from app.services.satellite_service import SatelliteDataService

async with SatelliteDataService() as service:
    metadata = await service.fetch_sentinel_indices(
        bbox=[5.2, 31.8, 5.5, 32.1],
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
# -> dict that validates as SatelliteLayerMetadata
```

It searches `SENTINEL-2`, discards scenes at or above `max_cloud_cover` (20 % by
default), picks the clearest remaining scene, samples its B04 (red) and B08
(NIR) bands, and returns mean NDVI as `(NIR - RED) / (NIR + RED)`.

**It degrades instead of raising.** An unreachable API, an empty result set, a
missing band, or an unreadable raster all return a valid
`SatelliteLayerMetadata` dictionary rather than an exception, so one dead
upstream cannot take an agent graph down. The sentinel values are:

| Signal                                      | Meaning                        |
| ------------------------------------------- | ------------------------------ |
| `stac_id == UNAVAILABLE_STAC_ID`            | the whole retrieval failed     |
| `ndvi_mean == NDVI_NO_DATA`                 | NDVI could not be computed     |
| `lst_celsius_max == LST_NOT_AVAILABLE`      | always — see below             |

Only malformed caller input (bad bbox, unparseable date) raises, as a
`ValidationError`.

Two caveats worth knowing:

- **Sentinel-2 has no thermal band**, so `lst_celsius_max` can never come from
  this collection and is always a placeholder. Real land-surface temperature
  needs Sentinel-3 SLSTR or Landsat TIRS.
- **Mean NDVI reads pixels.** Bands are decimated to at most
  `max_sample_pixels` (256) using average resampling, so the figure is an
  estimate over a coarsened grid, not a full-resolution statistic. Rasterio is
  blocking, so the read runs via `asyncio.to_thread`.

Both external dependencies are injectable for testing — `transport` for HTTP,
`band_reader` for pixels.

## Risk scoring

`app/services/risk_service.py` computes the **Heat Equity Priority Score**, a
0-1 composite of three lines of evidence blended by `RiskWeights` (which must
sum to 1):

| Component       | Derived from                                    | Default weight |
| --------------- | ----------------------------------------------- | -------------- |
| Heat exposure   | Land-surface temperature, normalised 15-50 °C   | 0.40           |
| Cooling deficit | Inverted NDVI, normalised 0.1-0.8               | 0.35           |
| Equity          | Population density + social vulnerability index | 0.25           |

`StressThresholds` maps the score onto `GridStressLevel`.

**Missing evidence is renormalised, never zero-filled.** This is the part that
matters: because Sentinel-2 has no thermal band, LST is never observed, and
feeding the `0.0` placeholder into the formula would score heat exposure at
zero and *understate* risk everywhere. Instead unavailable components drop out,
the remaining weights are rescaled to sum to 1, and the result reports
`data_complete=False` with an explanation in `notes`. The formula is a
documented, tunable heuristic — not a published standard.

Demographic input comes from a `VulnerabilityProvider`. Only
`StaticVulnerabilityProvider` ships, returning a neutral mid-range baseline —
**no real population dataset is wired up**, so equity weighting is a
placeholder until you supply WorldPop, GHSL or a census source.

## Orchestration

`CoordinatorAgent.execute_pipeline(request)` runs a `langgraph` graph:

```
START -> gather_satellite -> gather_vulnerability -> assess -> consolidate -> END
              |                     |
              +-- on error ---------+-- on error --> consolidate
```

Each phase logs at its own level (`phase=gather_satellite start/done`, warnings
for degraded evidence, errors for tool failures) and contains its own
exceptions, so no tool failure escapes. `audit_status` reports how much
evidence backed the verdict:

| `audit_status` | Meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `passed`       | every component observed (requires a thermal source)        |
| `escalated`    | score is usable but incomplete — needs human review         |
| `failed`       | a phase raised; no score, result describes the failure      |

A degraded satellite tool escalates rather than fails, because an
equity-only HEPS is still useful. A failed vulnerability provider *does* fail
the run: without an equity component it is not a Heat **Equity** score.

```python
from app.agents.coordinator import CoordinatorAgent
from app.core.schemas import SpatialQueryRequest

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
```

`provides_lst` defaults to False for exactly the reason above — flip it only
once a thermal source is wired in.

## REST API

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs at `/docs`, OpenAPI schema at `/openapi.json`.

| Method | Path             | Purpose                                             |
| ------ | ---------------- | --------------------------------------------------- |
| POST   | `/api/v1/query`  | Run the pipeline for a `SpatialQueryRequest`        |
| GET    | `/health`        | Process status and configuration summary            |

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"heat stress in dense housing","bbox":[5.2,31.8,5.5,32.1],
       "start_date":"2024-06-01","end_date":"2024-06-30"}'
```

A malformed body returns `422` with the offending field in `detail[].loc`.
Note that **a 200 does not mean the score is complete** — the pipeline degrades
rather than erroring, so read `audit_status` from the body.

CORS allows `http://localhost:3000` and `http://localhost:5173` for local
frontend work. Override with a comma-separated `CORS_ORIGINS`; the dev defaults
are a convenience, not a security boundary.

`/health` reports configuration, not liveness — it never probes the database or
the Copernicus API. Credentials appear as configured-or-not, and the DSN is
returned with its password masked.

---

## Dashboard (frontend)

`frontend/` is a **React 18 + Vite + TypeScript + Tailwind CSS 4 + react-leaflet**
dashboard that talks to the FastAPI service above.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (Vite proxies /api and /health to :8000)
```

Run the backend first (see above). The Vite dev server proxies `/api` and `/health`
to `http://127.0.0.1:8000`, so the browser never calls `localhost` cross-origin; the
app itself uses relative URLs only.

It renders the reference dashboard:

- **Agent workflow DAG** — Meta-Coordinator → Spatial Ingestion → Risk Reasoning →
  Sentinel Auditor, each with a status light: 🟢 done, 🟡 pulsing (running), 🔴 failed,
  ⚠ degraded, ⚫ idle — connected by animated glowing links that fill as the chain runs.
- **Query panel** (left) — a natural-language climate query, a bounding box
  (`south_west_lng, south_west_lat, north_east_lng, north_east_lat`) and a date
  window, submitted to `POST /api/v1/query`.
- **Metrics header** (top) — HEPS Score, Grid Stress Level, and an Audit Status badge
  (PASSED / ESCALATED / FAILED), plus the weights actually used.
- **Leaflet map** — the returned bbox subdivided into a grid painted with an
  emerald→amber→red→purple gradient driven by the per-cell HEPS.
- **Execution feed** — a timestamped trace of each phase and any degradation notes.

> Map tiles come from CartoDB's public basemap; with no internet in the sandbox the
> tiles stay blank, but the GeoJSON heat overlay still renders.

## Tests

```bash
pip install -r requirements-dev.txt

pytest                            # unit suite, fully offline
pytest --cov=app                  # with coverage
pytest -m integration             # live Copernicus Data Space API (needs egress)
```

The unit suite runs offline and deterministically: HTTP is stubbed with
`httpx.MockTransport` and rasters are real GeoTIFFs written into a temporary
directory, so the rasterio read path is exercised for real. The integration
tests are deselected by default and skip themselves if the API is unreachable.

419 tests, 100 % line coverage across `app/`.

## Database

`geo_context_embeddings` holds one row per geospatial context chunk:

| Column         | Type                    | Purpose                                          |
| -------------- | ----------------------- | ------------------------------------------------ |
| `bounding_box` | `GEOMETRY(Polygon,4326)`| Ground footprint of the observation (WGS84)       |
| `timestamp`    | `TIMESTAMPTZ`           | Observation / acquisition time                    |
| `embedding`    | `vector(1536)`          | `text-embedding-3-small` output width             |
| `metadata`     | `JSONB`                 | Provenance: source product, sensor, agent, scores |

Indexed for the three access patterns the platform actually uses: GiST for
spatial filtering, a descending btree for recency, GIN for JSONB containment,
and HNSW (`vector_cosine_ops`) for approximate nearest-neighbour search.

> `infra/init.sql` runs **only** against an empty data volume. After editing it,
> run `docker compose down -v` to re-initialise.

## Notes

- **No single official image ships both extensions.** `infra/postgres/Dockerfile`
  starts from `postgis/postgis:16-3.4` and installs `postgresql-16-pgvector`
  from the PGDG APT repo the base image already trusts. If you do not need
  PostGIS, swap the `build:` block for `image: pgvector/pgvector:pg16`.
- **Python 3.11 vs 3.12.** `geopandas` 1.2 requires Python >= 3.11, while
  `rasterio` 1.5 requires >= 3.12. `requirements.txt` uses environment markers
  to install `rasterio` 1.4.x on 3.11 and 1.5.x on 3.12+.
- `pydantic-settings` is a separate distribution in Pydantic v2 — it is where
  `BaseSettings` lives, so it is pinned alongside `pydantic`.
