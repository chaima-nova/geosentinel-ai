-- GeoSentinel-AI — database bootstrap
--
-- Mounted into the container at /docker-entrypoint-initdb.d/, so this runs
-- ONCE, on first start against an empty data directory. To re-apply after
-- changing it:  docker compose down -v && docker compose up -d
--
-- Requires PostGIS and pgvector to be present in the image
-- (see infra/postgres/Dockerfile).

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- geo_context_embeddings
--
-- One row per geospatial context chunk: the ground footprint it covers, when
-- it was observed, the embedding of its derived climate-risk description, and
-- free-form provenance (source product, band, sensor, agent that produced it).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS geo_context_embeddings (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Ground footprint of the observation, WGS84 lon/lat (EPSG:4326).
    bounding_box  GEOMETRY(Polygon, 4326) NOT NULL,

    -- Observation / acquisition time. TIMESTAMPTZ (not naive TIMESTAMP) so
    -- multi-source satellite data compares correctly across time zones.
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 1536 dims = OpenAI text-embedding-3-small output width.
    embedding     vector(1536),

    -- Free-form provenance and derived attributes.
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Reject rows that cannot be placed in space or time.
    CONSTRAINT geo_context_embeddings_bbox_is_polygon
        CHECK (GeometryType(bounding_box) = 'POLYGON'),
    CONSTRAINT geo_context_embeddings_bbox_is_valid
        CHECK (ST_IsValid(bounding_box))
);

COMMENT ON TABLE  geo_context_embeddings              IS 'Geospatial context chunks with vector embeddings for climate-risk retrieval.';
COMMENT ON COLUMN geo_context_embeddings.bounding_box IS 'Ground footprint of the observation, EPSG:4326.';
COMMENT ON COLUMN geo_context_embeddings.timestamp    IS 'Observation / acquisition time of the source data.';
COMMENT ON COLUMN geo_context_embeddings.embedding    IS '1536-dim embedding of the derived climate-risk description.';
COMMENT ON COLUMN geo_context_embeddings.metadata     IS 'Provenance and derived attributes (source product, sensor, agent, scores).';

-- --- Indexes ---------------------------------------------------------------

-- Spatial: bbox intersection / containment queries (ST_Intersects, ST_Within).
CREATE INDEX IF NOT EXISTS idx_geo_context_embeddings_bbox
    ON geo_context_embeddings USING GIST (bounding_box);

-- Temporal recency ordering, the usual companion to a spatial filter.
CREATE INDEX IF NOT EXISTS idx_geo_context_embeddings_timestamp
    ON geo_context_embeddings (timestamp DESC);

-- JSONB containment lookups on provenance keys (metadata @> '{"source": "..."}').
CREATE INDEX IF NOT EXISTS idx_geo_context_embeddings_metadata
    ON geo_context_embeddings USING GIN (metadata jsonb_path_ops);

-- Approximate nearest neighbour search. HNSW needs pgvector >= 0.5 and
-- supports up to 2000 dimensions, so 1536 fits. Cosine ops match the
-- normalised embeddings returned by OpenAI.
CREATE INDEX IF NOT EXISTS idx_geo_context_embeddings_embedding_hnsw
    ON geo_context_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Example hybrid query (spatial + temporal filter, then semantic ranking):
--
--   SELECT id, metadata,
--          1 - (embedding <=> $3) AS similarity
--     FROM geo_context_embeddings
--    WHERE bounding_box && ST_MakeEnvelope($4, $5, $6, $7, 4326)
--      AND timestamp >= now() - interval '90 days'
--    ORDER BY embedding <=> $3
--    LIMIT 10;
