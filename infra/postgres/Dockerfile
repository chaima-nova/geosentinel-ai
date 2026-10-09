# GeoSentinel-AI — PostgreSQL 16 with PostGIS *and* pgvector
#
# No single official image ships both extensions, so we start from the official
# PostGIS image (itself based on postgres:16) and add pgvector from the PGDG
# APT repository, which the base image already has configured and trusted.

FROM postgis/postgis:16-3.4

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends postgresql-16-pgvector; \
    rm -rf /var/lib/apt/lists/*

# Fail the build loudly if either extension is missing from the image.
RUN set -eux; \
    psql --version; \
    ls /usr/share/postgresql/16/extension/ | grep -E '^(postgis|vector)\.control$'
