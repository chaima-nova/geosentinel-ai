"""Application configuration for GeoSentinel-AI.

Settings are read from the process environment, with a local ``.env`` file as
the fallback (see ``.env.example`` at the repository root). Environment
variables win over ``.env`` values, which is what you want in containers.

Note on imports: Pydantic v2 moved ``BaseSettings`` out of the core ``pydantic``
package and into the separate ``pydantic-settings`` distribution, so the class
below comes from ``pydantic_settings``. Both are listed in
``requirements.txt``.

Usage::

    from app.core.config import get_settings

    settings = get_settings()
    print(settings.postgres_url)
    print(settings.openai_api_key.get_secret_value())
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["development", "staging", "production"]

#: Accepted URL schemes for ``POSTGRES_URL``.
_ALLOWED_POSTGRES_SCHEMES = frozenset(
    {
        "postgresql",
        "postgres",
        "postgresql+psycopg2",
        "postgresql+asyncpg",
        "postgresql+psycopg",
    }
)

#: Scheme prefix used when a driver must be stated explicitly.
_PSYCOPG2_SCHEME = "postgresql+psycopg2"


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / ``.env``.

    Field names are lowercase and matched case-insensitively, so
    ``postgres_url`` is filled by the ``POSTGRES_URL`` environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application metadata -------------------------------------------
    app_name: str = Field(default="GeoSentinel-AI", description="Service name.")
    environment: Environment = Field(
        default="development",
        description="Deployment stage; controls log verbosity and guardrails.",
    )
    # NoDecode stops pydantic-settings from JSON-parsing the raw environment
    # value first, so CORS_ORIGINS can be a plain comma-separated string.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("http://localhost:3000", "http://localhost:5173"),
        description=(
            "Browser origins allowed to call the API. Set CORS_ORIGINS to a "
            "comma-separated list to override; leave these defaults for local "
            "frontend development only."
        ),
    )

    # --- Credentials ----------------------------------------------------
    postgres_url: str = Field(
        default="postgresql://geosentinel:geosentinel@localhost:5432/geosentinel",
        description=(
            "DSN for the PostGIS + pgvector database created by docker-compose. "
            "Must carry a postgres:// or postgresql:// scheme."
        ),
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key, used for embeddings and agent inference.",
    )
    copernicus_client_id: str = Field(
        default="",
        description="Client ID for the Copernicus Data Space Ecosystem OAuth2 flow.",
    )
    copernicus_client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Client secret for the Copernicus Data Space Ecosystem OAuth2 flow.",
    )

    # --- Validation -----------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a comma-separated string so ``CORS_ORIGINS`` works from env.

        Args:
            value: A comma-separated string, a sequence, or ``None`` to fall
                back to the default.

        Returns:
            A tuple of origins, with blanks and surrounding whitespace dropped.

        Raises:
            ValueError: If the value is neither a string nor a sequence.
        """
        if value is None:  # unset: keep the declared default
            return cls.model_fields["cors_origins"].default
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        raise ValueError(
            "CORS_ORIGINS must be a comma-separated string or a list, got "
            f"{type(value).__name__}"
        )

    @field_validator("postgres_url", mode="after")
    @classmethod
    def _validate_postgres_url(cls, value: str) -> str:
        """Reject DSNs that cannot describe a PostgreSQL server."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("POSTGRES_URL must not be empty")

        parsed = urlsplit(stripped)
        if parsed.scheme.lower() not in _ALLOWED_POSTGRES_SCHEMES:
            raise ValueError(
                "POSTGRES_URL must start with one of "
                f"{sorted(_ALLOWED_POSTGRES_SCHEMES)}, got '{parsed.scheme}'"
            )
        if not parsed.hostname:
            raise ValueError("POSTGRES_URL must include a host")
        return stripped

    # --- Convenience ----------------------------------------------------
    @property
    def postgres_url_psycopg2(self) -> str:
        """``postgres_url`` with an explicit psycopg2 driver, for SQLAlchemy."""
        parsed = urlsplit(self.postgres_url)
        scheme = parsed.scheme.lower()
        if "+" in scheme:  # driver already pinned by the caller
            return self.postgres_url
        rest = parsed.netloc + parsed.path
        if parsed.query:
            rest = f"{rest}?{parsed.query}"
        if parsed.fragment:
            rest = f"{rest}#{parsed.fragment}"
        return f"{_PSYCOPG2_SCHEME}://{rest}"

    @property
    def has_openai_credentials(self) -> bool:
        """True when an OpenAI key is present."""
        return bool(self.openai_api_key.get_secret_value())

    @property
    def has_copernicus_credentials(self) -> bool:
        """True when both Copernicus OAuth2 values are present."""
        return bool(
            self.copernicus_client_id
            and self.copernicus_client_secret.get_secret_value()
        )

    def missing_secrets(self) -> list[str]:
        """Environment variable names still unset, for a startup health check.

        Useful for a fail-fast guard that names *every* missing credential at
        once instead of raising on the first one.
        """
        missing: list[str] = []
        if not self.has_openai_credentials:
            missing.append("OPENAI_API_KEY")
        if not self.copernicus_client_id:
            missing.append("COPERNICUS_CLIENT_ID")
        if not self.copernicus_client_secret.get_secret_value():
            missing.append("COPERNICUS_CLIENT_SECRET")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Cached so every module shares one instance and the ``.env`` file is parsed
    once. Call ``get_settings.cache_clear()`` in tests after mutating the
    environment.
    """
    return Settings()


#: Module-level singleton for simple ``from app.core.config import settings`` use.
settings: Settings = get_settings()


__all__ = ["Settings", "get_settings", "settings"]
