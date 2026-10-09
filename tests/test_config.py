"""Tests for :mod:`app.core.config`.

Ported from the throwaway verification script used to build the module, so the
behaviour is pinned permanently rather than only at authoring time.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings, get_settings

CREDS = ("POSTGRES_URL", "OPENAI_API_KEY", "COPERNICUS_CLIENT_ID", "COPERNICUS_CLIENT_SECRET", "ENVIRONMENT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Strip GeoSentinel variables and run in an empty directory.

    ``Settings`` reads ``.env`` relative to the working directory, so tests must
    not accidentally pick up a developer's real ``.env``.
    """
    for name in CREDS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


class TestDefaults:
    """Values used when nothing is configured."""

    def test_default_dsn_matches_compose(self) -> None:
        assert (
            Settings().postgres_url
            == "postgresql://geosentinel:geosentinel@localhost:5432/geosentinel"
        )

    def test_default_environment_is_development(self) -> None:
        assert Settings().environment == "development"

    def test_credentials_default_to_empty(self) -> None:
        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == ""
        assert settings.copernicus_client_id == ""
        assert settings.copernicus_client_secret.get_secret_value() == ""

    def test_missing_secrets_lists_everything(self) -> None:
        assert Settings().missing_secrets() == [
            "OPENAI_API_KEY",
            "COPERNICUS_CLIENT_ID",
            "COPERNICUS_CLIENT_SECRET",
        ]

    def test_credential_flags_are_false(self) -> None:
        settings = Settings()
        assert settings.has_openai_credentials is False
        assert settings.has_copernicus_credentials is False

    def test_psycopg2_url_injects_driver(self) -> None:
        assert (
            Settings().postgres_url_psycopg2
            == "postgresql+psycopg2://geosentinel:geosentinel@localhost:5432/geosentinel"
        )


class TestEnvironmentLoading:
    """UPPERCASE environment variables map onto lowercase fields."""

    @pytest.fixture
    def full(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        values = {
            "POSTGRES_URL": "postgresql://u:p@db.example:6543/gs?sslmode=require",
            "OPENAI_API_KEY": "sk-test-123",
            "COPERNICUS_CLIENT_ID": "cop-id-abc",
            "COPERNICUS_CLIENT_SECRET": "cop-secret-xyz",
            "ENVIRONMENT": "production",
        }
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return values

    def test_every_field_is_populated(self, full) -> None:
        settings = Settings()
        assert settings.postgres_url == full["POSTGRES_URL"]
        assert settings.openai_api_key.get_secret_value() == "sk-test-123"
        assert settings.copernicus_client_id == "cop-id-abc"
        assert settings.copernicus_client_secret.get_secret_value() == "cop-secret-xyz"
        assert settings.environment == "production"

    def test_nothing_is_missing(self, full) -> None:
        settings = Settings()
        assert settings.missing_secrets() == []
        assert settings.has_openai_credentials is True
        assert settings.has_copernicus_credentials is True

    def test_query_string_survives_driver_injection(self, full) -> None:
        assert Settings().postgres_url_psycopg2.endswith("/gs?sslmode=require")

    def test_explicit_driver_is_preserved(self) -> None:
        assert (
            Settings(postgres_url="postgresql+asyncpg://u:p@h/d").postgres_url_psycopg2
            == "postgresql+asyncpg://u:p@h/d"
        )

    def test_fragment_is_preserved(self) -> None:
        url = "postgresql://u:p@h/d#section"
        assert Settings(postgres_url=url).postgres_url_psycopg2 == f"postgresql+psycopg2://u:p@h/d#section"


class TestDotEnvFile:
    """``.env`` is a fallback; the process environment wins."""

    @pytest.fixture
    def dotenv(self, tmp_path) -> None:
        (tmp_path / ".env").write_text(
            "POSTGRES_URL=postgresql://file:file@filehost:5432/filedb\n"
            "OPENAI_API_KEY=sk-from-dotenv\n"
            "COPERNICUS_CLIENT_ID=cop-from-dotenv\n"
            "COPERNICUS_CLIENT_SECRET=sec-from-dotenv\n"
            "ENVIRONMENT=staging\n"
            "SOMETHING_UNRELATED=ignored\n"
        )

    def test_dotenv_is_read(self, dotenv) -> None:
        settings = Settings()
        assert settings.postgres_url == "postgresql://file:file@filehost:5432/filedb"
        assert settings.openai_api_key.get_secret_value() == "sk-from-dotenv"
        assert settings.environment == "staging"

    def test_unknown_keys_are_ignored(self, dotenv) -> None:
        assert Settings().missing_secrets() == []

    def test_process_environment_wins(self, dotenv, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-process-env")
        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == "sk-from-process-env"
        # ...while the rest still comes from the file.
        assert settings.postgres_url == "postgresql://file:file@filehost:5432/filedb"


class TestSecretHandling:
    """Credentials must never leak through serialisation."""

    @pytest.fixture
    def secreted(self, monkeypatch: pytest.MonkeyPatch) -> Settings:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "cop-secret-xyz")
        return Settings()

    @pytest.mark.parametrize("serialise", [repr, str])
    def test_not_leaked_by_repr_or_str(self, secreted, serialise) -> None:
        blob = serialise(secreted)
        assert "sk-test-123" not in blob
        assert "cop-secret-xyz" not in blob

    @pytest.mark.parametrize("serialise", [lambda s: s.model_dump_json(), lambda s: str(s.model_dump())])
    def test_not_leaked_by_model_dump(self, secreted, serialise) -> None:
        blob = serialise(secreted)
        assert "sk-test-123" not in blob
        assert "cop-secret-xyz" not in blob

    def test_repr_masks_with_secretstr(self, secreted) -> None:
        assert "SecretStr" in repr(secreted)

    def test_value_is_still_retrievable(self, secreted) -> None:
        assert secreted.openai_api_key.get_secret_value() == "sk-test-123"


class TestPostgresUrlValidation:
    """Malformed DSNs are rejected at construction time."""

    @pytest.mark.parametrize(
        "url, fragment",
        [
            ("mysql://u:p@h/db", "must start with one of"),
            ("", "must not be empty"),
            ("   ", "must not be empty"),
            ("postgresql:///nodb", "must include a host"),
            ("not-a-url", "must start with one of"),
        ],
    )
    def test_rejected_with_a_clear_message(self, url: str, fragment: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Settings(postgres_url=url)
        errors = excinfo.value.errors()
        assert errors[0]["loc"] == ("postgres_url",)
        assert fragment in errors[0]["msg"]

    def test_rejection_produces_no_spurious_errors(self) -> None:
        """A single bad field must not cascade into phantom ones.

        Guards the regression where ``default_factory=SecretStr`` made every
        unrelated failure also report ``default_factory_not_called``.
        """
        with pytest.raises(ValidationError) as excinfo:
            Settings(postgres_url="mysql://u:p@h/db")
        assert [e["type"] for e in excinfo.value.errors()] == ["value_error"]

    @pytest.mark.parametrize(
        "scheme",
        ["postgresql", "postgres", "postgresql+psycopg2", "postgresql+asyncpg", "postgresql+psycopg"],
    )
    def test_accepted_schemes(self, scheme: str) -> None:
        assert Settings(postgres_url=f"{scheme}://u:p@h/d").postgres_url.startswith(scheme)

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert Settings(postgres_url="  postgresql://u:p@h/d  ").postgres_url == "postgresql://u:p@h/d"


class TestSingleton:
    """``get_settings`` is cached so ``.env`` is parsed once."""

    def test_returns_the_same_instance(self) -> None:
        assert get_settings() is get_settings()

    def test_cache_clear_picks_up_new_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert get_settings().environment == "development"
        monkeypatch.setenv("ENVIRONMENT", "production")
        assert get_settings().environment == "development"  # still cached
        get_settings.cache_clear()
        assert get_settings().environment == "production"

    def test_secret_str_is_usable_as_a_default(self) -> None:
        """Guards the ``default=SecretStr('')`` choice explicitly."""
        assert isinstance(Settings().openai_api_key, SecretStr)

class TestCorsOrigins:
    """CORS_ORIGINS accepts a comma-separated string from the environment."""

    def test_default_covers_the_dev_frontend_origins(self) -> None:
        assert Settings().cors_origins == ("http://localhost:3000", "http://localhost:5173")

    def test_comma_separated_string_is_split(self) -> None:
        assert Settings(cors_origins="https://a.test,https://b.test").cors_origins == (
            "https://a.test",
            "https://b.test",
        )

    def test_whitespace_and_blanks_are_dropped(self) -> None:
        assert Settings(cors_origins=" https://a.test , , https://b.test ,").cors_origins == (
            "https://a.test",
            "https://b.test",
        )

    def test_a_list_passes_through(self) -> None:
        assert Settings(cors_origins=["https://c.test"]).cors_origins == ("https://c.test",)

    def test_none_falls_back_to_the_default(self) -> None:
        assert Settings(cors_origins=None).cors_origins == Settings().cors_origins

    def test_empty_string_yields_no_origins(self) -> None:
        assert Settings(cors_origins="").cors_origins == ()

    def test_wrong_type_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValidationError, match="comma-separated string or a list"):
            Settings(cors_origins=123)

    def test_loaded_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "https://env.test")
        get_settings.cache_clear()
        assert Settings().cors_origins == ("https://env.test",)
