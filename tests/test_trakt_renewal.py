"""Tests for server-side renewal of Trakt sign-ins."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, cast

import httpx
import pytest

from app.config import Settings
from app.database import Database
from app.db_models import Profile
from app.services.catalog_generator import CatalogService, ManifestConfig
from app.services.metadata_addon import MetadataAddonClient
from app.services.openai import OpenAIClient
from app.services.openrouter import OpenRouterClient
from app.services.trakt import TraktAuthExpired, TraktClient


class _FakeTrakt:
    """Records refresh calls and answers with queued results."""

    def __init__(self, results: list[Any]):
        self.results = results
        self.calls: list[str] = []

    async def refresh_access_token(self, refresh_token, *, client_id=None):
        self.calls.append(refresh_token)
        await asyncio.sleep(0)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _build_service(tmp_path, trakt: _FakeTrakt) -> tuple[CatalogService, Database]:
    settings = Settings(_env_file=None, TRAKT_CLIENT_ID="client")  # type: ignore[call-arg]
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'renewal.db'}")
    service = CatalogService(
        settings,
        cast(TraktClient, trakt),
        cast(OpenRouterClient, object()),
        cast(OpenAIClient, object()),
        cast(MetadataAddonClient, object()),
        database.session_factory,
    )
    return service, database


def _epoch(moment: datetime) -> int:
    """Seconds since the epoch for a naive UTC datetime."""

    return int((moment - datetime(1970, 1, 1)).total_seconds())


def _sign_in_config(access: str, refresh: str, created_at: datetime) -> ManifestConfig:
    return ManifestConfig.model_validate(
        {
            "profileId": "trakt-joe",
            "traktAccessToken": access,
            "traktRefreshToken": refresh,
            "traktTokenCreatedAt": _epoch(created_at),
            "traktTokenExpiresIn": 7 * 24 * 3600,
        }
    )


async def _profile(database: Database) -> Profile:
    async with database.session_factory() as session:
        profile = await session.get(Profile, "trakt-joe")
        assert profile is not None
        return profile


def _run(coro):
    return asyncio.run(coro)


def test_token_renewed_when_close_to_expiry(tmp_path) -> None:
    async def scenario() -> None:
        now = datetime.utcnow()
        trakt = _FakeTrakt(
            [
                {
                    "access_token": "access-2",
                    "refresh_token": "refresh-2",
                    "created_at": _epoch(now),
                    "expires_in": 7 * 24 * 3600,
                }
            ]
        )
        service, database = _build_service(tmp_path, trakt)
        await database.create_all()
        # Signed in 6.5 days ago, so the token is inside the one-day margin.
        context = await service.resolve_profile(
            _sign_in_config("access-1", "refresh-1", now - timedelta(days=6, hours=12))
        )

        # Two callers at once must share one renewal: refresh tokens are single-use.
        first, second = await asyncio.gather(
            service._ensure_fresh_trakt_token(context.state),
            service._ensure_fresh_trakt_token(context.state),
        )

        assert trakt.calls == ["refresh-1"]
        assert first.trakt_access_token == "access-2"
        assert second.trakt_access_token == "access-2"
        stored = await _profile(database)
        assert stored.trakt_refresh_token == "refresh-2"
        assert stored.trakt_token_expires_at > now + timedelta(days=6)
        await database.dispose()

    _run(scenario())


def test_fresh_token_is_not_renewed(tmp_path) -> None:
    async def scenario() -> None:
        trakt = _FakeTrakt([])
        service, database = _build_service(tmp_path, trakt)
        await database.create_all()
        context = await service.resolve_profile(
            _sign_in_config("access-1", "refresh-1", datetime.utcnow())
        )
        state = await service._ensure_fresh_trakt_token(context.state)
        assert trakt.calls == []
        assert state.trakt_access_token == "access-1"
        await database.dispose()

    _run(scenario())


def test_stale_browser_copy_does_not_overwrite_renewed_token(tmp_path) -> None:
    async def scenario() -> None:
        now = datetime.utcnow()
        signed_in_at = now - timedelta(days=6, hours=12)
        trakt = _FakeTrakt(
            [
                {
                    "access_token": "access-2",
                    "refresh_token": "refresh-2",
                    "created_at": _epoch(now),
                    "expires_in": 7 * 24 * 3600,
                }
            ]
        )
        service, database = _build_service(tmp_path, trakt)
        await database.create_all()
        original = _sign_in_config("access-1", "refresh-1", signed_in_at)
        context = await service.resolve_profile(original)
        await service._ensure_fresh_trakt_token(context.state)

        # The config page still holds the first sign-in and sends it again.
        await service.resolve_profile(original)
        stored = await _profile(database)
        assert stored.trakt_access_token == "access-2"
        assert stored.trakt_refresh_token == "refresh-2"

        # A brand new sign-in is newer than the renewal and replaces it.
        await service.resolve_profile(
            _sign_in_config("access-3", "refresh-3", now + timedelta(minutes=5))
        )
        stored = await _profile(database)
        assert stored.trakt_access_token == "access-3"
        assert stored.trakt_refresh_token == "refresh-3"
        await database.dispose()

    _run(scenario())


def test_new_sign_in_adopts_current_client_id(tmp_path) -> None:
    async def scenario() -> None:
        service, database = _build_service(tmp_path, _FakeTrakt([]))
        await database.create_all()
        now = datetime.utcnow()
        await service.resolve_profile(
            _sign_in_config("access-1", "refresh-1", now - timedelta(days=1))
        )

        # The deployment switches to a new Trakt app, then the user signs in again.
        service._settings.trakt_client_id = "new-client"
        await service.resolve_profile(_sign_in_config("access-2", "refresh-2", now))

        stored = await _profile(database)
        assert stored.trakt_client_id == "new-client"
        assert stored.trakt_access_token == "access-2"
        await database.dispose()

    _run(scenario())


def test_rejected_refresh_token_is_dropped(tmp_path) -> None:
    async def scenario() -> None:
        trakt = _FakeTrakt([TraktAuthExpired("invalid_grant")])
        service, database = _build_service(tmp_path, trakt)
        await database.create_all()
        context = await service.resolve_profile(
            _sign_in_config(
                "access-1", "refresh-1", datetime.utcnow() - timedelta(days=8)
            )
        )
        state = await service._ensure_fresh_trakt_token(context.state)
        assert state.trakt_refresh_token is None
        # Nothing left to renew with, so later calls don't hit Trakt again.
        await service._ensure_fresh_trakt_token(state)
        assert trakt.calls == ["refresh-1"]
        await database.dispose()

    _run(scenario())


@pytest.mark.parametrize(
    ("status", "body", "expect_expired"),
    [
        (200, {"access_token": "new", "refresh_token": "r2"}, False),
        (401, {"error": "invalid_grant"}, True),
        (500, {}, False),
    ],
)
def test_trakt_client_refresh_request(status, body, expect_expired) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(status, json=body)

    async def scenario() -> Any:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            TRAKT_CLIENT_ID="client",
            TRAKT_REDIRECT_URI="https://example.test/callback",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.trakt.tv"
        ) as http_client:
            client = TraktClient(settings, http_client)
            return await client.refresh_access_token("r1")

    if expect_expired:
        with pytest.raises(TraktAuthExpired):
            _run(scenario())
    else:
        result = _run(scenario())
        assert result == (body if status == 200 else None)
    assert seen["url"] == "https://auth.trakt.tv/oauth/token"
    assert '"grant_type":"refresh_token"' in seen["body"].replace(" ", "")
    assert '"redirect_uri":"https://example.test/callback"' in seen["body"].replace(" ", "")
    assert "client_secret" not in seen["body"]
