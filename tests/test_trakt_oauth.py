"""Tests for the Trakt device-code sign-in endpoints."""

from __future__ import annotations

from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import settings
from app.main import CatalogService, create_app


@contextmanager
def _test_client(monkeypatch, responses):
    """Yield a client whose Trakt calls answer from ``responses`` by path."""

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(settings, "trakt_client_id", "client")

    async def _noop_start(self):  # type: ignore[override]
        return None

    async def _fake_post(path, payload):
        calls.append((path, payload))
        status, body = responses[path].pop(0)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(CatalogService, "start", _noop_start)
    monkeypatch.setattr(main_module, "_post_trakt_oauth", _fake_post)

    app = create_app()
    with TestClient(app) as client:
        client.calls = calls  # type: ignore[attr-defined]
        yield client


_DEVICE_CODE = (
    200,
    {
        "device_code": "device-secret",
        "user_code": "ABCD1234",
        "verification_url": "https://trakt.tv/activate",
        "expires_in": 600,
        "interval": 5,
    },
)


def test_device_code_hides_device_code_from_browser(monkeypatch) -> None:
    with _test_client(monkeypatch, {"/oauth/device/code": [_DEVICE_CODE]}) as client:
        response = client.post("/api/trakt/device/code")
        assert response.status_code == 200
        payload = response.json()
        assert payload["user_code"] == "ABCD1234"
        assert payload["interval"] == 5
        assert "device-secret" not in response.text
        assert client.calls == [("/oauth/device/code", {"client_id": "client"})]


def test_device_token_pending_then_success(monkeypatch) -> None:
    responses = {
        "/oauth/device/code": [_DEVICE_CODE],
        "/oauth/device/token": [
            (400, {}),
            (200, {"access_token": "access", "refresh_token": "refresh"}),
        ],
    }
    with _test_client(monkeypatch, responses) as client:
        session = client.post("/api/trakt/device/code").json()["session"]

        pending = client.post("/api/trakt/device/token", json={"session": session})
        assert pending.json() == {"status": "pending"}

        done = client.post("/api/trakt/device/token", json={"session": session})
        assert done.status_code == 200
        tokens = done.json()["tokens"]
        assert tokens["access_token"] == "access"
        assert tokens["refresh_token"] == "refresh"
        # Filled in when Trakt omits it, so the server can order sign-ins.
        assert isinstance(tokens["created_at"], int)
        path, body = client.calls[-1]
        assert path == "/oauth/device/token"
        assert body == {"code": "device-secret", "client_id": "client"}
        assert "client_secret" not in body

        reused = client.post("/api/trakt/device/token", json={"session": session})
        assert reused.status_code == 410


def test_device_token_denied(monkeypatch) -> None:
    responses = {
        "/oauth/device/code": [_DEVICE_CODE],
        "/oauth/device/token": [(418, {})],
    }
    with _test_client(monkeypatch, responses) as client:
        session = client.post("/api/trakt/device/code").json()["session"]
        response = client.post("/api/trakt/device/token", json={"session": session})
        assert response.status_code == 400
        assert response.json()["detail"]["description"] == "Access was denied on Trakt."


def test_device_code_requires_client_id(monkeypatch) -> None:
    with _test_client(monkeypatch, {}) as client:
        monkeypatch.setattr(settings, "trakt_client_id", None)
        response = client.post("/api/trakt/device/code")
        assert response.status_code == 503
