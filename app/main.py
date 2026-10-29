"""Entry point for the FastAPI-powered Stremio addon."""

from __future__ import annotations
import json
import logging
import secrets
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from .config import settings
from .database import Database
from .services.metadata_addon import MetadataAddonClient
from .services.catalog_generator import CatalogService
from .services.catalog_generator import ManifestConfig
from .services.openrouter import OpenRouterClient
from .services.openai import OpenAIClient
from .services.trakt import TraktClient
from .web import render_config_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app: FastAPI


@asynccontextmanager
async def lifespan(_: FastAPI):
    exit_stack = AsyncExitStack()
    trakt_client = await exit_stack.enter_async_context(
        httpx.AsyncClient(
            base_url=str(settings.trakt_api_url),
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
    )
    openrouter_client = await exit_stack.enter_async_context(
        httpx.AsyncClient(
            base_url=str(settings.openrouter_api_url),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    )
    openai_client = await exit_stack.enter_async_context(
        httpx.AsyncClient(
            base_url=str(settings.openai_api_url),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    )
    metadata_client_kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(15.0, connect=5.0)
    }
    if settings.metadata_addon_url:
        metadata_client_kwargs["base_url"] = str(settings.metadata_addon_url)
    metadata_http_client = await exit_stack.enter_async_context(
        httpx.AsyncClient(**metadata_client_kwargs)
    )
    database = Database(settings.database_url)
    await database.create_all()

    trakt = TraktClient(settings, trakt_client)
    openrouter = OpenRouterClient(settings, openrouter_client)
    openai = OpenAIClient(settings, openai_client)
    default_metadata_addon = (
        str(settings.metadata_addon_url)
        if settings.metadata_addon_url is not None
        else None
    )
    metadata_client = MetadataAddonClient(
        metadata_http_client, default_metadata_addon
    )
    catalog_service = CatalogService(
        settings, trakt, openrouter, openai, metadata_client, database.session_factory
    )

    app.state.catalog_service = catalog_service
    app.state.database = database
    await catalog_service.start()

    try:
        yield
    finally:  # pragma: no cover - teardown path exercised at runtime
        await catalog_service.stop()
        await database.dispose()
        await exit_stack.aclose()


def create_app() -> FastAPI:
    fastapi_app = FastAPI(
        title=settings.app_name,
        description="AI-personalized catalogs for Stremio powered by OpenRouter",
        version="1.0.0",
        lifespan=lifespan,
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    fastapi_app.state.trakt_device_sessions: dict[str, dict[str, Any]] = {}

    register_routes(fastapi_app)
    return fastapi_app


def get_catalog_service(app: FastAPI) -> CatalogService:
    service = getattr(app.state, "catalog_service", None)
    if not isinstance(service, CatalogService):
        raise RuntimeError("Catalog service not initialised")
    return service


def register_routes(fastapi_app: FastAPI) -> None:
    async def _manifest_endpoint(
        request: Request,
        *,
        profile_id: str | None = None,
        extra_params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        service = get_catalog_service(fastapi_app)
        try:
            query_params = dict(request.query_params)
            if query_params:
                raise HTTPException(
                    status_code=400,
                    detail="Query parameters are not supported for manifest overrides.",
                )
            payload: dict[str, str] = {}
            if extra_params:
                payload.update(extra_params)
            config = ManifestConfig.from_request(payload, profile_id=profile_id)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc

        try:
            profile_state, catalogs = await service.list_manifest_catalogs(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        manifest_suffix = getattr(profile_state, "id", None)
        manifest_id = "com.aiopicks.python"
        if isinstance(manifest_suffix, str) and manifest_suffix.strip():
            manifest_id = f"{manifest_id}.{manifest_suffix.strip()}"
        manifest_name = (config.manifest_name or "").strip()
        if not manifest_name:
            manifest_name = settings.app_name
        return {
            "id": manifest_id,
            "version": "1.0.0",
            "name": manifest_name,
            "description": "Dynamic catalogs tailored to your Trakt history.",
            "catalogs": catalogs,
            "resources": ["catalog"],
            "types": ["movie", "series"],
            "idPrefixes": ["aiopicks", "tt", "trakt"],
        }

    async def _catalog_endpoint(
        request: Request,
        content_type: str,
        catalog_id: str,
        *,
        profile_id: str | None = None,
    ) -> JSONResponse:
        if content_type not in {"movie", "series"}:
            raise HTTPException(status_code=400, detail="Unsupported content type")
        service = get_catalog_service(fastapi_app)
        if profile_id is None:
            inferred = service.profile_id_from_catalog_id(catalog_id)
            if inferred:
                profile_id = inferred
        try:
            config = ManifestConfig.from_request(
                request.query_params, profile_id=profile_id
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        try:
            payload = await service.get_catalog_payload(
                config, content_type, catalog_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(payload)

    @fastapi_app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @fastapi_app.get("/config", response_class=HTMLResponse)
    async def config_page() -> HTMLResponse:
        return HTMLResponse(render_config_page(settings))

    def _parse_path_overrides(raw_segments: str) -> dict[str, str]:
        if not raw_segments:
            return {}
        segments = [segment for segment in raw_segments.split("/") if segment]
        if not segments:
            return {}
        if len(segments) % 2 != 0:
            raise ValueError("Path overrides must use key/value pairs")
        overrides: dict[str, str] = {}
        for index in range(0, len(segments), 2):
            key = segments[index]
            value = segments[index + 1]
            if not key:
                raise ValueError("Override keys may not be empty")
            overrides[key] = value
        return overrides

    @fastapi_app.get("/manifest.json")
    async def manifest(request: Request) -> dict[str, Any]:
        return await _manifest_endpoint(request)

    @fastapi_app.get("/manifest/{path_params:path}/manifest.json")
    async def manifest_with_path_overrides(
        request: Request, path_params: str
    ) -> dict[str, Any]:
        try:
            overrides = _parse_path_overrides(path_params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _manifest_endpoint(request, extra_params=overrides)

    @fastapi_app.get("/profiles/{profile_id}/manifest.json")
    async def manifest_with_profile(
        request: Request, profile_id: str
    ) -> dict[str, Any]:
        return await _manifest_endpoint(request, profile_id=profile_id)

    @fastapi_app.get("/catalog/{content_type}/{catalog_id}.json")
    async def catalog(
        request: Request, content_type: str, catalog_id: str
    ) -> JSONResponse:
        return await _catalog_endpoint(request, content_type, catalog_id)

    @fastapi_app.get(
        "/profiles/{profile_id}/catalog/{content_type}/{catalog_id}.json"
    )
    async def catalog_with_profile(
        request: Request, profile_id: str, content_type: str, catalog_id: str
    ) -> JSONResponse:
        return await _catalog_endpoint(
            request, content_type, catalog_id, profile_id=profile_id
        )

    @fastapi_app.post("/api/profile/prepare")
    async def prepare_profile_endpoint(request: Request) -> JSONResponse:
        service = get_catalog_service(fastapi_app)
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid payload")

        def _coerce_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "on"}
            if isinstance(value, (int, float)):
                return bool(value)
            return False

        try:
            config = ManifestConfig.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc

        force_flag = _coerce_bool(payload.get("force"))
        wait_for_completion = _coerce_bool(payload.get("waitForCompletion", False))
        context = await service.resolve_profile(config)
        state = context.state
        status = await service.get_profile_status(state.id)
        if status is None:
            raise HTTPException(status_code=404, detail="Profile could not be resolved")

        should_refresh = force_flag or context.force_refresh or status.needs_refresh
        if should_refresh:
            if wait_for_completion:
                await service.ensure_catalogs(state, force=True, wait=True)
            else:
                service.request_refresh(state, force=True)
            status = await service.get_profile_status(state.id)
            if status is None:
                raise HTTPException(status_code=404, detail="Profile not found after refresh")

        return JSONResponse(status.to_payload())

    @fastapi_app.get("/api/profile/status")
    async def profile_status_endpoint(request: Request) -> JSONResponse:
        service = get_catalog_service(fastapi_app)
        try:
            config = ManifestConfig.from_query(request.query_params)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc

        def _has_overrides(cfg: ManifestConfig) -> bool:
            return any(
                value is not None
                for value in (
                    cfg.generator_mode,
                    cfg.openrouter_key,
                    cfg.openrouter_model,
                    cfg.catalog_keys,
                    cfg.catalog_item_count,
                    cfg.refresh_interval,
                    cfg.response_cache,
                    cfg.trakt_client_id,
                    cfg.trakt_access_token,
                    cfg.metadata_addon_url,
                )
            )

        profile_id = await service.determine_profile_id(config)
        status = await service.get_profile_status(profile_id)

        def _requires_resolution(existing_status) -> bool:
            if existing_status is None:
                return _has_overrides(config)

            state = existing_status.state
            if (
                config.generator_mode is not None
                and getattr(state, "generator_mode", None) != config.generator_mode
            ):
                return True
            if (
                config.trakt_access_token is not None
                and state.trakt_access_token != config.trakt_access_token
            ):
                return True
            if (
                config.trakt_client_id is not None
                and state.trakt_client_id != config.trakt_client_id
            ):
                return True
            # trakt_history_limit is ignored; full history is always used
            if (
                config.openrouter_key is not None
                and state.openrouter_api_key != config.openrouter_key
            ):
                return True
            if (
                config.openrouter_model is not None
                and state.openrouter_model != config.openrouter_model
            ):
                return True
            if (
                config.catalog_item_count is not None
                and state.catalog_item_count != config.catalog_item_count
            ):
                return True
            if (
                config.refresh_interval is not None
                and state.refresh_interval_seconds != config.refresh_interval
            ):
                return True
            if (
                config.response_cache is not None
                and state.response_cache_seconds != config.response_cache
            ):
                return True
            if config.catalog_keys is not None:
                if tuple(config.catalog_keys) != state.catalog_keys:
                    return True
            if config.metadata_addon_url is not None:
                metadata_url = str(config.metadata_addon_url)
                if (state.metadata_addon_url or None) != metadata_url:
                    return True
            return False

        if _requires_resolution(status):
            try:
                context = await service.resolve_profile(config)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            profile_id = context.state.id
            status = await service.get_profile_status(profile_id)

        if status is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        return JSONResponse(status.to_payload())

    @fastapi_app.post("/api/trakt/device/code")
    async def trakt_device_code() -> dict[str, Any]:
        if not settings.trakt_client_id:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "trakt_credentials_missing",
                    "description": (
                        "Trakt client ID must be configured on the server "
                        "to enable sign in."
                    ),
                },
            )

        _prune_expired_device_sessions(fastapi_app)
        try:
            response = await _post_trakt_oauth(
                "/oauth/device/code", {"client_id": settings.trakt_client_id}
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "network_error",
                    "description": "Unable to reach Trakt. Please try again shortly.",
                },
            ) from exc

        data = _response_json(response)
        if response.status_code >= 400 or not data.get("device_code"):
            raise HTTPException(
                status_code=502,
                detail=_format_trakt_error(data, "Trakt could not start the sign in."),
            )

        expires_in = _coerce_int(data.get("expires_in"), default=600) or 600
        interval = _coerce_int(data.get("interval"), default=5) or 5
        # The device code stays on the server; the browser only gets an opaque id.
        session_id = secrets.token_urlsafe(32)
        fastapi_app.state.trakt_device_sessions[session_id] = {
            "device_code": data["device_code"],
            "expires_at": time.time() + expires_in,
        }
        return {
            "session": session_id,
            "user_code": data.get("user_code"),
            "verification_url": data.get("verification_url")
            or "https://trakt.tv/activate",
            "expires_in": expires_in,
            "interval": interval,
        }

    @fastapi_app.post("/api/trakt/device/token")
    async def trakt_device_token(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        session_id = payload.get("session") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(status_code=400, detail="Invalid payload")

        _prune_expired_device_sessions(fastapi_app)
        sessions = fastapi_app.state.trakt_device_sessions
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "expired",
                    "description": _TRAKT_DEVICE_ERRORS[410],
                },
            )

        try:
            response = await _post_trakt_oauth(
                "/oauth/device/token",
                {"code": session["device_code"], "client_id": settings.trakt_client_id},
            )
        except httpx.HTTPError:
            # A network blip shouldn't end the sign in; the browser polls again.
            return {"status": "pending"}

        if response.status_code == 400:
            return {"status": "pending"}
        if response.status_code == 429:
            return {"status": "slow_down"}

        sessions.pop(session_id, None)
        if response.status_code in _TRAKT_DEVICE_ERRORS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "trakt_device_error",
                    "description": _TRAKT_DEVICE_ERRORS[response.status_code],
                },
            )

        data = _response_json(response)
        if response.status_code >= 400 or not data.get("access_token"):
            raise HTTPException(
                status_code=502,
                detail=_format_trakt_error(data, "Trakt rejected the sign in request."),
            )

        token_payload = {
            "access_token": data.get("access_token"),
            "refresh_token": data.get("refresh_token"),
            "expires_in": _coerce_int(data.get("expires_in")),
            "scope": data.get("scope"),
            "token_type": data.get("token_type"),
            # The server uses this to tell a new sign-in from a stale copy.
            "created_at": _coerce_int(data.get("created_at"))
            or int(time.time()),
        }
        return {
            "status": "success",
            "tokens": {
                key: value
                for key, value in token_payload.items()
                if value not in {None, ""}
            },
        }


# Trakt's device-token poll answers: 400 pending and 429 slow down are retried.
_TRAKT_DEVICE_ERRORS: dict[int, str] = {
    404: "Trakt did not recognise the code. Start again.",
    409: "That code was already used. Start again.",
    410: "The code expired before it was approved. Start again.",
    418: "Access was denied on Trakt.",
}


def _prune_expired_device_sessions(fastapi_app: FastAPI) -> None:
    store = getattr(fastapi_app.state, "trakt_device_sessions", {})
    now = time.time()
    expired = [key for key, info in store.items() if info.get("expires_at", 0) <= now]
    for key in expired:
        store.pop(key, None)


def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def _post_trakt_oauth(path: str, payload: dict[str, Any]) -> httpx.Response:
    headers = {
        "trakt-api-version": "2",
        "Content-Type": "application/json",
    }
    client_id = payload.get("client_id")
    if isinstance(client_id, str) and client_id:
        headers["trakt-api-key"] = client_id
    async with httpx.AsyncClient(
        base_url=str(settings.trakt_auth_url),
        timeout=httpx.Timeout(20.0, connect=10.0),
    ) as client:
        return await client.post(path, json=payload, headers=headers)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _format_trakt_error(data: dict[str, Any], fallback: str) -> dict[str, str]:
    if not data:
        return {"error": "trakt_error", "description": fallback}
    error = str(
        data.get("error")
        or data.get("error_description")
        or data.get("message")
        or "trakt_error"
    )
    description = str(
        data.get("error_description")
        or data.get("message")
        or data.get("hint")
        or data.get("description")
        or fallback
    )
    return {"error": error, "description": description}


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual execution
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.environment == "development",
    )
