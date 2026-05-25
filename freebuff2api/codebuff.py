from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from .config import Settings
from .logging_config import redact_headers, render_debug
from .models import agent_validation_payload


logger = logging.getLogger("freebuff2api.codebuff")


class CodebuffError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class FreebuffSession:
    instance_id: str
    model: str
    expires_at: str | None = None
    remaining_ms: int | None = None

    @property
    def is_fresh(self) -> bool:
        return self.remaining_ms is None or self.remaining_ms > 60_000


@dataclass
class FreebuffRun:
    run_id: str
    agent_id: str
    started_at: str
    child_run_id: str | None = None
    chat_run_id: str | None = None
    chat_started_at: str | None = None

    @property
    def payload_run_id(self) -> str:
        return self.chat_run_id or self.run_id


@dataclass
class FreebuffSessionLease:
    session: FreebuffSession
    _lock: asyncio.Lock
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lock.release()


class CodebuffClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, read=None),
            follow_redirects=True,
            proxy=settings.upstream_proxy_url,
            trust_env=False,
        )
        self._agents_validated = False
        self._validate_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(
        self,
        *,
        json_body: bool = False,
        user_agent: str = "Bun/1.3.11",
        require_auth: bool = True,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if require_auth and not self.settings.codebuff_token:
            raise CodebuffError("FREEBUFF_TOKEN or CODEBUFF_TOKEN is required", 500)

        headers = {
            "Accept": "*/*",
            "User-Agent": user_agent,
        }
        if require_auth:
            headers["Authorization"] = f"Bearer {self.settings.codebuff_token}"
        if json_body:
            headers["Content-Type"] = "application/json"
        if extra:
            headers.update(extra)
        return headers

    async def _json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.codebuff_api_url}{path}"
        request_headers = headers or self._headers(json_body=body is not None)
        try:
            response = await self._client.request(
                method,
                url,
                json=body,
                headers=request_headers,
            )
        except httpx.RequestError as error:
            raise _network_error(method, url, error) from error
        if self.settings.debug:
            logger.debug(
                "upstream json request method=%s url=%s headers=%s body=%s",
                method,
                url,
                redact_headers(request_headers),
                render_debug(body, self.settings.log_body_chars),
            )
            logger.debug(
                "upstream json response status=%s body=%s",
                response.status_code,
                render_debug(response.text, self.settings.log_body_chars),
            )
        if response.status_code >= 400:
            raise _upstream_error(response)
        if not response.content:
            return {}
        return response.json()

    async def validate_agents(self) -> None:
        if self._agents_validated:
            return
        async with self._validate_lock:
            if self._agents_validated:
                return
            try:
                data = await self._json(
                    "POST",
                    "/api/agents/validate",
                    body=agent_validation_payload(),
                    headers=self._headers(json_body=True, require_auth=False),
                )
            except CodebuffError:
                logger.warning(
                    "agent validation failed; continuing with server configs",
                    exc_info=self.settings.debug,
                )
                self._agents_validated = True
                return
            error_count = int(data.get("errorCount") or 0)
            if error_count:
                logger.warning(
                    "agent validation returned errors count=%s body=%s",
                    error_count,
                    render_debug(data, self.settings.log_body_chars),
                )
            else:
                logger.info(
                    "agent validation completed configs=%s",
                    len(data.get("configs") or []),
                )
            self._agents_validated = True

    async def health(self) -> dict[str, Any]:
        return await self._json("GET", "/api/healthz", headers={"Accept": "*/*"})

    async def get_session(self, instance_id: str | None = None) -> dict[str, Any]:
        headers_extra = {}
        if instance_id:
            headers_extra["x-freebuff-instance-id"] = instance_id
        return await self._json(
            "GET",
            "/api/v1/freebuff/session",
            headers=self._headers(extra=headers_extra),
        )

    async def create_session(self, model: str) -> FreebuffSession:
        data = await self._json(
            "POST",
            "/api/v1/freebuff/session",
            headers=self._headers(extra={"x-freebuff-model": model}),
        )
        if data.get("status") == "queued":
            return await self._wait_for_active_session(data, model)
        return self._session_from_data(data, model)

    def _session_from_data(
        self,
        data: dict[str, Any],
        model: str,
        instance_id: str | None = None,
    ) -> FreebuffSession:
        resolved_instance_id = data.get("instanceId") or instance_id
        if data.get("status") != "active" or not resolved_instance_id:
            raise CodebuffError(f"Freebuff session is not active: {data}", 502)
        return FreebuffSession(
            instance_id=resolved_instance_id,
            model=data.get("model") or model,
            expires_at=data.get("expiresAt"),
            remaining_ms=data.get("remainingMs"),
        )

    async def _wait_for_active_session(
        self,
        data: dict[str, Any],
        model: str,
    ) -> FreebuffSession:
        instance_id = data.get("instanceId")
        if not instance_id:
            raise CodebuffError(f"Freebuff queued session id missing: {data}", 502)

        deadline = time.monotonic() + self.settings.request_timeout
        attempts = 0
        while data.get("status") == "queued":
            logger.info(
                "freebuff session queued model=%s instance_id=%s position=%s estimated_wait_ms=%s",
                model,
                instance_id,
                data.get("position"),
                data.get("estimatedWaitMs"),
            )
            if time.monotonic() >= deadline:
                raise CodebuffError(
                    f"Freebuff session did not become active before timeout: {data}",
                    502,
                )
            if attempts:
                await asyncio.sleep(_queue_poll_delay(data.get("estimatedWaitMs")))
            data = await self.get_session(instance_id)
            attempts += 1

        return self._session_from_data(data, model, instance_id=instance_id)

    async def delete_session(self) -> None:
        await self._json(
            "DELETE",
            "/api/v1/freebuff/session",
            headers=self._headers(),
        )
        logger.info("deleted active freebuff session")

    async def request_ads(
        self,
        provider: str,
        messages: list[dict[str, Any]] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "provider": provider,
            "messages": _ad_messages(messages),
            "sessionId": self.settings.session_id,
            "device": {
                "os": self.settings.os_name,
                "timezone": self.settings.timezone,
                "locale": self.settings.locale,
            },
            "userAgent": self.settings.browser_user_agent,
        }
        if surface:
            body["surface"] = surface
        return await self._json(
            "POST",
            "/api/v1/ads",
            body=body,
            headers=self._headers(
                json_body=True,
                user_agent="Freebuff-CLI/0.0.95",
            ),
        )

    async def request_ad_chain(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        for provider in self.settings.ad_providers:
            try:
                ads_data = await self.request_ads(
                    provider,
                    messages=messages,
                    surface=surface,
                )
                ads = ads_data.get("ads") or []
                ad = ads[0] if ads else None
                logger.info(
                    "ads provider=%s messages=%s count=%s selected=%s",
                    provider,
                    len(messages or []),
                    len(ads),
                    bool(ad),
                )
                if not ad:
                    continue
                await self.report_zeroclick_impressions(
                    list(ad.get("impressionIds") or [])
                )
                await self.report_codebuff_impression(ad.get("impUrl") or "")
                return
            except CodebuffError as error:
                logger.warning(
                    "ads provider=%s failed; continuing without blocking chat: %s",
                    provider,
                    error,
                    exc_info=self.settings.debug,
                )

    async def report_zeroclick_impressions(self, ids: list[str]) -> None:
        if not ids:
            return
        url = f"{self.settings.zeroclick_api_url}/api/v2/impressions"
        try:
            response = await self._client.post(
                url,
                json={"ids": ids},
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "User-Agent": "Bun/1.3.11",
                },
            )
        except httpx.RequestError as error:
            raise _network_error("POST", url, error) from error
        if self.settings.debug:
            logger.debug(
                "zeroclick impression ids=%s status=%s body=%s",
                ids,
                response.status_code,
                render_debug(response.text, self.settings.log_body_chars),
            )
        if response.status_code >= 400:
            raise CodebuffError(
                f"Zeroclick impression failed: {response.status_code} {response.text[:500]}",
                502,
            )

    async def report_codebuff_impression(self, imp_url: str) -> None:
        if not imp_url:
            return
        await self._json(
            "POST",
            "/api/v1/ads/impression",
            body={"impUrl": imp_url, "mode": "LITE"},
            headers=self._headers(
                json_body=True,
                user_agent="Freebuff-CLI/0.0.95",
            ),
        )

    async def start_run(
        self,
        agent_id: str,
        ancestor_run_ids: list[str] | None = None,
    ) -> str:
        data = await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "START",
                "agentId": agent_id,
                "ancestorRunIds": ancestor_run_ids or [],
            },
        )
        run_id = data.get("runId")
        if not run_id:
            raise CodebuffError(f"Codebuff run id missing: {data}", 502)
        logger.info(
            "agent run started agent_id=%s run_id=%s ancestors=%s",
            agent_id,
            run_id,
            ancestor_run_ids or [],
        )
        return run_id

    async def record_run_step(
        self,
        run_id: str,
        *,
        step_number: int,
        message_id: str | None,
        start_time: str,
        child_run_ids: list[str] | None = None,
    ) -> None:
        await self._json(
            "POST",
            f"/api/v1/agent-runs/{run_id}/steps",
            body={
                "stepNumber": step_number,
                "credits": 0,
                "childRunIds": child_run_ids or [],
                "messageId": message_id,
                "status": "completed",
                "startTime": start_time,
            },
        )
        logger.info(
            "agent run step recorded run_id=%s step=%s message_id=%s children=%s",
            run_id,
            step_number,
            message_id,
            child_run_ids or [],
        )

    async def finish_run(self, run_id: str, *, total_steps: int) -> None:
        await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "FINISH",
                "runId": run_id,
                "status": "completed",
                "totalSteps": total_steps,
                "directCredits": 0,
                "totalCredits": 0,
            },
        )
        logger.info("agent run finished run_id=%s total_steps=%s", run_id, total_steps)

    async def chat_events(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        url = f"{self.settings.codebuff_api_url}/api/v1/chat/completions"
        request_headers = self._headers(
            json_body=True,
            user_agent=(
                "ai-sdk/openai-compatible/0.0.0-test/codebuff "
                "ai-sdk/provider-utils/3.0.20 runtime/browser"
            ),
        )
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=request_headers,
            ) as response:
                if self.settings.debug:
                    logger.debug(
                        "chat stream request url=%s headers=%s payload=%s",
                        url,
                        redact_headers(request_headers),
                        render_debug(payload, self.settings.log_body_chars),
                    )
                    logger.debug(
                        "chat stream response status=%s headers=%s",
                        response.status_code,
                        redact_headers(dict(response.headers)),
                    )
                if response.status_code >= 400:
                    text = await response.aread()
                    raise _upstream_error(
                        response,
                        body=text,
                        prefix="Codebuff chat failed",
                    )
                async for line in response.aiter_lines():
                    if self.settings.debug:
                        logger.debug(
                            "chat stream line=%s",
                            render_debug(line, self.settings.log_body_chars),
                        )
                    yield line
        except httpx.RequestError as error:
            raise _network_error("POST", url, error) from error


class SessionManager:
    def __init__(self, client: CodebuffClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._sessions: dict[str, FreebuffSession] = {}
        self._lock = asyncio.Lock()

    async def ensure_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        async with self._lock:
            return await self._ensure_session_locked(model, messages)

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSessionLease:
        await self._lock.acquire()
        try:
            session = await self._ensure_session_locked(model, messages)
        except Exception:
            self._lock.release()
            raise
        return FreebuffSessionLease(session=session, _lock=self._lock)

    async def _ensure_session_locked(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        cached = self._sessions.get(model)
        if cached and cached.is_fresh:
            try:
                data = await self.client.get_session(cached.instance_id)
                if data.get("status") == "active" and data.get("model") in {
                    None,
                    model,
                }:
                    cached.remaining_ms = data.get("remainingMs")
                    logger.debug(
                        "reuse freebuff session model=%s instance_id=%s remaining_ms=%s",
                        model,
                        cached.instance_id,
                        cached.remaining_ms,
                    )
                    return cached
                if data.get("status") == "active":
                    logger.info(
                        "cached freebuff session model mismatch cached=%s upstream=%s",
                        model,
                        data.get("model"),
                    )
                    self._sessions.pop(model, None)
            except CodebuffError:
                logger.debug(
                    "cached freebuff session invalid model=%s instance_id=%s",
                    model,
                    cached.instance_id,
                    exc_info=self.settings.debug,
                )
                self._sessions.pop(model, None)

        active_session = await self._delete_locked_session(model)
        if active_session:
            return active_session
        await self.client.request_ad_chain(messages=[], surface="waiting_room")

        try:
            session = await self.client.create_session(model)
        except CodebuffError as error:
            if "model_locked" not in str(error):
                raise
            logger.info(
                "freebuff session locked during create; delete and retry model=%s",
                model,
            )
            await self.client.delete_session()
            self._sessions.clear()
            await self.client.request_ad_chain(messages=[], surface="waiting_room")
            session = await self.client.create_session(model)
        self._sessions[model] = session
        logger.debug(
            "created freebuff session model=%s instance_id=%s remaining_ms=%s",
            model,
            session.instance_id,
            session.remaining_ms,
        )
        return session

    async def _delete_locked_session(
        self,
        requested_model: str,
    ) -> FreebuffSession | None:
        try:
            data = await self.client.get_session()
        except CodebuffError:
            logger.debug(
                "could not inspect active freebuff session before create",
                exc_info=self.settings.debug,
            )
            return None

        if data.get("status") != "active":
            return None

        current_model = data.get("model")
        instance_id = data.get("instanceId")
        if current_model == requested_model and instance_id:
            session = FreebuffSession(
                instance_id=instance_id,
                model=current_model,
                expires_at=data.get("expiresAt"),
                remaining_ms=data.get("remainingMs"),
            )
            self._sessions[requested_model] = session
            logger.info(
                "discovered active freebuff session model=%s instance_id=%s remaining_ms=%s",
                requested_model,
                session.instance_id,
                session.remaining_ms,
            )
            return session

        if not current_model or current_model == requested_model:
            return None

        logger.info(
            "switch freebuff session current_model=%s requested_model=%s instance_id=%s",
            current_model,
            requested_model,
            instance_id,
        )
        await self.client.delete_session()
        self._sessions.clear()
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _queue_poll_delay(estimated_wait_ms: Any) -> float:
    if isinstance(estimated_wait_ms, int | float) and estimated_wait_ms > 0:
        return min(max(float(estimated_wait_ms) / 1000.0, 0.25), 2.0)
    return 0.25


def _network_error(method: str, url: str, error: httpx.RequestError) -> CodebuffError:
    detail = str(error).strip()
    suffix = f": {detail}" if detail else ""
    return CodebuffError(
        f"Codebuff request failed: {method} {url} network error "
        f"({type(error).__name__}){suffix}",
        502,
    )


def _upstream_error(
    response: httpx.Response,
    *,
    body: bytes | None = None,
    prefix: str = "Codebuff request failed",
) -> CodebuffError:
    raw_text = (
        body.decode("utf-8", errors="replace")
        if body is not None
        else response.text
    )
    text = raw_text[:500]
    if response.status_code == 409:
        try:
            data = (
                response.json()
                if body is None
                else httpx.Response(
                    response.status_code,
                    content=body,
                    headers=response.headers,
                ).json()
            )
        except ValueError:
            data = {}
        if data.get("error") == "session_model_mismatch":
            upstream_message = data.get("message") or text
            return CodebuffError(
                "Codebuff 409 session_model_mismatch: "
                f"{upstream_message} 当前 IP/区域受限；请换用 US 服务器或 US 出口 IP 后重试。",
                409,
            )

    return CodebuffError(
        f"{prefix}: {response.status_code} {text}",
        502,
    )


def _ad_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role") or "user"),
            "content": _ad_message_content(message.get("content")),
        }
        for message in messages or []
    ]


def _ad_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [
            str(part.get("text"))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)
