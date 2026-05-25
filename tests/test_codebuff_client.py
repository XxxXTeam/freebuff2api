import unittest
from unittest.mock import patch

import httpx

from freebuff2api.codebuff import CodebuffError
from freebuff2api.codebuff import CodebuffClient
from freebuff2api.config import Settings


class QueuedSessionClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        self.calls = []
        self.responses = [
            {
                "status": "queued",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "position": 0,
                "queueDepth": 0,
                "estimatedWaitMs": 0,
            },
            {
                "status": "active",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "expiresAt": "2026-05-23T16:04:31.177Z",
                "remainingMs": 3_000_000,
            },
        ]

    async def _json(self, method, path, *, body=None, headers=None):
        self.calls.append((method, path))
        return self.responses.pop(0)


class CapturingAdsClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        self.body = None

    async def _json(self, method, path, *, body=None, headers=None):
        self.body = body
        return {"ads": []}


class FailingAdsClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                ad_providers=("gravity", "zeroclick"),
                request_timeout=1,
            )
        )
        self.providers = []

    async def request_ads(self, provider, messages=None, surface=None):
        self.providers.append(provider)
        raise CodebuffError(f"{provider} unavailable", 502)


class CodebuffClientTests(unittest.IsolatedAsyncioTestCase):
    def test_client_uses_explicit_proxy_only_when_enabled(self) -> None:
        captured = {}

        class FakeAsyncClient:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=True,
            proxy_url="socks5://127.0.0.1:1080",
        )

        with patch("freebuff2api.codebuff.httpx.AsyncClient", FakeAsyncClient):
            CodebuffClient(settings)

        self.assertEqual(captured["proxy"], "socks5://127.0.0.1:1080")
        self.assertFalse(captured["trust_env"])

    async def test_create_session_polls_queued_session_until_active(self) -> None:
        client = QueuedSessionClient()
        try:
            session = await client.create_session("moonshotai/kimi-k2.6")
        finally:
            await client.aclose()

        self.assertEqual(session.instance_id, "queued-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.6")
        self.assertEqual(
            client.calls,
            [
                ("POST", "/api/v1/freebuff/session"),
                ("GET", "/api/v1/freebuff/session"),
            ],
        )

    async def test_request_ads_converts_openai_content_parts_to_string(self) -> None:
        client = CapturingAdsClient()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "text", "text": "world"},
                ],
            },
            {"role": "assistant", "content": None},
        ]

        try:
            await client.request_ads("gravity", messages=messages)
        finally:
            await client.aclose()

        self.assertEqual(
            client.body["messages"],
            [
                {"role": "user", "content": "hello\nworld"},
                {"role": "assistant", "content": ""},
            ],
        )
        self.assertIsInstance(messages[0]["content"], list)

    async def test_request_ad_chain_does_not_block_when_all_providers_fail(self) -> None:
        client = FailingAdsClient()

        try:
            with self.assertLogs("freebuff2api.codebuff", level="WARNING") as logs:
                await client.request_ad_chain(
                    messages=[{"role": "user", "content": "hi"}]
                )
        finally:
            await client.aclose()

        self.assertEqual(client.providers, ["gravity", "zeroclick"])
        self.assertIn("ads provider=gravity failed", logs.output[0])
        self.assertIn("ads provider=zeroclick failed", logs.output[1])

    async def test_json_wraps_network_error_as_codebuff_error(self) -> None:
        def raise_connect_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("proxy connect failed", request=request)

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(raise_connect_error),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                await client.get_session()
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("network error", str(ctx.exception))
        self.assertIn("ConnectError", str(ctx.exception))

    async def test_json_explains_session_model_mismatch_as_region_limit(self) -> None:
        def session_model_mismatch(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "error": "session_model_mismatch",
                    "message": (
                        "Limited free access is only available with DeepSeek V4 Flash."
                    ),
                },
            )

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(session_model_mismatch),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                await client.create_session("deepseek/deepseek-v4-pro")
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("当前 IP/区域", str(ctx.exception))
        self.assertIn("US", str(ctx.exception))

    async def test_chat_stream_explains_session_model_mismatch_as_region_limit(self) -> None:
        def session_model_mismatch(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "error": "session_model_mismatch",
                    "message": (
                        "Limited free access is only available with DeepSeek V4 Flash."
                    ),
                },
            )

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(session_model_mismatch),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                async for _ in client.chat_events({"messages": []}):
                    pass
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("当前 IP/区域", str(ctx.exception))
        self.assertIn("US", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
