import asyncio
import unittest

from freebuff2api.codebuff import CodebuffError, FreebuffSession, SessionManager
from freebuff2api.config import Settings


class SwitchModelClient:
    def __init__(self) -> None:
        self.deleted = False
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id))
        if self.deleted:
            return {"status": "none"}
        return {
            "status": "active",
            "instanceId": "deepseek-instance",
            "model": "deepseek/deepseek-v4-pro",
            "expiresAt": "2026-05-23T15:27:34.581Z",
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session",))
        self.deleted = True

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        if not self.deleted:
            raise CodebuffError(
                'Codebuff request failed: 409 {"status":"model_locked"}',
                502,
            )
        return FreebuffSession(
            instance_id="kimi-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class LeaseSwitchModelClient:
    def __init__(self) -> None:
        self.current_model = "deepseek/deepseek-v4-flash"
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id, self.current_model))
        return {
            "status": "active",
            "instanceId": f"{self.current_model}-instance",
            "model": self.current_model,
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session", self.current_model))
        self.current_model = ""

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        self.current_model = model
        return FreebuffSession(
            instance_id=f"{model}-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_model_deletes_active_upstream_session_before_create(self):
        client = SwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        session = await manager.ensure_session("moonshotai/kimi-k2.6")

        self.assertEqual(session.instance_id, "kimi-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.6")
        self.assertEqual(
            client.calls,
            [
                ("get_session", None),
                ("delete_session",),
                ("request_ad_chain", [], "waiting_room"),
                ("create_session", "moonshotai/kimi-k2.6"),
            ],
        )

    async def test_session_lease_blocks_model_switch_until_chat_releases(self):
        client = LeaseSwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        first = await manager.acquire_session("deepseek/deepseek-v4-flash")
        started = asyncio.Event()

        async def acquire_second():
            started.set()
            return await manager.acquire_session("moonshotai/kimi-k2.6")

        task = asyncio.create_task(acquire_second())
        await started.wait()
        await asyncio.sleep(0.05)

        self.assertFalse(task.done())
        self.assertNotIn(
            ("delete_session", "deepseek/deepseek-v4-flash"),
            client.calls,
        )

        await first.aclose()
        second = await asyncio.wait_for(task, timeout=1)
        try:
            self.assertEqual(second.session.model, "moonshotai/kimi-k2.6")
            self.assertIn(
                ("delete_session", "deepseek/deepseek-v4-flash"),
                client.calls,
            )
        finally:
            await second.aclose()


if __name__ == "__main__":
    unittest.main()
