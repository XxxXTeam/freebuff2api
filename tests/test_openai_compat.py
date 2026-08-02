import unittest

from freebuff2api.codebuff import FreebuffSession
from freebuff2api.models import (
    ALL_MODELS,
    CONTEXT_PRUNER_AGENT_ID,
    GEMINI_THINKER_AGENT_ID,
    agent_validation_payload,
    models_response,
    resolve_model,
)
from freebuff2api.openai_compat import (
    BUFFY_SYSTEM_PROMPT_OPENING,
    CompletionAccumulator,
    build_upstream_payload,
    sanitize_stream_chunk,
)


class OpenAICompatTests(unittest.TestCase):
    def test_models_response_lists_all_freebuff_models(self) -> None:
        response = models_response()

        self.assertEqual(
            [item["id"] for item in response["data"]],
            [model.id for model in ALL_MODELS],
        )

    def test_models_response_filters_to_upstream_available_models(self) -> None:
        response = models_response(
            {"minimax/minimax-m3", "openai/gpt-5.6-luna"}
        )

        ids = [item["id"] for item in response["data"]]
        self.assertIn("minimax/minimax-m3", ids)
        self.assertIn("openai/gpt-5.6-luna", ids)
        # 基准模型即使不在上游 rateLimitsByModel 里也应保留
        self.assertIn("deepseek/deepseek-v4-flash", ids)
        self.assertIn("mimo/mimo-v2.5", ids)
        # 上游未报告的模型应从列表消失
        self.assertNotIn("deepseek/deepseek-v4-pro", ids)
        self.assertNotIn("moonshotai/kimi-k2.6", ids)

    def test_models_response_maps_entitlement_id_to_catalog_id(self) -> None:
        # 上游按 crof/glm-5.2 记账，目录里是可对话的 z-ai/glm-5.2
        response = models_response({"crof/glm-5.2"})

        self.assertIn(
            "z-ai/glm-5.2", [item["id"] for item in response["data"]]
        )

    def test_resolve_model_maps_agent_id(self) -> None:
        model = resolve_model("moonshotai/kimi-k2.6")

        self.assertEqual(model.agent_id, "base2-free-kimi")

    def test_resolve_minimax_m3_maps_har_agent_id(self) -> None:
        model = resolve_model("minimax/minimax-m3")

        self.assertEqual(model.agent_id, "base2-free-minimax-m3")

    def test_resolve_gemini_model_maps_allowed_agent_combo(self) -> None:
        model = resolve_model("google/gemini-3.1-pro-preview")

        self.assertEqual(model.agent_id, GEMINI_THINKER_AGENT_ID)
        self.assertEqual(model.parent_agent_id, "base2-free-kimi")
        self.assertEqual(model.session_id, "moonshotai/kimi-k2.6")
        self.assertEqual(model.upstream_id, "google/gemini-3.1-pro-preview")

    def test_resolve_gemini_flash_lite_runs_under_session_root(self) -> None:
        model = resolve_model("google/gemini-2.5-flash-lite")

        self.assertEqual(model.agent_id, "file-picker")
        self.assertEqual(model.parent_agent_id, "base2-free-deepseek-flash")
        self.assertEqual(model.session_id, "deepseek/deepseek-v4-flash")

    def test_resolve_gemini_flash_preview_uses_program_default_agent(self) -> None:
        model = resolve_model("google/gemini-3.1-flash-lite-preview")

        self.assertEqual(model.agent_id, "file-picker-max")
        self.assertEqual(model.parent_agent_id, "base2-free-deepseek-flash")
        self.assertEqual(model.upstream_id, "google/gemini-3.1-flash-lite-preview")

    def test_agent_validation_payload_defines_spawnable_agents(self) -> None:
        payload = agent_validation_payload()
        definitions = payload["agentDefinitions"]
        ids = {definition["id"] for definition in definitions}
        spawnable_ids = {
            agent_id
            for definition in definitions
            for agent_id in definition.get("spawnableAgents", [])
        }

        self.assertIn(CONTEXT_PRUNER_AGENT_ID, ids)
        self.assertLessEqual(spawnable_ids, ids)

    def test_agent_validation_payload_has_spawn_agent_tool_when_spawnable(self) -> None:
        payload = agent_validation_payload()

        for definition in payload["agentDefinitions"]:
            if definition.get("spawnableAgents"):
                self.assertIn("spawn_agents", definition["toolNames"])

    def test_build_upstream_payload_uses_explicit_client_id(self) -> None:
        payload = build_upstream_payload(
            {"model": "deepseek/deepseek-v4-pro", "messages": []},
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(
            payload["codebuff_metadata"],
            {
                "freebuff_instance_id": "instance-1",
                "trace_session_id": "trace-1",
                "run_id": "run-1",
                "client_id": "client-1",
                "cost_mode": "free",
            },
        )

    def test_build_upstream_payload_can_override_upstream_model(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "google/gemini-3.1-flash-lite-preview",
                "messages": [],
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-flash",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
            upstream_model_id="google/gemini-3.1-flash-lite-preview",
        )

        self.assertEqual(payload["model"], "google/gemini-3.1-flash-lite-preview")

    def test_build_upstream_payload_maps_developer_role_to_system(self) -> None:
        body = {
            "model": "deepseek/deepseek-v4-pro",
            "messages": [
                {"role": "developer", "content": "be helpful"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.2,
        }

        payload = build_upstream_payload(
            body,
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(body["messages"][0]["role"], "developer")

    def test_injected_system_prompt_opens_with_canonical_buffy_identity(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "deepseek/deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hello"}],
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        system_message = payload["messages"][0]
        self.assertEqual(system_message["role"], "system")
        self.assertTrue(
            system_message["content"].startswith(BUFFY_SYSTEM_PROMPT_OPENING),
            "upstream rejects chat (403 free_mode_cli_required) unless the "
            "system prompt opens with the canonical Buffy identity",
        )

    def test_custom_system_message_gets_canonical_buffy_prefix(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "deepseek/deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Always answer in Chinese."},
                    {"role": "user", "content": "hello"},
                ],
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertTrue(
            payload["messages"][0]["content"].startswith(BUFFY_SYSTEM_PROMPT_OPENING)
        )
        self.assertIn("Always answer in Chinese.", payload["messages"][0]["content"])

    def test_build_upstream_payload_filters_unknown_request_fields(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "deepseek/deepseek-v4-pro",
                "messages": [],
                "temperature": 0.2,
                "provider": {"data_collection": "allow"},
                "codebuff_metadata": {"cost_mode": "paid"},
                "unexpected": "client-owned",
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn("unexpected", payload)
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(payload["codebuff_metadata"]["cost_mode"], "free")

    def test_accumulator_keeps_reasoning_content_separate(self) -> None:
        accumulator = CompletionAccumulator("deepseek/deepseek-v4-flash")

        accumulator.add(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
        )

        response = accumulator.final_response()

        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning_content"], "hello")

    def test_accumulator_keeps_final_answer_as_content(self) -> None:
        accumulator = CompletionAccumulator("deepseek/deepseek-v4-flash")

        accumulator.add(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "thinking"},
                        "finish_reason": None,
                    },
                    {
                        "index": 0,
                        "delta": {"content": "answer", "reasoning_content": None},
                        "finish_reason": "stop",
                    },
                ],
            }
        )

        message = accumulator.final_response()["choices"][0]["message"]

        self.assertEqual(message["content"], "answer")
        self.assertEqual(message["reasoning_content"], "thinking")

    def test_stream_chunk_keeps_reasoning_content_separate(self) -> None:
        chunk = sanitize_stream_chunk(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
        )

        delta = chunk["choices"][0]["delta"]
        self.assertNotIn("content", delta)
        self.assertEqual(delta["reasoning_content"], "hello")


if __name__ == "__main__":
    unittest.main()
