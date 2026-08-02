from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreebuffModel:
    id: str
    agent_id: str
    owned_by: str = "freebuff"
    upstream_model_id: str | None = None
    session_model_id: str | None = None
    parent_agent_id: str | None = None

    @property
    def upstream_id(self) -> str:
        return self.upstream_model_id or self.id

    @property
    def session_id(self) -> str:
        return self.session_model_id or self.upstream_id


FREEBUFF_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel("deepseek/deepseek-v4-flash", "base2-free-deepseek-flash"),
    FreebuffModel("deepseek/deepseek-v4-pro", "base2-free-deepseek"),
    FreebuffModel("moonshotai/kimi-k2.6", "base2-free-kimi"),
    FreebuffModel("minimax/minimax-m2.7", "base2-free"),
    FreebuffModel("minimax/minimax-m3", "base2-free-minimax-m3"),
    FreebuffModel("openai/gpt-5.6-luna", "base2-free-luna"),
    FreebuffModel("z-ai/glm-5.2", "base2-free-glm"),
    FreebuffModel("mimo/mimo-v2.5", "base2-free-mimo"),
    FreebuffModel("mimo/mimo-v2.5-pro", "base2-free-mimo-pro"),
)

DEFAULT_MODEL = FREEBUFF_MODELS[0]
CONTEXT_PRUNER_AGENT_ID = "context-pruner"
GEMINI_THINKER_AGENT_ID = "thinker-with-files-gemini"
GEMINI_THINKER_PARENT_AGENT_ID = "base2-free-kimi"
GEMINI_THINKER_PARENT_MODEL_ID = "moonshotai/kimi-k2.6"
GEMINI_FLASH_LITE_SESSION_MODEL_ID = DEFAULT_MODEL.id

GEMINI_FREE_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel(
        "google/gemini-2.5-flash-lite",
        "file-picker",
        owned_by="google",
        session_model_id=GEMINI_FLASH_LITE_SESSION_MODEL_ID,
        parent_agent_id=DEFAULT_MODEL.agent_id,
    ),
    FreebuffModel(
        "google/gemini-3.1-flash-lite-preview",
        "file-picker-max",
        owned_by="google",
        session_model_id=GEMINI_FLASH_LITE_SESSION_MODEL_ID,
        parent_agent_id=DEFAULT_MODEL.agent_id,
    ),
    FreebuffModel(
        "google/gemini-3.1-pro-preview",
        GEMINI_THINKER_AGENT_ID,
        owned_by="google",
        session_model_id=GEMINI_THINKER_PARENT_MODEL_ID,
        parent_agent_id=GEMINI_THINKER_PARENT_AGENT_ID,
    ),
)

ALL_MODELS = FREEBUFF_MODELS + GEMINI_FREE_MODELS

# 无论账号当前配额如何都可用、且不会出现在上游 rateLimitsByModel 里的基准模型。
ALWAYS_AVAILABLE_MODEL_IDS = frozenset(
    {"deepseek/deepseek-v4-flash", "mimo/mimo-v2.5"}
    | {model.id for model in GEMINI_FREE_MODELS}
)

# 上游 rateLimitsByModel 里的配额 id 与可对话的 model id 不一致时的映射。
_UPSTREAM_ID_TO_CATALOG_ID = {
    "crof/glm-5.2": "z-ai/glm-5.2",
}


def resolve_model(requested: str | None) -> FreebuffModel:
    if not requested:
        return DEFAULT_MODEL
    for model in ALL_MODELS:
        if model.id == requested:
            return model
    raise ValueError(f"Unsupported Freebuff model: {requested}")


def models_response(upstream_model_ids: set[str] | None = None) -> dict[str, object]:
    if upstream_model_ids is None:
        models = ALL_MODELS
    else:
        upstream = {
            _UPSTREAM_ID_TO_CATALOG_ID.get(model_id, model_id)
            for model_id in upstream_model_ids
        }
        models = [
            model
            for model in ALL_MODELS
            if model.id in upstream or model.id in ALWAYS_AVAILABLE_MODEL_IDS
        ]
    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": model.owned_by,
            }
            for model in models
        ],
    }


def agent_validation_payload() -> dict[str, object]:
    models_by_agent: dict[str, FreebuffModel] = {}
    spawnable_by_agent: dict[str, set[str]] = {}
    for model in ALL_MODELS:
        models_by_agent.setdefault(model.agent_id, model)
        spawnable_by_agent.setdefault(model.agent_id, set()).add(CONTEXT_PRUNER_AGENT_ID)
        if model.parent_agent_id:
            spawnable_by_agent.setdefault(model.parent_agent_id, set()).add(model.agent_id)

    definitions = [
        _agent_definition(
            agent_id=model.agent_id,
            model_id=model.upstream_id,
            display_name=f"Freebuff {model.upstream_id}",
            spawnable_agents=sorted(spawnable_by_agent.get(model.agent_id, set())),
        )
        for model in models_by_agent.values()
    ]
    definitions.append(
        _agent_definition(
            agent_id=CONTEXT_PRUNER_AGENT_ID,
            model_id=DEFAULT_MODEL.id,
            display_name="Context Pruner",
            spawnable_agents=[],
        )
    )

    return {"agentDefinitions": definitions}


def _agent_definition(
    *,
    agent_id: str,
    model_id: str,
    display_name: str,
    spawnable_agents: list[str],
) -> dict[str, object]:
    return {
        "id": agent_id,
        "publisher": "codebuff",
        "model": model_id,
        "displayName": display_name,
        "spawnerPrompt": "Freebuff OpenAI-compatible orchestrator",
        "inputSchema": {
            "prompt": {
                "type": "string",
                "description": "A coding task to complete",
            },
            "params": {"type": "object", "properties": {}, "required": []},
        },
        "outputMode": "last_message",
        "includeMessageHistory": True,
        "toolNames": ["spawn_agents"] if spawnable_agents else [],
        "spawnableAgents": spawnable_agents,
        "systemPrompt": "Act as a helpful coding assistant.",
    }
