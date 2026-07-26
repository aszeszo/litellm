#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "litellm>=1.77",
#   "pydantic>=2,<3",
#   "pyyaml>=6,<7",
# ]
# ///

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast
from uuid import uuid4

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot import common_utils as copilot_common_utils


ProviderName = Literal["chatgpt", "github_copilot", "deepseek", "zai", "xiaomi", "moonshotai", "local"]
ModelKey = tuple[ProviderName, str]


class ModelOverride(TypedDict, total=False):
    skip: bool
    model_id: str
    aliases: list[str]
    codex_visibility: Literal["list", "hide"]
    codex_description: str
    codex_priority: int


CONFIG: dict[str, JsonValue] = {
    "general_settings": {"master_key": "dummy"},
    "litellm_settings": {"drop_params": True, "set_verbose": False},
    "model_list": [],
}
MODEL_OVERRIDES_WORK: dict[ModelKey, ModelOverride] = {
    ("github_copilot", "gpt-5.6-sol"): {"codex_priority": 101, "codex_description": "GPT-5.6 Sol"},
    ("github_copilot", "gpt-5.6-terra"): {"codex_priority": 102, "codex_description": "GPT-5.6 Terra"},
    ("github_copilot", "gpt-5.6-luna"): {"codex_priority": 103, "codex_description": "GPT-5.6 Luna"},
    ("github_copilot", "gpt-5.5"): {"codex_priority": 104, "codex_description": "GPT-5.5"},
    ("github_copilot", "gpt-5.4"): {"codex_priority": 105, "codex_description": "GPT-5.4"},
    ("github_copilot", "gpt-5.4-mini"): {"codex_priority": 106, "codex_description": "GPT-5.4 mini"},
    ("github_copilot", "gpt-5.3-codex"): {"codex_priority": 107, "codex_description": "GPT-5.3-Codex"},
    ("github_copilot", "gpt-5-mini"): {"codex_priority": 108, "codex_description": "GPT-5 mini"},
    ("github_copilot", "mai-code-1-flash-picker"): {
        "codex_priority": 200,
        "codex_description": "MAI-Code-1-Flash",
    },
    ("github_copilot", "claude-opus-5"): {
        "aliases": ["claude-opus-5[1m]"],
        "codex_priority": 301,
        "codex_description": "Claude Opus 5",
    },
    ("github_copilot", "claude-opus-4.8"): {"codex_priority": 302, "codex_description": "Claude Opus 4.8"},
    ("github_copilot", "claude-opus-4.8-fast"): {
        "codex_priority": 303,
        "codex_description": "Claude Opus 4.8 (fast mode)",
    },
    ("github_copilot", "claude-opus-4.7"): {"codex_priority": 304, "codex_description": "Claude Opus 4.7"},
    ("github_copilot", "claude-opus-4.6"): {"codex_priority": 305, "codex_description": "Claude Opus 4.6"},
    ("github_copilot", "claude-opus-4.5"): {"codex_priority": 306, "codex_description": "Claude Opus 4.5"},
    ("github_copilot", "claude-sonnet-5"): {
        "aliases": ["claude-sonnet-5[1m]"],
        "codex_priority": 307,
        "codex_description": "Claude Sonnet 5",
    },
    ("github_copilot", "claude-sonnet-4.6"): {"codex_priority": 308, "codex_description": "Claude Sonnet 4.6"},
    ("github_copilot", "claude-sonnet-4.5"): {"codex_priority": 309, "codex_description": "Claude Sonnet 4.5"},
    ("github_copilot", "claude-haiku-4.5"): {
        "aliases": ["claude-haiku-4-5-20251001"],
        "codex_priority": 310,
        "codex_description": "Claude Haiku 4.5",
    },
    ("github_copilot", "gemini-3.1-pro-preview"): {
        "codex_priority": 401,
        "codex_description": "Gemini 3.1 Pro (Preview)",
    },
    ("github_copilot", "gemini-3.5-flash"): {"codex_priority": 402, "codex_description": "Gemini 3.5 Flash"},
    ("github_copilot", "gemini-3.6-flash"): {"codex_priority": 403, "codex_description": "Gemini 3.6 Flash"},
    ("github_copilot", "gemini-3-flash-preview"): {
        "codex_priority": 404,
        "codex_description": "Gemini 3 Flash (Preview)",
    },
    ("github_copilot", "gemini-2.5-pro"): {"codex_priority": 405, "codex_description": "Gemini 2.5 Pro"},
    **{
        ("github_copilot", model_id): {"skip": True}
        for model_id in (
            "trajectory-compaction",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "text-embedding-3-small",
            "text-embedding-3-small-inference",
            "gpt-4.1-2025-04-14",
            "gpt-41-copilot",
            "gpt-3.5-turbo-0613",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-0125-preview",
            "gpt-4o-2024-05-13",
            "gpt-4-o-preview",
            "gpt-4.1",
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "text-embedding-ada-002",
        )
    },
}
MODEL_OVERRIDES: dict[ModelKey, ModelOverride] = {
    (provider, model_id): cast(
        ModelOverride,
        {
            **override,
            **(
                {"model_id": model_id}
                if provider == "github_copilot"
                and not model_id.startswith("gpt-")
                and not override.get("skip", False)
                else {}
            ),
        },
    )
    for (provider, model_id), override in MODEL_OVERRIDES_WORK.items()
}
ENABLE_GITHUB_COPILOT = True
DUMMY_MODELS: frozenset[str] = frozenset()
DUMMY_MODELS_WORK: frozenset[str] = frozenset({"gpt-5.2"})


class LocalModel(TypedDict):
    model_name: str
    model: str
    api_base: str
    api_key: str
    reasoning_levels: list[str]
    default_reasoning_level: str | None


LOCAL_MODELS: list[LocalModel] = []
LOCAL_MODELS_WORK: list[LocalModel] = []

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CODEX_MODELS_FILE = SCRIPT_DIRECTORY / "generate-config-codex-models.json"
PI_AUTH_FILE = Path.home() / ".pi/agent/auth.json"
OUTPUT_FILE = Path.home() / ".litellm/config.yaml"
JsonObject = dict[str, JsonValue]
JsonObjectAdapter = TypeAdapter(JsonObject)


class CopilotSupports(BaseModel):
    model_config = ConfigDict(extra="allow")

    adaptive_thinking: bool = False
    max_thinking_budget: int | None = None
    min_thinking_budget: int | None = None
    reasoning_effort: tuple[str, ...] = ()


class CopilotCapabilities(BaseModel):
    model_config = ConfigDict(extra="allow")

    limits: JsonObject = Field(default_factory=dict)
    supports: CopilotSupports = Field(default_factory=CopilotSupports)


class CopilotModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    capabilities: CopilotCapabilities
    supported_endpoints: tuple[str, ...] = ()


class CopilotModelsResponse(BaseModel):
    data: tuple[CopilotModel, ...]


class ChatGPTModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    slug: str
    priority: int


class ChatGPTModelsResponse(BaseModel):
    models: tuple[ChatGPTModel, ...]


class ProviderModel(BaseModel):
    id: str


class ProviderModelsResponse(BaseModel):
    data: tuple[ProviderModel, ...]


class Arguments(argparse.Namespace):
    work: bool = False


def copilot_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "copilot-integration-id": "vscode-chat",
        "editor-version": "vscode/1.95.0",
        "editor-plugin-version": copilot_common_utils.EDITOR_PLUGIN_VERSION,
        "user-agent": copilot_common_utils.USER_AGENT,
        "openai-intent": "conversation-panel",
        "x-github-api-version": copilot_common_utils.API_VERSION,
        "x-request-id": str(uuid4()),
        "x-vscode-user-agent-library-version": "electron-fetch",
    }


def load_chatgpt_models(path: Path = CODEX_MODELS_FILE) -> tuple[ChatGPTModel, ...]:
    return ChatGPTModelsResponse.model_validate_json(path.read_text()).models


def load_copilot_models() -> tuple[CopilotModel, ...]:
    access_token_file = Path.home() / ".config/litellm/github_copilot/access-token"
    if not access_token_file.read_text().strip():
        raise RuntimeError(f"No GitHub Copilot credentials found in {access_token_file}")

    authenticator = Authenticator()
    api_key = authenticator.get_api_key()
    api_base = authenticator.get_api_base() or copilot_common_utils.DEFAULT_GITHUB_COPILOT_API_BASE
    response = httpx.get(
        f"{api_base.rstrip('/')}/models",
        headers=copilot_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    return CopilotModelsResponse.model_validate(response.json()).data


def apply_copilot_limits(metadata: JsonObject, limits: JsonObject) -> JsonObject:
    max_prompt_tokens = limits.get("max_prompt_tokens")
    max_context_window_tokens = limits.get("max_context_window_tokens")
    max_output_tokens = limits.get("max_output_tokens")
    max_non_streaming_output_tokens = limits.get("max_non_streaming_output_tokens")
    non_limit_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "context_window",
            "limits",
            "max_context_window",
            "max_non_streaming_output_tokens",
            "max_output_tokens",
        }
    }
    return {
        **non_limit_metadata,
        **({"context_window": max_prompt_tokens} if isinstance(max_prompt_tokens, int) else {}),
        **({"max_context_window": max_context_window_tokens} if isinstance(max_context_window_tokens, int) else {}),
        **({"max_output_tokens": max_output_tokens} if isinstance(max_output_tokens, int) else {}),
        **(
            {"max_non_streaming_output_tokens": max_non_streaming_output_tokens}
            if isinstance(max_non_streaming_output_tokens, int)
            else {}
        ),
        "limits": limits,
    }


def codex_metadata(model: ChatGPTModel) -> JsonObject:
    return JsonObjectAdapter.validate_python(model.model_dump(mode="json", exclude_none=False))


def merge_json(default: JsonValue, supplied: JsonValue) -> JsonValue:
    if not isinstance(default, dict) or not isinstance(supplied, dict):
        return supplied
    return {
        key: (
            merge_json(default[key], supplied[key])
            if key in default and key in supplied
            else supplied[key]
            if key in supplied
            else default[key]
        )
        for key in default.keys() | supplied.keys()
    }


def copilot_codex_metadata(model: CopilotModel, default_model: ChatGPTModel) -> JsonObject:
    copilot = JsonObjectAdapter.validate_python(model.model_dump(mode="json", exclude_unset=True))
    name = copilot.get("name")
    policy = copilot.get("policy")
    description = policy.get("terms") if isinstance(policy, dict) else None
    aliases: JsonObject = {
        "slug": model.id,
        **({"display_name": name} if isinstance(name, str) else {}),
        **({"description": description} if isinstance(description, str) else {}),
    }
    merged = merge_json(codex_metadata(default_model), {**copilot, **aliases})
    return JsonObjectAdapter.validate_python(merged)


def codex_api_metadata(metadata: JsonObject, mode: str) -> JsonObject:
    if mode == "responses":
        return metadata
    return {**metadata, "use_responses_lite": False}


def apply_codex_overrides(
    provider: ProviderName,
    model_id: str,
    metadata: JsonObject,
    overrides: dict[ModelKey, ModelOverride],
) -> JsonObject:
    override = overrides.get((provider, model_id), {})
    override_model_id = override.get("model_id")
    visibility = override.get("codex_visibility")
    description = override.get("codex_description")
    priority = override.get("codex_priority")
    return {
        **metadata,
        **({"id": override_model_id, "slug": override_model_id} if override_model_id is not None else {}),
        **({"visibility": visibility} if visibility is not None else {}),
        **({"description": description} if description is not None else {}),
        **({"priority": priority} if priority is not None else {}),
    }


def normalized_endpoints(model: CopilotModel) -> tuple[str, ...]:
    aliases = {
        "/responses": "/v1/responses",
        "/chat/completions": "/v1/chat/completions",
    }
    normalized = frozenset(aliases.get(endpoint, endpoint) for endpoint in model.supported_endpoints)
    priorities = {"/v1/responses": 0, "/v1/messages": 1, "/v1/chat/completions": 2}
    return tuple(sorted(normalized, key=lambda endpoint: (priorities.get(endpoint, 3), endpoint)))


def model_mode(model: CopilotModel, endpoints: tuple[str, ...]) -> str:
    if "/v1/responses" in endpoints:
        return "responses"
    if (model.capabilities.model_extra or {}).get("type") == "embeddings":
        return "embedding"
    return "chat"


def public_model_id(
    provider: ProviderName,
    model_id: str,
    overrides: dict[ModelKey, ModelOverride],
    prefix: str,
) -> str:
    return overrides.get((provider, model_id), {}).get("model_id", f"{prefix}{model_id}")


def reasoning_model_info(model: CopilotModel) -> JsonObject:
    supports = model.capabilities.supports
    effort_fields: JsonObject = {
        f"supports_{effort}_reasoning_effort": True
        for effort in supports.reasoning_effort
        if effort in {"minimal", "low", "medium", "high", "xhigh", "max"}
    }
    supports_reasoning = bool(supports.reasoning_effort) or supports.max_thinking_budget is not None
    return {
        "supports_reasoning": supports_reasoning,
        "supports_adaptive_thinking": supports.adaptive_thinking,
        "supports_output_config": supports.adaptive_thinking,
        **({"max_thinking_budget": supports.max_thinking_budget} if supports.max_thinking_budget is not None else {}),
        **({"min_thinking_budget": supports.min_thinking_budget} if supports.min_thinking_budget is not None else {}),
        **effort_fields,
    }


def dummy_codex_metadata(model_id: str) -> JsonObject:
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": None,
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "hide",
        "supported_in_api": True,
        "priority": 0,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": "",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }


def dummy_model_entries(
    copilot_models: tuple[CopilotModel, ...], prefix: str, dummy_models: frozenset[str]
) -> list[JsonValue]:
    available_model_ids = frozenset(model.id for model in copilot_models)
    return [
        {
            "model_name": f"{prefix}{model_id}",
            "litellm_params": {"model": f"github_copilot/{model_id}"},
            "model_info": {
                "cache_creation_input_token_cost": 0,
                "cache_read_input_token_cost": 0,
                "codex": dummy_codex_metadata(f"{prefix}{model_id}"),
            },
        }
        for model_id in sorted(dummy_models - available_model_ids)
    ]


def model_entry(
    model: CopilotModel,
    default_model: ChatGPTModel,
    references: dict[str, ChatGPTModel],
    model_name: str,
    overrides: dict[ModelKey, ModelOverride],
    hidden_alias: bool = False,
) -> JsonObject:
    endpoints = normalized_endpoints(model)
    mode = model_mode(model, endpoints)
    metadata = apply_codex_overrides(
        "github_copilot",
        model.id,
        codex_api_metadata(
            apply_copilot_limits(
                codex_metadata(references[model.id])
                if model.id in references
                else copilot_codex_metadata(model, default_model),
                model.capabilities.limits,
            ),
            mode,
        ),
        overrides,
    )
    return {
        "model_name": model_name,
        "litellm_params": {"model": f"github_copilot/{model.id}"},
        "model_info": {
            "cache_creation_input_token_cost": 0,
            "cache_read_input_token_cost": 0,
            "mode": mode,
            "supported_endpoints": list(endpoints),
            **reasoning_model_info(model),
            "codex": {
                **metadata,
                "slug": model_name,
                **({"visibility": "hide"} if hidden_alias else {}),
            },
        },
    }


def copilot_entries(
    copilot_models: tuple[CopilotModel, ...],
    chatgpt_models: tuple[ChatGPTModel, ...],
    overrides: dict[ModelKey, ModelOverride],
    prefix: str,
    dummy_models: frozenset[str],
) -> list[JsonValue]:
    if not chatgpt_models:
        raise RuntimeError("ChatGPT model metadata does not contain any models")

    default_model = min(chatgpt_models, key=lambda model: model.priority)
    references = {model.slug: model for model in chatgpt_models}
    entries = [
        model_entry(
            model,
            default_model,
            references,
            public_model_id("github_copilot", model.id, overrides, prefix),
            overrides,
        )
        for model in copilot_models
        if not overrides.get(("github_copilot", model.id), {}).get("skip", False)
    ]
    aliases = [
        model_entry(model, default_model, references, alias, overrides, hidden_alias=True)
        for model in copilot_models
        if not overrides.get(("github_copilot", model.id), {}).get("skip", False)
        for alias in overrides.get(("github_copilot", model.id), {}).get("aliases", [])
    ]
    return [*entries, *aliases, *dummy_model_entries(copilot_models, prefix, dummy_models)]


ADDITIONAL_PROVIDER_BASE_URLS: dict[ProviderName, str] = {
    "deepseek": "https://api.deepseek.com",
    "zai": "https://api.z.ai/api/coding/paas/v4",
    "xiaomi": "https://api.xiaomimimo.com/v1",
    "moonshotai": "https://api.moonshot.ai/v1",
}


def load_pi_api_keys(path: Path = PI_AUTH_FILE) -> dict[ProviderName, str]:
    auth = JsonObjectAdapter.validate_json(path.read_text())
    return {
        provider: key
        for provider in ADDITIONAL_PROVIDER_BASE_URLS
        if isinstance((entry := auth.get(provider)), dict) and isinstance((key := entry.get("key")), str) and key
    }


def load_additional_provider_models(provider: ProviderName, api_key: str) -> tuple[str, ...]:
    response = httpx.get(
        f"{ADDITIONAL_PROVIDER_BASE_URLS[provider].rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    model_ids = tuple(sorted(model.id for model in ProviderModelsResponse.model_validate(response.json()).data))
    if provider != "xiaomi":
        return model_ids
    return tuple(model_id for model_id in model_ids if not any(token in model_id for token in ("-asr", "-tts")))


def reasoning_levels(provider: ProviderName, model_id: str) -> tuple[str | None, tuple[str, ...]]:
    if provider == "deepseek":
        return "high", ("none", "high", "max")
    if provider == "zai" and model_id == "glm-5.2":
        return "high", ("none", "low", "medium", "high", "max")
    if provider == "zai":
        return "high", ("none", "high")
    if provider == "xiaomi":
        return "medium", ("none", "minimal", "low", "medium", "high")
    if provider == "moonshotai" and model_id == "kimi-k3":
        return "high", ("low", "high", "max")
    if provider == "moonshotai" and model_id.startswith("kimi-k2.7-code"):
        return "medium", ("minimal", "low", "medium", "high")
    if provider == "moonshotai":
        return "high", ("none", "high")
    return None, ()


def reasoning_presets(levels: tuple[str, ...]) -> list[JsonValue]:
    descriptions = {
        "none": "Disable thinking",
        "minimal": "Minimal thinking",
        "low": "Low thinking effort",
        "medium": "Medium thinking effort",
        "high": "High thinking effort",
        "max": "Maximum thinking effort",
    }
    return [{"effort": level, "description": descriptions[level]} for level in levels]


def provider_codex_metadata(
    default_model: ChatGPTModel,
    provider: ProviderName,
    model_id: str,
    model_name: str,
    overrides: dict[ModelKey, ModelOverride],
) -> JsonObject:
    default_effort, levels = reasoning_levels(provider, model_id)
    metadata: JsonObject = {
        **codex_metadata(default_model),
        "slug": model_name,
        "display_name": model_name,
        "description": f"{model_id} via {provider}",
        "default_reasoning_level": default_effort,
        "supported_reasoning_levels": reasoning_presets(levels),
        "visibility": "list",
        "use_responses_lite": False,
    }
    return apply_codex_overrides(provider, model_id, metadata, overrides)


def provider_litellm_model(provider: ProviderName, model_id: str) -> str:
    if provider == "chatgpt":
        return f"chatgpt/{model_id}"
    if provider == "github_copilot":
        return f"github_copilot/{model_id}"
    if provider == "zai":
        return f"zai/{model_id}"
    if provider == "moonshotai" and model_id == "kimi-k3":
        return f"openai/{model_id}"
    if provider in {"deepseek", "xiaomi", "moonshotai"}:
        return f"deepseek/{model_id}"
    raise RuntimeError(f"unsupported provider: {provider}")


def configured_entry(
    provider: ProviderName,
    model_id: str,
    model_name: str,
    metadata: JsonObject,
    api_key: str | None = None,
) -> JsonObject:
    mode = "responses" if provider == "chatgpt" else "chat"
    litellm_params: JsonObject = {"model": provider_litellm_model(provider, model_id)}
    if provider in ADDITIONAL_PROVIDER_BASE_URLS:
        litellm_params = {
            **litellm_params,
            "api_base": ADDITIONAL_PROVIDER_BASE_URLS[provider],
            "api_key": api_key,
        }
    return {
        "model_name": model_name,
        "litellm_params": litellm_params,
        "model_info": {
            "cache_creation_input_token_cost": 0,
            "cache_read_input_token_cost": 0,
            "mode": mode,
            "supported_endpoints": ["/v1/responses" if mode == "responses" else "/v1/chat/completions"],
            "supports_reasoning": bool(metadata.get("supported_reasoning_levels")),
            "codex": metadata,
        },
    }


def configured_entries_with_aliases(
    provider: ProviderName,
    model_id: str,
    model_name: str,
    metadata: JsonObject,
    overrides: dict[ModelKey, ModelOverride],
    api_key: str | None = None,
) -> list[JsonValue]:
    aliases = overrides.get((provider, model_id), {}).get("aliases", [])
    return [
        configured_entry(provider, model_id, model_name, metadata, api_key),
        *(
            configured_entry(
                provider,
                model_id,
                alias,
                {**metadata, "slug": alias, "visibility": "hide"},
                api_key,
            )
            for alias in aliases
        ),
    ]


def chatgpt_entries(
    models: tuple[ChatGPTModel, ...], overrides: dict[ModelKey, ModelOverride]
) -> list[JsonValue]:
    return [
        entry
        for model in models
        if not overrides.get(("chatgpt", model.slug), {}).get("skip", False)
        for model_name in (public_model_id("chatgpt", model.slug, overrides, "chatgpt/"),)
        for metadata in (apply_codex_overrides("chatgpt", model.slug, codex_metadata(model), overrides),)
        for entry in configured_entries_with_aliases(
            "chatgpt", model.slug, model_name, {**metadata, "slug": model_name}, overrides
        )
    ]


def additional_provider_entries(
    models: dict[ProviderName, tuple[str, ...]],
    api_keys: dict[ProviderName, str],
    default_model: ChatGPTModel,
    overrides: dict[ModelKey, ModelOverride],
) -> list[JsonValue]:
    return [
        entry
        for provider, model_ids in models.items()
        for model_id in model_ids
        if not overrides.get((provider, model_id), {}).get("skip", False)
        for model_name in (public_model_id(provider, model_id, overrides, f"{provider}/"),)
        for metadata in (provider_codex_metadata(default_model, provider, model_id, model_name, overrides),)
        for entry in configured_entries_with_aliases(
            provider, model_id, model_name, metadata, overrides, api_keys[provider]
        )
    ]


def local_entry(model: LocalModel, model_name: str, metadata: JsonObject) -> JsonObject:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/{model['model']}",
            "api_base": model["api_base"],
            "api_key": model["api_key"],
        },
        "model_info": {
            "mode": "chat",
            "supported_endpoints": ["/v1/chat/completions"],
            "supports_reasoning": bool(model["reasoning_levels"]),
            "codex": metadata,
        },
    }


def local_entries(
    models: list[LocalModel], default_model: ChatGPTModel, overrides: dict[ModelKey, ModelOverride]
) -> list[JsonValue]:
    return [
        entry
        for model in models
        if not overrides.get(("local", model["model_name"]), {}).get("skip", False)
        for model_name in (public_model_id("local", model["model_name"], overrides, ""),)
        for metadata in (
            apply_codex_overrides(
                "local",
                model["model_name"],
                {
                    **codex_metadata(default_model),
                    "slug": model_name,
                    "display_name": model_name,
                    "description": f"{model['model']} via local OpenAI-compatible API",
                    "default_reasoning_level": model["default_reasoning_level"],
                    "supported_reasoning_levels": reasoning_presets(tuple(model["reasoning_levels"])),
                    "use_responses_lite": False,
                },
                overrides,
            ),
        )
        for entry in [
            local_entry(model, model_name, metadata),
            *(
                local_entry(model, alias, {**metadata, "slug": alias, "visibility": "hide"})
                for alias in overrides.get(("local", model["model_name"]), {}).get("aliases", [])
            ),
        ]
    ]


def build_config(
    copilot_models: tuple[CopilotModel, ...],
    chatgpt_models: tuple[ChatGPTModel, ...],
    additional_models: dict[ProviderName, tuple[str, ...]],
    api_keys: dict[ProviderName, str],
    work: bool,
) -> dict[str, JsonValue]:
    if not chatgpt_models:
        raise RuntimeError("Codex model metadata does not contain any models")
    overrides = MODEL_OVERRIDES_WORK if work else MODEL_OVERRIDES
    default_model = min(chatgpt_models, key=lambda model: model.priority)
    copilot = (
        copilot_entries(
            copilot_models,
            chatgpt_models,
            overrides,
            "" if work else "copilot/",
            DUMMY_MODELS_WORK if work else DUMMY_MODELS,
        )
        if work or ENABLE_GITHUB_COPILOT
        else []
    )
    if work:
        return {**CONFIG, "model_list": [*copilot, *local_entries(LOCAL_MODELS_WORK, default_model, overrides)]}
    return {
        **CONFIG,
        "model_list": [
            *chatgpt_entries(chatgpt_models, overrides),
            *copilot,
            *additional_provider_entries(additional_models, api_keys, default_model, overrides),
            *local_entries(LOCAL_MODELS, default_model, overrides),
        ],
    }


def write_config(config: dict[str, JsonValue], path: Path = OUTPUT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", action="store_true")
    work = parser.parse_args(namespace=Arguments()).work
    try:
        chatgpt_models = load_chatgpt_models()
        api_keys: dict[ProviderName, str] = {} if work else load_pi_api_keys()
        missing_keys = ADDITIONAL_PROVIDER_BASE_URLS.keys() - api_keys.keys()
        if missing_keys:
            raise RuntimeError(f"Missing API keys in {PI_AUTH_FILE}: {', '.join(sorted(missing_keys))}")
        additional_models: dict[ProviderName, tuple[str, ...]] = (
            {}
            if work
            else {
                provider: load_additional_provider_models(provider, api_key)
                for provider, api_key in api_keys.items()
            }
        )
        copilot_models = load_copilot_models() if work or ENABLE_GITHUB_COPILOT else ()
        write_config(build_config(copilot_models, chatgpt_models, additional_models, api_keys, work))
    except Exception as error:
        sys.stderr.write(f"generate-config: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
