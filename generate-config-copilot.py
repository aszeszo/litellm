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

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    get_copilot_default_headers,
)


class ModelOverride(TypedDict, total=False):
    skip: bool
    model_id: str
    codex_visibility: Literal["list", "hide"]
    codex_description: str
    codex_priority: int


CONFIG: dict[str, JsonValue] = {
    "general_settings": {"master_key": "dummy"},
    "litellm_settings": {"drop_params": True, "set_verbose": False},
    "model_list": [],
}
MODEL_OVERRIDES: dict[str, ModelOverride] = {
    "gpt-5.6-sol": {"codex_priority": 101, "codex_description": "GPT-5.6 Sol"},
    "gpt-5.6-terra": {"codex_priority": 102, "codex_description": "GPT-5.6 Terra"},
    "gpt-5.6-luna": {"codex_priority": 103, "codex_description": "GPT-5.6 Luna"},
    "gpt-5.5": {"codex_priority": 104, "codex_description": "GPT-5.5"},
    "gpt-5.4": {"codex_priority": 105, "codex_description": "GPT-5.4"},
    "gpt-5.4-mini": {"codex_priority": 106, "codex_description": "GPT-5.4 mini"},
    "gpt-5.3-codex": {"codex_priority": 107, "codex_description": "GPT-5.3-Codex"},
    "gpt-5-mini": {"codex_priority": 108, "codex_description": "GPT-5 mini"},
    "mai-code-1-flash-picker": {"codex_priority": 200, "codex_description": "MAI-Code-1-Flash"},
    "claude-opus-5": {"codex_priority": 301, "codex_description": "Claude Opus 5"},
    "claude-opus-4.8": {"codex_priority": 302, "codex_description": "Claude Opus 4.8"},
    "claude-opus-4.8-fast": {"codex_priority": 303, "codex_description": "Claude Opus 4.8 (fast mode)"},
    "claude-opus-4.7": {"codex_priority": 304, "codex_description": "Claude Opus 4.7"},
    "claude-opus-4.6": {"codex_priority": 305, "codex_description": "Claude Opus 4.6"},
    "claude-opus-4.5": {"codex_priority": 306, "codex_description": "Claude Opus 4.5"},
    "claude-sonnet-5": {"codex_priority": 307, "codex_description": "Claude Sonnet 5"},
    "claude-sonnet-4.6": {"codex_priority": 308, "codex_description": "Claude Sonnet 4.6"},
    "claude-sonnet-4.5": {"codex_priority": 309, "codex_description": "Claude Sonnet 4.5"},
    "claude-haiku-4.5": {"codex_priority": 310, "codex_description": "Claude Haiku 4.5"},
    "gemini-3.1-pro-preview": {"codex_priority": 401, "codex_description": "Gemini 3.1 Pro (Preview)"},
    "gemini-3.5-flash": {"codex_priority": 402, "codex_description": "Gemini 3.5 Flash"},
    "gemini-3.6-flash": {"codex_priority": 403, "codex_description": "Gemini 3.6 Flash"},
    "gemini-3-flash-preview": {"codex_priority": 404, "codex_description": "Gemini 3 Flash (Preview)"},
    "gemini-2.5-pro": {"codex_priority": 405, "codex_description": "Gemini 2.5 Pro"},
    "trajectory-compaction": {"skip": True},
    "gpt-4o-mini-2024-07-18": {"skip": True},
    "gpt-4o-2024-11-20": {"skip": True},
    "gpt-4o-2024-08-06": {"skip": True},
    "text-embedding-3-small": {"skip": True},
    "text-embedding-3-small-inference": {"skip": True},
    "gpt-4.1-2025-04-14": {"skip": True},
    "gpt-41-copilot": {"skip": True},
    "gpt-3.5-turbo-0613": {"skip": True},
    "gpt-4": {"skip": True},
    "gpt-4-0613": {"skip": True},
    "gpt-4-0125-preview": {"skip": True},
    "gpt-4o-2024-05-13": {"skip": True},
    "gpt-4-o-preview": {"skip": True},
    "gpt-4.1": {"skip": True},
    "gpt-3.5-turbo": {"skip": True},
    "gpt-4o-mini": {"skip": True},
    "gpt-4o": {"skip": True},
    "text-embedding-ada-002": {"skip": True},
}
DUMMY_MODELS: frozenset[str] = frozenset({"gpt-5.2"})

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CHATGPT_MODELS_FILE = SCRIPT_DIRECTORY / "list-chatgpt-models.json"
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


def load_chatgpt_models(path: Path = CHATGPT_MODELS_FILE) -> tuple[ChatGPTModel, ...]:
    return ChatGPTModelsResponse.model_validate_json(path.read_text()).models


def load_copilot_models() -> tuple[CopilotModel, ...]:
    access_token_file = Path.home() / ".config/litellm/github_copilot/access-token"
    if not access_token_file.read_text().strip():
        raise RuntimeError(f"No GitHub Copilot credentials found in {access_token_file}")

    authenticator = Authenticator()
    api_key = authenticator.get_api_key()
    api_base = authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE
    response = httpx.get(
        f"{api_base.rstrip('/')}/models",
        headers=get_copilot_default_headers(api_key),
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


def apply_codex_overrides(model: CopilotModel, metadata: JsonObject) -> JsonObject:
    override = MODEL_OVERRIDES.get(model.id, {})
    model_id = override.get("model_id")
    visibility = override.get("codex_visibility")
    description = override.get("codex_description")
    priority = override.get("codex_priority")
    return {
        **metadata,
        **({"id": model_id, "slug": model_id} if model_id is not None else {}),
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
    if model.capabilities.model_extra.get("type") == "embeddings":
        return "embedding"
    return "chat"


def public_model_id(model: CopilotModel) -> str:
    return MODEL_OVERRIDES.get(model.id, {}).get("model_id", model.id)


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
        "upgrade": None,
        "base_instructions": "",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }


def dummy_model_entries(copilot_models: tuple[CopilotModel, ...]) -> list[JsonValue]:
    available_model_ids = frozenset(model.id for model in copilot_models)
    return [
        {
            "model_name": model_id,
            "litellm_params": {"model": f"github_copilot/{model_id}"},
            "model_info": {
                "cache_creation_input_token_cost": 0,
                "cache_read_input_token_cost": 0,
                "codex": dummy_codex_metadata(model_id),
            },
        }
        for model_id in sorted(DUMMY_MODELS - available_model_ids)
    ]


def build_config(
    copilot_models: tuple[CopilotModel, ...], chatgpt_models: tuple[ChatGPTModel, ...]
) -> dict[str, JsonValue]:
    if not chatgpt_models:
        raise RuntimeError("ChatGPT model metadata does not contain any models")

    default_model = min(chatgpt_models, key=lambda model: model.priority)
    references = {model.slug: model for model in chatgpt_models}
    entries: list[JsonValue] = [
        {
            "model_name": public_model_id(model),
            "litellm_params": {"model": f"github_copilot/{model.id}"},
            "model_info": {
                "cache_creation_input_token_cost": 0,
                "cache_read_input_token_cost": 0,
                "mode": mode,
                "supported_endpoints": list(endpoints),
                **reasoning_model_info(model),
                "codex": apply_codex_overrides(
                    model,
                    codex_api_metadata(
                        apply_copilot_limits(
                            codex_metadata(references[model.id])
                            if model.id in references
                            else copilot_codex_metadata(model, default_model),
                            model.capabilities.limits,
                        ),
                        mode,
                    ),
                ),
            },
        }
        for model in copilot_models
        if not MODEL_OVERRIDES.get(model.id, {}).get("skip", False)
        for endpoints in (normalized_endpoints(model),)
        for mode in (model_mode(model, endpoints),)
    ]
    return {**CONFIG, "model_list": [*entries, *dummy_model_entries(copilot_models)]}


def write_config(config: dict[str, JsonValue], path: Path = OUTPUT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    temporary_path.replace(path)


def main() -> int:
    try:
        write_config(build_config(load_copilot_models(), load_chatgpt_models()))
    except Exception as error:
        sys.stderr.write(f"generate-config-copilot: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
