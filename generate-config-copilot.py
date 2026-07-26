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

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    get_copilot_default_headers,
)


CONFIG: dict[str, JsonValue] = {
    "general_settings": {"master_key": "dummy"},
    "litellm_settings": {"drop_params": True, "set_verbose": False},
    "model_list": [],
}

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CHATGPT_MODELS_FILE = SCRIPT_DIRECTORY / "list-chatgpt-models.json"
OUTPUT_FILE = Path.home() / ".litellm/config.yaml"
JsonObject = dict[str, JsonValue]
JsonObjectAdapter = TypeAdapter(JsonObject)


class CopilotCapabilities(BaseModel):
    model_config = ConfigDict(extra="allow")

    limits: JsonObject = Field(default_factory=dict)


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
    copilot = JsonObjectAdapter.validate_python(model.model_dump(mode="json", exclude_none=False))
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


def build_config(
    copilot_models: tuple[CopilotModel, ...], chatgpt_models: tuple[ChatGPTModel, ...]
) -> dict[str, JsonValue]:
    if not chatgpt_models:
        raise RuntimeError("ChatGPT model metadata does not contain any models")

    default_model = min(chatgpt_models, key=lambda model: model.priority)
    references = {model.slug: model for model in chatgpt_models}
    entries: list[JsonValue] = [
        {
            "model_name": model.id,
            "litellm_params": {"model": f"github_copilot/{model.id}"},
            "model_info": {
                "cache_creation_input_token_cost": 0,
                "cache_read_input_token_cost": 0,
                "mode": mode,
                "supported_endpoints": list(endpoints),
                "codex": codex_api_metadata(
                    apply_copilot_limits(
                        codex_metadata(references[model.id])
                        if model.id in references
                        else copilot_codex_metadata(model, default_model),
                        model.capabilities.limits,
                    ),
                    mode,
                ),
            },
        }
        for model in copilot_models
        for endpoints in (normalized_endpoints(model),)
        for mode in (model_mode(model, endpoints),)
    ]
    return {**CONFIG, "model_list": entries}


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
