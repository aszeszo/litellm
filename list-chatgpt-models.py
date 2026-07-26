#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "litellm>=1.77",
#   "pyyaml>=6,<7",
# ]
# ///

import json
import sys
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import httpx
import yaml

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import get_chatgpt_default_headers


class ChatGPTModelsResponse(TypedDict):
    models: list[dict[str, object]]


class ModelAlias(TypedDict):
    model_name: str
    model: str
    display_name: NotRequired[str]


class ModelOverrides(TypedDict):
    aliases: list[ModelAlias]


MODEL_OVERRIDES: ModelOverrides = {
    "aliases": [
        {"model_name": "codex", "model": "chatgpt/gpt-5", "display_name": "Codex"},
        {
            "model_name": "codex-claude",
            "model": "chatgpt/claude-opus-4",
            "display_name": "Codex Claude",
        },
        {
            "model_name": "codex-sonnet",
            "model": "chatgpt/claude-sonnet-4",
            "display_name": "Codex Sonnet",
        },
    ],
}


REQUIRED_CODEX_METADATA: dict[str, object] = {
    "slug": "",
    "display_name": "",
    "description": None,
    "supported_reasoning_levels": [],
    "shell_type": "shell_command",
    "visibility": "hide",
    "supported_in_api": True,
    "priority": 99,
    "availability_nux": None,
    "upgrade": None,
    "base_instructions": "",
    "support_verbosity": False,
    "default_verbosity": None,
    "apply_patch_tool_type": None,
    "truncation_policy": {"mode": "bytes", "limit": 10_000},
    "supports_parallel_tool_calls": False,
    "experimental_supported_tools": [],
}


def load_models() -> list[dict[str, object]]:
    auth_file = Path.home() / ".config/litellm/chatgpt/auth.json"
    auth_data = json.loads(auth_file.read_text())
    if not auth_data.get("access_token") and not auth_data.get("refresh_token"):
        raise RuntimeError(f"No ChatGPT credentials found in {auth_file}")

    authenticator = Authenticator()
    access_token = authenticator.get_access_token()
    response = httpx.get(
        f"{authenticator.get_api_base().rstrip('/')}/models",
        headers=get_chatgpt_default_headers(access_token, authenticator.get_account_id()),
        params={"client_version": "0.0.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = cast(ChatGPTModelsResponse, response.json())
    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("ChatGPT models response does not contain a models list")
    return models


def model_list(models: list[dict[str, object]]) -> dict[str, object]:
    template = next((model for model in models if isinstance(model.get("slug"), str)), {})
    return {
        "model_list": [
            {
                "model_name": slug,
                "litellm_params": {"model": f"chatgpt/{slug}"},
                "model_info": {"codex": model},
            }
            for model in models
            if isinstance((slug := model.get("slug")), str) and slug
        ]
        + [
            {
                "model_name": name,
                "litellm_params": {"model": target},
                "model_info": {
                    "codex": REQUIRED_CODEX_METADATA
                    | template
                    | {
                        "slug": name,
                        "display_name": alias.get("display_name", name),
                        "description": f"Hidden alias for {target}",
                        "visibility": "hide",
                    }
                },
            }
            for alias in MODEL_OVERRIDES["aliases"]
            if (name := alias["model_name"]) and (target := alias["model"])
        ],
    }


def main() -> int:
    try:
        yaml.safe_dump(model_list(load_models()), sys.stdout, sort_keys=False, allow_unicode=True)
    except Exception as error:
        sys.stderr.write(f"list-chatgpt-models: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
