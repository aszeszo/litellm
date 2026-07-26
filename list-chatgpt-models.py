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
from typing import TypedDict, cast

import httpx
import yaml

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import get_chatgpt_default_headers


class ChatGPTModelsResponse(TypedDict):
    models: list[dict[str, object]]


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
