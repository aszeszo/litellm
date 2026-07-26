#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "litellm>=1.77",
#   "pyyaml>=6,<7",
# ]
# ///

import argparse
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


def parse_alias(value: str) -> tuple[str, str]:
    name, separator, model = value.partition("=")
    if not separator or not name or not model:
        raise argparse.ArgumentTypeError("alias must use NAME=MODEL")
    return name, model


def model_list(models: list[dict[str, object]], aliases: list[tuple[str, str]]) -> dict[str, object]:
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
                    "codex": template
                    | {
                        "slug": name,
                        "display_name": name,
                        "description": f"Hidden alias for {target}",
                        "visibility": "hide",
                    }
                },
            }
            for name, target in aliases
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", action="append", default=[], type=parse_alias, metavar="NAME=MODEL")
    aliases = cast(list[tuple[str, str]], parser.parse_args().alias)
    try:
        yaml.safe_dump(model_list(load_models(), aliases), sys.stdout, sort_keys=False, allow_unicode=True)
    except Exception as error:
        sys.stderr.write(f"list-chatgpt-models: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
