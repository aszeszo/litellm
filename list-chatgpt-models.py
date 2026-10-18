#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "litellm>=1.77",
#   "pydantic>=2,<3",
#   "pydantic-settings>=2.9,<3",
# ]
# ///

import json
import os
import platform
import sys
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.chatgpt.authenticator import Authenticator


class CodexRelease(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag_name: str = Field(pattern=r"^rust-v[0-9]+\.[0-9]+\.[0-9]+$")


def codex_user_agent(client: httpx.Client) -> str:
    response: Final = client.get(
        "https://api.github.com/repos/openai/codex/releases/latest",
        headers={"Accept": "application/vnd.github+json"},
        timeout=30,
    )
    response.raise_for_status()
    release: Final = CodexRelease.model_validate_json(response.content)
    version: Final = release.tag_name.removeprefix("rust-v")
    os_name: Final = "Mac OS" if platform.system() == "Darwin" else platform.system()
    os_version: Final = platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release()
    terminal: Final = os.getenv("TERM_PROGRAM") or os.getenv("TERM") or "unknown"
    terminal_version: Final = os.getenv("TERM_PROGRAM_VERSION") if os.getenv("TERM_PROGRAM") else None
    terminal_token: Final = f"{terminal}/{terminal_version}" if terminal_version else terminal
    safe_terminal: Final = "".join(
        character
        for character in terminal_token
        if character.isascii() and (character.isalnum() or character in "-_./")
    )
    return f"codex_cli_rs/{version} ({os_name} {os_version}; {platform.machine()}) {safe_terminal or 'unknown'}"


def load_models(client: httpx.Client, authenticator: Authenticator, user_agent: str) -> JsonValue:
    access_token: Final = authenticator.get_access_token()
    account_id: Final = authenticator.get_account_id()
    response: Final = client.get(
        f"{authenticator.get_api_base().rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {access_token}",
            "originator": "codex_cli_rs",
            "User-Agent": user_agent,
            **({"ChatGPT-Account-ID": account_id} if account_id else {}),
        },
        params={"client_version": user_agent.split(" ", 1)[0].split("/", 1)[1]},
        timeout=30,
    )
    response.raise_for_status()
    return TypeAdapter(JsonValue).validate_json(response.content)


def main() -> int:
    try:
        authenticator: Final = Authenticator()
        auth_data: Final = TypeAdapter(dict[str, JsonValue]).validate_json(Path(authenticator.auth_file).read_text())
        if not auth_data.get("access_token") and not auth_data.get("refresh_token"):
            sys.stderr.write(f"list-chatgpt-models: No ChatGPT credentials found in {authenticator.auth_file}\n")
            return 1
        with httpx.Client() as client:
            user_agent: Final = codex_user_agent(client)
            json.dump(load_models(client, authenticator, user_agent), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    except Exception as error:
        sys.stderr.write(f"list-chatgpt-models: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
