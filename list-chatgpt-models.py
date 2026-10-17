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
import subprocess
import sys
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.llms.chatgpt.authenticator import Authenticator


class CodexIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    userAgent: str


class InitializeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: CodexIdentity


def codex_user_agent() -> str:
    request: Final = {
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "codex_cli_rs", "version": "0.0.0"}},
    }
    process: Final = subprocess.run(
        ["codex", "app-server"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    response: Final = InitializeResponse.model_validate_json(process.stdout.splitlines()[0])
    return response.result.userAgent.removesuffix(" (codex_cli_rs; 0.0.0)")


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
        user_agent: Final = codex_user_agent()
        with httpx.Client() as client:
            json.dump(load_models(client, authenticator, user_agent), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    except Exception as error:
        sys.stderr.write(f"list-chatgpt-models: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
