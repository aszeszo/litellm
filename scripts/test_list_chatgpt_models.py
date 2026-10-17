import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue

from litellm.llms.chatgpt.authenticator import Authenticator


@pytest.mark.parametrize("account_id", [None, "test-account"])
def test_models_request_uses_codex_identity_and_preserves_full_response(account_id: str | None) -> None:
    namespace: Final = runpy.run_path(str(Path(__file__).resolve().parents[1] / "list-chatgpt-models.py"))
    load_models: Final = cast(Callable[[httpx.Client, Authenticator, str], JsonValue], namespace["load_models"])
    authenticator: Final = Mock(spec=Authenticator)
    authenticator.get_access_token.return_value = "test-token"
    authenticator.get_account_id.return_value = account_id
    authenticator.get_api_base.return_value = "https://example.test/backend-api/codex/"
    user_agent: Final = "codex_cli_rs/1.2.3 (Test OS 1; test) terminal"
    payload: Final = {"models": [{"slug": "test-model", "extra": {"enabled": True}}], "metadata": [1, None]}

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/backend-api/codex/models?client_version=1.2.3"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["originator"] == "codex_cli_rs"
        assert request.headers["User-Agent"] == user_agent
        assert request.headers.get("ChatGPT-Account-ID") == account_id
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert load_models(client, authenticator, user_agent) == payload
