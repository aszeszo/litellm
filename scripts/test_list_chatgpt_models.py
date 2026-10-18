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


@pytest.mark.parametrize("version", ["1.2.3", "9.8.7"])
def test_latest_release_version_is_used_for_models_request(version: str) -> None:
    namespace: Final = runpy.run_path(str(Path(__file__).resolve().parents[1] / "list-chatgpt-models.py"))
    codex_user_agent: Final = cast(Callable[[httpx.Client], str], namespace["codex_user_agent"])
    load_models: Final = cast(Callable[[httpx.Client, Authenticator, str], JsonValue], namespace["load_models"])
    authenticator: Final = Mock(spec=Authenticator)
    authenticator.get_access_token.return_value = "test-token"
    authenticator.get_account_id.return_value = None
    authenticator.get_api_base.return_value = "https://example.test/codex"

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            assert request.url.path == "/repos/openai/codex/releases/latest"
            return httpx.Response(200, json={"tag_name": f"rust-v{version}"})
        assert request.url.params["client_version"] == version
        assert request.headers["User-Agent"].startswith(f"codex_cli_rs/{version} (")
        return httpx.Response(200, json={"models": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert load_models(client, authenticator, codex_user_agent(client)) == {"models": []}


@pytest.mark.parametrize("status", [403, 503])
def test_release_lookup_failure_is_reported(status: int) -> None:
    namespace: Final = runpy.run_path(str(Path(__file__).resolve().parents[1] / "list-chatgpt-models.py"))
    codex_user_agent: Final = cast(Callable[[httpx.Client], str], namespace["codex_user_agent"])
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status))) as client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            codex_user_agent(client)
    assert error.value.response.status_code == status
