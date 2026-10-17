# Local changes

Branch `litellm_client_specific_model_lists`, rebased onto upstream `main` at `ef1d3959a0`. 33 local commits, about 1,700 added lines

## Proxy: client-specific model lists

`GET /v1/models` returns a client-specific payload when the deployment's `model_info` has matching metadata. Codex requests (identified by a `client_version` query param) get `{"models": [...]}` built from `model_info.codex`. Claude Code requests (identified by an `anthropic-version` header) get `{"data": [...]}` built from `model_info.claude`. Other clients see the normal OpenAI-style listing. The code lives in `_client_specific_model_list` in `litellm/proxy/proxy_server.py`

## Config generation scripts

`generate-config.py` is a standalone `uv` script that writes `~/.litellm/config.yaml`. It pulls the ChatGPT and GitHub Copilot model catalogues and adds DeepSeek, Z.AI, Xiaomi MiMo, Moonshot Kimi and local llama.cpp models. Each entry gets Codex and Claude picker metadata and cache costs. The script also applies per-model overrides (public IDs, priorities, hidden aliases such as `claude-opus-5[1m]`, skipped models), routes Copilot models by their native API and disables Responses Lite for bridged models. Unavailable models from the bundled Codex list get dummy entries. `--work` switches to a Copilot-plus-local-only profile without provider prefixes. Tests are in `tests/test_generate_config.py`

`list-chatgpt-models.py` dumps the ChatGPT model catalogue, and `run.sh` syncs dependencies and starts the proxy on port 4000, warning when login tokens are missing

## Provider fixes

DeepSeek always repairs incomplete tool-call history: missing tool results get a placeholder, using `sanitize_messages_for_tool_calling(..., force=True)`, which now ignores `litellm.modify_params`. In thinking mode, the `reasoning_content` placeholder or restoration only applies to assistant messages with tool calls, and it logs at most one warning per request. `reasoning_effort` maps to `thinking` and is passed through for `deepseek-v4-*`, `kimi-k2.7-code*` and `mimo-*`

Z.AI always advertises `thinking` and `reasoning_effort`. `reasoning_effort` maps to `thinking` enabled or disabled, and `glm-5.2` also gets an effort of `high` or `max`

## Responses API bridge

When Responses API input goes through chat completions, `reasoning` items are now attached to the next assistant message as `provider_specific_fields.reasoning_content`. This uses the raw reasoning text, or the joined summary as a fallback. Providers that need reasoning history, such as DeepSeek, can then restore it

## Cost map warnings

`register_model` takes a new `suppress_missing_cache_cost_warning` flag, which the router and per-request custom pricing set when they register stripped shared backend entries. This stops misleading "cache cost will default to 0" warnings for deployments that already declare cache costs
