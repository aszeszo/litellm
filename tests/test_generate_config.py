import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict, cast

import pytest
from pydantic import BaseModel


class GeneratedModelInfo(TypedDict):
    cache_creation_input_token_cost: int
    cache_read_input_token_cost: int
    codex: dict[str, object]


class GeneratedEntry(TypedDict):
    model_name: str
    litellm_params: dict[str, object]
    model_info: GeneratedModelInfo


class GeneratedConfig(TypedDict):
    model_list: list[GeneratedEntry]


class GeneratorModule(Protocol):
    ChatGPTModel: type[BaseModel]
    CopilotModel: type[BaseModel]
    ENABLE_GITHUB_COPILOT: bool
    MODEL_OVERRIDES: dict[tuple[str, str], dict[str, object]]
    MODEL_OVERRIDES_WORK: dict[tuple[str, str], dict[str, object]]
    build_config: Callable[
        [tuple[BaseModel, ...], tuple[BaseModel, ...], dict[str, tuple[str, ...]], dict[str, str], bool],
        GeneratedConfig,
    ]


def load_generator() -> GeneratorModule:
    path = Path(__file__).parents[1] / "generate-config.py"
    spec = importlib.util.spec_from_file_location("generate_config_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(GeneratorModule, module)


@pytest.fixture(scope="module")
def generator() -> GeneratorModule:
    return load_generator()


def codex_model(generator: GeneratorModule, slug: str = "gpt-5.5") -> BaseModel:
    return generator.ChatGPTModel.model_validate(
        {
            "slug": slug,
            "priority": 1,
            "display_name": slug,
            "description": slug,
            "supported_reasoning_levels": [],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": "instructions",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": False,
            "experimental_supported_tools": [],
        }
    )


def copilot_model(generator: GeneratorModule, model_id: str = "gpt-5.5") -> BaseModel:
    return generator.CopilotModel.model_validate(
        {
            "id": model_id,
            "capabilities": {"limits": {}, "supports": {"reasoning_effort": ["low", "high"]}},
            "supported_endpoints": ["/responses"],
        }
    )


def test_work_mode_contains_only_unprefixed_copilot_models(generator: GeneratorModule):
    config = generator.build_config(
        (copilot_model(generator),),
        (codex_model(generator),),
        {},
        {},
        True,
    )

    rows = config["model_list"]
    assert rows[0]["model_name"] == "gpt-5.5"
    assert rows[0]["model_info"]["codex"]["slug"] == "gpt-5.5"
    assert {row["model_name"] for row in rows} == {"gpt-5.5", "gpt-5.2"}
    assert all("/" not in row["model_name"] for row in rows)


def test_default_mode_prefixes_providers_and_maps_reasoning(generator: GeneratorModule):
    config = generator.build_config(
        (copilot_model(generator),),
        (codex_model(generator),),
        {
            "deepseek": ("deepseek-v4-pro",),
            "zai": ("glm-5.2",),
            "xiaomi": ("mimo-v2.5-pro",),
            "moonshotai": ("kimi-k3",),
        },
        {"deepseek": "d", "zai": "z", "xiaomi": "x", "moonshotai": "m"},
        False,
    )

    rows = {row["model_name"]: row for row in config["model_list"]}
    assert set(rows) == {
        "gpt-5.5",
        "copilot/gpt-5.5",
        "deepseek-v4-pro",
        "glm-5.2",
        "mimo-v2.5-pro",
        "kimi-k3",
        "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M",
        "unsloth/Qwen3.5-35B-A3B-MTP-GGUF:Q4_K_M",
    }
    reasoning = cast(list[dict[str, str]], rows["glm-5.2"]["model_info"]["codex"]["supported_reasoning_levels"])
    assert [preset["effort"] for preset in reasoning] == ["none", "low", "medium", "high", "max"]
    assert rows["mimo-v2.5-pro"]["litellm_params"]["model"] == "deepseek/mimo-v2.5-pro"
    assert rows["unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M"]["litellm_params"]["api_base"] == "http://localhost:9090/v1"
    assert rows["unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M"]["model_info"]["cache_creation_input_token_cost"] == 0
    assert rows["unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M"]["model_info"]["cache_read_input_token_cost"] == 0
    assert rows["gpt-5.5"]["model_info"]["codex"]["priority"] == 4
    assert rows["deepseek-v4-pro"]["model_info"]["codex"]["priority"] == 8
    assert rows["glm-5.2"]["model_info"]["codex"]["priority"] == 10
    assert rows["mimo-v2.5-pro"]["model_info"]["codex"]["priority"] == 18
    assert rows["kimi-k3"]["model_info"]["codex"]["priority"] == 20
    assert rows["unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M"]["model_info"]["codex"]["priority"] == 24
    assert rows["deepseek-v4-pro"]["model_info"]["codex"]["description"] == "DeepSeek-V4-Pro"
    assert rows["glm-5.2"]["model_info"]["codex"]["description"] == "GLM-5.2"
    assert rows["mimo-v2.5-pro"]["model_info"]["codex"]["description"] == "MiMo-V2.5-Pro"
    assert rows["kimi-k3"]["model_info"]["codex"]["description"] == "Kimi K3"


def test_provider_scoped_overrides_do_not_collide(generator: GeneratorModule, monkeypatch: pytest.MonkeyPatch):
    overrides = {
        ("chatgpt", "gpt-5.5"): {"model_id": "personal-gpt"},
        ("github_copilot", "gpt-5.5"): {"model_id": "work-gpt"},
    }
    for key, override in overrides.items():
        monkeypatch.setitem(generator.MODEL_OVERRIDES, key, override)

    config = generator.build_config(
        (copilot_model(generator),),
        (codex_model(generator),),
        {},
        {},
        False,
    )

    names = {row["model_name"] for row in config["model_list"]}
    assert {"personal-gpt", "work-gpt"} <= names


def test_default_copilot_overrides_preserve_work_skips_and_unprefix_non_gpt_models(generator: GeneratorModule):
    skipped_work = {key for key, override in generator.MODEL_OVERRIDES_WORK.items() if override.get("skip") is True}
    skipped_default = {key for key, override in generator.MODEL_OVERRIDES.items() if override.get("skip") is True}

    assert skipped_default == skipped_work
    assert generator.MODEL_OVERRIDES[("github_copilot", "gpt-5.5")].get("model_id") is None
    assert generator.MODEL_OVERRIDES[("github_copilot", "gpt-5.5")]["codex_priority"] == 47
    assert generator.MODEL_OVERRIDES_WORK[("github_copilot", "gpt-5.5")]["codex_priority"] == 104
    assert generator.MODEL_OVERRIDES[("github_copilot", "claude-opus-5")]["model_id"] == "claude-opus-5"
    assert generator.MODEL_OVERRIDES[("github_copilot", "gemini-3.5-flash")]["model_id"] == "gemini-3.5-flash"


def test_default_claude_priorities_include_aliases_in_requested_order(generator: GeneratorModule):
    ordered = (
        ("claude-opus-5", "claude-opus-5[1m]"),
        ("claude-sonnet-5", "claude-sonnet-5[1m]"),
        ("claude-opus-4.8",),
        ("claude-opus-4.8-fast",),
        ("claude-opus-4.7",),
        ("claude-opus-4.6",),
        ("claude-sonnet-4.6",),
        ("claude-opus-4.5",),
        ("claude-sonnet-4.5",),
        ("claude-haiku-4.5", "claude-haiku-4-5-20251001"),
    )
    priorities = tuple(
        priority
        for source_and_aliases in ordered
        for source in source_and_aliases[:1]
        for override in (generator.MODEL_OVERRIDES[("github_copilot", source)],)
        for priority in (
            override["codex_priority"],
            *(alias["codex_priority"] for alias in cast(list[dict[str, object]], override.get("aliases", []))),
        )
    )

    assert priorities == tuple(range(26, 39))


def test_default_mode_can_disable_copilot_models(generator: GeneratorModule, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generator, "ENABLE_GITHUB_COPILOT", False)

    config = generator.build_config(
        (copilot_model(generator),),
        (codex_model(generator),),
        {},
        {},
        False,
    )

    assert [row["model_name"] for row in config["model_list"]] == [
        "gpt-5.5",
        "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M",
        "unsloth/Qwen3.5-35B-A3B-MTP-GGUF:Q4_K_M",
    ]
