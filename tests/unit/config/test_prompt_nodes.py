from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import PROMPT_MODEL_CONFIG_TYPE, ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import build_model

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

LIBRARY_NAME = "Griptape Nodes Library"

SECRETS = {
    "GT_CLOUD_API_KEY": "gt-test",
    "OPENAI_API_KEY": "sk-test",
    "ANTHROPIC_API_KEY": "anthropic-test",
    "COHERE_API_KEY": "cohere-test",
    "GROK_API_KEY": "grok-test",
    "GROQ_API_KEY": "groq-test",
    "NVIDIA_API_KEY": "nvidia-test",
    "AWS_ACCESS_KEY_ID": "AKIATEST",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-test",
    "AWS_DEFAULT_REGION": "us-east-1",
}


@pytest.fixture(autouse=True)
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def openai_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OpenAiPrompt` lists the account's models when constructed."""
    models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="gpt-4.1"), SimpleNamespace(id="o3")])
    )
    monkeypatch.setattr("openai.Client", lambda: SimpleNamespace(models=models))


def _create_node(node_type: str) -> BaseNode:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


def _process(node: BaseNode) -> ModelConfig:
    node.process()
    config = node.parameter_output_values["prompt_model_config"]
    assert isinstance(config, ModelConfig)
    return config


@pytest.mark.parametrize(
    ("node_type", "model", "provider", "api_key_secret", "base_url", "settings", "model_class"),
    [
        (
            "OpenAiPrompt",
            "gpt-4.1",
            ModelProvider.OPENAI,
            "OPENAI_API_KEY",
            None,
            {"temperature": 0.1, "top_p": 0.9},
            OpenAIChatModel,
        ),
        (
            "AnthropicPrompt",
            "claude-sonnet-4-6",
            ModelProvider.ANTHROPIC,
            "ANTHROPIC_API_KEY",
            None,
            {"temperature": 0.1, "top_k": 50, "top_p": 0.9},
            AnthropicModel,
        ),
        (
            "CoherePrompt",
            "command-r-plus",
            ModelProvider.COHERE,
            "COHERE_API_KEY",
            None,
            {"temperature": 0.1, "top_k": 50, "top_p": 0.9},
            CohereModel,
        ),
        (
            "GrokPrompt",
            "grok-3-mini-beta",
            ModelProvider.GROK,
            "GROK_API_KEY",
            None,
            {"temperature": 0.1, "top_p": 0.9},
            OpenAIChatModel,
        ),
        (
            "GroqPrompt",
            "llama-3.3-70b-versatile",
            ModelProvider.GROQ,
            "GROQ_API_KEY",
            "https://api.groq.com/openai/v1",
            {"temperature": 0.1, "top_p": 0.9},
            OpenAIChatModel,
        ),
        (
            "NimPrompt",
            "openai/gpt-oss-20b",
            ModelProvider.NIM,
            "NVIDIA_API_KEY",
            "https://integrate.api.nvidia.com/v1",
            {"temperature": 0.1, "top_p": 0.9},
            OpenAIChatModel,
        ),
        (
            "GriptapeCloudPrompt",
            "gpt-4.1-mini",
            ModelProvider.GRIPTAPE_CLOUD,
            None,
            None,
            {"temperature": 0.1, "top_p": 0.9},
            OpenAIChatModel,
        ),
    ],
)
def test_process_emits_model_config(  # noqa: PLR0913
    node_type: str,
    model: str,
    provider: ModelProvider,
    api_key_secret: str | None,
    base_url: str | None,
    settings: dict[str, Any],
    model_class: type,
) -> None:
    node = _create_node(node_type)
    node.set_parameter_value("model", model)

    config = _process(node)

    assert config.provider == provider
    assert config.model == model
    assert config.api_key_secret == api_key_secret
    assert config.base_url == base_url
    assert config.settings == settings
    assert config.max_retries == 2
    assert config.api_key is None
    assert isinstance(build_model(config), model_class)


@pytest.mark.parametrize(
    "node_type",
    ["OpenAiPrompt", "AnthropicPrompt", "CoherePrompt", "GrokPrompt", "GroqPrompt", "NimPrompt", "GriptapeCloudPrompt"],
)
def test_output_holds_secret_names_not_values(node_type: str) -> None:
    config = _process(_create_node(node_type))

    assert not any(value in config.model_dump_json() for value in SECRETS.values())


@pytest.mark.parametrize("node_type", ["OpenAiPrompt", "AnthropicPrompt", "GroqPrompt"])
def test_output_type_is_prompt_model_config(node_type: str) -> None:
    node = _create_node(node_type)

    output = node.get_parameter_by_name("prompt_model_config")

    assert output is not None
    assert output.output_type == PROMPT_MODEL_CONFIG_TYPE


def test_max_tokens_and_retries_are_forwarded() -> None:
    node = _create_node("AnthropicPrompt")
    node.set_parameter_value("max_tokens", 512)
    node.set_parameter_value("max_attempts_on_fail", 5)
    node.set_parameter_value("temperature", 0.7)

    config = _process(node)

    assert config.settings["max_tokens"] == 512
    assert config.settings["temperature"] == 0.7
    assert config.max_retries == 5


def test_non_positive_max_tokens_is_omitted() -> None:
    node = _create_node("OpenAiPrompt")
    node.set_parameter_value("max_tokens", -1)

    assert "max_tokens" not in _process(node).settings


def test_openai_seed_parameter_is_not_offered() -> None:
    assert _create_node("OpenAiPrompt").get_parameter_by_name("seed") is None


class TestCohere:
    def test_p_and_k_map_to_top_p_and_top_k(self) -> None:
        node = _create_node("CoherePrompt")
        node.set_parameter_value("p", 0.4)
        node.set_parameter_value("k", 12)

        settings = _process(node).settings

        assert settings["top_p"] == 0.4
        assert settings["top_k"] == 12


class TestGriptapeCloud:
    def test_catalog_max_tokens_overrides_node_value(self) -> None:
        node = _create_node("GriptapeCloudPrompt")
        node.set_parameter_value("model", "claude-haiku-4-5")
        node.set_parameter_value("max_tokens", 100)

        assert _process(node).settings["max_tokens"] == 64000

    def test_o_series_model_drops_top_p(self) -> None:
        node = _create_node("GriptapeCloudPrompt")
        node.set_parameter_value("model", "o3")

        assert "top_p" not in _process(node).settings

    def test_missing_credential_fails_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GT_CLOUD_API_KEY")
        monkeypatch.delenv("GRIPTAPE_NODES_LICENSE", raising=False)
        node = _create_node("GriptapeCloudPrompt")

        exceptions = node.validate_before_workflow_run()

        assert exceptions is not None


class TestAmazonBedrock:
    def test_process_emits_aws_secret_names(self) -> None:
        node = _create_node("AmazonBedrockPrompt")

        config = _process(node)

        assert config.provider == ModelProvider.BEDROCK
        assert config.model == "us.anthropic.claude-opus-4-7"
        assert config.settings == {"temperature": 0.1, "max_tokens": 100}
        assert config.options == {
            "access_key_id_secret": "AWS_ACCESS_KEY_ID",
            "secret_access_key_secret": "AWS_SECRET_ACCESS_KEY",
            "region_secret": "AWS_DEFAULT_REGION",
        }
        assert isinstance(build_model(config), BedrockConverseModel)

    def test_process_fails_when_session_cannot_be_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(**_: Any) -> None:
            msg = "bad region"
            raise ValueError(msg)

        monkeypatch.setattr("boto3.Session", explode)
        node = _create_node("AmazonBedrockPrompt")

        with pytest.raises(RuntimeError, match="Failed to create AWS session"):
            node.process()

    def test_validation_reports_each_missing_aws_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID")
        monkeypatch.delenv("AWS_DEFAULT_REGION")
        node = _create_node("AmazonBedrockPrompt")

        exceptions = node.validate_before_workflow_run()

        assert exceptions is not None
        assert len(exceptions) == 2


def _tags_response(models: list[str]) -> httpx.Response:
    request = httpx.Request("GET", "http://127.0.0.1:11434/api/tags")
    return httpx.Response(200, json={"models": [{"name": m, "model": m} for m in models]}, request=request)


class TestOllama:
    def test_lists_models_from_tags_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        urls: list[str] = []

        def get(url: str, **_: Any) -> httpx.Response:
            urls.append(url)
            return _tags_response(["mistral:latest", "llama3.2:latest"])

        monkeypatch.setattr(httpx, "get", get)

        node = _create_node("OllamaPrompt")

        assert urls[0] == "http://127.0.0.1:11434/api/tags"
        assert node.get_parameter_value("model") == "llama3.2:latest"

    def test_connection_failure_raises_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get(url: str, **_: Any) -> httpx.Response:
            msg = "refused"
            raise httpx.ConnectError(msg)

        monkeypatch.setattr(httpx, "get", get)
        node = _create_node("OllamaPrompt")

        # The library loads node modules dynamically, so the exception class can't be imported here.
        with pytest.raises(Exception, match="Unable to get available models from Ollama"):
            cast("Any", node)._get_models()
        assert "Ollama connection error" in node.get_parameter_value("model")

    def test_process_builds_openai_compatible_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["llama3.2:latest"]))
        node = _create_node("OllamaPrompt")
        node.set_parameter_value("base_url", "http://gpu-box")
        node.set_parameter_value("port", "9999")

        config = _process(node)

        assert config.provider == ModelProvider.OLLAMA
        assert config.model == "llama3.2:latest"
        assert config.base_url == "http://gpu-box:9999/v1"
        assert config.api_key_secret is None
        assert config.settings == {"temperature": 0.1}
        assert isinstance(build_model(config), OpenAIChatModel)


def test_base_prompt_has_no_provider_to_emit() -> None:
    node = BasePrompt(name="base")

    with pytest.raises(NotImplementedError):
        node.process()
