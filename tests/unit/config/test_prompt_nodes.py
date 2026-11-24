from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.traits.options import Options
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.config.prompt.openai_prompt import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
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
    assert config.max_retries == 1
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
    assert config.max_retries == 4


def test_non_positive_max_tokens_is_omitted() -> None:
    node = _create_node("OpenAiPrompt")
    node.set_parameter_value("max_tokens", -1)

    assert "max_tokens" not in _process(node).settings


def test_openai_prompt_builds_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")

    def _no_network(*_args: object, **_kwargs: object) -> None:
        msg = "OpenAiPrompt must not call OpenAI when constructed"
        raise AssertionError(msg)

    monkeypatch.setattr("openai.Client", _no_network)

    node = _create_node("OpenAiPrompt")

    assert node.get_parameter_value("model") == OPENAI_DEFAULT_MODEL


def test_openai_prompt_keeps_unlisted_saved_model() -> None:
    node = _create_node("OpenAiPrompt")
    node.set_parameter_value("model", "gpt-3.5-turbo", initial_setup=True)

    exceptions = node.validate_before_workflow_run()

    assert node.get_parameter_value("model") == "gpt-3.5-turbo"
    assert exceptions is not None
    assert len(exceptions) == 1
    assert "'gpt-3.5-turbo'" in str(exceptions[0])
    with pytest.raises(ValueError, match="gpt-3.5-turbo"):
        node.process()


def test_openai_prompt_listed_model_passes_validation() -> None:
    node = _create_node("OpenAiPrompt")
    node.set_parameter_value("model", "gpt-4.1", initial_setup=True)

    assert node.validate_before_workflow_run() is None


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

        with pytest.raises(RuntimeError, match="Could not create an AWS session"):
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


def _refuse(url: str, **_: Any) -> httpx.Response:
    msg = "Connection refused"
    raise httpx.ConnectError(msg)


def _model_choices(node: Any) -> list[str]:
    return node.get_parameter_by_name("model").find_elements_by_type(Options)[0].choices


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

    def test_unreachable_server_keeps_a_model_name_and_fails_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", _refuse)
        node = cast("Any", _create_node("OllamaPrompt"))

        assert node.get_parameter_value("model") == "llama3.2:latest"
        assert _model_choices(node) == ["llama3.2:latest"]

        exceptions = node.validate_before_node_run()

        assert exceptions is not None
        assert len(exceptions) == 1
        assert "Can't reach the Ollama server at http://127.0.0.1:11434" in str(exceptions[0])

    def test_port_change_while_unreachable_keeps_selected_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["mistral:latest"]))
        node = cast("Any", _create_node("OllamaPrompt"))
        monkeypatch.setattr(httpx, "get", _refuse)

        node.set_parameter_value("port", "9999")

        assert node.get_parameter_value("model") == "mistral:latest"
        assert _model_choices(node) == ["mistral:latest"]

    def test_refresh_button_and_port_change_offer_the_same_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["mistral:latest", "llama3.2:latest"]))
        node = cast("Any", _create_node("OllamaPrompt"))

        node._on_refresh_clicked(None, None)
        from_button = _model_choices(node)
        node.set_parameter_value("port", "9999")

        assert from_button == ["llama3.2:latest", "mistral:latest"]
        assert _model_choices(node) == from_button

    def test_refresh_replaces_a_saved_status_message_with_a_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["mistral:latest"]))
        node = cast("Any", _create_node("OllamaPrompt"))
        legacy = "⚠️ Ollama connection error\nUnable to get available models"
        node._update_option_choices(param="model", choices=[legacy], default=legacy)

        node._on_refresh_clicked(None, None)

        assert node.get_parameter_value("model") == "mistral:latest"

    def test_server_without_models_fails_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response([]))
        node = cast("Any", _create_node("OllamaPrompt"))

        exceptions = node.validate_before_node_run()

        assert exceptions is not None
        assert "ollama pull" in str(exceptions[0])

    def test_uninstalled_model_fails_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["llama3.2:latest"]))
        node = cast("Any", _create_node("OllamaPrompt"))
        assert node.validate_before_node_run() is None

        monkeypatch.setattr(httpx, "get", lambda url, **_: _tags_response(["mistral:latest"]))
        exceptions = node.validate_before_node_run()

        assert exceptions is not None
        assert "'llama3.2:latest' is not installed" in str(exceptions[0])

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
