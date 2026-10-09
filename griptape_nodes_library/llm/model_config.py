"""Serializable description of an LLM, passed between nodes as a `Prompt Model Config`.

Holds secret *names*, never secret values, so it is safe to persist inside an
`Agent` wire value. :func:`griptape_nodes_library.llm.models.build_model` resolves
the secrets and returns a live pydantic-ai model at the point of use.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from pydantic import BaseModel, Field

PROMPT_MODEL_CONFIG_TYPE = "Prompt Model Config"
ENGINE_PROVIDER_OPTION = "engine_provider"
USE_NATIVE_TOOLS_OPTION = "use_native_tools"


class ModelProvider(StrEnum):
    GRIPTAPE_CLOUD = "griptape_cloud"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    GROQ = "groq"
    GROK = "grok"
    NIM = "nim"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI_COMPATIBLE = "openai_compatible"


DEFAULT_API_KEY_SECRETS: dict[ModelProvider, str] = {
    ModelProvider.OPENAI: "OPENAI_API_KEY",
    ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    ModelProvider.COHERE: "COHERE_API_KEY",
    ModelProvider.GROQ: "GROQ_API_KEY",
    ModelProvider.GROK: "GROK_API_KEY",
    ModelProvider.NIM: "NVIDIA_API_KEY",
}

DEFAULT_BASE_URLS: dict[ModelProvider, str] = {
    ModelProvider.GROQ: "https://api.groq.com/openai/v1",
    ModelProvider.GROK: "https://api.x.ai/v1",
    ModelProvider.NIM: "https://integrate.api.nvidia.com/v1",
    ModelProvider.OLLAMA: "http://localhost:11434/v1",
    ModelProvider.LMSTUDIO: "http://localhost:1234/v1",
}


class ModelConfig(BaseModel):
    provider: ModelProvider
    model: str
    base_url: str | None = None
    api_key_secret: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    max_retries: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = Field(default=None, exclude=True, repr=False)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: Any) -> ModelConfig | None:
        if isinstance(value, ModelConfig):
            return value
        if isinstance(value, dict) and "provider" in value and "model" in value:
            return cls.model_validate(value)
        return None


# griptape drivers' default temperature, applied where no prompt config sets one.
DEFAULT_TEMPERATURE = 0.1


def default_settings() -> dict[str, Any]:
    return {"temperature": DEFAULT_TEMPERATURE}


def cloud_model_config(model: str) -> ModelConfig:
    return ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=model, settings=default_settings())


def _engine_provider_kind(provider_type: str | None) -> ModelProvider:
    match provider_type:
        case ModelProvider.OLLAMA | ModelProvider.LMSTUDIO:
            return ModelProvider(provider_type)
        case _:
            return ModelProvider.OPENAI_COMPATIBLE  # "custom" and anything else speaks the OpenAI API


def model_config_for_engine_provider(provider_config: ProviderConfig, model: str) -> ModelConfig:
    return ModelConfig(
        provider=_engine_provider_kind(provider_config.type),
        model=model,
        settings=default_settings(),
        base_url=provider_config.base_url or None,
        api_key_secret=provider_config.api_key_secret_name or None,
    )


# Griptape driver `type` tags written by `to_dict()` in saved workflows.
_LEGACY_DRIVER_PROVIDERS: dict[str, ModelProvider] = {
    "GriptapeCloudPromptDriver": ModelProvider.GRIPTAPE_CLOUD,
    "OpenAiChatPromptDriver": ModelProvider.OPENAI,
    "AnthropicPromptDriver": ModelProvider.ANTHROPIC,
    "AmazonBedrockPromptDriver": ModelProvider.BEDROCK,
    "CoherePromptDriver": ModelProvider.COHERE,
    "GrokPromptDriver": ModelProvider.GROK,
    "OllamaPromptDriver": ModelProvider.OLLAMA,
}

_LEGACY_SETTING_KEYS = ("temperature", "max_tokens", "seed", "top_p", "top_k")
# Sampling settings griptape drivers kept in `extra_params`; Cohere names them `p` and `k`.
_LEGACY_EXTRA_SETTING_KEYS = {"top_p": "top_p", "top_k": "top_k", "p": "top_p", "k": "top_k"}


def model_config_from_legacy_driver(driver: dict[str, Any], provider: dict[str, Any] | None = None) -> ModelConfig:
    model = str(driver.get("model") or "")
    settings: dict[str, Any] = {k: driver[k] for k in _LEGACY_SETTING_KEYS if driver.get(k) is not None}
    if settings.get("max_tokens") is not None and settings["max_tokens"] <= 0:
        settings.pop("max_tokens")
    extra = driver.get("extra_params") or {}
    if isinstance(extra, dict):
        for key, setting in _LEGACY_EXTRA_SETTING_KEYS.items():
            if extra.get(key) is not None:
                settings[setting] = extra[key]

    if provider:
        return ModelConfig(
            provider=_engine_provider_kind(provider.get("type")),
            model=model,
            base_url=provider.get("base_url") or None,
            api_key=provider.get("api_key") or None,
            settings=settings,
            # Keep the provider name so wire round-trips can resolve a secret omitted from serialization.
            options={ENGINE_PROVIDER_OPTION: provider["name"]} if provider.get("name") else {},
        )

    driver_type = str(driver.get("type"))
    if driver_type not in _LEGACY_DRIVER_PROVIDERS:
        msg = f"Unsupported prompt driver '{driver_type}'; connect a Prompt Model Config."
        raise ValueError(msg)
    kind = _LEGACY_DRIVER_PROVIDERS[driver_type]
    base_url = driver.get("base_url") or None
    if kind == ModelProvider.OPENAI and base_url:
        kind = next(
            (p for p, url in DEFAULT_BASE_URLS.items() if url.rstrip("/") == str(base_url).rstrip("/")),
            ModelProvider.OPENAI_COMPATIBLE,
        )
    if kind == ModelProvider.GRIPTAPE_CLOUD:
        base_url = None
    if kind == ModelProvider.OLLAMA and driver.get("host"):
        base_url = f"{str(driver['host']).rstrip('/')}/v1"
    options = {USE_NATIVE_TOOLS_OPTION: False} if driver.get("use_native_tools") is False else {}
    return ModelConfig(provider=kind, model=model, base_url=base_url, settings=settings, options=options)


def model_config_from_input(value: Any) -> ModelConfig | None:
    """A connected `model` port value as a config, or None for a dropdown selection.

    Accepts a config, its wire dict, or a griptape prompt driver (live or `to_dict()`).

    Raises:
        TypeError: `value` is none of these.
        ValueError: `value` is a driver this library cannot run.
    """
    if value is None or isinstance(value, str):
        return None
    config = ModelConfig.from_wire(value)
    if config is not None:
        return config
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if isinstance(value, dict) and "type" in value:
        return model_config_from_legacy_driver(value)
    msg = f"Unsupported model value of type {type(value).__name__}; choose a model or connect a Prompt Model Config."
    raise TypeError(msg)
