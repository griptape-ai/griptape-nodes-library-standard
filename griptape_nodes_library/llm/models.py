from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import boto3
from anthropic import AsyncAnthropic
from botocore.config import Config as BotocoreConfig
from cohere import AsyncClientV2
from griptape_nodes.drivers.cloud_models import model_settings_for
from griptape_nodes.retained_mode.events.agent_events import ListAgentProvidersRequest, ListAgentProvidersResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from openai import AsyncOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.cohere import CohereProvider
from pydantic_ai.providers.openai import OpenAIProvider

from griptape_nodes_library.llm.budget import cloud_root
from griptape_nodes_library.llm.model_config import (
    DEFAULT_API_KEY_SECRETS,
    DEFAULT_BASE_URLS,
    ENGINE_PROVIDER_OPTION,
    ModelConfig,
    ModelProvider,
)
from griptape_nodes_library.utils.cloud_credential_utils import missing_credential_message, resolve_cloud_api_key
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


AWS_ACCESS_KEY_ID_SECRET = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_SECRET = "AWS_SECRET_ACCESS_KEY"  # noqa: S105
AWS_SESSION_TOKEN_SECRET = "AWS_SESSION_TOKEN"  # noqa: S105
AWS_DEFAULT_REGION_SECRET = "AWS_DEFAULT_REGION"


def _secret(name: str | None) -> str | None:
    if not name:
        return None
    return GriptapeNodes.SecretsManager().get_secret(name, should_error_on_not_found=False) or None


def resolve_api_key(config: ModelConfig) -> str | None:
    """The API key `config` will authenticate with, or None if none is set."""
    if config.provider == ModelProvider.GRIPTAPE_CLOUD:
        return config.api_key or resolve_cloud_api_key() or None
    if config.api_key:
        return config.api_key
    secret_name = config.api_key_secret or DEFAULT_API_KEY_SECRETS.get(config.provider)
    if secret_name is None and config.options.get(ENGINE_PROVIDER_OPTION):
        secret_name = _engine_provider_secret(str(config.options[ENGINE_PROVIDER_OPTION]))
    return _secret(secret_name)


def _engine_provider_secret(provider_name: str) -> str | None:
    result = GriptapeNodes.handle_request(ListAgentProvidersRequest())
    if not isinstance(result, ListAgentProvidersResultSuccess):
        return None
    provider = next((p for p in result.providers if p.name == provider_name), None)
    return provider.api_key_secret_name if provider else None


def _settings(config: ModelConfig) -> ModelSettings | None:
    settings: dict[str, Any] = dict(config.settings)
    if config.provider == ModelProvider.GRIPTAPE_CLOUD:
        preset = model_settings_for(config.model) or {}
        settings = {**preset, **settings}
    return cast("ModelSettings", settings) if settings else None


def _anthropic_settings(config: ModelConfig) -> ModelSettings | None:
    """Send `top_p` or `temperature`, not both, as griptape's Anthropic driver did. Anthropic rejects both together."""
    settings = _settings(config)
    if settings and settings.get("top_p") is not None and "temperature" in settings:
        settings = cast("ModelSettings", {k: v for k, v in settings.items() if k != "temperature"})
    return settings


def _openai_compatible(config: ModelConfig, *, base_url: str, api_key: str, headers: dict[str, str] | None = None):
    client_kwargs: dict[str, Any] = {"base_url": base_url.rstrip("/"), "api_key": api_key}
    if headers:
        client_kwargs["default_headers"] = headers
    if config.max_retries is not None:
        client_kwargs["max_retries"] = config.max_retries
    provider = OpenAIProvider(openai_client=AsyncOpenAI(**client_kwargs))
    return OpenAIChatModel(config.model, provider=provider, settings=_settings(config))


_model_override: Model | None = None


@contextlib.contextmanager
def override_model(model: Model) -> Iterator[None]:
    """Make every `build_model` call return `model`. For tests (pydantic-ai `TestModel`/`FunctionModel`)."""
    global _model_override  # noqa: PLW0603
    previous, _model_override = _model_override, model
    try:
        yield
    finally:
        _model_override = previous


def build_model(config: ModelConfig) -> Model:
    """Return a pydantic-ai model for `config`, resolving credentials now.

    Raises:
        KeyError: A required credential is missing.
    """
    if _model_override is not None:
        return _model_override
    match config.provider:
        case ModelProvider.GRIPTAPE_CLOUD:
            api_key = resolve_api_key(config)
            if not api_key:
                raise KeyError(missing_credential_message(f"run model '{config.model}' on Griptape Cloud"))
            root = cloud_root(config.base_url)
            return _openai_compatible(
                config,
                base_url=f"{root}/api/v1",
                api_key=api_key,
                headers=build_griptape_cloud_headers(api_key, attribution=True),
            )
        case ModelProvider.OPENAI:
            api_key = _require_key(config)
            return _openai_compatible(config, base_url=config.base_url or "https://api.openai.com/v1", api_key=api_key)
        case ModelProvider.GROQ | ModelProvider.GROK | ModelProvider.NIM | ModelProvider.OPENAI_COMPATIBLE:
            api_key = _require_key(config) if config.provider != ModelProvider.OPENAI_COMPATIBLE else None
            base_url = config.base_url or DEFAULT_BASE_URLS.get(config.provider)
            if not base_url:
                msg = f"Model '{config.model}' needs a base_url for provider '{config.provider}'."
                raise ValueError(msg)
            return _openai_compatible(
                config, base_url=base_url, api_key=api_key or resolve_api_key(config) or "not-needed"
            )
        case ModelProvider.OLLAMA | ModelProvider.LMSTUDIO:
            # No auth, but the OpenAI client needs a non-empty key.
            base_url = config.base_url or DEFAULT_BASE_URLS[config.provider]
            return _openai_compatible(config, base_url=base_url, api_key=resolve_api_key(config) or "not-needed")
        case ModelProvider.ANTHROPIC:
            client_kwargs: dict[str, Any] = {"api_key": _require_key(config), "base_url": config.base_url}
            if config.max_retries is not None:
                client_kwargs["max_retries"] = config.max_retries
            provider = AnthropicProvider(anthropic_client=AsyncAnthropic(**client_kwargs))
            return AnthropicModel(config.model, provider=provider, settings=_anthropic_settings(config))
        case ModelProvider.COHERE:
            provider = CohereProvider(cohere_client=_cohere_client(config))
            return CohereModel(config.model, provider=provider, settings=_settings(config))
        case ModelProvider.BEDROCK:
            provider = BedrockProvider(bedrock_client=_bedrock_client(config))
            return BedrockConverseModel(config.model, provider=provider, settings=_settings(config))
        case _:
            msg = f"Unknown model provider: {config.provider!r}"
            raise ValueError(msg)


def _cohere_client(config: ModelConfig) -> AsyncClientV2:
    client = AsyncClientV2(api_key=_require_key(config))
    if config.max_retries is not None:
        # The Cohere SDK takes retries per request, not per client.
        chat = client.chat
        retries = config.max_retries

        async def chat_with_retries(*args: Any, **kwargs: Any) -> Any:
            kwargs["request_options"] = {"max_retries": retries, **(kwargs.get("request_options") or {})}
            return await chat(*args, **kwargs)

        client.chat = chat_with_retries  # type: ignore[method-assign]
    return client


def _bedrock_client(config: ModelConfig) -> Any:
    options = config.options
    session = boto3.Session(
        aws_access_key_id=_secret(options.get("access_key_id_secret", AWS_ACCESS_KEY_ID_SECRET)),
        aws_secret_access_key=_secret(options.get("secret_access_key_secret", AWS_SECRET_ACCESS_KEY_SECRET)),
        aws_session_token=_secret(options.get("session_token_secret", AWS_SESSION_TOKEN_SECRET)),
        region_name=options.get("region") or _secret(options.get("region_secret", AWS_DEFAULT_REGION_SECRET)),
    )
    # botocore's `max_attempts` counts retries after the first call.
    retries = {"mode": "standard", "max_attempts": config.max_retries} if config.max_retries is not None else None
    # Timeouts match BedrockProvider's defaults.
    botocore_config = BotocoreConfig(read_timeout=300, connect_timeout=60, retries=retries)
    return session.client("bedrock-runtime", config=botocore_config)


def _require_key(config: ModelConfig) -> str:
    api_key = resolve_api_key(config)
    if not api_key:
        secret = config.api_key_secret or DEFAULT_API_KEY_SECRETS.get(config.provider, "API key")
        msg = f"Cannot run model '{config.model}': secret '{secret}' for provider '{config.provider}' is not set."
        raise KeyError(msg)
    return api_key
