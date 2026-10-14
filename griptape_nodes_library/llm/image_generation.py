"""Image generation without the griptape framework: a serializable config and one call.

:class:`ImageGenerationConfig` is the `Image Generation Driver` parameter value. Like
:class:`~griptape_nodes_library.llm.model_config.ModelConfig` it holds secret *names*, never
values; :func:`generate_image` resolves them at the point of use.
"""

from __future__ import annotations

import base64
from enum import StrEnum
from typing import Any

import httpx
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from openai import OpenAI
from pydantic import BaseModel

from griptape_nodes_library.llm.budget import cloud_root, raise_budget_halt
from griptape_nodes_library.utils.cloud_credential_utils import missing_credential_message, resolve_cloud_api_key
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

IMAGE_GENERATION_DRIVER_TYPE = "Image Generation Driver"

GRIPTAPE_CLOUD_IMAGE_SIZES = ("1024x1024", "1536x1024", "1024x1536")
GRIPTAPE_CLOUD_REQUEST_TIMEOUT_SECONDS = 300.0

GROK_BASE_URL = "https://api.x.ai/v1"
OPENAI_STYLE_MODELS = ("dall-e-3",)
OPENAI_GPT_IMAGE_PREFIX = "gpt-image"


class ImageProvider(StrEnum):
    GRIPTAPE_CLOUD = "griptape_cloud"
    OPENAI = "openai"
    GROK = "grok"


DEFAULT_API_KEY_SECRETS: dict[ImageProvider, str] = {
    ImageProvider.OPENAI: "OPENAI_API_KEY",
    ImageProvider.GROK: "GROK_API_KEY",
}


class ImageGenerationConfig(BaseModel):
    """Image model configuration.

    OpenAI sends `style` only for DALL-E 3 and sends `background`, `moderation`,
    `output_format`, and `output_compression` only for GPT Image models.
    """

    provider: ImageProvider
    model: str
    base_url: str | None = None
    api_key_secret: str | None = None
    image_size: str | None = None
    quality: str | None = None
    style: str | None = None
    background: str | None = None
    moderation: str | None = None
    output_format: str | None = None
    output_compression: int | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: Any) -> ImageGenerationConfig | None:
        if isinstance(value, ImageGenerationConfig):
            return value
        if isinstance(value, dict) and "provider" in value and "model" in value:
            return cls.model_validate(value)
        return None


def generate_image(config: ImageGenerationConfig, prompt: str) -> bytes:
    match config.provider:
        case ImageProvider.GRIPTAPE_CLOUD:
            return _generate_griptape_cloud(config, prompt)
        case ImageProvider.OPENAI | ImageProvider.GROK:
            return _generate_openai_compatible(config, prompt)
        case _:
            msg = f"Unknown image provider: {config.provider!r}"
            raise ValueError(msg)


def _generate_griptape_cloud(config: ImageGenerationConfig, prompt: str) -> bytes:
    if config.image_size is not None and config.image_size not in GRIPTAPE_CLOUD_IMAGE_SIZES:
        msg = f"Image size, {config.image_size}, must be one of the following: {GRIPTAPE_CLOUD_IMAGE_SIZES}"
        raise ValueError(msg)
    api_key = resolve_cloud_api_key()
    if not api_key:
        raise KeyError(missing_credential_message(f"generate an image with model '{config.model}' on Griptape Cloud"))

    root = cloud_root(config.base_url)
    driver_configuration = {
        "model": config.model,
        "image_size": config.image_size,
        "quality": config.quality,
        "background": config.background,
        "moderation": config.moderation,
        "output_compression": config.output_compression,
        "output_format": config.output_format,
    }
    response = httpx.post(
        f"{root}/api/images/generations",
        headers=build_griptape_cloud_headers(api_key, attribution=True),
        json={
            "prompts": [prompt],
            "driver_configuration": {k: v for k, v in driver_configuration.items() if v is not None},
        },
        timeout=GRIPTAPE_CLOUD_REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise_budget_halt(exc, base_url=root)
        raise
    return _decode_b64(response.json()["artifact"]["value"])


def _generate_openai_compatible(config: ImageGenerationConfig, prompt: str) -> bytes:
    secret = config.api_key_secret or DEFAULT_API_KEY_SECRETS[config.provider]
    api_key = GriptapeNodes.SecretsManager().get_secret(secret, should_error_on_not_found=False)
    if not api_key:
        msg = f"Cannot generate an image with model '{config.model}': secret '{secret}' is not set."
        raise KeyError(msg)
    base_url = config.base_url or (GROK_BASE_URL if config.provider == ImageProvider.GROK else None)

    is_gpt_image = config.model.startswith(OPENAI_GPT_IMAGE_PREFIX)
    params: dict[str, Any] = {"size": config.image_size, "quality": config.quality}
    if config.model in OPENAI_STYLE_MODELS:
        params["style"] = config.style
    # GPT Image models always return base64 and reject `response_format`.
    if is_gpt_image:
        params |= {
            "background": config.background,
            "moderation": config.moderation,
            "output_compression": config.output_compression,
            "output_format": config.output_format,
        }
    else:
        params["response_format"] = "b64_json"

    response = OpenAI(api_key=api_key, base_url=base_url).images.generate(
        model=config.model, prompt=prompt, n=1, **{k: v for k, v in params.items() if v is not None}
    )
    if not response.data or response.data[0].b64_json is None:
        msg = "Failed to generate image"
        raise RuntimeError(msg)
    return _decode_b64(response.data[0].b64_json)


def _decode_b64(value: str) -> bytes:
    return base64.b64decode(value)
