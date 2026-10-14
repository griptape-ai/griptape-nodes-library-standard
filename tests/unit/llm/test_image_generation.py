from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import griptape_nodes_library.llm.image_generation as image_generation
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider, generate_image

PNG_BYTES = b"\x89PNG fake image"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


class TestImageGenerationConfig:
    def test_wire_round_trip_drops_unset_fields(self) -> None:
        config = ImageGenerationConfig(provider=ImageProvider.OPENAI, model="gpt-image-1", quality="low")

        wire = config.to_wire()

        assert wire == {"provider": "openai", "model": "gpt-image-1", "quality": "low"}
        assert ImageGenerationConfig.from_wire(wire) == config

    def test_from_wire_accepts_instance_and_rejects_other_values(self) -> None:
        config = ImageGenerationConfig(provider=ImageProvider.GROK, model="grok-2-image-1212")

        assert ImageGenerationConfig.from_wire(config) is config
        assert ImageGenerationConfig.from_wire("gpt-image-1-mini") is None
        assert ImageGenerationConfig.from_wire({"model": "x"}) is None


class TestGriptapeCloud:
    @pytest.fixture
    def posted(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return httpx.Response(
                200, json={"artifact": {"value": PNG_B64, "format": "png"}}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(image_generation.httpx, "post", fake_post)
        monkeypatch.setattr(image_generation, "resolve_cloud_api_key", lambda: "gt-key")
        monkeypatch.setattr(
            image_generation,
            "build_griptape_cloud_headers",
            lambda token, *, attribution: {"Authorization": f"Bearer {token}", "attribution": str(attribution)},
        )
        monkeypatch.delenv("GT_CLOUD_BASE_URL", raising=False)
        return captured

    def test_posts_prompt_and_non_empty_driver_configuration(self, posted: dict[str, Any]) -> None:
        config = ImageGenerationConfig(
            provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1-mini", image_size="1536x1024", quality="high"
        )

        assert generate_image(config, "a cat") == PNG_BYTES

        assert posted["url"] == "https://cloud.griptape.ai/api/images/generations"
        assert posted["headers"] == {"Authorization": "Bearer gt-key", "attribution": "True"}
        assert posted["json"] == {
            "prompts": ["a cat"],
            "driver_configuration": {"model": "gpt-image-1-mini", "image_size": "1536x1024", "quality": "high"},
        }

    def test_base_url_comes_from_environment(self, posted: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GT_CLOUD_BASE_URL", "https://staging.example/")

        generate_image(ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="m"), "p")

        assert posted["url"] == "https://staging.example/api/images/generations"

    def test_missing_credential_raises_before_request(
        self, posted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(image_generation, "resolve_cloud_api_key", lambda: "")

        with pytest.raises(KeyError, match="generate an image"):
            generate_image(ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="m"), "p")

        assert "url" not in posted

    def test_rejects_unsupported_size(self, posted: dict[str, Any]) -> None:
        config = ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="m", image_size="64x64")

        with pytest.raises(ValueError, match="must be one of"):
            generate_image(config, "p")

    def test_http_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_post(url: str, **_: Any) -> httpx.Response:
            return httpx.Response(402, request=httpx.Request("POST", url))

        monkeypatch.setattr(image_generation.httpx, "post", fake_post)
        monkeypatch.setattr(image_generation, "resolve_cloud_api_key", lambda: "gt-key")
        monkeypatch.setattr(image_generation, "build_griptape_cloud_headers", lambda *_, **__: {})

        with pytest.raises(httpx.HTTPStatusError):
            generate_image(ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="m"), "p")


class TestOpenAiCompatible:
    @pytest.fixture
    def openai_calls(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class FakeImages:
            def generate(self, **kwargs: Any) -> SimpleNamespace:
                captured["generate"] = kwargs
                return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG_B64)])

        class FakeOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                captured["client"] = kwargs
                self.images = FakeImages()

        class FakeSecrets:
            def get_secret(self, name: str, *, should_error_on_not_found: bool = True) -> str | None:  # noqa: ARG002
                return {"OPENAI_API_KEY": "sk-openai", "GROK_API_KEY": "xai-key"}.get(name)

        monkeypatch.setattr(image_generation, "OpenAI", FakeOpenAI)
        monkeypatch.setattr(image_generation.GriptapeNodes, "SecretsManager", staticmethod(FakeSecrets))
        return captured

    def test_gpt_image_sends_gpt_image_options_without_response_format(self, openai_calls: dict[str, Any]) -> None:
        config = ImageGenerationConfig(
            provider=ImageProvider.OPENAI,
            model="gpt-image-1",
            image_size="1024x1024",
            quality="low",
            style="vivid",
            background="opaque",
            moderation="auto",
            output_format="jpeg",
            output_compression=80,
        )

        assert generate_image(config, "a dog") == PNG_BYTES

        assert openai_calls["client"] == {"api_key": "sk-openai", "base_url": None}
        assert openai_calls["generate"] == {
            "model": "gpt-image-1",
            "prompt": "a dog",
            "n": 1,
            "size": "1024x1024",
            "quality": "low",
            "background": "opaque",
            "moderation": "auto",
            "output_format": "jpeg",
            "output_compression": 80,
        }

    def test_dall_e_3_sends_style_and_b64_response_format(self, openai_calls: dict[str, Any]) -> None:
        config = ImageGenerationConfig(
            provider=ImageProvider.OPENAI, model="dall-e-3", image_size="1024x1792", quality="hd", style="natural"
        )

        generate_image(config, "p")

        assert openai_calls["generate"] == {
            "model": "dall-e-3",
            "prompt": "p",
            "n": 1,
            "size": "1024x1792",
            "quality": "hd",
            "style": "natural",
            "response_format": "b64_json",
        }

    def test_grok_uses_xai_endpoint_and_secret(self, openai_calls: dict[str, Any]) -> None:
        generate_image(ImageGenerationConfig(provider=ImageProvider.GROK, model="grok-2-image-1212"), "p")

        assert openai_calls["client"] == {"api_key": "xai-key", "base_url": "https://api.x.ai/v1"}
        assert openai_calls["generate"]["response_format"] == "b64_json"

    def test_missing_secret_raises(self, openai_calls: dict[str, Any]) -> None:
        config = ImageGenerationConfig(provider=ImageProvider.OPENAI, model="dall-e-2", api_key_secret="MISSING")

        with pytest.raises(KeyError, match="MISSING"):
            generate_image(config, "p")

        assert "client" not in openai_calls
