"""Tests for build_prompt_driver host-conversion logic."""

import pytest
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape_nodes.drivers.cloud_models import ProviderID

from griptape_nodes_library.utils.agent_utils import build_prompt_driver


@pytest.mark.parametrize(
    ("base_url", "expected_host"),
    [
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("http://localhost:11434/v1/", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("", None),
    ],
)
def test_ollama_host_from_base_url(base_url: str, expected_host: str | None) -> None:
    driver = build_prompt_driver(
        provider_type=ProviderID.OLLAMA,
        model="llama3",
        base_url=base_url,
    )
    assert isinstance(driver, OllamaPromptDriver)
    assert driver.host == expected_host


def test_non_ollama_type_returns_openai_compat_driver() -> None:
    driver = build_prompt_driver(
        provider_type="some_other_provider",
        model="gpt-4o",
        base_url="http://example.com/v1",
        api_key="sk-test",
    )
    assert isinstance(driver, OpenAiChatPromptDriver)


def test_missing_type_falls_through_to_openai_compat() -> None:
    driver = build_prompt_driver(
        provider_type=None,
        model="gpt-4o",
        base_url="http://example.com/v1",
    )
    assert isinstance(driver, OpenAiChatPromptDriver)


def test_openai_compat_uses_not_needed_fallback_when_no_api_key() -> None:
    driver = build_prompt_driver(
        provider_type=None,
        model="gpt-4o",
        base_url="http://example.com/v1",
        api_key=None,
    )
    assert isinstance(driver, OpenAiChatPromptDriver)
    assert driver.api_key == "not-needed"
