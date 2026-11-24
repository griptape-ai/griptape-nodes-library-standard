from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry

LIBRARY_NAME = "Griptape Nodes Library"
_LICENSE = "header.payload.signature"
"""A Griptape Nodes License is a JWT: three dot-separated segments."""

TASK_NODE_TYPES = ["ScrapeWeb", "SearchWeb", "DateAndTime", "SummarizeText", "Askulator", "EvaluateTextResult"]


def _create_node(node_type: str) -> Any:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]) -> None:
    """Make secret resolution see only the Cloud credentials in *secrets*."""
    for name in ("GRIPTAPE_NODES_LICENSE", "GT_CLOUD_API_KEY"):
        if name in secrets:
            monkeypatch.setenv(name, secrets[name])
        else:
            monkeypatch.delenv(name, raising=False)


def _assert_names_both_credentials(exceptions: list[Exception] | None) -> None:
    assert exceptions is not None
    assert len(exceptions) == 1
    message = str(exceptions[0])
    assert "license" in message.lower()
    assert "GT_CLOUD_API_KEY" in message


@pytest.mark.parametrize("node_type", TASK_NODE_TYPES)
def test_task_node_fails_validation_without_a_credential(node_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {})
    node = _create_node(node_type)

    _assert_names_both_credentials(node.validate_before_workflow_run())
    _assert_names_both_credentials(node.validate_before_node_run())


@pytest.mark.parametrize("node_type", TASK_NODE_TYPES)
@pytest.mark.parametrize(
    "secrets", [{"GT_CLOUD_API_KEY": "gt-cloud-key"}, {"GRIPTAPE_NODES_LICENSE": _LICENSE}], ids=["api_key", "license"]
)
def test_task_node_passes_validation_with_either_credential(
    node_type: str, secrets: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secrets(monkeypatch, secrets)
    node = _create_node(node_type)

    assert node.validate_before_workflow_run() is None
    assert node.validate_before_node_run() is None


def test_search_web_reports_the_credential_and_the_engine_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {})
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    node = _create_node("SearchWeb")
    node.set_parameter_value("search_engine", "Exa")

    exceptions = node.validate_before_workflow_run()

    assert exceptions is not None
    messages = [str(e) for e in exceptions]
    assert any("EXA_API_KEY" in m for m in messages)
    assert any("license" in m.lower() for m in messages)


@pytest.mark.parametrize(
    ("input_text", "selection_type"), [("", "sentence"), ("", "paragraph"), ("! ", "sentence"), ("\n\n", "paragraph")]
)
def test_random_text_fails_validation_without_a_credential_when_it_generates(
    input_text: str, selection_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secrets(monkeypatch, {})
    node = _create_node("RandomText")
    node.parameter_values["input_text"] = input_text
    node.parameter_values["selection_type"] = selection_type

    _assert_names_both_credentials(node.validate_before_node_run())


@pytest.mark.parametrize(
    ("input_text", "selection_type"),
    [("", "word"), ("", "character"), ("One. Two.", "sentence"), ("One\n\nTwo", "paragraph")],
)
def test_random_text_needs_no_credential_when_it_does_not_generate(
    input_text: str, selection_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secrets(monkeypatch, {})
    node = _create_node("RandomText")
    node.parameter_values["input_text"] = input_text
    node.parameter_values["selection_type"] = selection_type

    assert node.validate_before_node_run() is None


def test_random_text_passes_validation_with_a_license_when_it_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {"GRIPTAPE_NODES_LICENSE": _LICENSE})
    node = _create_node("RandomText")
    node.parameter_values["input_text"] = ""
    node.parameter_values["selection_type"] = "sentence"

    assert node.validate_before_node_run() is None
