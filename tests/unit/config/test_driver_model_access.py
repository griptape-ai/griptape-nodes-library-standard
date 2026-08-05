"""Tests that the config driver nodes wire their model dropdown through
``ModelAccessComponent``.

Every prompt/image driver node under ``griptape_nodes_library/config`` offers
its model list via ``BaseDriver._install_model_access``, which hands the
node's ``model`` parameter to a ``ModelAccessComponent``: the component owns
the ``Options`` + refresh ``Button`` traits, decorates each row with the
caller's license entitlement, and gates ``process()`` against the current
policy. These tests cover that wiring and the runtime gate across every node
that installs it; the component's own behavior is covered in the engine test
suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import requests
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import (
    AuthorizationCheckpoint,
    CheckpointAction,
    CheckpointDenial,
    CheckpointFailure,
)
from griptape_nodes.traits.button import Button
from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.base_driver import BaseDriver

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from griptape_nodes.exe_types.node_types import BaseNode

LIBRARY_NAME = "Griptape Nodes Library"

type AuthorizationHook = Callable[[AuthorizationCheckpoint], CheckpointDenial | None]

# (node type, catalog model id to deny). The dropdown stores this same catalog
# key, so there is no separate "dropdown value" column to track. The denied id
# is deliberately not the node's default everywhere it can be, so the
# ModelAccessComponent constructor's default-relocation logic (which moves the
# stored value off a denied default) never fires and can't confuse these
# assertions. CoherePrompt and GrokImage each declare exactly one model, so
# their only choice is necessarily also their default.
NODE_MODEL_CASES: list[tuple[str, str]] = [
    ("AnthropicPrompt", "gtc_claude_haiku_4_5"),
    ("CoherePrompt", "cohere_command_r_plus"),  # CoherePrompt's only declared model
    ("GriptapeCloudPrompt", "gtc_claude_sonnet_4_6"),
    ("GrokPrompt", "xai_grok_3_mini_beta"),
    ("GroqPrompt", "groq_llama_3_3_70b_versatile"),
    ("NimPrompt", "nim_gpt_oss_20b"),
    ("GriptapeCloudImage", "gtc_gpt_image_1_5"),
    ("GrokImage", "xai_grok_2_image_1212"),  # GrokImage's only declared model
    ("OpenAiImage", "openai_dall_e_3"),
]


def _create_node(node_type: str) -> BaseNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    The engine's model-access query reads those two metadata keys to resolve a
    node's declared models; a bare `NodeClass(name=...)` construction does not
    set them and would leave the dropdown undecorated.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


def _deny_hook(action: CheckpointAction, subject_id: str) -> AuthorizationHook:
    """Build a hook that denies one action against one catalog model id, else allows."""

    def hook(checkpoint: AuthorizationCheckpoint) -> CheckpointDenial | None:
        if checkpoint.action == action and checkpoint.subject_id == subject_id:
            return CheckpointDenial(failures=(CheckpointFailure(detail="denied for test"),))
        return None

    return hook


class _FakeCloudModelsResponse:
    """Stand-in for the `requests.Response` `GriptapeCloudPrompt._list_models` reads."""

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": [{"model_name": "gpt-4.1-mini", "default": True}]}


@pytest.fixture(autouse=True)
def _stub_griptape_cloud_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GriptapeCloudPrompt.__init__` calls `_list_models`, which hits Griptape Cloud's
    API over HTTP; stub `requests.get` so constructing the node never depends on network
    access. The library loader reimports each node's module from its file path, so
    patching the `GriptapeCloudPrompt` class imported here would miss the module
    instance the library actually registers; `requests.get` is shared regardless.
    """
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _FakeCloudModelsResponse())  # noqa: ARG005


@pytest.fixture
def authorization_hook() -> Iterator[Callable[[AuthorizationHook], None]]:
    """Register an authorization hook for the test, guaranteed removed afterward."""
    registered: list[AuthorizationHook] = []

    def register(hook: AuthorizationHook) -> None:
        GriptapeNodes.EventManager().add_authorization_hook(hook)
        registered.append(hook)

    try:
        yield register
    finally:
        for hook in registered:
            GriptapeNodes.EventManager().remove_authorization_hook(hook)


@pytest.mark.parametrize("node_type", [case[0] for case in NODE_MODEL_CASES])
def test_model_param_wired_to_model_access_component(node_type: str) -> None:
    node = cast("BaseDriver", _create_node(node_type))

    assert node._model_access is not None
    assert isinstance(node._model_access.component, ModelAccessComponent)

    model_param = node.get_parameter_by_name("model")
    assert model_param is not None
    # The component installs the Options dropdown and a refresh Button.
    assert model_param.find_elements_by_type(Options)
    assert model_param.find_elements_by_type(Button)
    # Per-row decoration the frontend uses to flag denied models.
    ui_options = model_param.ui_options
    assert ui_options.get("dropdown_row_icons") is True
    assert ui_options.get("dropdown_row_subtitles") is True
    assert isinstance(ui_options.get("data"), list)
    assert ui_options["data"]


@pytest.mark.parametrize(("node_type", "denied_catalog_id"), NODE_MODEL_CASES)
def test_denied_model_row_carries_denial_icon(
    node_type: str,
    denied_catalog_id: str,
    authorization_hook: Callable[[AuthorizationHook], None],
) -> None:
    # Registered before construction so the component's constructor-time snapshot
    # already carries the denial, and decorates the row on the first build.
    authorization_hook(_deny_hook(CheckpointAction.OFFER_MODEL, denied_catalog_id))
    node = _create_node(node_type)

    model_param = node.get_parameter_by_name("model")
    assert model_param is not None
    data = model_param.ui_options["data"]
    assert data

    for row in data:
        if row["name"] == denied_catalog_id:
            assert row.get("icon") == "shield-off"
        else:
            assert row.get("icon") is None


@pytest.mark.parametrize(("node_type", "denied_catalog_id"), NODE_MODEL_CASES)
def test_process_raises_when_selected_model_is_denied(
    node_type: str,
    denied_catalog_id: str,
    authorization_hook: Callable[[AuthorizationHook], None],
) -> None:
    """A denied model must not reach a downstream node as a driver.

    Construct the node BEFORE registering the deny hook, so the component's
    constructor-time snapshot is clean and doesn't relocate the stored value
    off the about-to-be-denied selection.
    """
    node = _create_node(node_type)
    authorization_hook(_deny_hook(CheckpointAction.OFFER_MODEL, denied_catalog_id))
    node.set_parameter_value("model", denied_catalog_id)

    with pytest.raises(RuntimeError, match="is not permitted"):
        node.process()
