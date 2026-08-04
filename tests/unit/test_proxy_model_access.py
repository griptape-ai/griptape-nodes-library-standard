"""Tests that the proxy nodes wire their model dropdown through
``ModelAccessComponent``.

Every proxy node under ``griptape_nodes_library`` that offers a choice of
models routes its ``model``/``model_id`` parameter through
``GriptapeProxyNode._install_model_access``, which hands the parameter to a
``ModelDropdownAccess``: the component owns the ``Options`` + refresh
``Button`` traits, decorates each row with the caller's license entitlement,
and gates ``_submit_and_poll`` against the current policy. These tests cover
that wiring and the runtime gate across every node that installs it; the
component's own behavior is covered in the engine test suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
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

from griptape_nodes_library.proxy.griptape_proxy_node import GriptapeProxyNode
from griptape_nodes_library.video.sora_video_generation import SoraVideoGeneration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from griptape_nodes.exe_types.node_types import BaseNode

LIBRARY_NAME = "Griptape Nodes Library"

type AuthorizationHook = Callable[[AuthorizationCheckpoint], CheckpointDenial | None]

# (node type, catalog model id to deny, dropdown value that id resolves to, model
# parameter name). The denied id is deliberately not the node's default everywhere
# it can be, so the ModelAccessComponent constructor's default-relocation logic
# (which moves the stored value off a denied default) never fires and can't
# confuse these assertions. OmnihumanSubjectRecognition, OmnihumanSubjectDetection,
# WanReferenceToVideoGeneration, FluxImageGeneration, and TranscribeAudio each
# declare exactly one model, so their only choice is necessarily also their default.
NODE_MODEL_CASES: list[tuple[str, str, str, str]] = [
    ("TranscribeAudio", "gtc_whisper_1", "whisper-1", "model"),  # only declared model
    ("ElevenLabsTextToSpeechGeneration", "gtc_eleven_multilingual_v2", "eleven_multilingual_v2", "model"),
    (
        "OmnihumanSubjectRecognition",
        "gtc_omnihuman_1_5_subject_recognition",
        "omnihuman-1-5-subject-recognition",
        "model_id",
    ),  # only declared model
    (
        "OmnihumanSubjectDetection",
        "gtc_omnihuman_1_5_subject_detection",
        "omnihuman-1-5-subject-detection",
        "model_id",
    ),  # only declared model
    ("OmnihumanVideoGeneration", "gtc_omnihuman_1_0", "omnihuman-1-0", "model_id"),
    ("SoraVideoGeneration", "gtc_sora_2_pro", "sora-2-pro", "model"),
    ("WanTextToVideoGeneration", "gtc_wan_2_5_t2v_preview", "wan2.5-t2v-preview", "model"),
    ("WanImageToVideoGeneration", "gtc_wan_2_5_i2v_preview", "wan2.5-i2v-preview", "model"),
    ("WanAnimateGeneration", "gtc_wan_2_2_animate_move", "wan2.2-animate-move", "model"),
    ("WanReferenceToVideoGeneration", "gtc_wan_2_6_r2v", "wan2.6-r2v", "model"),  # only declared model
    ("FluxImageGeneration", "gtc_flux_kontext_pro", "flux-kontext-pro", "model"),  # only declared model
    ("QwenImageGeneration", "gtc_qwen_image_plus", "qwen-image-plus", "model"),
    ("QwenImageEdit", "gtc_qwen_image_edit", "qwen-image-edit", "model"),
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


@pytest.mark.parametrize(
    ("node_type", "param_name"),
    [(case[0], case[3]) for case in NODE_MODEL_CASES],
    ids=[case[0] for case in NODE_MODEL_CASES],
)
def test_model_param_wired_to_model_access_component(node_type: str, param_name: str) -> None:
    node = cast("GriptapeProxyNode", _create_node(node_type))

    assert node._model_access is not None
    assert isinstance(node._model_access.component, ModelAccessComponent)
    assert node._model_access.parameter_name == param_name

    model_param = node.get_parameter_by_name(param_name)
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


@pytest.mark.parametrize(("node_type", "denied_catalog_id", "dropdown_value", "param_name"), NODE_MODEL_CASES)
def test_denied_model_row_carries_denial_icon(
    node_type: str,
    denied_catalog_id: str,
    dropdown_value: str,
    param_name: str,
    authorization_hook: Callable[[AuthorizationHook], None],
) -> None:
    # Registered before construction so the component's constructor-time snapshot
    # already carries the denial, and decorates the row on the first build.
    authorization_hook(_deny_hook(CheckpointAction.OFFER_MODEL, denied_catalog_id))
    node = _create_node(node_type)

    model_param = node.get_parameter_by_name(param_name)
    assert model_param is not None
    data = model_param.ui_options["data"]
    assert data

    for row in data:
        if row["name"] == dropdown_value:
            assert row.get("icon") == "shield-off"
        else:
            assert row.get("icon") is None


@pytest.mark.asyncio
async def test_submit_and_poll_gates_on_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_submit_and_poll` re-checks the selection against the current policy.

    Construct the node BEFORE registering the deny hook, so the component's
    constructor-time snapshot is clean and doesn't relocate the stored value
    off the about-to-be-denied selection. Only one representative node
    (SoraVideoGeneration) is exercised here: the gate itself lives in
    `GriptapeProxyNode._submit_and_poll`, shared by every adopting node, so this
    isn't re-checked per node.
    """
    node = cast("SoraVideoGeneration", _create_node("SoraVideoGeneration"))
    node.set_parameter_value("model", "sora-2-pro")

    async def fake_build_payload() -> dict[str, Any]:
        return {}

    submit_calls: list[dict[str, Any]] = []

    async def fake_submit_generation(payload: dict[str, Any], headers: dict[str, str], api_model_id: str) -> None:
        submit_calls.append({"payload": payload, "headers": headers, "api_model_id": api_model_id})
        # None short-circuits `_submit_and_poll` before it reaches the poll loop,
        # which this test has no interest in exercising.
        return None

    monkeypatch.setattr(node, "_build_payload", fake_build_payload)
    monkeypatch.setattr(node, "_submit_generation", fake_submit_generation)

    hook = _deny_hook(CheckpointAction.OFFER_MODEL, "gtc_sora_2_pro")
    GriptapeNodes.EventManager().add_authorization_hook(hook)
    try:
        result = await node._submit_and_poll({})
    finally:
        GriptapeNodes.EventManager().remove_authorization_hook(hook)

    assert result is None
    assert node.parameter_output_values.get("was_successful") is False
    assert "denied for test" in node.parameter_output_values.get("result_details", "")
    assert submit_calls == []

    # Mirror case: with the hook removed, the same selection is permitted again
    # and the flow reaches `_submit_generation`.
    await node._submit_and_poll({})

    assert len(submit_calls) == 1
    assert submit_calls[0]["api_model_id"] == "sora-2-pro"
