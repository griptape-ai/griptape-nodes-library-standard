"""Tests that ``GenerateImage`` wires its model dropdown through ``ModelAccessComponent``.

The component owns the ``model`` parameter's ``Options`` + refresh ``Button``
traits, decorates each row with the caller's license entitlement, and gates the
run at execute time. These tests cover the wiring and the runtime gate; the
component's own behavior is covered in the engine test suite.
"""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial, CheckpointFailure
from griptape_nodes.traits.button import Button
from griptape_nodes.traits.options import Options

import griptape_nodes_library.image.create_image as create_image_module
from griptape_nodes_library.image.create_image import GenerateImage


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    class _FakeSecrets:
        def get_secret(self, _name: str) -> str | None:
            return value

    monkeypatch.setattr(create_image_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


@pytest.fixture
def generate_image_node(monkeypatch: pytest.MonkeyPatch) -> GenerateImage:
    _stub_secret(monkeypatch, "gt-cloud-key")
    node = GenerateImage(name="GenerateImage")
    node.set_parameter_value("model", "gpt-image-1-mini")
    node.set_parameter_value("prompt", "a cat wearing a hat")
    return node


def test_model_param_wired_to_model_access_component(generate_image_node: GenerateImage) -> None:
    assert isinstance(generate_image_node._model_access, ModelAccessComponent)

    model_param = generate_image_node.get_parameter_by_name("model")
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


def test_process_raises_when_model_access_denies_selected_model(
    generate_image_node: GenerateImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A denied image model fails closed before any driver is built."""
    denial = CheckpointDenial(failures=(CheckpointFailure(detail="GPT family is not in your plan."),))
    monkeypatch.setattr(generate_image_node._model_access, "query_for_denial", lambda _value: denial)

    gen = generate_image_node.process()
    with pytest.raises(RuntimeError, match="is not permitted"):
        next(gen)
