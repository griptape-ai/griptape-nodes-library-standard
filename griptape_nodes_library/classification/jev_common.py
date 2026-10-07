from __future__ import annotations

import json
from typing import TYPE_CHECKING

from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

__all__ = [
    "CONTEXT_INPUT_TYPES",
    "DEFAULT_MODEL",
    "MODEL_CHOICES",
    "QUESTION_KEY",
    "add_context_parameter",
    "add_model_group",
    "to_state",
]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
QUESTION_KEY = "answer"
CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]


def to_state(text: str | None) -> str | dict | list | None:
    """Convert a context string to JEV state, or None if empty.

    JSON objects and arrays are parsed back out so JEV reads them as structured
    state rather than as raw text.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def add_context_parameter(node: BaseNode) -> None:
    """Add the shared Context parameter to a node."""
    node.add_parameter(
        ParameterString(
            name="context",
            display_name="Context",
            tooltip="The text JEV reads. JSON works too.",
            default_value="",
            multiline=True,
            placeholder_text="Text to ask about",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            input_types=CONTEXT_INPUT_TYPES,
        )
    )


def add_model_group(node: BaseNode) -> None:
    """Add the collapsed Advanced group with the model dropdown to a node."""
    with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced_group:
        ParameterString(
            name="model",
            display_name="Model",
            tooltip="jev-latest is the newest stable model. jev-preview is the newest release.",
            default_value=DEFAULT_MODEL,
            allow_input=False,
            allow_output=False,
            traits={Options(choices=MODEL_CHOICES)},
        )
    node.add_node_element(advanced_group)
