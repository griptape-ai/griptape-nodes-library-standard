"""Shared helpers for situation dropdown parameters in save nodes."""

import logging

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import FileDestinationProvider
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.project_events import (
    GetAllSituationsForProjectRequest,
    GetAllSituationsForProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")

DEFAULT_SITUATION = ProjectFileParameter.DEFAULT_SITUATION
_OUTPUT_FILE_PARAM = "output_file"


def on_output_file_connected(node: BaseNode, source_node: BaseNode, target_parameter: Parameter) -> None:
    """Call from after_incoming_connection to hide the situation dropdown.

    Only hides when the source is a FileDestinationProvider — the same predicate
    build_file() uses to bypass situation resolution. Plain-string connections to
    output_file still let the situation govern the save path, so the dropdown must
    remain visible.
    """
    if target_parameter.name == _OUTPUT_FILE_PARAM and isinstance(source_node, FileDestinationProvider):
        param = node.get_parameter_by_name("situation")
        if param is not None:
            param.hide = True


def on_output_file_disconnected(node: BaseNode, target_parameter: Parameter) -> None:
    """Call from after_incoming_connection_removed to restore the situation dropdown."""
    if target_parameter.name == _OUTPUT_FILE_PARAM:
        param = node.get_parameter_by_name("situation")
        if param is not None:
            param.hide = False


def fetch_situations() -> tuple[list[str], dict[str, str]]:
    """Return (sorted_names, {name: description}) from the current project.

    Uses GetAllSituationsForProjectRequest, which is safe to call from __init__
    (does not trigger the reentrant-bus-in-init strict-mode rule). Falls back to
    ([DEFAULT_SITUATION], {}) when the project template is not yet loaded.
    """
    try:
        # Request only the fields we need. The `fields` parameter was added in a later engine
        # release — see https://github.com/griptape-ai/griptape-nodes-engine/pull/5047.
        # Falls back to the unfiltered call on older engines that raise TypeError for the
        # unknown kwarg.
        result = GriptapeNodes.handle_request(GetAllSituationsForProjectRequest(fields=["situations", "descriptions"]))  # type: ignore[call-arg]
    except TypeError:
        result = GriptapeNodes.handle_request(GetAllSituationsForProjectRequest())
    if not isinstance(result, GetAllSituationsForProjectResultSuccess):
        logger.warning("Could not fetch situations; using default situation")
        return [DEFAULT_SITUATION], {}
    names = sorted(result.situations.keys())
    descriptions: dict[str, str] = getattr(result, "descriptions", {})
    return (names if names else [DEFAULT_SITUATION]), descriptions


def build_situation_data(names: list[str], descriptions: dict[str, str]) -> list[dict[str, str]]:
    """Build the ui_options data list for dropdown_row_subtitles display."""
    return [{"name": n, "subtitle": descriptions.get(n, "")} for n in names]


def add_situation_parameter(node: BaseNode, file_param: ProjectFileParameter) -> None:
    """Add a situation dropdown and refresh button to a save node.

    ``file_param`` must be created before calling this (it seeds ``_situation_name``
    to the dropdown default). Callers must resolve the situation fresh at process time:
    set ``file_param._situation_name = node.get_parameter_value("situation")`` at the
    top of each process/aprocess before calling ``build_file()``. Do NOT sync via
    ``after_value_set`` — that hook is skipped on workflow reload, causing files to
    land in the wrong situation directory after a save/reload cycle.

    TODO: remove the per-node sync lines once the engine resolves situation natively
    in ProjectFileParameter.build_file() — see
    https://github.com/griptape-ai/griptape-nodes-engine/issues/5105
    """
    names, descriptions = fetch_situations()
    default = DEFAULT_SITUATION if DEFAULT_SITUATION in names else names[0]

    options_trait = Options(choices=names)

    def _on_refresh(_button: Button, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        refreshed_names, refreshed_descriptions = fetch_situations()

        # Update trait choices (for validation) and ui_options (for subtitle display).
        # Both are needed: the trait attribute governs value validation; update_ui_options
        # drives the rich dropdown rendering.
        options_trait.choices = refreshed_names
        situation_param.update_ui_options(
            {
                "data": build_situation_data(refreshed_names, refreshed_descriptions),
                "dropdown_row_subtitles": True,
            }
        )

        current = node.get_parameter_value("situation")
        if current not in refreshed_names:
            new_val = DEFAULT_SITUATION if DEFAULT_SITUATION in refreshed_names else refreshed_names[0]
            node.set_parameter_value("situation", new_val)
            file_param._situation_name = new_val

        return NodeMessageResult(
            success=True,
            details=f"Refreshed {len(refreshed_names)} situation(s)",
            response=button_details,
            altered_workflow_state=True,
        )

    situation_param = ParameterString(
        name="situation",
        default_value=default,
        allowed_modes={ParameterMode.PROPERTY},
        tooltip="File save situation — determines the output directory and naming conventions.",
        traits={
            options_trait,
            Button(
                icon="refresh-cw",
                size="icon",
                variant="secondary",
                on_click=_on_refresh,
            ),
        },
        settable=True,
    )
    node.add_parameter(situation_param)
    # update_ui_options must follow add_parameter — subtitle data is pushed to the
    # UI layer after the param is registered.
    situation_param.update_ui_options(
        {
            "data": build_situation_data(names, descriptions),
            "dropdown_row_subtitles": True,
        }
    )
    file_param._situation_name = default
