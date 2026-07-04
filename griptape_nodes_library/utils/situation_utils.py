"""Shared helpers for situation dropdown parameters in save nodes."""

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.project_events import (
    GetCurrentProjectRequest,
    GetCurrentProjectResultSuccess,
    GetProjectTemplateRequest,
    GetProjectTemplateResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")

DEFAULT_SITUATION = ProjectFileParameter.DEFAULT_SITUATION


def fetch_situations_with_descriptions() -> tuple[list[str], dict[str, str]]:
    """Return (sorted_names, {name: description}) from the current project.

    Falls back to a single-entry list containing the default situation when the
    project template is not yet loaded.
    """
    current = GriptapeNodes.handle_request(GetCurrentProjectRequest())
    if not isinstance(current, GetCurrentProjectResultSuccess):
        logger.warning("Could not fetch current project; using default situation")
        return [DEFAULT_SITUATION], {}

    template_result = GriptapeNodes.handle_request(
        GetProjectTemplateRequest(project_id=current.project_info.project_id)
    )
    if not isinstance(template_result, GetProjectTemplateResultSuccess):
        logger.warning("Could not fetch project template; using default situation")
        return [DEFAULT_SITUATION], {}

    situations = template_result.template.situations
    names = sorted(situations.keys())
    descriptions = {name: (sit.description or "") for name, sit in situations.items()}
    return names, descriptions


def build_situation_data(names: list[str], descriptions: dict[str, str]) -> list[dict[str, str]]:
    """Build the ui_options data list for dropdown_row_subtitles display."""
    return [{"name": n, "subtitle": descriptions.get(n, "")} for n in names]


def add_situation_parameter(node: Any, file_param: ProjectFileParameter) -> None:
    """Add a situation dropdown and refresh button to a save node.

    ``file_param`` must be created (but NOT yet added via ``add_parameter()``)
    before calling this so that ``after_value_set`` can safely reference it.
    """
    names, descriptions = fetch_situations_with_descriptions()
    default = DEFAULT_SITUATION if DEFAULT_SITUATION in names else (names[0] if names else DEFAULT_SITUATION)

    options_trait = Options(choices=names)

    def _on_refresh(_button: Button, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        refreshed_names, refreshed_descriptions = fetch_situations_with_descriptions()

        if not refreshed_names:
            return NodeMessageResult(
                success=False,
                details="No situations available — project template could not be loaded",
                response=button_details,
                altered_workflow_state=False,
            )

        # Update trait choices (for validation) and ui_options (for subtitle display).
        # Both are needed: the trait attribute governs value validation; update_ui_options
        # drives the rich dropdown rendering. FileOutputSettings omits the trait mutation
        # because it never refreshes in-place; here we need both.
        options_trait.choices = refreshed_names
        situation_param.update_ui_options({
            "data": build_situation_data(refreshed_names, refreshed_descriptions),
            "dropdown_row_subtitles": True,
        })

        current = node.get_parameter_value("situation")
        if current not in refreshed_names:
            new_val = (
                DEFAULT_SITUATION
                if DEFAULT_SITUATION in refreshed_names
                else refreshed_names[0]
            )
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
    situation_param.update_ui_options({
        "data": build_situation_data(names, descriptions),
        "dropdown_row_subtitles": True,
    })
    file_param._situation_name = default
