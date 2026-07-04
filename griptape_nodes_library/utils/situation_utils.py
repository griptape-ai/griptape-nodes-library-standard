"""Shared helpers for situation dropdown parameters in save nodes."""

import logging

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.project_events import (
    GetAllSituationsForProjectRequest,
    GetAllSituationsForProjectResultSuccess,
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


def fetch_situation_names() -> list[str]:
    """Return sorted situation names from the current project.

    Uses GetAllSituationsForProjectRequest, which is safe to call from __init__
    (does not trigger the reentrant-bus-in-init strict-mode rule). Falls back to
    [DEFAULT_SITUATION] when the project template is not yet loaded.
    """
    result = GriptapeNodes.handle_request(GetAllSituationsForProjectRequest())
    if not isinstance(result, GetAllSituationsForProjectResultSuccess):
        logger.warning("Could not fetch situation names; using default situation")
        return [DEFAULT_SITUATION]
    names = sorted(result.situations.keys())
    return names if names else [DEFAULT_SITUATION]


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


def update_file_param_situation(
    file_param: ProjectFileParameter, parameter: Parameter, value: object, **kwargs: object
) -> None:
    """Sync situation name to file_param when the situation parameter changes.

    Centralises the single write to the private ``_situation_name`` attribute so
    all five save nodes share one call site instead of five.  The ``initial_setup``
    guard mirrors ``FileOutputSettings.after_value_set`` — skip internal framework
    setup calls because ``add_situation_parameter`` already seeds the correct default.
    """
    if parameter.name == "situation" and not kwargs.get("initial_setup"):
        file_param._situation_name = str(value)


def add_situation_parameter(node: BaseNode, file_param: ProjectFileParameter) -> None:
    """Add a situation dropdown and refresh button to a save node.

    ``file_param`` must be created (but NOT yet added via ``add_parameter()``)
    before calling this so that ``after_value_set`` can safely reference it.

    Uses GetAllSituationsForProjectRequest (init-safe) for the initial list.
    Descriptions are not available until the user clicks the refresh button,
    which calls the slower two-request fetch that includes them.
    """
    names = fetch_situation_names()
    default = DEFAULT_SITUATION if DEFAULT_SITUATION in names else names[0]

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
    file_param._situation_name = default
