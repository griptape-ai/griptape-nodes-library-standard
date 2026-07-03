"""Shared helpers for situation dropdown parameters in save nodes."""

import logging

from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.retained_mode.events.project_events import (
    GetCurrentProjectRequest,
    GetCurrentProjectResultSuccess,
    GetProjectTemplateRequest,
    GetProjectTemplateResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

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


def build_situation_data(names: list[str], descriptions: dict[str, str]) -> list[dict]:
    """Build the ui_options data list for dropdown_row_subtitles display."""
    return [{"name": n, "subtitle": descriptions.get(n, "")} for n in names]
