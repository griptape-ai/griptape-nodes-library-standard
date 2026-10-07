from typing import Any

import requests
from griptape.artifacts import BaseArtifact, ErrorArtifact
from griptape.structures import Structure
from griptape.tasks import ActionsSubtask, BaseTask, PromptTask
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.utils.budget_refusal import BudgetExceededError

from griptape_nodes_library.utils.node_error_utils import (
    FieldValue,
    error_fields,
    error_response,
    parse_error_body,
    provider_error_fields,
    provider_error_message,
    request_id_from_headers,
)


def raise_if_budget_halt(agent_output: BaseArtifact | None) -> None:
    """Re-raise a budget halt that a griptape task caught and stored as its output.

    Griptape's `BaseTask.run` keeps a driver's exception on an `ErrorArtifact`, so
    without this a node would carry on or report a generic failure.
    """
    if isinstance(agent_output, ErrorArtifact) and isinstance(agent_output.exception, BudgetExceededError):
        raise agent_output.exception


def raise_if_budget_halt_in_run(run: Structure | BaseTask) -> None:
    """Re-raise a budget halt caught anywhere in a finished run, tool calls included.

    Griptape hands a tool's refusal back to the model as the tool's result (the
    Extraction Tool spends through Cloud), so each action is checked, not just the
    task output.
    """
    tasks = run.tasks if isinstance(run, Structure) else [run]
    for task in tasks:
        if isinstance(task, PromptTask):
            for subtask in task.subtasks:
                if isinstance(subtask, ActionsSubtask):
                    for action in subtask.actions:
                        raise_if_budget_halt(action.output)
        raise_if_budget_halt(task.output)


def try_throw_error(agent_output: BaseArtifact) -> None:
    """Raise a NodeError if the agent output is an ErrorArtifact.

    The message is the provider's own explanation. Its error body, status code, and request ID
    are attached to the NodeError.
    """
    raise_if_budget_halt(agent_output)
    if not isinstance(agent_output, ErrorArtifact):
        return
    exc = agent_output.exception
    response = _provider_error_body(exc)
    response = _find_embedded_body(exc, response, agent_output) or response
    msg = _provider_error_reason(exc, response, agent_output)
    raise NodeError(msg, fields=_provider_error_fields(exc, response), response=error_response(response)) from exc


def _provider_error_body(exc: BaseException | None) -> dict[str, Any] | None:
    """The provider's JSON error body, read from the exception's HTTP response.

    `requests.HTTPError` and the OpenAI and Anthropic SDK errors keep the response as `.response`.
    The SDK errors also keep the parsed body as `.body`, used when the response can't be read.
    """
    read_json = getattr(getattr(exc, "response", None), "json", None)
    if callable(read_json):
        try:
            body = read_json()
        # A body that isn't JSON, or a streamed response that was never read.
        except (ValueError, RuntimeError):
            body = None
        if isinstance(body, dict) and body:
            return body
    sdk_body = getattr(exc, "body", None)
    if isinstance(sdk_body, dict) and sdk_body:
        return sdk_body
    return None


def _find_embedded_body(
    exc: BaseException | None, response: dict[str, Any] | None, agent_output: ErrorArtifact
) -> dict[str, Any] | None:
    """The provider body Griptape Cloud passed on as text, wherever it ended up.

    It can be a string under the JSON body's "error", "detail", or "message" key, the raw
    response text, or only the exception's message.
    """
    candidates = [response.get(key) for key in ("error", "detail", "message")] if response else []
    http_response = getattr(exc, "response", None)
    candidates.extend([getattr(http_response, "text", None), str(exc) if exc else None, str(agent_output.value)])
    for candidate in candidates:
        if isinstance(candidate, str):
            body = parse_error_body(candidate)
            if body is not None:
                return body
    return None


def _provider_error_reason(
    exc: BaseException | None, response: dict[str, Any] | None, agent_output: ErrorArtifact
) -> str:
    """The provider's own explanation, used as the whole message.

    Without one, the response text or the exception's text, saying the agent run failed.
    """
    message = provider_error_message(response)
    if message is not None:
        return message
    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.text:
        return f"Agent run failed: {exc.response.text.strip()}"
    return f"Agent run failed: {agent_output.value}"


def _provider_error_fields(exc: BaseException | None, response: dict[str, Any] | None) -> dict[str, FieldValue]:
    """The status code, the provider's error code and the parameter it rejected, and the request ID."""
    http_response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(http_response, "status_code", None)
    fields = error_fields(status_code=status_code if isinstance(status_code, int) else None)
    fields.update(provider_error_fields(response))
    request_id = getattr(exc, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        request_id = request_id_from_headers(getattr(http_response, "headers", None))
    if request_id and "request_id" not in fields:
        fields["request_id"] = request_id
    return fields
