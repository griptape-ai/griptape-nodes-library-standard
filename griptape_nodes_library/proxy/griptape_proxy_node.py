from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.proxy.provider_asset_access import (
    missing_proxy_credential_message,
    resolve_proxy_base,
    resolve_proxy_credential,
)
from griptape_nodes_library.proxy.proxy_api_key_providers import get_proxy_api_key_provider_config
from griptape_nodes_library.proxy.proxy_auth_provider_parameter import ProxyAuthProviderParameter
from griptape_nodes_library.utils.model_invocation import declare_model_invocation

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial

logger = logging.getLogger("griptape_nodes")

__all__ = ["GriptapeProxyNode"]

STATUS_QUEUED = "QUEUED"
STATUS_CANCELLED = "CANCELLED"
STATUS_RUNNING = "RUNNING"
STATUS_ERRORED = "ERRORED"
STATUS_FAILED = "FAILED"
STATUS_COMPLETED = "COMPLETED"
STATUS_TIMED_OUT = "TIMED_OUT"

# Number of HTTP client errors (4xx) that indicate a permanent failure not worth retrying.
HTTP_CLIENT_ERROR_MIN = 400
HTTP_CLIENT_ERROR_MAX = 500
# Short delay before the single retry on a transient download failure.
DOWNLOAD_RETRY_DELAY_SECONDS = 1.0
# The proxy accepts cancellation only while a generation is still QUEUED; it answers
# 400 once the work has been dispatched to the provider.
HTTP_BAD_REQUEST = 400
# Cancellation is cleanup on the way out of a cancelled node, so it gets a short timeout.
CANCEL_REQUEST_TIMEOUT_SECONDS = 10


class CancelOutcome(StrEnum):
    """What a best-effort server-side cancel request actually achieved."""

    # The proxy dropped the generation while it was still queued; no billable work ran.
    CANCELLED = "cancelled"
    # The generation had already been dispatched, so it runs to completion and is billed.
    ALREADY_STARTED = "already_started"
    # The cancel request never got an answer; the generation's fate is unknown.
    UNKNOWN = "unknown"


class GriptapeProxyNode(SuccessFailureNode, ABC):
    """Base class for nodes that use the Griptape Cloud v2 async model proxy API.

    This class provides common functionality for nodes that:
    1. Submit generation requests to POST /api/proxy/v2/models/{model_id}
    2. Poll generation status via GET /api/proxy/v2/generations/{generation_id}
    3. Handle terminal states (COMPLETED, FAILED, ERRORED, CANCELLED)
    4. Fetch final results from GET /api/proxy/v2/generations/{generation_id}/result
    5. Cancel a still-queued generation via POST /api/proxy/v2/generations/{generation_id}/cancel
       when the node's execution is cancelled

    Subclasses must implement:
    - _build_payload(): Build the request payload for generation submission
    - _parse_result(): Parse the model-specific result data
    - _set_safe_defaults(): Clear output parameters on error

    This base class handles all polling logic, API error handling, and status management.
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    # Polling configuration
    DEFAULT_POLL_INTERVAL = 5
    DEFAULT_MAX_ATTEMPTS = 120  # 10 minutes with 5s intervals

    # Subclasses that download media set this to the destination for saved output.
    _output_file: ProjectFileParameter

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Compute API base once; GT_CLOUD_PROXY_BASE_URL overrides just the proxy
        # without affecting other engine systems that use GT_CLOUD_BASE_URL.
        self._proxy_base = resolve_proxy_base()
        self._user_auth_info: str | None = None
        self._api_key_provider: ProxyAuthProviderParameter | None = None
        self._initialize_api_key_provider()

        # Assigned by subclasses whose model selection is a license-filtered dropdown:
        # they construct a `ModelAccessComponent` over their model parameter and store it
        # here, which is what wires the dropdown into `after_value_set`,
        # `_get_api_model_id`, `_get_catalog_model_id`, and the `_submit_and_poll` gate.
        # Stays None on the subclasses bound to a single model.
        self._model_access: ModelAccessComponent | None = None

        default_timeout = self.DEFAULT_MAX_ATTEMPTS * self.DEFAULT_POLL_INTERVAL
        self.add_parameter(
            ParameterInt(
                name="timeout",
                default_value=default_timeout,
                tooltip="Polling timeout in seconds. Set to 0 for no timeout.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=0,
                max_val=86400,
            )
        )

    def _initialize_api_key_provider(self) -> None:
        provider_config = get_proxy_api_key_provider_config(type(self).__name__)
        if not provider_config:
            return

        self._api_key_provider = ProxyAuthProviderParameter(node=self, provider_config=provider_config)
        self._api_key_provider.add_parameters()

    def _create_status_parameters(
        self,
        *,
        result_details_tooltip: str = "Details about the operation result",
        result_details_placeholder: str = "Details on the operation will be presented here.",
        parameter_group_initially_collapsed: bool = True,
    ) -> None:
        super()._create_status_parameters(
            result_details_tooltip=result_details_tooltip,
            result_details_placeholder=result_details_placeholder,
            parameter_group_initially_collapsed=parameter_group_initially_collapsed,
        )
        # Inject generation_id, generation_status, and a Refresh button into the Status group.
        # The button is the affordance that lets users recover a result after timeout
        # without re-running the workflow.
        status_group = self.status_component.get_parameter_group()
        status_group.add_child(
            ParameterString(
                name="generation_id",
                default_value="",
                tooltip="Griptape Cloud generation ID. Preserved across timeouts and failures so the result can be recovered via the Refresh button.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False,
                hide=True,
                hide_property=True,
            )
        )
        status_group.add_child(
            ParameterString(
                name="generation_status",
                default_value="",
                tooltip="Latest known status of the generation (e.g., RUNNING, COMPLETED, TIMED_OUT).",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False,
            )
        )
        status_group.add_child(
            ParameterButton(
                name="generation_refresh",
                label="Refresh / Retrieve Result",
                icon="refresh-cw",
                variant="secondary",
                full_width=True,
                tooltip="Re-check the generation status and pull the result onto the node if completed.",
                on_click=self._on_refresh_clicked,
            )
        )

    def register_user_auth_info(self, user_auth_info: str | None) -> None:
        """Register optional user auth info to send with generation submissions."""
        self._user_auth_info = user_auth_info

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if self._api_key_provider:
            self._api_key_provider.after_value_set(parameter, value)
        if self._model_access is not None:
            self._model_access.on_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Refuse a model the caller's license denies, alongside the node's input checks.

        Subclasses append their own input validation to the list this returns, so
        reporting the denial here stops a missing prompt or image from being the only
        thing an artist hears about when the real blocker is the license. `super()`
        also resets the status parameters, so it must run either way.
        `_submit_and_poll` re-checks the selection for execution paths that skip
        validation entirely.
        """
        exceptions = super().validate_before_node_run() or []
        if self._model_access is None:
            return exceptions or None
        denial = self._model_access.selection_denial()
        if denial is None:
            return exceptions or None
        exceptions.append(RuntimeError(f"{self.name}: {denial.reason()}"))
        return exceptions

    def _get_selected_model_id(self) -> str:
        """The provider model id the model dropdown currently stores.

        The dropdown stores the provider's own id for the model, which is what a
        node building a request URL or payload needs. Reading it through here
        rather than by name keeps the parameter's name in one place: it is
        `model` on most nodes but `model_name` or `model_id` on others.

        Returns:
            str: The stored provider model id, or `""` when there is no
                model-access component installed or nothing is selected.
        """
        if self._model_access is None:
            return ""
        return self._model_access.selected_value or ""

    def _prepare_user_auth_info(self) -> None:
        self.register_user_auth_info(None)
        if not self._api_key_provider or not self._api_key_provider.is_user_auth_enabled():
            return

        user_auth_info = self._api_key_provider.get_user_auth_info()
        self.register_user_auth_info(user_auth_info)

    @abstractmethod
    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for generation submission.

        This method must be implemented by subclasses to build the model-specific
        payload that will be sent to POST /api/proxy/v2/models/{model_id}.

        This method is async to support operations like image downloading/encoding.

        Returns:
            dict: The request payload to send to the API
        """

    @abstractmethod
    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the model-specific result data and set output parameters.

        This method must be implemented by subclasses to parse the result data
        from GET /api/proxy/v2/generations/{generation_id}/result and set the
        appropriate output parameters.

        Args:
            result_json: The JSON response from the /result endpoint
            generation_id: The generation ID for this request
        """

    @abstractmethod
    def _set_safe_defaults(self) -> None:
        """Clear all output parameters on error.

        This method must be implemented by subclasses to reset all output
        parameters to safe default values when an error occurs.
        """

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from failed/errored generation response.

        Default implementation follows this hierarchy:
        1. status_detail.details (user-oriented message)
        2. entire status_detail object
        3. top-level error field
        4. full response

        Subclasses can override this to add model-specific error extraction logic.

        Args:
            response_json: The JSON response from the generation status endpoint
                          when status is FAILED or ERRORED

        Returns:
            str: A formatted error message to display to the user
        """
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        # First, try to extract from status_detail.details (user-oriented message)
        status_detail = response_json.get("status_detail")
        if status_detail and isinstance(status_detail, dict):
            details = status_detail.get("details")
            if details:
                return f"{self.name} {details}"

        # Try top-level error field
        error = response_json.get("error")
        if error:
            if isinstance(error, dict):
                error_msg = error.get("message") or error.get("error") or str(error)
                return f"{self.name} {error_msg}"
            return f"{self.name} {error}"

        # Try entire status_detail object
        if status_detail:
            return f"{self.name} generation failed.\n\nError details:\n{status_detail}"

        # Final fallback: show the full response
        return f"{self.name} generation failed.\n\nFull API response:\n{response_json}"

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation.

        Subclasses can override this if they need to map the dropdown value to a
        differently-shaped API ID (e.g. an operation suffix in the URL path). By
        default, returns the dropdown's stored provider model id; falls back to
        the raw 'model' parameter value when no model-access component is
        installed.

        Returns:
            str: The model ID to use in the API request
        """
        if self._model_access is not None:
            return self._get_selected_model_id()
        return self.get_parameter_value("model") or ""

    def _get_catalog_model_id(self) -> str:
        """Get the model ID used to resolve this node's declared catalog model.

        The declaration layer resolves this through the catalog's
        `provider_model_id`, so the bare stored value is what it needs. Falls
        back to `_get_api_model_id()` when no model-access component is installed.

        Subclasses whose `_get_api_model_id()` decorates the id with an operation
        suffix for the URL path (e.g. `grok-imagine-video:generate`) do not need
        to override this: the stored value is already the bare provider id.

        Returns:
            str: The model ID to match against declared catalog models
        """
        if self._model_access is not None:
            return self._get_selected_model_id()
        return self._get_api_model_id()

    def _validate_api_key(self) -> str:
        """Validate and return the credential used for proxy requests.

        Resolution spans GT_CLOUD_PROXY_API_KEY, the Griptape Nodes License, and
        GT_CLOUD_API_KEY (see `resolve_proxy_credential`), so the failure message names every
        credential the proxy accepts rather than only the last one checked.

        Returns:
            str: The credential to send as the bearer token

        Raises:
            ValueError: If no source holds a usable credential
        """
        credential = resolve_proxy_credential(self.API_KEY_NAME)
        if not credential.value:
            self._set_safe_defaults()
            msg = missing_proxy_credential_message(credential, attempted=f"run {self.name}")
            raise ValueError(msg)
        return credential.value

    def _log(self, message: str) -> None:
        """Log a message with error suppression."""
        with suppress(Exception):
            logger.info(message)

    def _log_auth_header_summary(self, context: str, headers: dict[str, str]) -> None:
        authorization = headers.get("Authorization", "")
        auth_scheme, _, auth_value = authorization.partition(" ")
        proxy_auth_info = headers.get("X-GTC-PROXY-AUTH-INFO", "")
        self._log(
            f"{context} auth headers: "
            f"authorization_present={bool(authorization)}, "
            f"authorization_scheme={auth_scheme or 'missing'}, "
            f"authorization_value_length={len(auth_value)}, "
            f"proxy_auth_info_present={bool(proxy_auth_info)}, "
            f"proxy_auth_info_length={len(proxy_auth_info)}"
        )

    def _elide_base64_in_payload(self, payload: dict[str, Any]) -> str:
        """Create a log-safe version of payload with base64 data elided.

        Replaces base64 strings in data URIs with length indicators to make logs readable.
        Example: "data:image/png;base64,iVBORw0K..." becomes "data:image/png;base64,[123 chars]"

        Args:
            payload: The payload dictionary to process

        Returns:
            JSON string with base64 data elided
        """

        def elide_value(obj: Any, *, key: str | None = None) -> Any:
            if isinstance(obj, str):
                # Match data URIs with base64 encoding
                match = re.match(r"^(data:[^;]+;base64,)(.+)$", obj)
                if match:
                    prefix, b64_data = match.groups()
                    return f"{prefix}[{len(b64_data)} chars]"
                if key == "bytesBase64Encoded":
                    return f"[{len(obj)} chars base64]"
                # Truncate any long string (>100 chars) to first 100 chars
                if len(obj) > 100:
                    return f"{obj[:100]}... [{len(obj)} chars total]"
                return obj
            elif isinstance(obj, dict):
                return {k: elide_value(v, key=k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [elide_value(item) for item in obj]
            return obj

        elided = elide_value(payload)
        return json.dumps(elided, indent=2)

    async def _submit_generation(
        self, payload: dict[str, Any], headers: dict[str, str], api_model_id: str
    ) -> str | None:
        """Submit generation request to the v2 API.

        Args:
            payload: The request payload
            headers: HTTP headers including Authorization
            api_model_id: The model ID to use in the URL

        Returns:
            str | None: The generation ID if successful, None otherwise

        Raises:
            RuntimeError: If the API request fails
        """
        proxy_url = urljoin(self._proxy_base, f"models/{api_model_id}")
        self._log(f"Submitting generation request to {proxy_url}")
        self._log(f"Request payload:\n{self._elide_base64_in_payload(payload)}")

        try:
            async with httpx.AsyncClient() as client:
                request_headers = headers.copy()
                if self._user_auth_info:
                    request_headers["X-GTC-PROXY-AUTH-INFO"] = self._user_auth_info
                self._log_auth_header_summary("Submitting generation request", request_headers)
                response = await client.post(proxy_url, json=payload, headers=request_headers, timeout=60)
                response.raise_for_status()
                response_json = response.json()
                self._log("Request submitted successfully")
        except httpx.HTTPStatusError as e:
            self._log(f"HTTP error: {e.response.status_code} - {e.response.text}")
            error_msg = self._extract_http_error_message(e.response)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            self._log(f"Request failed: {e}")
            msg = f"{self.name} request failed: {e}"
            raise RuntimeError(msg) from e

        generation_id = response_json.get("generation_id")
        if generation_id:
            self._log(f"Submitted. generation_id={generation_id}")
            return str(generation_id)

        self._log("No generation_id returned from POST response")
        return None

    def _extract_http_error_message(self, response: httpx.Response) -> str:
        """Extract error message from HTTP error response.

        Args:
            response: The HTTP response object

        Returns:
            str: Formatted error message
        """
        try:
            error_json = response.json()
        except Exception:
            return f"{self.name}: API error: {response.status_code} - {response.text}"
        else:
            error_message = self._extract_error_message(error_json)
            return f"{self.name}: {error_message}"

    def _handle_terminal_status(self, status: str, result_json: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        """Handle terminal generation statuses.

        Returns:
            tuple: (is_terminal, result_json_or_none)
        """
        if status == STATUS_COMPLETED:
            return True, result_json

        generation_id = self.parameter_output_values.get("generation_id", "") or ""

        if status in [STATUS_FAILED, STATUS_ERRORED]:
            logger.error("%s: Generation failed with status: %s", self.name, status)
            logger.error("%s: Error response: %s", self.name, result_json)
            self._set_safe_defaults()
            self.parameter_output_values["generation_id"] = generation_id
            self.parameter_output_values["generation_status"] = status
            error_message = self._extract_error_message(result_json)
            logger.error("%s: Extracted error message: %s", self.name, error_message)
            if not error_message:
                error_message = (
                    f"{self.name} generation failed with status {status} but no error details were provided."
                )
            self._set_status_results(was_successful=False, result_details=error_message)
            return True, None

        if status == STATUS_CANCELLED:
            logger.info("%s: Generation cancelled.", self.name)
            self._set_safe_defaults()
            self.parameter_output_values["generation_id"] = generation_id
            self.parameter_output_values["generation_status"] = status
            status_detail = result_json.get("status_detail", {})
            details = ""
            if isinstance(status_detail, dict):
                details = status_detail.get("details") or ""
            cancel_message = (
                f"{self.name} generation was cancelled."
                if not details
                else f"{self.name} generation was cancelled: {details}"
            )
            self._set_status_results(was_successful=False, result_details=cancel_message)
            return True, None

        return False, None

    def _resolve_timeout_seconds(self) -> int:
        try:
            value = self.get_parameter_value("timeout")
        except Exception:
            value = None
        if value is None:
            return self.DEFAULT_MAX_ATTEMPTS * self.DEFAULT_POLL_INTERVAL
        return max(0, int(value))

    async def _request_generation_cancel(self, generation_id: str, headers: dict[str, str]) -> CancelOutcome:
        """POST the proxy's cancel endpoint for a generation.

        Never raises. Cancellation is cleanup, and a failure here must not replace
        the reason the node is unwinding, so every failure mode collapses into a
        ``CancelOutcome`` the caller can report.

        Args:
            generation_id: The generation to cancel
            headers: HTTP headers including Authorization

        Returns:
            CancelOutcome: What the request achieved
        """
        cancel_url = urljoin(self._proxy_base, f"generations/{generation_id}/cancel")
        self._log(f"Requesting cancellation of generation {generation_id}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(cancel_url, headers=headers, timeout=CANCEL_REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            self._log(f"Cancel request for generation {generation_id} failed: {e}")
            return CancelOutcome.UNKNOWN

        # Both the accepted and the rejected response report the generation's status, and
        # the proxy is authoritative about it — record it so `generation_status` reflects
        # where the work actually ended up rather than the last status polling happened to see.
        with suppress(Exception):
            reported_status = response.json().get("status")
            if reported_status:
                self.parameter_output_values["generation_status"] = reported_status

        # A generation that has already left the queue cannot be cancelled. That is the
        # expected answer whenever the work was picked up quickly, so it is reported as
        # an outcome rather than surfaced as a node error.
        if response.status_code == HTTP_BAD_REQUEST:
            return CancelOutcome.ALREADY_STARTED
        if response.is_success:
            return CancelOutcome.CANCELLED

        self._log(f"Cancel request for generation {generation_id} returned HTTP {response.status_code}")
        return CancelOutcome.UNKNOWN

    async def _cancel_generation_best_effort(self, generation_id: str, headers: dict[str, str]) -> CancelOutcome:
        """Ask the proxy to drop a generation, then record what that achieved.

        The request is shielded because the usual caller is a cancellation unwind: a
        second ``task.cancel()`` landing on this node while the POST is in flight
        would otherwise abandon the request before it reaches the server.

        Args:
            generation_id: The generation to cancel
            headers: HTTP headers including Authorization

        Returns:
            CancelOutcome: What the request achieved
        """
        outcome = await asyncio.shield(self._request_generation_cancel(generation_id, headers))
        self._report_cancellation(generation_id, outcome)
        return outcome

    def _report_cancellation(self, generation_id: str, outcome: CancelOutcome) -> None:
        """Record on the node what the cancel attempt achieved.

        A cancel that could not stop billable work is the case worth telling the user
        about, so each outcome gets its own message. The generation_id is preserved so
        a generation that outlived the cancel can still be recovered via Refresh.

        Args:
            generation_id: The generation the cancel was requested for
            outcome: What the cancel request achieved

        Raises:
            ValueError: If the outcome is not a known CancelOutcome
        """
        match outcome:
            case CancelOutcome.CANCELLED:
                details = (
                    f"Generation `{generation_id}` was cancelled on Griptape Cloud before it started running, "
                    f"so it will not be billed."
                )
            case CancelOutcome.ALREADY_STARTED:
                details = (
                    f"Generation `{generation_id}` had already started and could not be cancelled. It will run to "
                    f"completion and be billed — click the refresh icon on `generation_status` to retrieve its result."
                )
            case CancelOutcome.UNKNOWN:
                details = (
                    f"Cancellation of generation `{generation_id}` could not be confirmed. It may still be running "
                    f"and be billed — click the refresh icon on `generation_status` to check."
                )
            case _:
                msg = f"Unknown cancel outcome: {outcome!r}"
                raise ValueError(msg)

        logger.info("%s: %s", self.name, details)
        self._set_safe_defaults()
        self.parameter_output_values["generation_id"] = generation_id
        self._set_status_results(was_successful=False, result_details=details)

    async def _poll_generation_status(self, generation_id: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """Poll generation status until terminal state is reached.

        Args:
            generation_id: The generation ID to poll
            headers: HTTP headers including Authorization

        Returns:
            dict | None: The final status response, or None if polling failed
        """
        get_url = urljoin(self._proxy_base, f"generations/{generation_id}")
        poll_interval = self.DEFAULT_POLL_INTERVAL
        timeout_s = self._resolve_timeout_seconds()
        # None means unbounded (timeout=0 set by user)
        max_attempts = max(1, (timeout_s + poll_interval - 1) // poll_interval) if timeout_s > 0 else None

        attempt = 0
        try:
            async with httpx.AsyncClient() as client:
                while True:
                    # Cooperative cancellation: covers a cancel that lands between awaits,
                    # and callers that set the flag without cancelling the asyncio task.
                    if self.is_cancellation_requested:
                        self._log(f"Cancellation requested while polling generation {generation_id}")
                        await self._cancel_generation_best_effort(generation_id, headers)
                        return None

                    try:
                        self._log(f"Polling attempt #{attempt + 1} for generation {generation_id}")

                        response = await client.get(get_url, headers=headers, timeout=60)
                        response.raise_for_status()
                        result_json = response.json()

                        status = result_json.get("status", "unknown")
                        self._log(f"Status: {status}")
                        self.parameter_output_values["generation_status"] = status

                        is_terminal, terminal_result = self._handle_terminal_status(status, result_json)
                        if is_terminal:
                            return terminal_result

                        attempt += 1

                        # Timeout reached (only when max_attempts is set)
                        if max_attempts is not None and attempt >= max_attempts:
                            break

                        # Still processing (QUEUED or RUNNING), wait before next poll
                        await asyncio.sleep(poll_interval)

                    except httpx.HTTPStatusError as e:
                        self._log(f"HTTP error while polling: {e.response.status_code} - {e.response.text}")
                        attempt += 1
                        if max_attempts is not None and attempt >= max_attempts:
                            self._set_safe_defaults()
                            error_msg = f"Failed to poll generation status: HTTP {e.response.status_code}"
                            self._set_status_results(was_successful=False, result_details=error_msg)
                            return None
                        await asyncio.sleep(poll_interval)
                    except Exception as e:
                        self._log(f"Error while polling: {e}")
                        attempt += 1
                        if max_attempts is not None and attempt >= max_attempts:
                            self._set_safe_defaults()
                            error_msg = f"Failed to poll generation status: {e}"
                            self._set_status_results(was_successful=False, result_details=error_msg)
                            return None
                        await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            # The engine cancels this node's task, which with a 5s poll interval almost
            # always lands mid-sleep — so this, not the flag check above, is the load-bearing
            # path. Ask the proxy to drop the generation before unwinding so queued work is
            # not billed for a result nobody will see, then let the cancellation stand.
            with suppress(asyncio.CancelledError):
                await self._cancel_generation_best_effort(generation_id, headers)
            raise

        # Timeout reached — preserve generation_id so the user can recover via Refresh
        self._log("Polling timed out waiting for result")
        self._set_safe_defaults()
        self.parameter_output_values["generation_id"] = generation_id
        self.parameter_output_values["generation_status"] = STATUS_TIMED_OUT
        self._set_status_results(
            was_successful=False,
            result_details=(
                f"Generation `{generation_id}` did not finish within {timeout_s} seconds. "
                f"It may still be running on Griptape Cloud — click the refresh icon on the "
                f"`generation_status` parameter to re-check and pull the result onto this node."
            ),
        )
        return None

    async def _fetch_generation_result(self, generation_id: str) -> dict[str, Any] | None:
        """Fetch the final result from the /result endpoint.

        Args:
            generation_id: The generation ID

        Returns:
            dict | None: The result JSON or dict containing raw bytes, or None if fetch failed
        """
        result_url = urljoin(self._proxy_base, f"generations/{generation_id}/result")
        self._log(f"Fetching result from {result_url}")

        try:
            api_key = self._validate_api_key()
        except ValueError as e:
            self._handle_api_key_validation_error(e)
            return None

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._log_auth_header_summary("Fetching generation result", headers)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(result_url, headers=headers, timeout=300)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._log(f"HTTP error fetching result: {e.response.status_code} - {e.response.text}")
            self._set_safe_defaults()
            error_msg = f"Failed to fetch generation result: HTTP {e.response.status_code}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            return None
        except Exception as e:
            self._log(f"Error fetching result: {e}")
            self._set_safe_defaults()
            error_msg = f"Failed to fetch generation result: {e}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            return None
        else:
            # Check Content-Type to determine if response is JSON or binary
            content_type = response.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                result_json = response.json()
                self._log("Result fetched successfully (JSON)")
                return result_json

            # Handle binary responses (raw audio, video, etc.)
            self._log(f"Result fetched successfully (binary, content-type: {content_type})")
            return {"raw_bytes": response.content}

    def _handle_api_key_validation_error(self, e: ValueError) -> None:
        """Handle API key validation errors."""
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details=str(e))
        logger.error("%s API key validation failed: %s", self.name, e)
        self._handle_failure_exception(e)

    def _handle_payload_build_error(self, e: Exception) -> None:
        """Handle payload building errors."""
        self._set_safe_defaults()
        error_msg = f"{self.name}: Failed to build request payload: {e}"
        self._set_status_results(was_successful=False, result_details=error_msg)
        self._handle_failure_exception(e)

    def _handle_missing_model_id(self) -> None:
        """Handle missing model ID error."""
        self._set_safe_defaults()
        error_msg = f"{self.name}: No model ID provided"
        self._set_status_results(was_successful=False, result_details=error_msg)

    def _handle_denied_model(self, denial: CheckpointDenial) -> None:
        """Handle a dropdown selection the license policy does not permit."""
        self._set_safe_defaults()
        error_msg = f"{self.name}: {denial.reason()}"
        self._set_status_results(was_successful=False, result_details=error_msg)

    def _handle_submission_error(self, e: RuntimeError) -> None:
        """Handle generation submission errors."""
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details=str(e))
        self._handle_failure_exception(e)

    def _handle_result_parsing_error(self, e: Exception) -> None:
        """Handle result parsing errors."""
        self._log(f"Error parsing result: {e}")
        self._set_safe_defaults()
        error_msg = f"Failed to parse generation result: {e}"
        self._set_status_results(was_successful=False, result_details=error_msg)
        self._handle_failure_exception(e)

    async def _submit_and_poll(self, headers: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
        """Submit generation request and poll for completion.

        Args:
            headers: HTTP headers including Authorization

        Returns:
            tuple | None: (generation_id, status_response) if successful, None otherwise
        """
        # Re-check the dropdown selection against the license policy: it may have
        # been permitted when the node was built and denied since. Both gates run
        # ahead of `_build_payload`, which uploads input images and videos to public
        # storage on the nodes that hand the provider a URL rather than bytes; a
        # denied model must not cost the caller that upload. The dropdown check runs
        # before the invocation declaration so the failure carries the dropdown's
        # own reason.
        if self._model_access is not None:
            selection_denial = self._model_access.selection_denial()
            if selection_denial is not None:
                self._handle_denied_model(selection_denial)
                return None

        # Declare the invocation so the engine's permission layer can gate it
        # before any network call. The proxy still enforces server-side; this is
        # the engine-side gate, so a denied invocation fails fast here. The
        # declaration resolves the bare provider model id, which may differ from
        # the URL-path id (e.g. when the latter carries an operation suffix).
        declaration = await declare_model_invocation(self, self._get_catalog_model_id())
        if declaration.failed():
            self._set_safe_defaults()
            details = str(declaration.result_details or f"{self.name}: model invocation was not permitted.")
            self._set_status_results(was_successful=False, result_details=details)
            return None

        # Build payload
        try:
            payload = await self._build_payload()
        except Exception as e:
            self._handle_payload_build_error(e)
            return None

        # Get API model ID
        api_model_id = self._get_api_model_id()
        if not api_model_id:
            self._handle_missing_model_id()
            return None

        # Submit request to get generation ID
        try:
            generation_id = await self._submit_generation(payload, headers, api_model_id)
            if not generation_id:
                self._set_safe_defaults()
                self._set_status_results(
                    was_successful=False,
                    result_details="No generation_id returned from API. Cannot proceed with generation.",
                )
                return None
        except RuntimeError as e:
            self._handle_submission_error(e)
            return None

        # Store generation_id so the Refresh affordance can recover the result on timeout/failure.
        # Subclasses declare a `generation_id` output parameter; writing here surfaces the value to the UI.
        self.parameter_output_values["generation_id"] = generation_id

        # Poll for completion
        status_response = await self._poll_generation_status(generation_id, headers)
        if not status_response:
            return None

        return generation_id, status_response

    async def _process_generation(self) -> None:
        """Main processing logic that orchestrates the generation flow.

        This method handles:
        1. API key validation
        2. Payload building
        3. Generation submission
        4. Status polling
        5. Result fetching and parsing
        """
        # Clear execution status at the start
        self._clear_execution_status()
        # A node's cancellation flag is only cleared by BaseNode.clear_node(), which the
        # flow's cancel path does not reach for a node cancelled mid-resolution — so the
        # flag outlives the run it belonged to. Left set, it would make the poll loop
        # cancel the generation this run is about to submit. A cancellation that really
        # applies to this run is requested after the run starts and also cancels the
        # asyncio task, which the poll loop's CancelledError path handles.
        self.clear_cancellation()
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["generation_status"] = ""

        try:
            self._prepare_user_auth_info()
            api_key = self._validate_api_key()
        except ValueError as e:
            self._handle_api_key_validation_error(e)
            return

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # Submit and poll
        result = await self._submit_and_poll(headers)
        if not result:
            return

        generation_id, _status_response = result

        # Fetch and parse result
        result_json = await self._fetch_generation_result(generation_id)
        if not result_json:
            return

        # Store provider_response if output parameter exists
        if "provider_response" in self.parameter_output_values:
            self.parameter_output_values["provider_response"] = result_json

        # Parse model-specific result
        try:
            await self._parse_result(result_json, generation_id)
        except Exception as e:
            self._handle_result_parsing_error(e)

    def _on_refresh_clicked(self, _button: Any, _details: Any) -> None:
        """Sync entry point for the Refresh button — bridges into the async refresh flow.

        Button.on_click_callback is invoked synchronously from a thread that may already
        have a running event loop, so we run the coroutine on a dedicated worker thread
        with its own fresh loop to avoid `RuntimeError: This event loop is already running`.
        """

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._refresh_async())
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, name=f"{self.name}-refresh", daemon=True)
        thread.start()
        thread.join()

    async def _fetch_status_for_refresh(self, generation_id: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """Single GET against the generations status endpoint for the Refresh flow.

        Sets failure status and returns None on HTTP/transport errors.
        """
        get_url = urljoin(self._proxy_base, f"generations/{generation_id}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(get_url, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            self._set_status_results(
                was_successful=False,
                result_details=f"Failed to fetch status for `{generation_id}`: HTTP {e.response.status_code}",
            )
        except Exception as e:
            self._set_status_results(
                was_successful=False,
                result_details=f"Failed to fetch status for `{generation_id}`: {e}",
            )
        return None

    async def _refresh_completed(self, generation_id: str) -> None:
        """Fetch and parse the result onto the node."""
        result_json = await self._fetch_generation_result(generation_id)
        if not result_json:
            self._set_status_results(
                was_successful=False,
                result_details=f"Generation `{generation_id}` is COMPLETED, but fetching the result failed. See node logs.",
            )
            return
        if "provider_response" in self.parameter_output_values:
            self.parameter_output_values["provider_response"] = result_json
        try:
            await self._parse_result(result_json, generation_id)
        except Exception as e:
            self._handle_result_parsing_error(e)
            self._set_status_results(
                was_successful=False,
                result_details=f"Generation `{generation_id}` completed, but parsing the result failed: {e}",
            )
            return
        self._set_status_results(
            was_successful=True,
            result_details=f"Refreshed: generation `{generation_id}` completed and result was retrieved.",
        )

    def _refresh_render_status(self, generation_id: str, status: str, status_json: dict[str, Any]) -> None:
        """Update result_details for non-completed states."""
        if status in (STATUS_FAILED, STATUS_ERRORED):
            error_message = self._extract_error_message(status_json)
            self._set_status_results(
                was_successful=False,
                result_details=f"Generation `{generation_id}` ended with status {status}.\n\n{error_message}",
            )
            return

        if status == STATUS_CANCELLED:
            status_detail = status_json.get("status_detail", {})
            details = ""
            if isinstance(status_detail, dict):
                details = status_detail.get("details") or ""
            body = (
                f"Generation `{generation_id}` was cancelled.\n\n{details}"
                if details
                else f"Generation `{generation_id}` was cancelled."
            )
            self._set_status_results(was_successful=False, result_details=body)
            return

        # QUEUED / RUNNING / unknown — still in flight
        self._set_status_results(
            was_successful=False,
            result_details=(
                f"Generation `{generation_id}` is still in progress (status: {status}). "
                f"Click the refresh icon again to re-check."
            ),
        )

    async def _refresh_async(self) -> None:
        """Re-check the generation status and pull the result if it has completed.

        A single GET to /generations/{id}; never re-enters the polling loop.
        """
        generation_id = (self.parameter_output_values.get("generation_id") or "").strip()
        if not generation_id:
            self._set_status_results(
                was_successful=False,
                result_details="No generation ID is available on this node yet. Run the node first to submit a generation.",
            )
            return

        try:
            api_key = self._validate_api_key()
        except ValueError as e:
            self._set_status_results(was_successful=False, result_details=f"Cannot refresh: {e}")
            return

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        status_json = await self._fetch_status_for_refresh(generation_id, headers)
        if status_json is None:
            return

        status = status_json.get("status", "unknown")
        self.parameter_output_values["generation_status"] = status

        if status == STATUS_COMPLETED:
            await self._refresh_completed(generation_id)
            return

        self._refresh_render_status(generation_id, status, status_json)

    async def aprocess(self) -> None:
        """Async processing entry point."""
        await self._process_generation()

    @staticmethod
    async def _download_bytes_from_url(url: str) -> bytes:
        """Download bytes from a URL, retrying once on transient failures.

        Transient failures (timeouts, connection errors, and 5xx responses) are
        retried a single time after a short delay. Permanent failures (4xx, e.g.
        an expired or missing provider URL) are raised immediately, since
        retrying cannot help. The underlying exception is propagated so callers
        can surface an actionable reason rather than a bare ``None``.

        Args:
            url: The URL to download from

        Returns:
            bytes: The downloaded bytes.

        Raises:
            httpx.HTTPError: If the download fails (after a retry for transient errors).
        """
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=120)
                    resp.raise_for_status()
                    return resp.content
            except httpx.HTTPStatusError as e:
                # 4xx are permanent (expired/missing URL); do not retry.
                if HTTP_CLIENT_ERROR_MIN <= e.response.status_code < HTTP_CLIENT_ERROR_MAX:
                    raise
                if attempt >= attempts:
                    raise
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= attempts:
                    raise
            await asyncio.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)

        # Unreachable: the loop either returns or raises on the final attempt.
        msg = f"Failed to download from {url}"
        raise httpx.HTTPError(msg)

    async def _download_and_save(
        self,
        url: str,
        output_param: str,
        artifact_factory: Callable[[str, str], Any],
        *,
        media_kind: str = "video",
        action: str = "generated",
    ) -> None:
        """Download media from a provider URL, save it to project storage, and set status.

        On success, saves the bytes via ``self._output_file`` and sets the given
        output parameter to the artifact produced by ``artifact_factory``. On any
        download or save failure, clears the output parameter and reports failure
        with an actionable message that names the provider URL, so the user can
        retrieve the asset manually. A generation that completed (and was billed)
        upstream but whose output cannot be retrieved is a failure, not a success.

        Args:
            url: The provider URL to download from.
            output_param: Name of the output parameter to set with the saved artifact.
            artifact_factory: Callable taking (value, name) and returning a ``*UrlArtifact``.
            media_kind: Human-readable media type for log and status messages.
            action: Past-tense verb describing what the node produced (e.g. "generated",
                "edited", "extended"), used in the success message.
        """
        try:
            logger.info("%s downloading %s from provider URL", self.name, media_kind)
            media_bytes = await self._download_bytes_from_url(url)
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(media_bytes)
            self.parameter_output_values[output_param] = artifact_factory(saved.location, saved.name)
            logger.info("%s saved %s as %s", self.name, media_kind, saved.name)
            self._set_status_results(
                was_successful=True,
                result_details=f"{media_kind.capitalize()} {action} successfully and saved as {saved.name}.",
            )
        except Exception as e:
            logger.error("%s failed to retrieve %s: %s", self.name, media_kind, e)
            self.parameter_output_values[output_param] = None
            self._set_status_results(
                was_successful=False,
                result_details=(
                    f"{self.name} generation completed upstream but the {media_kind} could not be retrieved: {e}. "
                    f"Provider URL (may be temporary): {url}"
                ),
            )
