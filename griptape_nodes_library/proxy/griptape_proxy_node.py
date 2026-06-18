from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin

import httpx
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.proxy.provider_asset_access import resolve_proxy_api_key, resolve_proxy_base
from griptape_nodes_library.proxy.proxy_api_key_providers import get_proxy_api_key_provider_config
from griptape_nodes_library.proxy.proxy_auth_provider_parameter import ProxyAuthProviderParameter

logger = logging.getLogger("griptape_nodes")

__all__ = ["GriptapeProxyNode"]

STATUS_QUEUED = "QUEUED"
STATUS_CANCELLED = "CANCELLED"
STATUS_RUNNING = "RUNNING"
STATUS_ERRORED = "ERRORED"
STATUS_FAILED = "FAILED"
STATUS_COMPLETED = "COMPLETED"
STATUS_TIMED_OUT = "TIMED_OUT"


class GriptapeProxyNode(SuccessFailureNode, ABC):
    """Base class for nodes that use the Griptape Cloud v2 async model proxy API.

    This class provides common functionality for nodes that:
    1. Submit generation requests to POST /api/proxy/v2/models/{model_id}
    2. Poll generation status via GET /api/proxy/v2/generations/{generation_id}
    3. Handle terminal states (COMPLETED, FAILED, ERRORED, CANCELLED)
    4. Fetch final results from GET /api/proxy/v2/generations/{generation_id}/result

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Compute API base once; GT_CLOUD_PROXY_BASE_URL overrides just the proxy
        # without affecting other engine systems that use GT_CLOUD_BASE_URL.
        self._proxy_base = resolve_proxy_base()
        self._user_auth_info: str | None = None
        self._api_key_provider: ProxyAuthProviderParameter | None = None
        self._initialize_api_key_provider()

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

        Subclasses can override this if they need to map friendly names to API IDs.
        By default, returns the value of the 'model' parameter if it exists.

        Returns:
            str: The model ID to use in the API request
        """
        return self.get_parameter_value("model") or ""

    def _validate_api_key(self) -> str:
        """Validate and return the API key.

        GT_CLOUD_PROXY_API_KEY overrides the key used for proxy requests
        without affecting other engine systems that use GT_CLOUD_API_KEY.

        Returns:
            str: The API key

        Raises:
            ValueError: If API key is missing
        """
        api_key = resolve_proxy_api_key(self.API_KEY_NAME)
        if not api_key:
            self._set_safe_defaults()
            msg = f"{self.name} is missing {self.API_KEY_NAME}. Ensure it's set in the environment/config."
            raise ValueError(msg)
        return api_key

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

        def elide_value(obj: Any) -> Any:
            if isinstance(obj, str):
                # Match data URIs with base64 encoding
                match = re.match(r"^(data:[^;]+;base64,)(.+)$", obj)
                if match:
                    prefix, b64_data = match.groups()
                    return f"{prefix}[{len(b64_data)} chars]"
                # Truncate any long string (>100 chars) to first 100 chars
                if len(obj) > 100:
                    return f"{obj[:100]}... [{len(obj)} chars total]"
                return obj
            elif isinstance(obj, dict):
                return {k: elide_value(v) for k, v in obj.items()}
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
        async with httpx.AsyncClient() as client:
            while True:
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
    async def _download_bytes_from_url(url: str) -> bytes | None:
        """Download bytes from a URL.

        Args:
            url: The URL to download from

        Returns:
            bytes | None: The downloaded bytes, or None if download failed
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=120)
                resp.raise_for_status()
                return resp.content
        except Exception:
            return None
