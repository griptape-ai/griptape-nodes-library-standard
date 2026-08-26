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
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy.provider_asset_access import resolve_proxy_api_key, resolve_proxy_base
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
# Local placeholder for a status response that names no status at all. Never published:
# `generation_status` is reconciled against the cloud, so it only ever carries a status the
# cloud actually reported (plus our own TIMED_OUT marker).
STATUS_UNKNOWN = "unknown"

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
# Upper bound on polling, and the meaning of `timeout=0`. This is also the `timeout`
# parameter's max_val, so 0 means "the longest this node is allowed to wait" rather than
# literally forever, which guarantees the poll loop always terminates and always reaches
# the code that reports a recoverable generation ID.
#
# Note what this does and does not buy: at 24 hours, a user who sets `timeout=0` still
# holds a concurrency slot far longer than any org would want. Bounding the loop does not
# make slot-holding safe — it makes it recoverable, because the loop now always reaches
# the timeout path that leaves a published generation ID behind.
MAX_TIMEOUT_SECONDS = 86400


class CancelOutcome(StrEnum):
    """What a best-effort server-side cancel request actually achieved."""

    # The proxy dropped the generation while it was still queued; no billable work ran.
    CANCELLED = "cancelled"
    # The generation had already been dispatched, so it runs to completion and is billed.
    ALREADY_STARTED = "already_started"
    # The cancel request never got an answer; the generation's fate is unknown.
    UNKNOWN = "unknown"


def _loop_time() -> float:
    """Monotonic event-loop time.

    Wrapped in a module-level function so tests can substitute a controllable clock without
    monkeypatching the stdlib `asyncio` module globally.
    """
    return asyncio.get_running_loop().time()


# Shortest dropdown value worth treating as the varying part of a model id. A one- or
# two-character value ("2", "v3") occurs inside ids by coincidence, and substituting over a
# coincidence manufactures candidate ids the node cannot actually produce.
MIN_SUBSTITUTABLE_CHOICE_LENGTH = 3
# The characters that delimit one segment of a model id from the next, in every id this
# library uses: `topaz-denoise`, `kling:motion-control`, `grok/flux-2-pro`.
MODEL_ID_SEGMENT_DELIMITERS = "-:/"


def _bare_model_id(model_id: str) -> str:
    """Normalize a model id for comparison: case-folded, unpadded, no leading/trailing slash."""
    return model_id.strip().lower().strip("/")


def _model_ids_match(left: str, right: str) -> bool:
    """Whether two normalized model ids name the same model.

    Three forms of the same id have to compare equal, because the id the node holds and the id
    the cloud reports are produced by different layers:

    * exactly equal;
    * one is a provider-prefixed form of the other (``grok/flux-2-pro`` vs ``flux-2-pro``) —
      deliberately segment-aligned rather than comparing final path segments, since
      ``kling/v2-1/master`` and ``wan/v2-2/master`` are different models that share a
      final ``master``;
    * one carries an operation suffix the other omits (``grok-imagine-video:generate`` vs
      ``grok-imagine-video``). The suffix is only dropped from a side that *has* one against a
      side that does not: dropping it from both would collapse ``kling:motion-control`` and
      ``kling:video-extend`` — the library's only two colon-bearing ids — onto ``kling`` and
      make the check a no-op between exactly the two nodes it exists to separate.
    """
    if not left or not right:
        return False
    if left == right or left.endswith(f"/{right}") or right.endswith(f"/{left}"):
        return True
    left_head, _, left_suffix = left.partition(":")
    right_head, _, right_suffix = right.partition(":")
    if bool(left_suffix) == bool(right_suffix):
        return False
    return _model_ids_match(left_head, right_head)


def _model_id_family(model_id: str) -> str:
    """The part of a model id a node's own dropdown cannot vary: everything but the last segment.

    Used only for nodes that cannot enumerate their model family, and only as a fallback after
    exact matching and dropdown substitution have both failed. Dropping the final segment is what
    lets ``topaz-video-slp-2.6`` and ``topaz-video-slp-2.5`` — the same node, one version
    dropdown apart — compare equal, while keeping ``gemini-omni-flash-preview`` out of
    ``gemini-3-pro-image``'s family and every cross-provider paste refused. Ids with no
    delimiter have no family and fall through to exact matching, which is what a single
    hardcoded id wants anyway.
    """
    head, delimiter, _tail = _rpartition_any(model_id, MODEL_ID_SEGMENT_DELIMITERS)
    return head if delimiter else ""


def _contains_whole_segment(model_id: str, candidate: str) -> bool:
    """Whether ``candidate`` occurs in ``model_id`` bounded by segment delimiters or the ends.

    ``topaz-denoise`` contains the operation ``denoise`` as a whole segment; it does not contain
    ``enoise``, and ``flux-2-pro`` does not contain ``ro``. Bounding the match is what keeps
    substitution from splicing a dropdown value into the middle of an unrelated segment.
    """
    if not candidate:
        return False
    start = model_id.find(candidate)
    while start != -1:
        end = start + len(candidate)
        before_ok = start == 0 or model_id[start - 1] in MODEL_ID_SEGMENT_DELIMITERS
        after_ok = end == len(model_id) or model_id[end] in MODEL_ID_SEGMENT_DELIMITERS
        if before_ok and after_ok:
            return True
        start = model_id.find(candidate, start + 1)
    return False


def _rpartition_any(value: str, delimiters: str) -> tuple[str, str, str]:
    """``str.rpartition`` against whichever of ``delimiters`` occurs last in ``value``."""
    index = max((value.rfind(delimiter) for delimiter in delimiters), default=-1)
    if index < 0:
        return "", "", value
    return value[:index], value[index], value[index + 1 :]


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
                tooltip=(
                    "Polling timeout in seconds — how long this node waits for the generation before giving up "
                    "and leaving it recoverable via Refresh. 0 means wait as long as this node is allowed to "
                    "(24 hours). Giving up does not cancel the generation on Griptape Cloud."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=0,
                max_val=MAX_TIMEOUT_SECONDS,
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
                tooltip=(
                    "Griptape Cloud generation ID. Preserved across timeouts and failures so the result can be "
                    "recovered via the Refresh button. You can also paste an ID copied from the editor's generation "
                    "tray and click Refresh to pull that generation's result onto this node — for binary results "
                    "this is the only way to retrieve them, since the editor cannot rehydrate raw bytes itself."
                ),
                # PROPERTY (settable, visible) is what makes adoption possible: a user who
                # lost the session that owned a generation can paste its ID here and click
                # Refresh. OUTPUT is retained so the node still reports the ID it submitted.
                # Being settable has a consequence worth knowing before touching any of this:
                # the engine unresolves the node and everything downstream of it on any
                # non-output set, so the paste that precedes Refresh invalidates the graph. That
                # is why `_retire_adopted_generation_id` pops `parameter_values` directly
                # instead of routing through `set_parameter_value` — "fixing" it into a proper
                # setter would unresolve downstream nodes in the middle of our own execution.
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
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

    @staticmethod
    def _reported_status(status_json: dict[str, Any]) -> str | None:
        """The status the cloud actually reported, or None if it named none.

        Distinguishing "no status" from a status matters because ``generation_status`` is
        reconciled against the cloud: publishing our own :data:`STATUS_UNKNOWN` sentinel
        would replace a good RUNNING badge with a neutral one on a single malformed response.
        Callers use the return value directly when publishing and fall back to the sentinel
        only for logging and local branching.
        """
        status = status_json.get("status")
        if isinstance(status, str) and status:
            return status
        return None

    def _publish_one(self, name: str, value: str) -> None:
        """Record one generation output value and push it to the editor immediately.

        Both channels matter and they are not the same. ``parameter_output_values`` is what
        downstream nodes read and what the editor displays; assigning to it also emits an
        ``AlterElementEvent`` for the displayed value. ``publish_update_to_parameter`` emits
        a ``ParameterValueUpdateEvent``, which is the channel the editor's generation tray
        registers on. A value that must reach the tray has to go through both, which is why
        every write in this class goes through here.

        The output write comes first and is unconditional: publishing is telemetry, and a
        failure to notify the editor must never be the reason a paid generation fails.

        Neither channel is deduplicated, so an unchanged status is re-announced on every poll —
        one event per poll interval for as long as the node waits. That is intentional: the
        output-value channel *is* change-gated by the engine, so an editor that connects
        part-way through a generation would otherwise see nothing at all until the status next
        changed, which for a long QUEUED job is the whole point of the tray. The volume is
        bounded by :data:`MAX_TIMEOUT_SECONDS` over the poll interval and is a rounding error
        next to the generation itself.
        """
        self.parameter_output_values[name] = value
        if self.get_parameter_by_name(name) is None:
            return
        with suppress(Exception):
            self.publish_update_to_parameter(name, value)

    def _publish_generation_state(self, *, generation_id: str | None = None, status: str | None = None) -> None:
        """Record generation identity/status and push them to the editor now.

        ``parameter_output_values`` is only flushed to downstream nodes when the node
        resolves, and the tray's channel is not written at all. A node that polls for an
        hour and then never resolves — because the session died — therefore never tells
        anyone which generation it started, which is precisely how a batch of paid
        generations became unreachable. Publishing on arrival is what makes the ID survive
        the session that created it.

        ``COMPLETED`` is dropped entirely here, on both channels. The editor reads
        ``COMPLETED`` as "the node has the result, stop offering recovery", so announcing it
        before the result is on the node would hide the recovery affordance for a result the
        node does not yet hold. Nothing needs the intermediate value — leaving
        ``generation_status`` at the last non-terminal status until ``_parse_result``
        succeeds is also the more honest badge. Use :meth:`_publish_generation_completed`.

        Args:
            generation_id: The generation ID to record, or None to leave it unchanged.
            status: The status to record, or None to leave it unchanged.
        """
        if generation_id is not None:
            self._publish_one("generation_id", generation_id)
        if status is not None and status != STATUS_COMPLETED:
            self._publish_one("generation_status", status)

    def _publish_generation_completed(self, generation_id: str | None = None) -> None:
        """Announce ``COMPLETED``, once the result is actually on the node.

        Ordering is load-bearing: the editor stops offering result recovery as soon as it
        sees ``COMPLETED``. Call this only after ``_parse_result`` has succeeded, so the
        affordance disappears exactly when it stops being needed.

        A node that has reported failure is refused here rather than trusted to the caller.
        "``_parse_result`` returned" is not the same as "the result is on the node": the most
        likely post-billing failure in this library — a media download that dies part-way — is
        *reported*, not raised (see ``_download_and_save``), so both callers would otherwise
        announce ``COMPLETED`` for a node holding nothing. ``_execution_succeeded`` is the same
        signal the engine routes control flow on and is only ever written by
        ``_set_status_results``, so this makes the badge agree with the node's own verdict.

        Args:
            generation_id: The cloud generation ID to re-assert alongside the status.
                ``_parse_result`` runs between the ID being published and this call, and
                some subclasses overwrite ``generation_id`` there with a provider-side task
                id — a value that 404s when the editor reconciles it against the cloud.
                This parameter means the cloud generation ID, so restore it last.
        """
        if generation_id:
            self._publish_one("generation_id", generation_id)
        if self._execution_succeeded is False:
            return
        self._publish_one("generation_status", STATUS_COMPLETED)

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
            self._publish_generation_state(generation_id=generation_id, status=status)
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
            self._publish_generation_state(generation_id=generation_id, status=status)
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
        """Resolve the polling timeout in seconds.

        ``0`` is clamped to :data:`MAX_TIMEOUT_SECONDS` rather than meaning "poll forever",
        so the loop always terminates and always reaches the code that leaves a recoverable
        generation ID behind. A 24-hour ceiling does not by itself stop a node from holding
        a concurrency slot for a punishing length of time; it stops the node from holding one
        *with no way out*, and the caller loses nothing by us giving up, because the result
        stays recoverable via Refresh.
        """
        try:
            value = self.get_parameter_value("timeout")
        except Exception:
            value = None
        if value is None:
            return self.DEFAULT_MAX_ATTEMPTS * self.DEFAULT_POLL_INTERVAL
        seconds = max(0, int(value))
        if seconds == 0:
            return MAX_TIMEOUT_SECONDS
        return min(seconds, MAX_TIMEOUT_SECONDS)

    @staticmethod
    def _polling_exhausted(attempt: int, max_attempts: int, deadline: float) -> bool:
        """Whether polling should stop, by either of its two independent bounds.

        Attempt-counting alone does not bound elapsed time: every iteration costs the HTTP
        request — up to 60s when a poll hangs — *plus* ``poll_interval``, so the old
        attempt cap let a 600s ``timeout`` keep a node polling for over two hours. The
        deadline bounds the time the user actually asked for, while the attempt cap keeps
        the loop finite regardless of the clock. Both live here so the loop's three exit
        checks cannot drift apart.

        Args:
            attempt: Number of poll attempts completed so far.
            max_attempts: Maximum number of attempts permitted.
            deadline: Event-loop monotonic time after which polling must stop.

        Returns:
            bool: True when either bound has been reached.
        """
        return attempt >= max_attempts or _loop_time() >= deadline

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
        #
        # Published rather than assigned, because this runs on a cancellation unwind: the node
        # is not going to resolve, and `parameter_output_values` alone is only flushed when it
        # does. A cancel that reports COMPLETED is dropped by `_publish_generation_state` like
        # any other, which is right here too — the node does not hold that result.
        with suppress(Exception):
            reported_status = response.json().get("status")
            if reported_status:
                self._publish_generation_state(status=reported_status)

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

        The ID is *published*, not just assigned. This method's own ALREADY_STARTED and
        UNKNOWN messages tell the user to click Refresh to retrieve a generation that is
        still running and still billing — and this runs on a cancellation unwind, where the
        node never resolves and so `parameter_output_values` is never flushed. Assigning
        alone would leave the ID nowhere the editor can see it, which is the exact way paid
        generations were lost before.

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
        # After `_set_safe_defaults()`, which blanks the ID — same ordering as every other
        # recovery path in this class.
        self._publish_generation_state(generation_id=generation_id)
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
        # Two independent bounds; whichever trips first ends polling. See _polling_exhausted.
        max_attempts = max(1, (timeout_s + poll_interval - 1) // poll_interval)
        deadline = _loop_time() + timeout_s

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

                    # Checked here as well as after each attempt: the post-attempt checks run
                    # before the sleep, so without this the loop can overshoot `timeout` by a
                    # whole request plus a whole interval (~65s) on its final pass.
                    if attempt > 0 and self._polling_exhausted(attempt, max_attempts, deadline):
                        break

                    try:
                        self._log(f"Polling attempt #{attempt + 1} for generation {generation_id}")

                        response = await client.get(get_url, headers=headers, timeout=60)
                        response.raise_for_status()
                        result_json = response.json()

                        reported_status = self._reported_status(result_json)
                        status = reported_status or STATUS_UNKNOWN
                        self._log(f"Status: {status}")
                        # Publish every intermediate status so the editor can track this
                        # generation even if the node never resolves. COMPLETED is withheld
                        # here and announced after _parse_result — see _publish_generation_state.
                        #
                        # An absent status is not a status: `generation_status` is reconciled
                        # against the cloud, so publishing our own "unknown" sentinel would
                        # replace a good RUNNING badge with a neutral one on a single malformed
                        # response. Leave the last status the cloud actually reported standing.
                        self._publish_generation_state(status=reported_status)

                        is_terminal, terminal_result = self._handle_terminal_status(status, result_json)
                        if is_terminal:
                            return terminal_result

                        attempt += 1

                        if self._polling_exhausted(attempt, max_attempts, deadline):
                            break

                        # Still processing (QUEUED or RUNNING), wait before next poll
                        await asyncio.sleep(poll_interval)

                    except httpx.HTTPStatusError as e:
                        self._log(f"HTTP error while polling: {e.response.status_code} - {e.response.text}")
                        attempt += 1
                        if self._polling_exhausted(attempt, max_attempts, deadline):
                            self._set_safe_defaults()
                            error_msg = f"Failed to poll generation status: HTTP {e.response.status_code}"
                            self._publish_generation_state(generation_id=generation_id, status=STATUS_TIMED_OUT)
                            self._set_status_results(was_successful=False, result_details=error_msg)
                            return None
                        await asyncio.sleep(poll_interval)
                    except Exception as e:
                        self._log(f"Error while polling: {e}")
                        attempt += 1
                        if self._polling_exhausted(attempt, max_attempts, deadline):
                            self._set_safe_defaults()
                            error_msg = f"Failed to poll generation status: {e}"
                            self._publish_generation_state(generation_id=generation_id, status=STATUS_TIMED_OUT)
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

        # Timeout reached — preserve generation_id so the user can recover via Refresh.
        # The generation is deliberately NOT cancelled here: the proxy only honours cancel
        # while a generation is QUEUED, and anything still running at this point is billed
        # either way, so cancelling would discard a result the user has already paid for.
        self._log("Polling timed out waiting for result")
        self._set_safe_defaults()
        self._publish_generation_state(generation_id=generation_id, status=STATUS_TIMED_OUT)
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
            # Same restore as the two transport branches below, and for the same reason: the
            # generation is COMPLETED and billed by the time this method runs. Reached when the
            # key is rotated or cleared between the final status poll and the result fetch.
            #
            # `finally` rather than a parameter on the handler, because twelve subclasses
            # override `_handle_api_key_validation_error` with the two-argument signature and
            # never call `super()`; passing a keyword through would raise `TypeError` on exactly
            # this path for those nodes. `finally` also satisfies both halves of the ordering
            # constraint on its own: it runs after the handler's `_set_safe_defaults()` has
            # blanked the ID, and it still runs when `_handle_failure_exception` re-raises
            # because the Failed output is unconnected. It restores the ID for the overriding
            # subclasses too, which their own handlers would never have done.
            try:
                self._handle_api_key_validation_error(e)
            finally:
                # Suppressed only in the `finally`: an exception raised from here would replace
                # the in-flight error the handler is re-raising, so the user would see an
                # unrelated failure and the flow would route on it.
                with suppress(Exception):
                    self._publish_generation_state(generation_id=generation_id)
            return None

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._log_auth_header_summary("Fetching generation result", headers)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(result_url, headers=headers, timeout=300)
                response.raise_for_status()
        # Both failure branches restore the generation ID after `_set_safe_defaults()` blanks
        # it. This is the likeliest of all the recovery paths to actually fire — a plain
        # network failure part-way through a large media download — and the generation is
        # COMPLETED and billed by the time we get here, so losing the ID would strand paid
        # work. `generation_status` is deliberately left at the last status the cloud
        # reported rather than COMPLETED: the node does not have the result.
        except httpx.HTTPStatusError as e:
            self._log(f"HTTP error fetching result: {e.response.status_code} - {e.response.text}")
            self._set_safe_defaults()
            self._publish_generation_state(generation_id=generation_id)
            error_msg = f"Failed to fetch generation result: HTTP {e.response.status_code}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            return None
        except Exception as e:
            self._log(f"Error fetching result: {e}")
            self._set_safe_defaults()
            self._publish_generation_state(generation_id=generation_id)
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
        """Handle API key validation errors.

        The signature is fixed by twelve subclasses that override this method without calling
        ``super()``, so the result-fetch caller restores ``generation_id`` around the call rather
        than through a parameter here — see :meth:`_fetch_generation_result`.
        """
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

    def _handle_result_parsing_error(self, e: Exception, *, generation_id: str | None = None) -> None:
        """Handle result parsing errors.

        Args:
            e: The exception raised by ``_parse_result``.
            generation_id: The generation whose result failed to parse, restored *after*
                ``_set_safe_defaults()`` has blanked it. Ordering is the whole point: every
                subclass's ``_set_safe_defaults`` sets ``generation_id`` to ``""``, so a
                caller that publishes the ID before calling this loses it again immediately,
                and Refresh then reports that the node has no generation to recover. The
                generation itself ran and was billed — only our parsing of it failed.

                No status accompanies it, deliberately. ``generation_status`` mirrors the
                cloud's vocabulary, in which ``ERRORED`` means "an internal failure; nothing
                is billed" — the opposite of what happened here. Leaving the last
                cloud-reported status in place keeps the badge honest and, being short of
                ``COMPLETED``, keeps recovery on offer. The parse failure itself is reported
                through ``result_details``.
        """
        self._log(f"Error parsing result: {e}")
        self._set_safe_defaults()
        if generation_id:
            self._publish_generation_state(generation_id=generation_id)
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

        # Publish the generation ID the moment we have one, before polling starts. This is
        # the single most important write in the class: until the editor has the ID, the
        # generation is billable work that nothing can point at, and a session that dies
        # mid-poll takes the only reference to it with it. Publishing (rather than writing
        # parameter_output_values, which is flushed only at node-resolve) is what makes the
        # ID survive a node that never resolves.
        #
        # The status is the one value published here that the cloud did not report: a
        # just-submitted generation is queued, and the tray needs something to show for the
        # interval before the first poll can replace it with a reconciled status.
        #
        # Retiring any pasted ID happens *here*, against the new ID, not at the top of the run:
        # a run that fails before this point (denied model, bad payload, submission error) must
        # leave the user's pasted ID intact, since it is the only pointer they have to the work
        # they were trying to recover. From this line on the new generation is the one that
        # needs recovering, so the paste has to go — including when polling later times out.
        self._retire_adopted_generation_id()
        self._publish_generation_state(generation_id=generation_id, status=STATUS_QUEUED)

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
        # Clear execution status at the start. Publish the cleared state too, so a re-run
        # retires the previous generation from the editor's tray instead of leaving a stale
        # ID that reconciles against work this node no longer represents. Only the *published*
        # state is cleared here; a pasted ID survives until there is a new one to replace it
        # with — see `_retire_adopted_generation_id`'s call site in `_submit_and_poll`.
        self._clear_execution_status()
        # A node's cancellation flag is only cleared by BaseNode.clear_node(), which the
        # flow's cancel path does not reach for a node cancelled mid-resolution — so the
        # flag outlives the run it belonged to. Left set, it would make the poll loop
        # cancel the generation this run is about to submit. A cancellation that really
        # applies to this run is requested after the run starts and also cancels the
        # asyncio task, which the poll loop's CancelledError path handles.
        self.clear_cancellation()
        self._publish_generation_state(generation_id="", status="")

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
            landed = await self._parse_result_onto_node(result_json, generation_id)
        except Exception as e:
            # The handler restores and publishes the ID itself, after its own
            # `_set_safe_defaults()` call and before `_handle_failure_exception` re-raises.
            # See `_handle_result_parsing_error` for why both orderings matter.
            self._handle_result_parsing_error(e, generation_id=generation_id)
            return

        # Only now is it true that the node has the result, which is what COMPLETED tells
        # the editor. The ID is re-asserted either way because `_parse_result` may have
        # overwritten it — see `_publish_generation_completed`.
        if not landed:
            self._publish_generation_state(generation_id=generation_id)
            return
        self._publish_generation_completed(generation_id)

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

    async def _parse_result_onto_node(self, result_json: dict[str, Any], generation_id: str) -> bool:
        """Run the subclass's ``_parse_result`` and report whether the result actually landed.

        Subclasses signal a *reported* failure — one they handled and explained rather than
        raised — by calling ``_set_status_results(was_successful=False)``, which is the only
        writer of ``_execution_succeeded``. Clearing that verdict first is what makes the return
        value describe this parse and not a previous run: ``_process_generation`` clears it at the
        top, but the Refresh path never does, and refresh sets it on nearly every branch.

        A subclass that reports nothing is treated as having succeeded, which is the behaviour
        every caller had before this seam existed.

        Args:
            result_json: The result payload to hand to ``_parse_result``.
            generation_id: The cloud generation ID, passed through unchanged.

        Returns:
            bool: True when the node holds the result, False when the subclass reported failure.

        Raises:
            Exception: Whatever ``_parse_result`` raises, for the callers' own handlers.
        """
        self._execution_succeeded = None
        await self._parse_result(result_json, generation_id)
        return self._execution_succeeded is not False

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
            landed = await self._parse_result_onto_node(result_json, generation_id)
        except Exception as e:
            self._handle_result_parsing_error(e, generation_id=generation_id)
            self._set_status_results(
                was_successful=False,
                result_details=f"Generation `{generation_id}` completed, but parsing the result failed: {e}",
            )
            return
        if not landed:
            # The subclass reported the failure and named the cause (typically a provider URL the
            # user can still fetch by hand), so leave its message standing rather than overwrite
            # it with a success line. Re-assert the ID so Refresh stays available.
            self._publish_generation_state(generation_id=generation_id)
            return
        # The result is on the node now, so COMPLETED is finally true to publish.
        self._publish_generation_completed(generation_id)
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

    def _retire_adopted_generation_id(self) -> None:
        """Drop a pasted generation ID when this node submits a generation of its own.

        Adoption has to be one-shot. A pasted ID lands in ``parameter_values``, which nothing
        else in this class ever writes, so without this it would outrank the node's own
        ``generation_id`` *forever* — and `_refresh_async` would then overwrite the real ID
        with the stale pasted one. Concretely: paste `gen-OLD` and refresh; run the node, so
        `gen-NEW` is submitted and billed and then times out; click Refresh, and the node
        fetches `gen-OLD` and destroys its only pointer to `gen-NEW`. That reproduces the
        original incident through the very mechanism meant to fix it, and the stale value
        persists into the saved workflow file.

        ``parameter_values`` is popped directly rather than going through
        ``set_parameter_value``, which would fire the value-changed machinery (and, on the
        request path, unresolve downstream nodes) in the middle of our own execution.
        """
        with suppress(Exception):
            self.parameter_values.pop("generation_id", None)

    def _resolve_refresh_generation_id(self) -> str:
        """Resolve which generation ID the Refresh button should act on.

        The property value wins. A user adopting a generation types the ID into the
        property, which lands in ``parameter_values``, not ``parameter_output_values`` — so
        reading only the output value (as this flow originally did) would silently refresh
        the node's own last generation instead of the one that was pasted. The output value
        remains the fallback for the ordinary post-timeout case, where the node submitted
        the generation itself and the property was never touched.

        The precedence is safe only because ``_retire_adopted_generation_id`` clears the
        property whenever this node submits a generation, which keeps a paste from
        outranking the node's own work indefinitely.

        Returns:
            str: The generation ID to refresh, or "" if none is available.
        """
        with suppress(Exception):
            pasted = self.get_parameter_value("generation_id")
            if pasted and str(pasted).strip():
                return str(pasted).strip()
        return (self.parameter_output_values.get("generation_id") or "").strip()

    def _with_shadowing_note(self, refusal: str) -> str:
        """Add a way out when a refused paste is hiding a generation this node submitted.

        Every refusal path returns without clearing the pasted value, and the paste outranks the
        node's own ``generation_id`` — so a user who pasted a malformed or foreign ID onto a node
        that has since timed out cannot reach their own generation by clicking Refresh again, and
        nothing in the message says why. Naming the shadowed ID makes the field's precedence
        visible at the moment it starts costing something.

        Args:
            refusal: The refusal message to extend.

        Returns:
            str: The message, with a note appended when a different ID is being shadowed.
        """
        own = (self.parameter_output_values.get("generation_id") or "").strip()
        if not own or own == self._resolve_refresh_generation_id():
            return refusal
        return (
            f"{refusal} Clear the `generation_id` field to refresh the generation this node "
            f"submitted (`{own}`) instead."
        )

    @staticmethod
    def _unusable_generation_id_reason(generation_id: str) -> str | None:
        """Reject a pasted value that would not address a generation, before it reaches a URL.

        Every ID is interpolated into ``generations/{id}`` and resolved with ``urljoin``, and
        making ``generation_id`` settable means this value is now user-typed rather than
        whatever the API handed back. ``urljoin`` resolves dot segments and honours ``?``/``#``,
        so a paste that includes a whole URL, a trailing query string, or ``../`` silently
        addresses a *different* endpoint with the user's key attached instead of failing with
        something a user can act on.

        Deliberately a denylist of the characters that change the request target rather than an
        allowlist of ID shapes: the asymmetry that governs the model check governs this one too
        — refusing a legitimate ID blocks the only route by which a binary result can be
        recovered, so the check may only reject input that could not have addressed a generation
        in the first place.

        Args:
            generation_id: The already-stripped candidate ID.

        Returns:
            str | None: An error message if the value cannot be an ID, else None.
        """
        # `%` is here because `urljoin` passes percent-escapes through undecoded, so whether
        # `%2e%2e%2f` redirects depends on gateway behaviour this library cannot see. Real IDs are
        # UUIDs, so rejecting `%` costs nothing.
        offending = {char for char in "/\\?#%" if char in generation_id}
        if any(char.isspace() for char in generation_id):
            offending.add("whitespace")
        if not offending:
            # A value made only of dots contains none of the above and still walks the path:
            # `..` resolves to the collection endpoint, `.` to `generations/`.
            if set(generation_id) == {"."}:
                return (
                    f"`{generation_id}` does not look like a generation ID (it is only dots). Paste just the ID "
                    f"from the editor's generation tray, without any surrounding URL or punctuation."
                )
            return None
        return (
            f"`{generation_id}` does not look like a generation ID (it contains "
            f"{', '.join(f'`{char}`' for char in sorted(offending))}). Paste just the ID from the editor's "
            f"generation tray, without any surrounding URL or punctuation."
        )

    def _extract_generation_model_id(self, status_json: dict[str, Any]) -> str:
        """Pull the model ID out of a generation status response, if it reports one.

        ``model_id`` is the field the proxy spec defines (and marks required) for a
        generation's model, so it is consulted first. The other keys are accepted because this
        library also runs against cloud deployments predating that spec, which is also why the
        caller treats an absent model as "cannot tell" rather than as a mismatch. Order
        matters: ``model`` is the key subclasses put a *friendly* label under in the request
        payload, so letting it win over the authoritative field would feed a display name into
        the comparison and manufacture a mismatch.

        Ordering is not enough on its own, though — the fallback keys exist precisely for
        deployments that send no ``model_id``, and on those the label is all that is left. So a
        value that cannot be an API id at all is treated as naming no model rather than as
        naming a foreign one: ``TopazImageEnhance`` sends ``"model": "Standard V2"`` and
        ``TopazVideoUpscale`` sends ``"Starlight Precise 2.6"``, and refusing on an echo of
        either would block recovery of the user's own generation.

        Args:
            status_json: The JSON response from the generation status endpoint.

        Returns:
            str: The reported model ID, or "" if the response does not name one.
        """
        for key in ("model_id", "model", "provider_model_id"):
            value = status_json.get(key)
            if isinstance(value, dict):
                value = value.get("model_id") or value.get("id") or value.get("name")
            if value and isinstance(value, str) and not any(char.isspace() for char in value):
                return value
        return ""

    def _supported_model_ids(self) -> set[str]:
        """Every API model id whose results this node's ``_parse_result`` can interpret.

        Result-shape compatibility is a property of the node *class*, not of whichever model
        the dropdown currently names: an ``LTXImageToVideoGeneration`` parses ``ltx-2-fast``
        and ``ltx-2-pro`` results identically. Comparing against only the current selection
        refuses a perfectly recoverable generation because an unrelated dropdown happens to
        have moved — and there is no other node type to send the user to, on the one route by
        which a raw-bytes result can reach them at all.

        The model-access component's ``model_choices`` is the node's list of *declared* models
        and stores provider model ids directly, so it is exactly the right set and is already
        the single source of truth for what a node can run. "Declared", not "permitted": an
        OFFER_MODEL denial decorates a dropdown row and gates execution, it does not remove the
        choice — which is what keeps a licence downgrade from making a node unable to recover a
        generation it ran before the downgrade. Nodes without a component fall back to their
        current ids, which is why the caller loosens the comparison for them.

        Every lookup is best-effort: the whole check exists to *widen* what Refresh accepts, so
        a subclass raising here must degrade to "cannot tell" (an empty set fails open) rather
        than take down the only route by which a binary result can be recovered.

        Returns:
            set[str]: Un-normalized candidate model ids, including this node's current API and
                catalog ids where those can be resolved.
        """
        candidates: set[str] = set()

        def offer(value: Any) -> None:
            # Guards the *return* of every source, not just the call: the annotations promise
            # `str` but nothing enforces them, and a `None` reaching the comparison would raise
            # inside `_refresh_model_mismatch` — on the refresh worker thread, where it surfaces
            # as no status at all rather than as a message the user can act on.
            if isinstance(value, str) and value.strip():
                candidates.add(value.strip())

        for get_id in (self._get_api_model_id, self._get_catalog_model_id):
            with suppress(Exception):
                offer(get_id())
        if self._model_access is not None:
            with suppress(Exception):
                for choice in self._model_access.model_choices:
                    offer(choice)
        with suppress(Exception):
            for model_id in self._dropdown_derived_model_ids():
                offer(model_id)
        return candidates

    def _dropdown_derived_model_ids(self) -> set[str]:
        """Every API model id this node's own dropdowns could produce, by substitution.

        Some subclasses build their API id by interpolating a dropdown value
        (``TopazImageEnhance`` returns ``f"topaz-{operation}"`` over nine operations) and have no
        ``ModelAccessComponent`` to enumerate, so ``_supported_model_ids`` would otherwise
        collapse to whatever the dropdown names *right now* — the exact basis the adoption check
        is documented not to use. Submit with ``enhance``, time out, move the dropdown, and your
        own billed generation becomes unrecoverable.

        Recovering the set by substitution is what lets the family fallback in
        :meth:`_refresh_model_mismatch` stay narrow. The alternative — widening that fallback far
        enough to cover nine operations, three of them hyphenated — also widened it for the nine
        single-hardcoded-id subclasses that gain nothing from it, and started accepting
        ``gemini-3-pro-image`` results on a ``gemini-omni-flash-preview`` node.

        Deliberately conservative, because a wrong candidate loosens the guard: a choice only
        substitutes when the node's *current* value for that dropdown appears in the current API
        id as a whole segment, and short values are skipped, so a coincidental match
        (``aspect_ratio`` of ``"2"`` inside ``flux-2-pro``) does not manufacture ids the node
        cannot produce.

        Returns:
            set[str]: API ids reachable by moving one dropdown, empty when the id is not
                dropdown-derived.
        """
        current = self._get_api_model_id()
        if not current:
            return set()

        derived: set[str] = set()
        for parameter in self.parameters:
            options = parameter.find_elements_by_type(Options)
            if not options:
                continue
            selected = self.get_parameter_value(parameter.name)
            if not isinstance(selected, str) or len(selected) < MIN_SUBSTITUTABLE_CHOICE_LENGTH:
                continue
            # Only a dropdown whose value is *part* of the id is a template. When the value is the
            # whole id the node is not interpolating anything — the dropdown already enumerates
            # every model, `model_choices` already contributes them, and substituting would just
            # re-add the choice list, which for a model dropdown also carries display labels
            # (`FLUX.2 [pro]`) and catalog keys (`gtc_flux_2_pro`) alongside the provider ids.
            if selected == current or not _contains_whole_segment(current, selected):
                continue
            for trait in options:
                for choice in trait.choices:
                    if isinstance(choice, str) and choice and not any(char.isspace() for char in choice):
                        derived.add(current.replace(selected, choice))
        derived.discard(current)
        return derived

    def _refresh_model_mismatch(self, status_json: dict[str, Any]) -> str | None:
        """Check that an adopted generation is one this node can actually interpret.

        Only a matching ``_parse_result`` knows how to rehydrate a given model's result
        shape, so pointing a node at an unrelated model's generation would at best fail
        confusingly and at worst write a mis-parsed result onto the node. Two deliberate
        biases, both because a false refusal is worse than a false accept here — a false
        accept surfaces as a parse error, while a false refusal blocks the only route by
        which a binary result can be recovered at all:

        * fail open when the response names no model;
        * compare against :meth:`_supported_model_ids`, the node's whole model family,
          rather than just its current selection;
        * for nodes that cannot enumerate that family, ignore the final id segment.

        Args:
            status_json: The JSON response from the generation status endpoint.

        Returns:
            str | None: An error message if the models definitely disagree, else None.
        """
        reported = self._extract_generation_model_id(status_json)
        if not reported:
            return None

        expected = {_bare_model_id(candidate) for candidate in self._supported_model_ids()} - {""}
        if not expected:
            return None

        reported_bare = _bare_model_id(reported)
        if any(_model_ids_match(reported_bare, candidate) for candidate in expected):
            return None

        # A node with no `ModelAccessComponent` cannot enumerate its declared models, so its
        # candidate set is only as good as what `_supported_model_ids` could reconstruct. Ignoring
        # the final segment covers the remaining case that reconstruction cannot: a version
        # dropdown whose values are *friendly labels* mapped to ids elsewhere
        # (`"Starlight Precise 2.6"` -> `topaz-video-slp-2.6`), where there is no substring to
        # substitute. `topaz-video-slp-2.5` and `topaz-video-slp-2.6` are one such dropdown apart.
        #
        # Kept to one segment because the widening is not free: every id in a family shares a
        # prefix, so a rule that drops more starts accepting foreign models
        # (`gemini-3-pro-image` on a `gemini-omni-flash-preview` node) for the nine subclasses
        # here whose id is a single hardcoded string and which gain nothing from any widening at
        # all. Cross-provider pastes — the actual footgun — are refused under either rule.
        if self._model_access is None:
            reported_family = _model_id_family(reported_bare)
            if reported_family and any(reported_family == _model_id_family(candidate) for candidate in expected):
                return None

        return (
            f"This generation was produced by model `{reported_bare}`, which this node cannot interpret, "
            f"so refresh was not performed. This node can read results from: "
            f"{self._advertised_model_ids(expected)}. Try a node that runs `{reported_bare}` instead."
        )

    def _advertised_model_ids(self, expected: set[str]) -> str:
        """The model list to name in a refusal, formatted for the message.

        Prefers the dropdown's own choices where there are any, because the point of the message
        is to name something the user can act on: ``expected`` is a union that also carries
        catalog ids and URL-path forms the dropdown never shows. Ids are listed with any operation
        suffix intact for the same reason — advertising `kling` rather than `kling:motion-control`
        names nothing a user could select.
        """
        listed: set[str] = set()
        if self._model_access is not None:
            with suppress(Exception):
                listed = {choice.strip() for choice in self._model_access.model_choices if isinstance(choice, str)}
        listed = {choice for choice in listed if choice} or expected
        return ", ".join(f"`{name}`" for name in sorted(listed))

    async def _refresh_async(self) -> None:
        """Re-check the generation status and pull the result if it has completed.

        A single GET to /generations/{id}; never re-enters the polling loop.

        Deliberately ungated by model-access policy, unlike ``_submit_and_poll``: this path
        invokes nothing and bills nothing, it retrieves a result the org has already paid for.
        Refusing here would strand paid work whenever a licence changed between submission and
        recovery, which is the exact failure this whole flow exists to prevent.
        """
        generation_id = self._resolve_refresh_generation_id()
        if not generation_id:
            self._set_status_results(
                was_successful=False,
                result_details=(
                    "No generation ID is available on this node yet. Run the node to submit a generation, or paste "
                    "an existing generation ID into `generation_id` and click Refresh."
                ),
            )
            return

        unusable = self._unusable_generation_id_reason(generation_id)
        if unusable is not None:
            self._set_status_results(was_successful=False, result_details=self._with_shadowing_note(unusable))
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

        mismatch = self._refresh_model_mismatch(status_json)
        if mismatch is not None:
            self._set_status_results(was_successful=False, result_details=self._with_shadowing_note(mismatch))
            return

        reported_status = self._reported_status(status_json)
        status = reported_status or STATUS_UNKNOWN
        # Record the ID we actually acted on, so an adopted generation is tracked by the
        # editor exactly like one this node submitted. An absent status is not published —
        # see the same guard in _poll_generation_status.
        self._publish_generation_state(generation_id=generation_id, status=reported_status)

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
