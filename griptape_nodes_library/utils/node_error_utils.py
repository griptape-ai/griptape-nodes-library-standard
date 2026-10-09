"""Helpers for building the parts of a NodeError.

The engine shows a NodeError's `fields`, `response`, and `links` on their own lines in the editor's
error panel. These helpers keep those parts consistent across the library.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any
from urllib.parse import quote

from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink

__all__ = [
    "FieldValue",
    "MissingSecretError",
    "error_fields",
    "error_response",
    "missing_secret_error",
    "missing_secret_message",
    "parse_error_body",
    "provider_error_fields",
    "provider_error_message",
    "request_id_from_headers",
    "secret_link",
]

FieldValue = str | int | float | bool
"""A value NodeError accepts in `fields`."""

# Strings longer than this in a response are cut down. The engine drops a whole response over
# 16 KB, so one long value would otherwise hide every other part of it.
MAX_RESPONSE_STRING_CHARS = 2000

_DATA_URI = re.compile(r"^(data:[^;,]+;base64,)(.+)$", re.DOTALL)

# A provider error passed on as text: "Error code: 400 - {'error': {...}}". The part after the
# dash is the provider's body as a Python repr, not JSON.
_EMBEDDED_ERROR = re.compile(r"Error code: \d+ - (\{.*\})", re.DOTALL)


class MissingSecretError(NodeError, ValueError):
    """A secret the node needs is not set.

    A ValueError too, so the call sites that already catch a missing key as one still do.
    """


# Response headers providers use for the ID of a request, in the order they are checked.
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray")


def secret_link(secret_name: str, label: str = "Add the API key") -> NodeErrorLink:
    """A link that opens Settings → API Keys & Secrets filtered to one secret."""
    return NodeErrorLink(label=label, url=f"#settings-secrets?filter={quote(secret_name)}")


def missing_secret_message(secret_name: str) -> str:
    """The message for a secret that is not set, naming the exact secret and where to add it."""
    return f"{secret_name} is not set. Add it in Settings → API Keys & Secrets, then run the node again."


def missing_secret_error(
    secret_name: str,
    *,
    message: str | None = None,
    key_url: str | None = None,
    key_url_label: str = "Get an API key",
) -> MissingSecretError:
    """Build the error for a secret that is not set, linking to it in Settings.

    Args:
        secret_name: The secret, e.g. "OPENAI_API_KEY".
        message: Replaces the default message from :func:`missing_secret_message`.
        key_url: The provider's page for getting a key, added as a second link.
        key_url_label: The label for `key_url`.
    """
    links = [secret_link(secret_name)]
    if key_url:
        links.append(NodeErrorLink(label=key_url_label, url=key_url))
    return MissingSecretError(message or missing_secret_message(secret_name), links=links)


def request_id_from_headers(headers: Any) -> str | None:
    """The request ID a provider sent back in its response headers, or None."""
    if headers is None:
        return None
    for name in _REQUEST_ID_HEADERS:
        value = headers.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def error_response(response: Any) -> dict[str, Any] | None:
    """Return a provider response ready to attach to a NodeError, or None if there is nothing to attach.

    Base64 data is replaced with its length and very long strings are cut down, so the response
    stays under the engine's 16 KB limit and is still readable.
    """
    if not isinstance(response, dict) or not response:
        return None
    return _shrink(response)


def error_fields(**values: FieldValue | None) -> dict[str, FieldValue]:
    """Fields for a NodeError, such as a status code or an ID a user may quote to support.

    Values that are None or empty are left out, so callers can pass what they have.
    """
    fields: dict[str, FieldValue] = {}
    for key, value in values.items():
        if value is not None and value != "":
            fields[key] = value
    return fields


def parse_error_body(text: str) -> dict[str, Any] | None:
    """The provider error body inside a string, or None if the string isn't one.

    Handles a JSON object, and an "Error code: NNN - {...}" string with a Python repr after the
    dash, which is how Griptape Cloud passes on some provider errors.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            body = json.loads(stripped)
        except ValueError:
            body = None
        if isinstance(body, dict) and body:
            return body
    match = _EMBEDDED_ERROR.search(text)
    if match is None:
        return None
    try:
        body = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return body if isinstance(body, dict) and body else None


def provider_error_message(body: dict[str, Any] | None) -> str | None:
    """The provider's own explanation in an error body, or None if it has none.

    Looks in `error` (OpenAI, Anthropic), `detail` (FastAPI-style APIs such as ElevenLabs), and a
    top-level `message`.
    """
    error = _error_object(body)
    if error is not None:
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if body is not None:
        for key in ("error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def provider_error_fields(body: dict[str, Any] | None) -> dict[str, FieldValue]:
    """The provider's error code, the parameter it rejected, and its request ID, where present."""
    error = _error_object(body) or {}
    values: dict[str, FieldValue | None] = {}
    for field, key in (("error_code", "code"), ("parameter", "param"), ("request_id", "request_id")):
        value = error.get(key)
        if value is None and body is not None:
            value = body.get(key)
        values[field] = value if isinstance(value, str) else None
    return error_fields(**values)


def _error_object(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """The object holding the error's message: `body["error"]`, `body["detail"]`, or the body itself."""
    if body is None:
        return None
    for key in ("error", "detail"):
        value = body.get(key)
        if isinstance(value, dict):
            return value
    if "message" in body:
        return body
    return None


def _shrink(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shrink(item) for item in value]
    if isinstance(value, str):
        match = _DATA_URI.match(value)
        if match:
            prefix, data = match.groups()
            return f"{prefix}[{len(data)} chars]"
        if len(value) > MAX_RESPONSE_STRING_CHARS:
            return f"{value[:MAX_RESPONSE_STRING_CHARS]}... [{len(value)} chars total]"
    return value
