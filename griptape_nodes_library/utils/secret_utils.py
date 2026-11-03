"""Read a secret the way node code must: by request.

A request rather than ``GriptapeNodes.SecretsManager()``: in a worker process the request reaches the
process that owns the secrets, while the manager would answer from the worker's own copy.
"""

import logging

from griptape_nodes.retained_mode.events.secrets_events import GetSecretValueRequest, GetSecretValueResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


def get_secret(name: str, *, error_on_missing: bool = True) -> str | None:
    """The value of the secret `name`, or None when it isn't set.

    Args:
        name: The secret to read.
        error_on_missing: Whether a missing secret is logged as an error.
    """
    # The SecretsManager already logs a missing secret when error_on_missing is set.
    result = GriptapeNodes.handle_request(
        GetSecretValueRequest(
            key=name,
            should_error_on_not_found=error_on_missing,
            broadcast_result=False,
            failure_log_level=logging.DEBUG,
        )
    )
    if not isinstance(result, GetSecretValueResultSuccess):
        return None
    return result.value
