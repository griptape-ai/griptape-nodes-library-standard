"""Griptape Cloud drivers that stop when a budget refuses the call.

A HARD budget with no room makes Cloud answer 403 with a body naming every
budget that refused. The engine turns that body into a halt the artist can act
on -- ``griptape_nodes.utils.budget_refusal`` -- but only if the refusal reaches
it intact, and the framework drivers this library hands to ``griptape`` lose it
twice on the way.

**They retry it.** ``ExponentialBackoffMixin`` re-runs anything outside
``ignored_exception_types``, and a ``requests`` HTTP error is outside it by
default. A settled refusal is not a transient failure: every attempt is another
call Cloud refuses, so the node hangs for the backoff and then fails anyway.
Each driver here adds the halt to that tuple in ``__attrs_post_init__`` rather
than as a field default, so a call site that sets its own fail-fast list -- the
image node does -- keeps its choice and still stops on a budget.

**Streaming loses the body.** ``try_stream`` raises for status inside
``with requests.post(..., stream=True)``, and the exception leaves that block
carrying a response whose connection has been released: status 403 survives,
the JSON naming the budgets does not. Which is why the streaming method here is
written out rather than delegated -- the refusal has to be read while the
response is still open, and there is no hook into the middle of upstream's.

The halt these raise names no node. A driver is several frames below whichever
node is spending through it and cannot know its name; ``NodeManager`` does, and
re-words the message once the halt reaches it.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import requests
from attrs import define
from griptape.artifacts import ImageArtifact
from griptape.common import DeltaMessage, observable
from griptape.drivers.image_generation.griptape_cloud import (
    GriptapeCloudImageGenerationDriver as GtGriptapeCloudImageGenerationDriver,
)
from griptape.drivers.prompt.griptape_cloud import (
    GriptapeCloudPromptDriver as GtGriptapeCloudPromptDriver,
)
from griptape.utils.griptape_cloud import griptape_cloud_url
from griptape_nodes.utils.budget_refusal import BudgetExceededError, refusal_from_body
from griptape_nodes.utils.budget_refusal import describe as describe_budget_refusal
from griptape_nodes.utils.budget_refusal import log_line as budget_log_line

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape.common import Message, PromptStack

logger = logging.getLogger("griptape_nodes")

__all__ = ["GriptapeCloudImageGenerationDriver", "GriptapeCloudPromptDriver"]

MODULE_NAME = __name__
"""Where ``griptape`` should look when rebuilding one of these from a saved dict.

``to_dict()`` records only the class name, and both classes here are named after
the upstream driver they replace, so a rebuild finds upstream's unless the dict
also carries ``module_name``. ``agent_utils`` writes this in as it repairs the
credentials on a serialized agent, which is the one way a driver of ours crosses
a node boundary.
"""

_HTTP_FORBIDDEN = 403


def _budget_halt_for_response(response: requests.Response, *, cloud_host: str) -> BudgetExceededError | None:
    """Return the halt a still-open response is refusing with, or None.

    Takes the response rather than the exception because the only caller that
    needs it is streaming, where the exception outlives the body.
    """
    if response.status_code != _HTTP_FORBIDDEN:
        return None
    if urlsplit(response.url).hostname != cloud_host:
        return None

    try:
        body = response.json()
    except ValueError:
        # Not JSON, or empty. A 403 with nothing to read is somebody else's
        # refusal to explain.
        return None

    refusal = refusal_from_body(body)
    if refusal is None:
        return None

    logger.error(budget_log_line(refusal))
    return BudgetExceededError(describe_budget_refusal(refusal), refusal)


@define
class GriptapeCloudPromptDriver(GtGriptapeCloudPromptDriver):
    """The Cloud prompt driver, stopping on a budget refusal instead of retrying it."""

    def __attrs_post_init__(self) -> None:
        self.ignored_exception_types = (*self.ignored_exception_types, BudgetExceededError)

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        try:
            return super().try_run(prompt_stack)
        except requests.exceptions.HTTPError as exc:
            halt = _budget_halt_for_response(exc.response, cloud_host=urlsplit(self.base_url).hostname or "")
            if halt is not None:
                raise halt from exc
            raise

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        url = griptape_cloud_url(self.base_url, "api/chat/messages/stream")
        params = self._base_params(prompt_stack)
        logger.debug(params)
        with requests.post(url, headers=self.headers, json=params, stream=True) as response:
            halt = _budget_halt_for_response(response, cloud_host=urlsplit(self.base_url).hostname or "")
            if halt is not None:
                raise halt
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode("utf-8")
                if not decoded_line.startswith("data:"):
                    continue
                message_payload = decoded_line.removeprefix("data:").strip()
                logger.debug("Event stream data message payload: %s", message_payload)
                message_payload_dict = json.loads(message_payload)
                if "error" in message_payload_dict:
                    logger.error("Error in event stream data message: %s", message_payload_dict["error"])
                    raise RuntimeError(message_payload_dict["error"])
                yield DeltaMessage.from_dict(message_payload_dict)


@define
class GriptapeCloudImageGenerationDriver(GtGriptapeCloudImageGenerationDriver):
    """The Cloud image driver, stopping on a budget refusal instead of retrying it.

    Neither method streams, so both keep a readable body and can delegate.
    """

    def __attrs_post_init__(self) -> None:
        self.ignored_exception_types = (*self.ignored_exception_types, BudgetExceededError)

    def try_text_to_image(self, prompts: list[str], negative_prompts: list[str] | None = None) -> ImageArtifact:
        try:
            return super().try_text_to_image(prompts, negative_prompts)
        except requests.exceptions.HTTPError as exc:
            halt = _budget_halt_for_response(exc.response, cloud_host=urlsplit(self.base_url).hostname or "")
            if halt is not None:
                raise halt from exc
            raise

    def try_image_variation(
        self,
        prompts: list[str],
        image: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        try:
            return super().try_image_variation(prompts, image, negative_prompts)
        except requests.exceptions.HTTPError as exc:
            halt = _budget_halt_for_response(exc.response, cloud_host=urlsplit(self.base_url).hostname or "")
            if halt is not None:
                raise halt from exc
            raise
