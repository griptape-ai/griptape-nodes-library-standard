"""Griptape Cloud drivers that stop when a budget refuses the call.

Upstream's drivers lose a budget refusal two ways, and these fix both:

- **They retry it.** Each driver adds ``BudgetExceededError`` to
  ``ignored_exception_types`` in ``__attrs_post_init__``, which keeps a call
  site's own list.
- **Streaming loses the body.** Upstream raises after the response is released,
  so the budget names are gone. ``try_stream`` is written out here to read the
  refusal while the response is open.

The halt names no node; ``NodeManager`` re-words it to name one.
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
from griptape_nodes.utils.budget_refusal import BudgetExceededError, refusal_from_exception
from griptape_nodes.utils.budget_refusal import describe as describe_budget_refusal
from griptape_nodes.utils.budget_refusal import log_line as budget_log_line

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape.common import Message, PromptStack

logger = logging.getLogger("griptape_nodes")

__all__ = ["GriptapeCloudImageGenerationDriver", "GriptapeCloudPromptDriver"]

MODULE_NAME = __name__
"""Written into a serialized driver's ``module_name`` so ``from_dict()`` rebuilds ours, not upstream's."""


def _budget_halt_for(exc: requests.exceptions.HTTPError, *, base_url: str) -> BudgetExceededError | None:
    """Return the halt a Cloud HTTP error is refusing with, or None.

    Only a response from ``base_url``'s host counts, and it has to be read while its body is
    still open.
    """
    refusal = refusal_from_exception(exc, cloud_host=urlsplit(base_url).hostname or "")
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
            halt = _budget_halt_for(exc, base_url=self.base_url)
            if halt is not None:
                raise halt from exc
            raise

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        url = griptape_cloud_url(self.base_url, "api/chat/messages/stream")
        params = self._base_params(prompt_stack)
        logger.debug(params)
        with requests.post(url, headers=self.headers, json=params, stream=True) as response:
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                halt = _budget_halt_for(exc, base_url=self.base_url)
                if halt is not None:
                    raise halt from exc
                raise

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
            halt = _budget_halt_for(exc, base_url=self.base_url)
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
            halt = _budget_halt_for(exc, base_url=self.base_url)
            if halt is not None:
                raise halt from exc
            raise
