"""Guard the model id the declaration/permission layer resolves against.

`GriptapeProxyNode._submit_and_poll` declares the impending invocation by
matching `_get_catalog_model_id()` against the catalog's `provider_model_id`.
Nodes whose `_get_api_model_id()` decorates the id with an operation suffix for
the URL path (e.g. `grok-imagine-video:generate`) must keep the catalog id bare
or the declaration fails closed with "not a declared catalog model".
"""

from __future__ import annotations

import pytest

from griptape_nodes_library.image.grok_image_edit import GrokImageEdit
from griptape_nodes_library.image.grok_image_generation import GrokImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import GriptapeProxyNode
from griptape_nodes_library.video.grok_video_edit import GrokVideoEdit
from griptape_nodes_library.video.grok_video_generation import GrokVideoGeneration

# (node class, bare catalog provider id, suffixed url-path id)
GROK_NODES = [
    (GrokVideoGeneration, "grok-imagine-video", "grok-imagine-video:generate"),
    (GrokVideoEdit, "grok-imagine-video", "grok-imagine-video:edit"),
    (GrokImageGeneration, "grok-imagine-image", "grok-imagine-image:generate"),
    (GrokImageEdit, "grok-imagine-image", "grok-imagine-image:edit"),
]


@pytest.mark.parametrize(("node_class", "bare_id", "suffixed_id"), GROK_NODES)
def test_grok_catalog_id_is_bare_provider_id(
    node_class: type[GriptapeProxyNode], bare_id: str, suffixed_id: str
) -> None:
    node = node_class(name=node_class.__name__)

    # The URL-path id keeps the operation suffix; the catalog id must not, so it
    # matches the bare `provider_model_id` declared in the catalog.
    assert node._get_api_model_id() == suffixed_id
    assert node._get_catalog_model_id() == bare_id


def test_base_catalog_id_defaults_to_api_model_id() -> None:
    """Nodes that don't suffix their id resolve the catalog against the same id."""

    class _PlainProxyNode(GriptapeProxyNode):
        def _get_api_model_id(self) -> str:
            return "plain-model"

        async def _build_payload(self) -> dict[str, object]:  # pragma: no cover - unused
            return {}

        async def _parse_result(
            self, result_json: dict[str, object], generation_id: str
        ) -> None:  # pragma: no cover - unused
            return None

        def _set_safe_defaults(self) -> None:  # pragma: no cover - unused
            return None

    node = _PlainProxyNode(name="Plain")
    assert node._get_catalog_model_id() == node._get_api_model_id() == "plain-model"
