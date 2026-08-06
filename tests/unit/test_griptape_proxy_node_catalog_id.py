"""Guard the model id the declaration/permission layer resolves against.

`GriptapeProxyNode._submit_and_poll` declares the impending invocation by
matching `_get_catalog_model_id()` against the catalog's `model_id` (via
`resolve_catalog_model_id`, which resolves a provider id to its catalog key).
By default `_get_catalog_model_id()` returns the model dropdown's stored value
verbatim -- that value already IS the upstream provider's own model id --
while `_get_api_model_id()` returns the same value and, on nodes that decorate
the URL path with an operation, appends a `:suffix`.

The per-node cases below pin the ids of the nodes that decorate their API id;
the sweep at the bottom of the module holds every proxy node in the library to
the same contract, so a new node cannot reintroduce a mismatch.

Nodes are constructed through `LibraryRegistry` so their metadata carries
`library` / `node_type`: `_get_selected_model_id()` (and so
`_get_api_model_id()`) resolves through the node's declared models, which
requires that metadata. A bare `NodeClass(name=...)` construction leaves it
unset and resolution returns `""`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry, get_declared_models

from griptape_nodes_library.image.grok_image_edit import GrokImageEdit
from griptape_nodes_library.image.grok_image_generation import GrokImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import GriptapeProxyNode
from griptape_nodes_library.utils.model_invocation import resolve_catalog_model_id
from griptape_nodes_library.video.grok_video_edit import GrokVideoEdit
from griptape_nodes_library.video.grok_video_generation import GrokVideoGeneration
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.kling_text_to_video_generation import KlingTextToVideoGeneration
from griptape_nodes_library.video.ltx_audio_to_video_generation import LTXAudioToVideoGeneration
from griptape_nodes_library.video.ltx_image_to_video_generation import LTXImageToVideoGeneration
from griptape_nodes_library.video.ltx_text_to_video_generation import LTXTextToVideoGeneration
from griptape_nodes_library.video.ltx_video_extend import LTXVideoExtend
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.ltx_video_to_video_hdr import LTXVideoToVideoHDR

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node(node_type: str) -> GriptapeProxyNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    `_get_selected_model_id()` reads a node's declared models to resolve
    the dropdown's stored value to the upstream provider's id; a bare
    `NodeClass(name=...)` construction does not set the metadata that lookup needs
    and would silently resolve to `""`.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("GriptapeProxyNode", library.create_node(node_type=node_type, name=node_type))


# (node class, default provider model id, expected "<provider id>:<suffix>" API id)
GROK_NODES = [
    (GrokVideoGeneration, "grok-imagine-video", "grok-imagine-video:generate"),
    (GrokVideoEdit, "grok-imagine-video", "grok-imagine-video:edit"),
    (GrokImageGeneration, "grok-imagine-image", "grok-imagine-image:generate"),
    (GrokImageEdit, "grok-imagine-image", "grok-imagine-image:edit"),
]


@pytest.mark.parametrize(("node_class", "catalog_id", "api_id"), GROK_NODES)
def test_grok_catalog_id_is_bare_provider_id(node_class: type[GriptapeProxyNode], catalog_id: str, api_id: str) -> None:
    node = _create_node(node_class.__name__)

    # The catalog id is the dropdown's stored value verbatim; the API id is that
    # same selection with the URL-path operation appended.
    assert node._get_catalog_model_id() == catalog_id
    assert node._get_api_model_id() == api_id


# (node class, default provider model id, expected "<provider id>:<suffix>" API id) for
# each node's default dropdown selection. The catalog ids are cross-checked against the
# `model_usage` ids each node declares in griptape_nodes_library.json's `model_catalog`
# metadata.
SUFFIXED_NODES = [
    (LTXTextToVideoGeneration, "ltx-2-3-fast", "ltx-2-3-fast:text-to-video"),
    (LTXImageToVideoGeneration, "ltx-2-3-fast", "ltx-2-3-fast:image-to-video"),
    (LTXAudioToVideoGeneration, "ltx-2-pro", "ltx-2-pro:audio-to-video"),
    (LTXVideoExtend, "ltx-2-3-pro", "ltx-2-3-pro:extend"),
    (LTXVideoRetake, "ltx-2-pro", "ltx-2-pro:retake"),
    (LTXVideoToVideoHDR, "ltx-2-3-pro", "ltx-2-3-pro:video-to-video-hdr"),
    (KlingTextToVideoGeneration, "kling-v3", "kling-v3:text2video"),
    (KlingImageToVideoGeneration, "kling-v3", "kling-v3:image2video"),
    (KlingOmniVideoGeneration, "kling-v3-omni", "kling-v3-omni:omnivideo"),
]


@pytest.mark.parametrize(("node_class", "catalog_id", "api_id"), SUFFIXED_NODES)
def test_suffixed_catalog_id_is_bare_provider_id(
    node_class: type[GriptapeProxyNode], catalog_id: str, api_id: str
) -> None:
    node = _create_node(node_class.__name__)

    assert node._get_catalog_model_id() == catalog_id
    assert node._get_api_model_id() == api_id


def test_base_catalog_id_defaults_to_api_model_id() -> None:
    """A node with no model-access component falls back to `_get_api_model_id()`."""

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


def _node_types_declaring_models() -> list[str]:
    """Every node type in the manifest that declares model usage, in manifest order.

    Read from the manifest rather than a hand-kept list so a node added later is
    swept in automatically. This runs at collection time, before the library is
    registered, so it cannot consult the registry for node classes.
    """
    manifest_path = Path(__file__).parents[2] / "griptape_nodes_library.json"
    manifest = json.loads(manifest_path.read_text())
    return [
        node["class_name"]
        for node in manifest["nodes"]
        if any(
            declaration.get("type") == "model_usage"
            for declaration in (node.get("metadata", {}).get("declarations") or [])
        )
    ]


@pytest.mark.parametrize("node_type", _node_types_declaring_models())
def test_every_proxy_node_catalog_id_resolves(node_type: str) -> None:
    """A proxy node's default selection must resolve to one of its declared catalog models.

    This is what `declare_model_invocation` matches on: an id that resolves to
    nothing falls back to the raw provider id, and a policy keyed on catalog ids
    never fires. Nodes are built through the library so their declarations are
    resolvable. Node types outside the proxy family declare their invocations
    elsewhere and are skipped without being constructed, since several of them
    reach for credentials in `__init__`.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    if not issubclass(library.get_node_class(node_type), GriptapeProxyNode):
        pytest.skip(f"{node_type} is not a proxy node")
    node = library.create_node(node_type=node_type, name=node_type)

    catalog_model_id = cast("GriptapeProxyNode", node)._get_catalog_model_id()
    declared = sorted(
        model.model.provider_model_id
        for model in get_declared_models(node)
        if model.model.provider_model_id is not None
    )
    assert resolve_catalog_model_id(node, catalog_model_id) is not None, (
        f"{node_type}'s catalog id {catalog_model_id!r} is not one of its declared models {declared}"
    )
