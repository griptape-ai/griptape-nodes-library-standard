from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.files.project_file import ProjectFileDestination

from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

if TYPE_CHECKING:
    from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")


def _extract_model_url(data: dict[str, Any]) -> str | None:
    """Find the best 3D model URL in a Tripo task payload's data block."""
    result = data.get("result") or {}
    if isinstance(result.get("pbr_model"), dict):
        url = result["pbr_model"].get("url")
        if url:
            return url
    if isinstance(result.get("model"), dict):
        url = result["model"].get("url")
        if url:
            return url

    output = data.get("output") or {}
    return output.get("pbr_model") or output.get("model") or output.get("base_model")


def _extract_preview_url(data: dict[str, Any]) -> str | None:
    """Find the best preview image URL in a Tripo task payload's data block."""
    result = data.get("result") or {}
    if isinstance(result.get("rendered_image"), dict):
        url = result["rendered_image"].get("url")
        if url:
            return url

    output = data.get("output") or {}
    return output.get("rendered_image") or output.get("generated_image")


async def parse_tripo_task_result(node: GriptapeProxyNode, result_json: dict[str, Any]) -> None:
    """Parse a completed Tripo task payload, saving the GLB and preview to project files.

    The proxy's `fetch_completed_generation` returns Tripo's raw task response:
        {"code": 0,
         "data": {"status": "success",
                  "output": {"pbr_model": "<signed URL>", "rendered_image": "<signed URL>"},
                  "result": {"pbr_model": {"url": "...", "type": "glb"}, ...},
                  "consumed_credit": 20}}

    Tripo's signed URLs expire within 5 minutes, so we download the bytes
    immediately and save them as project files rather than exposing the
    expiring URLs downstream.
    """
    data = result_json.get("data") if isinstance(result_json, dict) else None
    if not isinstance(data, dict):
        data = result_json if isinstance(result_json, dict) else {}

    model_url = _extract_model_url(data)
    if not model_url:
        node._set_safe_defaults()
        node._set_status_results(
            was_successful=False,
            result_details="Tripo task completed but no model URL was present in the response.",
        )
        return

    model_bytes = await node._download_bytes_from_url(model_url)
    if not model_bytes:
        node._set_safe_defaults()
        node._set_status_results(
            was_successful=False,
            result_details="Failed to download the generated 3D model from Tripo's signed URL.",
        )
        return

    output_file_value = node.get_parameter_value("output_file") or "tripo_model.glb"
    model_path = Path(output_file_value)
    if model_path.suffix.lower() != ".glb":
        model_path = model_path.with_suffix(".glb")
    model_dest = ProjectFileDestination.from_situation(
        filename=str(model_path),
        situation="save_node_output",
        node_name=node.name,
    )
    saved_model = await model_dest.awrite_bytes(model_bytes)
    node.parameter_output_values["model_url"] = ThreeDUrlArtifact(
        value=saved_model.location,
        meta={"filename": saved_model.name, "format": "glb"},
    )

    preview_url = _extract_preview_url(data)
    if preview_url:
        preview_bytes = await node._download_bytes_from_url(preview_url)
        if preview_bytes:
            preview_path = model_path.with_suffix(".webp")
            preview_dest = ProjectFileDestination.from_situation(
                filename=str(preview_path),
                situation="save_node_output",
                node_name=node.name,
            )
            saved_preview = await preview_dest.awrite_bytes(preview_bytes)
            node.parameter_output_values["preview_image"] = ImageUrlArtifact(
                value=saved_preview.location,
                meta={"filename": saved_preview.name},
            )

    consumed = data.get("consumed_credit")
    detail = "3D model generated successfully."
    if consumed:
        detail += f" Tripo charged {consumed} credits."
    node._set_status_results(was_successful=True, result_details=detail)
