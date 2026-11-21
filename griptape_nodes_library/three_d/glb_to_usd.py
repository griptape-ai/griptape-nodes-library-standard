"""Griptape Nodes node: convert a GLB/glTF model to USD (usdz / usdc / usda)."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_three_d import Parameter3D
from griptape_nodes.files.file import File, FileLoadError, FileWriteError
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.traits.options import Options

from griptape_nodes_library.three_d._gltf_to_usd import (
    ConversionResult,
    GltfConversionError,
    MaterialMode,
    OutputFormat,
    convert_gltf_to_usd,
)
from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

logger = logging.getLogger("griptape_nodes")

# usda by default: usdz packaging behaves oddly in Nuke, plain USD + texture folder loads cleanly.
_DEFAULT_FILENAME = "converted_model.usda"


class GlbToUsd(SuccessFailureNode):
    """Converts a GLB/glTF model into USD with UsdPreviewSurface and/or MaterialX materials."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter3D(
                name="model",
                tooltip=(
                    "The model to convert (e.g. the output of a Tripo node). Must be self-contained: "
                    "a .glb, or a .gltf with its buffers and images embedded."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                display_name="GLB / glTF",
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_format",
                default_value=OutputFormat.USDA.value,
                tooltip=(
                    "usdz: one self-contained file with textures packed inside. "
                    "usdc / usda: binary / text USD with textures written to a folder beside it."
                ),
                allow_output=False,
                traits={Options(choices=[f.value for f in OutputFormat])},
                ui_options={"display_name": "Output Format"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="material_mode",
                default_value=MaterialMode.PREVIEW_SURFACE.value,
                tooltip=(
                    "UsdPreviewSurface: works everywhere (Hydra Storm, Nuke viewer, usdview). "
                    "MaterialX: standard_surface network in the mtlx render context. "
                    "Both: one material carrying both, so each renderer picks its own."
                ),
                allow_output=False,
                traits={Options(choices=[m.value for m in MaterialMode])},
                ui_options={"display_name": "Material Type"},
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=_DEFAULT_FILENAME)
        self._output_file.add_parameter()

        self.add_parameter(
            Parameter3D(
                name="usd_model",
                tooltip="The converted USD file.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False,
                display_name="USD",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Conversion summary and anything that couldn't be converted.",
            result_details_placeholder="Conversion details will appear here.",
            parameter_group_initially_collapsed=False,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        # With Failed connected, input problems take the Failed path in process() instead of
        # blocking the run, so the flow can handle them.
        if not self._has_outgoing_connections(self.failure_output):
            errors.extend(self._input_errors())
        return errors or None

    def _input_errors(self) -> list[Exception]:
        errors: list[Exception] = []
        if self._input_location() is None:
            errors.append(
                ValueError(
                    "Attempted to convert a model to USD. Failed because no GLB/glTF model is connected or selected."
                )
            )
        # Both dropdowns accept wired inputs, so a connected value may not be one of the choices.
        for name, label, enum in (
            ("output_format", "Output Format", OutputFormat),
            ("material_mode", "Material Type", MaterialMode),
        ):
            value = self.get_parameter_value(name)
            valid = [e.value for e in enum]
            if value not in valid:
                errors.append(
                    ValueError(
                        f"Attempted to convert a model to USD. Failed because {label} '{value}' isn't one of: {', '.join(valid)}."
                    )
                )
        return errors

    def process(self) -> None:
        self._clear_execution_status()
        self.parameter_output_values["usd_model"] = None
        input_errors = self._input_errors()
        if input_errors:
            error = ValueError("\n".join(str(e) for e in input_errors))
            self._set_status_results(was_successful=False, result_details=str(error))
            self._handle_failure_exception(error)
            return
        try:
            artifact, details = self._convert()
        except (GltfConversionError, FileLoadError, FileWriteError, ValueError, OSError) as e:
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return
        self.parameter_output_values["usd_model"] = artifact
        self._set_status_results(was_successful=True, result_details=details)

    # -- conversion ---------------------------------------------------------------------------

    def _convert(self) -> tuple[ThreeDUrlArtifact, str]:
        location = self._input_location()
        if location is None:
            raise ValueError(
                "Attempted to convert a model to USD. Failed because no GLB/glTF model is connected or selected."
            )

        source_bytes = File(location).read_bytes()
        output_format = OutputFormat(self.get_parameter_value("output_format") or OutputFormat.USDA.value)
        material_mode = MaterialMode(self.get_parameter_value("material_mode") or MaterialMode.PREVIEW_SURFACE.value)
        filename = Path(self.get_parameter_value("output_file") or _DEFAULT_FILENAME).with_suffix(
            f".{output_format.value}"
        )
        # Content-addressed texture folder: re-converting the same model reuses it, a different
        # model saved under the same name can't overwrite textures an older USD still points at.
        texture_dir = f"{filename.stem}_textures_{hashlib.sha1(source_bytes).hexdigest()[:8]}"

        with tempfile.TemporaryDirectory(prefix="glb_to_usd_") as tmp:
            tmp_dir = Path(tmp)
            input_suffix = ".glb" if source_bytes[:4] == b"glTF" else ".gltf"
            # Keep the source file's name so conversion errors point at the user's file, not a temp name.
            source_stem = Path(location.split("?", 1)[0]).stem or "input"
            input_path = tmp_dir / f"{source_stem}{input_suffix}"
            input_path.write_bytes(source_bytes)
            output_path = tmp_dir / "out" / filename.name

            result = convert_gltf_to_usd(
                input_path, output_path, material_mode=material_mode, texture_dir_name=texture_dir
            )
            saved = self._save_outputs(result, filename, texture_dir)

        artifact = ThreeDUrlArtifact(value=saved.location, meta={"filename": saved.name, "format": output_format.value})
        return artifact, self._summary(result, saved, output_format, material_mode)

    def _save_outputs(self, result: ConversionResult, filename: Path, texture_dir: str) -> File:
        destination = ProjectFileDestination.from_situation_in_workflow(
            self.workflow, str(filename), "save_node_output", node_name=self.name
        )
        saved = destination.write_bytes(result.output_path.read_bytes())

        # usdc/usda reference textures relatively ("./<texture_dir>/..."), so they go beside the
        # file actually written, whatever name the project's write policy gave it.
        if result.texture_paths:
            target_dir = Path(saved.resolve()).parent / texture_dir
            for texture in result.texture_paths:
                # The converter names each texture by its sniffed format, so extension coercion has
                # nothing to fix; disabled so a mismatch fails loudly instead of being renamed away
                # from the path the USD references.
                File(str(target_dir / texture.name)).write_bytes(
                    texture.read_bytes(), coerce_extension_to_match_bytes=False
                )
        return saved

    def _input_location(self) -> str | None:
        value = self.get_parameter_value("model")
        if value is None:
            return None
        if isinstance(value, ThreeDUrlArtifact):
            return value.value or None
        if isinstance(value, dict):
            return value.get("value") or None
        if isinstance(value, str):
            return value or None
        return getattr(value, "value", None) or None

    @staticmethod
    def _summary(
        result: ConversionResult, saved: File, output_format: OutputFormat, material_mode: MaterialMode
    ) -> str:
        lines = [
            f"Saved {saved.name} ({output_format.value}, {material_mode.value}).",
            f"{result.mesh_count} mesh(es), {result.triangle_count:,} triangles, {result.material_count} material(s).",
        ]
        if result.texture_paths:
            lines.append(f"{len(result.texture_paths)} texture(s) written beside the USD.")
        if result.warnings:
            lines.append("Not converted:")
            lines.extend(f"  - {w}" for w in result.warnings)
        return "\n".join(lines)
