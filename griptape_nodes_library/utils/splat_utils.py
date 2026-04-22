import base64

from griptape_nodes.files.project_file import ProjectFileDestination

from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact


def dict_to_splat_url_artifact(splat_dict: dict, splat_format: str | None = None) -> SplatUrlArtifact:
    """Convert a dictionary representation of a splat file to a SplatUrlArtifact."""
    value = splat_dict["value"]
    if splat_dict.get("type") == "SplatUrlArtifact":
        return SplatUrlArtifact(value)

    # If the base64 string has a prefix like "data:application/octet-stream;base64,", remove it
    if "base64," in value:
        value = value.split("base64,")[1]

    splat_bytes = base64.b64decode(value)

    if splat_format is None:
        if "type" in splat_dict:
            mime_format = splat_dict["type"].split("/")[1] if "/" in splat_dict["type"] else None
            splat_format = mime_format
        else:
            splat_format = "spz"

    dest = ProjectFileDestination.from_situation(filename=f"input.{splat_format}", situation="copy_external_file")
    saved = dest.write_bytes(splat_bytes)
    return SplatUrlArtifact(saved.location)
