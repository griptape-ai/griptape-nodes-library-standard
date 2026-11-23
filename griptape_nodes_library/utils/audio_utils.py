import base64
from typing import Any

from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files.project_file import ProjectFileDestination

# Supported audio file extensions (with leading dots)
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".webm"}


def dict_to_audio_url_artifact(audio_dict: dict, audio_format: str | None = None) -> AudioUrlArtifact:
    """Convert a dictionary representation of audio to an AudioUrlArtifact."""
    value = audio_dict["value"]

    # If it already is an AudioUrlArtifact, just wrap and return
    if audio_dict.get("type") == "AudioUrlArtifact":
        return AudioUrlArtifact(value)

    # Remove any data URL prefix
    if "base64," in value:
        value = value.split("base64,")[1]

    # Decode the base64 payload
    audio_bytes = base64.b64decode(value)

    # Infer format/extension if not explicitly provided
    if audio_format is None:
        if "type" in audio_dict and "/" in audio_dict["type"]:
            # e.g. "audio/mpeg" -> "mpeg"
            audio_format = audio_dict["type"].split("/")[1]
        else:
            audio_format = "mp3"

    dest = ProjectFileDestination.from_situation(filename=f"input.{audio_format}", situation="copy_external_file")
    saved = dest.write_bytes(audio_bytes)
    return AudioUrlArtifact(saved.location)


def is_audio_url_artifact(obj: Any) -> bool:
    """Check if object is any kind of AudioUrlArtifact (regardless of library).

    This handles AudioUrlArtifacts from various libraries that follow
    the AudioUrlArtifact pattern.

    Args:
        obj: Object to check

    Returns:
        True if object appears to be an AudioUrlArtifact
    """
    if not obj:
        return False

    # Must have both 'value' attribute and class name containing 'AudioUrlArtifact'
    return hasattr(obj, "value") and hasattr(obj, "__class__") and "AudioUrlArtifact" in obj.__class__.__name__


def extract_url_from_audio_object(obj: Any) -> str | None:
    """Extract URL from audio object if it contains one.

    Args:
        obj: Audio object (string, AudioUrlArtifact, etc.)

    Returns:
        URL string if found, None otherwise
    """
    if isinstance(obj, str):
        return obj

    if is_audio_url_artifact(obj) and hasattr(obj, "value"):
        value = obj.value  # type: ignore[attr-defined]
        if isinstance(value, str):
            return value

    return None
