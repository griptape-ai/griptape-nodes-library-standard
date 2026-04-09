from typing import Any

from griptape.artifacts import BaseArtifact
from griptape.artifacts.url_artifact import UrlArtifact


class SplatArtifact(BaseArtifact):
    """A splat file artifact."""

    def __init__(self, value: bytes, meta: dict[str, Any] | None = None) -> None:
        super().__init__(value=value, meta=meta or {})

    def to_text(self) -> str:
        """Convert the splat file to text representation."""
        return f"Splat file with {len(self.value)} bytes"

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["metadata"] = data.get("meta", {})
        return data


class SplatUrlArtifact(UrlArtifact):
    """A splat file URL artifact."""

    def __init__(self, value: str, meta: dict[str, Any] | None = None) -> None:
        super().__init__(value=value, meta=meta or {})

    def to_text(self) -> str:
        """Convert the splat URL to text representation."""
        return f"Splat file URL: {self.value}"

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["metadata"] = data.get("meta", {})
        return data
