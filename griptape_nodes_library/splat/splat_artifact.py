from typing import Any

from griptape.artifacts.url_artifact import UrlArtifact


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
