"""Tests for SelectFromGrid's preview URL resolution."""

from unittest.mock import AsyncMock, patch

import pytest
from griptape_nodes.retained_mode.events.artifact_events import (
    GetPreviewForArtifactRequest,
    GetPreviewForArtifactResultFailure,
)

from griptape_nodes_library.lists.select_from_grid import SelectFromGrid

_AHANDLE_TARGET = "griptape_nodes_library.lists.select_from_grid.GriptapeNodes.ahandle_request"


@pytest.fixture
def node() -> SelectFromGrid:
    return SelectFromGrid(name="test_select_from_grid")


class TestResolvePreviewUrl:
    @pytest.mark.asyncio
    async def test_macro_path_reaches_engine_unresolved(self, node: SelectFromGrid) -> None:
        """A macro-form path must arrive at the engine as its template.

        The engine records the request's template verbatim as the preview
        sidecar's source_macro_path; resolving first would bake this machine's
        absolute path into a file that lives inside the project.
        """
        ahandle = AsyncMock(return_value=GetPreviewForArtifactResultFailure(result_details="no preview"))
        with patch(_AHANDLE_TARGET, new=ahandle):
            url = await node._resolve_preview_url_async("{inputs}/images/photo.png", "Image")

        assert url == ""
        assert ahandle.await_args is not None
        request = ahandle.await_args.args[0]
        assert isinstance(request, GetPreviewForArtifactRequest)
        assert request.macro_path.parsed_macro.template == "{inputs}/images/photo.png"

    @pytest.mark.asyncio
    async def test_unbalanced_brace_filename_degrades_to_no_preview(self, node: SelectFromGrid) -> None:
        """A legal filename with an unbalanced brace degrades to no thumbnail.

        The brace fails macro parsing both before and after resolution, so no
        preview request can be built. The caller falls back to a direct file
        URL; raising here would fail the whole grid over one odd filename.
        """
        ahandle = AsyncMock()
        with patch(_AHANDLE_TARGET, new=ahandle):
            url = await node._resolve_preview_url_async("images/photo{1.png", "Image")

        assert url == ""
        ahandle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_http_url_skips_preview(self, node: SelectFromGrid) -> None:
        """URLs are already displayable; no preview request is made."""
        ahandle = AsyncMock()
        with patch(_AHANDLE_TARGET, new=ahandle):
            url = await node._resolve_preview_url_async("http://example.com/photo.png", "Image")

        assert url == ""
        ahandle.assert_not_awaited()
