"""Tests for what SelectFromGrid saves in place of its widget value."""

from unittest.mock import patch

import pytest

from griptape_nodes_library.lists.select_from_grid import SelectFromGrid
from griptape_nodes_library.utils.macro_path_utils import MacroPathResult

_PREVIEW_URL = "http://localhost:8124/workspace/.griptape-nodes-previews/outputs/images/a.png"


@pytest.fixture
def node() -> SelectFromGrid:
    return SelectFromGrid(name="test_select_from_grid")


def _resolve_to_localhost(path: str) -> str:
    return f"http://localhost:9999/resolved/{path}"


class TestSavedGridState:
    def test_grid_is_not_serialized(self, node: SelectFromGrid) -> None:
        assert node.grid_param.serializable is False
        assert node.grid_state_param.serializable is True

    def test_saved_state_keeps_macro_source_and_drops_urls(self, node: SelectFromGrid) -> None:
        with patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost):
            node.set_parameter_value("list", ["{outputs}/images/a.png", "note"])

        grid = node.get_parameter_value("grid")
        assert grid["items"][0]["url"].startswith("http://")

        state = node.get_parameter_value("grid_state")
        assert state["items"][0] == {"type": "image", "label": "a.png", "source": "{outputs}/images/a.png"}
        assert state["items"][1] == {"type": "text", "value": "note"}

    def test_absolute_project_path_is_saved_in_macro_form(self, node: SelectFromGrid) -> None:
        mapped = MacroPathResult(resolved_path="{outputs}/images/a.png", is_external=False)
        with (
            patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost),
            patch(
                "griptape_nodes_library.lists.select_from_grid.resolve_to_macro_path", return_value=mapped
            ) as to_macro,
        ):
            node.set_parameter_value("list", ["/home/user/project/outputs/images/a.png"])

        to_macro.assert_called_once_with("/home/user/project/outputs/images/a.png")
        assert node.get_parameter_value("grid_state")["items"][0]["source"] == "{outputs}/images/a.png"

    def test_selection_change_reaches_saved_state(self, node: SelectFromGrid) -> None:
        with patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost):
            node.set_parameter_value("list", ["{outputs}/images/a.png", "{outputs}/images/b.png"])
        grid = node.get_parameter_value("grid")
        node.set_parameter_value("grid", {**grid, "selected_indices": [1], "columns": 5})

        state = node.get_parameter_value("grid_state")
        assert state["selected_indices"] == [1]
        assert state["columns"] == 5
        assert all("url" not in item for item in state["items"])

    def test_loading_saved_state_rebuilds_urls(self, node: SelectFromGrid) -> None:
        saved = {
            "items": [{"type": "image", "label": "a.png", "source": "{outputs}/images/a.png"}],
            "selected_indices": [0],
            "columns": 4,
            "layout": "grid",
            "settings": {"multi_select": True},
        }
        with patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost):
            # A workflow load sets values with initial_setup, which skips after_value_set.
            node.set_parameter_value("grid_state", saved, initial_setup=True)

        grid = node.get_parameter_value("grid")
        assert grid["items"][0]["url"] == "http://localhost:9999/resolved/{outputs}/images/a.png"
        assert grid["selected_indices"] == [0]
        assert grid["columns"] == 4
        assert node.get_parameter_value("grid_state") == saved

    def test_loading_absolute_source_saves_it_in_macro_form(self, node: SelectFromGrid) -> None:
        saved = {"items": [{"type": "image", "label": "a.png", "source": "/home/user/project/outputs/images/a.png"}]}
        mapped = MacroPathResult(resolved_path="{outputs}/images/a.png", is_external=False)
        with (
            patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost),
            patch("griptape_nodes_library.lists.select_from_grid.resolve_to_macro_path", return_value=mapped),
        ):
            node.set_parameter_value("grid_state", saved, initial_setup=True)

        assert node.get_parameter_value("grid_state")["items"][0]["source"] == "{outputs}/images/a.png"
        assert node.get_parameter_value("grid")["items"][0]["url"] == (
            "http://localhost:9999/resolved/{outputs}/images/a.png"
        )

    def test_grid_from_older_save_without_source_falls_back_to_url(self, node: SelectFromGrid) -> None:
        node.set_parameter_value(
            "grid",
            {"items": [{"type": "image", "url": _PREVIEW_URL}], "selected_indices": [0]},
            initial_setup=True,
        )

        state = node.get_parameter_value("grid_state")
        assert state["items"][0] == {"type": "image", "source": _PREVIEW_URL}
        assert state["selected_indices"] == [0]

    def test_grid_from_older_save_takes_source_from_list(self, node: SelectFromGrid) -> None:
        node.set_parameter_value("list", ["{outputs}/images/a.png"], initial_setup=True)
        node.set_parameter_value(
            "grid",
            {"items": [{"type": "image", "url": _PREVIEW_URL}], "selected_indices": []},
            initial_setup=True,
        )

        state = node.get_parameter_value("grid_state")
        assert state["items"][0] == {"type": "image", "source": "{outputs}/images/a.png"}

    def test_rerun_after_load_keeps_selection(self, node: SelectFromGrid) -> None:
        saved = {
            "items": [
                {"type": "image", "source": "{outputs}/images/a.png"},
                {"type": "image", "source": "{outputs}/images/b.png"},
            ],
            "selected_indices": [1],
        }
        with patch.object(SelectFromGrid, "_resolve_url_string", side_effect=_resolve_to_localhost):
            node.set_parameter_value("grid_state", saved, initial_setup=True)
            node.set_parameter_value("list", ["{outputs}/images/a.png", "{outputs}/images/b.png"])

        assert node.get_parameter_value("grid")["selected_indices"] == [1]
        assert node.get_parameter_value("grid_state")["selected_indices"] == [1]
