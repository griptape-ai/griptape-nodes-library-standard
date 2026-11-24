"""Tests for ListToDictKeys node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.convert.list_to_dict_keys import ListToDictKeys


class TestListToDictKeys:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> ListToDictKeys:  # noqa: ARG002
        return ListToDictKeys(name="test_list_to_dict_keys")

    def _run(
        self,
        node: ListToDictKeys,
        keys: list,
        values: list | None = None,
        existing: dict | None = None,
        *,
        maintain_duplicates: bool = True,
    ) -> dict:
        node.parameter_values["keys"] = keys
        node.parameter_values["optional_values"] = values
        node.parameter_values["maintain_duplicates"] = maintain_duplicates
        node.parameter_values["KeyValuePairs"] = existing or {}
        node.process()
        return node.parameter_output_values["KeyValuePairs"]

    def test_values_matched_by_position(self, node: ListToDictKeys) -> None:
        assert self._run(node, ["one", "two", "three"], ["happy", "sad"]) == {
            "one": "happy",
            "two": "sad",
            "three": "",
        }

    def test_duplicates_are_numbered(self, node: ListToDictKeys) -> None:
        assert self._run(node, ["A", "N", "A"]) == {"A": "", "N": "", "A_2": ""}

    def test_typed_value_kept_after_blank_key(self, node: ListToDictKeys) -> None:
        assert self._run(node, ["", "a"], ["v"], {"a": "typed"}) == {"a": "typed"}

    def test_typed_value_kept_after_merged_duplicate(self, node: ListToDictKeys) -> None:
        result = self._run(node, ["a", "a", "b"], ["1", "2"], {"b": "typed"}, maintain_duplicates=False)
        assert result == {"a": "2", "b": "typed"}

    def test_merged_duplicate_keeps_provided_value(self, node: ListToDictKeys) -> None:
        assert self._run(node, ["a", "a", "b"], ["1"], maintain_duplicates=False) == {"a": "1", "b": ""}

    def test_provided_value_overrides_typed_value(self, node: ListToDictKeys) -> None:
        assert self._run(node, ["a", "b"], ["1"], {"a": "typed", "b": "kept"}) == {"a": "1", "b": "kept"}
