"""Tests for the SetVariablesFromData node.

Covers turning a dict / JSON string / list of key-value pairs into workflow variables in one
step, plus name sanitization, duplicate-key last-write-wins, and overwrite behavior.
"""

from collections.abc import Generator

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    GetVariableRequest,
    GetVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

from griptape_nodes_library.variables.set_variables_from_data import (
    _infer_type,
    _sanitize_name,
    _source_to_pairs,
)

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_set_variables_from_data_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


@pytest.fixture
def node(flow: str) -> BaseNode:
    """Create a SetVariablesFromData node inside the test flow and return the instance.

    Returns the node as ``BaseNode`` because library nodes are loaded via a dynamic module,
    so the imported class and the instantiated class are different class objects even though
    they share a name.
    """
    result = GriptapeNodes.handle_request(
        CreateNodeRequest(node_type="SetVariablesFromData", override_parent_flow_name=flow)
    )
    assert isinstance(result, CreateNodeResultSuccess)
    instance = GriptapeNodes.NodeManager().get_node_by_name(result.node_name)
    assert type(instance).__name__ == "SetVariablesFromData"
    return instance


def _has_variable(name: str, flow_name: str) -> bool:
    result = GriptapeNodes.handle_request(
        HasVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow_name)
    )
    assert isinstance(result, HasVariableResultSuccess)
    return result.exists


def _get_variable_value(name: str, flow_name: str) -> object:
    result = GriptapeNodes.handle_request(
        GetVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow_name)
    )
    assert isinstance(result, GetVariableResultSuccess)
    return result.variable.value


class TestSourceToPairs:
    """The pure normalization helper — no engine required."""

    def test_dict_input(self) -> None:
        assert _source_to_pairs({"NAME": "Jason", "PHONE": "027"}) == [("NAME", "Jason"), ("PHONE", "027")]

    def test_json_object_string(self) -> None:
        assert _source_to_pairs('{"NAME": "Jason"}') == [("NAME", "Jason")]

    def test_json_array_string(self) -> None:
        assert _source_to_pairs('[["NAME", "Jason"]]') == [("NAME", "Jason")]

    def test_list_of_key_value_dicts(self) -> None:
        source = [{"key": "NAME", "value": "Jason"}, {"key": "PHONE", "value": "027"}]
        assert _source_to_pairs(source) == [("NAME", "Jason"), ("PHONE", "027")]

    def test_list_of_pairs(self) -> None:
        assert _source_to_pairs([["NAME", "Jason"], ["PHONE", "027"]]) == [("NAME", "Jason"), ("PHONE", "027")]

    def test_list_of_single_entry_dicts(self) -> None:
        assert _source_to_pairs([{"NAME": "Jason"}, {"PHONE": "027"}]) == [("NAME", "Jason"), ("PHONE", "027")]

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _source_to_pairs(None)

    def test_invalid_json_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _source_to_pairs("{not json}")

    def test_bad_list_item_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly two elements"):
            _source_to_pairs([["a", "b", "c"]])


class TestSanitizeName:
    def test_spaces_become_underscores(self) -> None:
        assert _sanitize_name("Full Name") == "Full_Name"

    def test_punctuation_stripped(self) -> None:
        assert _sanitize_name("e-mail@address") == "e_mail_address"

    def test_leading_digit_prefixed(self) -> None:
        assert _sanitize_name("123abc").startswith("_")


class TestInferType:
    def test_bool_before_int(self) -> None:
        # bool is a subclass of int; must not be reported as int.
        assert _infer_type(True) == "bool"

    def test_int(self) -> None:
        assert _infer_type(5) == "int"

    def test_str(self) -> None:
        assert _infer_type("x") == "str"

    def test_dict_is_json(self) -> None:
        assert _infer_type({"a": 1}) == "json"


class TestSetVariablesFromDataProcess:
    """Exercises ``aprocess()`` end-to-end against the engine."""

    @pytest.mark.asyncio
    async def test_dict_creates_variables(self, node: BaseNode, flow: str) -> None:
        node.set_parameter_value("source", {"NAME": "Jason", "PHONE": "027"})

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "Jason"
        assert _get_variable_value("PHONE", flow) == "027"
        assert node.parameter_output_values["variable_names"] == ["NAME", "PHONE"]

    @pytest.mark.asyncio
    async def test_json_string_creates_variables(self, node: BaseNode, flow: str) -> None:
        node.set_parameter_value("source", '{"NAME": "Jason"}')

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "Jason"

    @pytest.mark.asyncio
    async def test_list_of_pairs_creates_variables(self, node: BaseNode, flow: str) -> None:
        node.set_parameter_value("source", [["NAME", "Jason"], ["CITY", "Wellington"]])

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "Jason"
        assert _get_variable_value("CITY", flow) == "Wellington"

    @pytest.mark.asyncio
    async def test_duplicate_keys_last_write_wins(self, node: BaseNode, flow: str) -> None:
        node.set_parameter_value("source", [["NAME", "first"], ["NAME", "second"]])

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "second"
        # Only one variable is emitted for the collapsed key.
        assert node.parameter_output_values["variable_names"] == ["NAME"]

    @pytest.mark.asyncio
    async def test_sanitizes_names_by_default(self, node: BaseNode, flow: str) -> None:
        node.set_parameter_value("source", {"Full Name": "Jason Schleifer"})

        await node.aprocess()

        assert _has_variable("Full_Name", flow)
        assert _get_variable_value("Full_Name", flow) == "Jason Schleifer"

    @pytest.mark.asyncio
    async def test_invalid_name_without_sanitize_raises(self, node: BaseNode) -> None:
        node.set_parameter_value("source", {"Full Name": "Jason"})
        node.set_parameter_value("sanitize_names", False)

        with pytest.raises(ValueError, match="not a valid variable name"):
            await node.aprocess()

    @pytest.mark.asyncio
    async def test_overwrite_off_leaves_existing(self, node: BaseNode, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="NAME", type="str", is_global=False, value="original", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)

        node.set_parameter_value("source", {"NAME": "new"})
        node.set_parameter_value("overwrite_existing", False)

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "original"
        # The skipped variable is not reported as created/updated.
        assert node.parameter_output_values["variable_names"] == []

    @pytest.mark.asyncio
    async def test_overwrite_on_updates_existing(self, node: BaseNode, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="NAME", type="str", is_global=False, value="original", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)

        node.set_parameter_value("source", {"NAME": "new"})

        await node.aprocess()

        assert _get_variable_value("NAME", flow) == "new"
        assert node.parameter_output_values["variable_names"] == ["NAME"]

    @pytest.mark.asyncio
    async def test_none_source_raises(self, node: BaseNode) -> None:
        # source is unset (None) by default — should raise before touching the engine.
        with pytest.raises(ValueError, match="non-empty"):
            await node.aprocess()

    @pytest.mark.asyncio
    async def test_sanitized_collision_last_write_wins(self, node: BaseNode, flow: str) -> None:
        # "Full Name" and "Full_Name" both sanitize to "Full_Name"; last write wins and only
        # one variable entry is emitted.
        node.set_parameter_value("source", {"Full Name": "first", "Full_Name": "second"})

        await node.aprocess()

        assert _get_variable_value("Full_Name", flow) == "second"
        assert node.parameter_output_values["variable_names"] == ["Full_Name"]
