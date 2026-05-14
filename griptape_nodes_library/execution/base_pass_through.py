from abc import ABC, abstractmethod
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import BaseNode, DataNode


def contains_any(types: set[str]) -> bool:
    """Check if set of type strings contains the built in any type.

    Args:
        types: Set of type strings to check.

    Returns:
        True if set contains the built in any type, False otherwise.
    """
    for t in types:
        if t.lower() == ParameterTypeBuiltin.ANY.value:
            return True
    return False


def has_control_type(param: Parameter) -> bool:
    """Check if a parameter has a control type.

    Args:
        param: The parameter to check for control type.

    Returns:
        True if the parameter has a control type, False otherwise.
    """
    return param.type == ParameterTypeBuiltin.CONTROL_TYPE.value


class BasePassThroughNode(DataNode, ABC):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.incoming_source_parameter: Parameter | None = None
        self.outgoing_target_parameters: dict[str, Parameter] = {}

    @abstractmethod
    def get_pass_thru_parameter(self) -> Parameter:
        pass

    def after_incoming_connection(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection has been established TO this Node."""
        if has_control_type(source_parameter) or has_control_type(target_parameter):
            # No custom reroute logic for control parameters.
            return
        self.incoming_source_parameter = source_parameter
        self._propagate_forwards()

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection TO this Node was REMOVED."""
        if has_control_type(source_parameter) or has_control_type(target_parameter):
            # No custom reroute logic for control parameters.
            return
        self.incoming_source_parameter = None
        self._propagate_forwards()

    def after_outgoing_connection(
        self,
        source_parameter: Parameter,
        target_node: BaseNode,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection has been established OUT of this Node."""
        if has_control_type(source_parameter) or has_control_type(target_parameter):
            # No custom reroute logic for control parameters.
            return
        self._add_outgoing_target_parameter(target_node, target_parameter)
        self._propagate_backwards()

    def after_outgoing_connection_removed(
        self,
        source_parameter: Parameter,
        target_node: BaseNode,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection OUT of this Node was REMOVED."""
        if has_control_type(source_parameter) or has_control_type(target_parameter):
            # No custom reroute logic for control parameters.
            return
        self._remove_outgoing_target_parameter(target_node, target_parameter)
        self._propagate_backwards()

    def process(self) -> None:
        param = self.get_pass_thru_parameter()
        if param:
            self.parameter_output_values[param.name] = self.parameter_values.get(param.name)

    def _resolve_types(self) -> None:
        param = self.get_pass_thru_parameter()
        if not param:
            return

        if self.incoming_source_parameter is not None and len(self.outgoing_target_parameters) > 0:
            # Incoming and outgoing connections.
            self._use_outgoing_target_parameters_type_intersection()
            # The type/output_type of the input takes precedence, but the input
            # type comes from the outgoing connections for the case when you
            # directly connect a different incoming edge. In that case it should
            # allow any valid input type that the outgoing targets accept.
            param.type = self.incoming_source_parameter.type
            param.output_type = self.incoming_source_parameter.output_type
        elif self.incoming_source_parameter is not None:
            # Incoming connection only.
            self._use_incoming_source_parameter_types()
        elif len(self.outgoing_target_parameters) > 0:
            # Outgoing connections only.
            self._use_outgoing_target_parameters_type_intersection()
        else:
            # No connections.
            self._reset_parameter_types()

        self._ensure_output_uses_all_instead_of_any()

    def _propagate_backwards(self) -> None:
        # Resolve backward to propagate type changes from outgoing connections.
        self._resolve_types()
        incoming_node = None
        if self.incoming_source_parameter is not None:
            incoming_node = self.incoming_source_parameter.get_node()
        if isinstance(incoming_node, BasePassThroughNode):
            # If we haven't reached the root, keep going.
            incoming_node._propagate_backwards()
        else:
            # Reached root, prop forward now.
            # The type/value may have been updated with knowledge of the leaves.
            # If you don't prop forward then you will not be able to reset a serial
            # connected subgraph of only reroute nodes to there initial "Any" state.
            self._propagate_forwards()

    def _propagate_forwards(self) -> None:
        # Resolve forward to propagate type changes from incoming connections.
        self._resolve_types()
        param = self.get_pass_thru_parameter()
        for target_param in self.outgoing_target_parameters.values():
            node = target_param.get_node()
            if node and isinstance(node, BasePassThroughNode):
                # Outgoing connections also need values to propagate.
                target_node_param = node.get_pass_thru_parameter()
                if param and target_node_param:
                    value = self.get_parameter_value(param.name)
                    node.set_parameter_value(target_node_param.name, value)
                    node._propagate_forwards()

    def _use_incoming_source_parameter_types(self) -> None:
        if self.incoming_source_parameter is None:
            msg = "Invalid state: self.incoming_source_parameter must not be None"
            raise ValueError(msg)
        param = self.get_pass_thru_parameter()
        param.input_types = [ParameterTypeBuiltin.ANY.value]
        param.type = self.incoming_source_parameter.output_type
        param.output_type = self.incoming_source_parameter.output_type

    def _to_outgoing_target_parameters_key(
        self, outgoing_target_node: BaseNode, outgoing_target_parameter: Parameter
    ) -> str:
        node_name = outgoing_target_node.name
        parameter_name = outgoing_target_parameter.name
        return f"{node_name}__{parameter_name}"

    def _add_outgoing_target_parameter(
        self, outgoing_target_node: BaseNode, outgoing_target_parameter: Parameter
    ) -> None:
        key = self._to_outgoing_target_parameters_key(outgoing_target_node, outgoing_target_parameter)
        self.outgoing_target_parameters[key] = outgoing_target_parameter

    def _remove_outgoing_target_parameter(
        self, outgoing_target_node: BaseNode, outgoing_target_parameter: Parameter
    ) -> None:
        key = self._to_outgoing_target_parameters_key(outgoing_target_node, outgoing_target_parameter)
        if key in self.outgoing_target_parameters:
            del self.outgoing_target_parameters[key]

    def _reset_parameter_types(self) -> None:
        param = self.get_pass_thru_parameter()
        param.input_types = [ParameterTypeBuiltin.ANY.value]
        param.type = None
        param.output_type = ParameterTypeBuiltin.ALL.value

    def _use_outgoing_target_parameters_type_intersection(self) -> None:
        if len(self.outgoing_target_parameters) == 0:
            msg = "Invalid state: self.outgoing_target_parameters must have at least one entry"
            raise ValueError(msg)

        # 1. Determine the set of input types that all the output target parameters have in common.

        # The input types, should be the overlap / intersection of
        # the currently outbound connections.
        input_type_sets = [set(p.input_types or []) for p in self.outgoing_target_parameters.values()]

        # Handle edge case: one of the input_type_sets contains 'Any'.
        #
        # Remove any input_type_sets that contain the special 'Any' type.
        # Since they match anything, they don't need to be part of the intersection.
        # A better solution would actually be to check subtype relationships, but
        # I feel like such a solution is out of scope of this PR and really should be
        # done everywhere.
        input_type_sets = [s for s in input_type_sets if not contains_any(s)]

        if input_type_sets:
            input_types = list(set.intersection(*input_type_sets))
        else:
            # If we removed everything, then they could have only contained the builtin any type.
            input_types = [ParameterTypeBuiltin.ANY.value]

        param = self.get_pass_thru_parameter()
        param.input_types = input_types

        # Determine the type. The one selected must be compatible with all
        # of the current outbound connections. All of the input_types meet
        # this requirement, so just pick one, any one.
        param.type = input_types[0]
        param.output_type = param.type

    def _ensure_output_uses_all_instead_of_any(self) -> None:
        param = self.get_pass_thru_parameter()
        # Output types use a special ALL value instead of "Any".
        if param.output_type and param.output_type.lower() == ParameterTypeBuiltin.ANY.value:
            param.output_type = ParameterTypeBuiltin.ALL.value
