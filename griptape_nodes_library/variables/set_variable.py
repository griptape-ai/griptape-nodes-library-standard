import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import (
    BaseNode,
    ControlNode,
    NodeDependencies,
    NodeResolutionState,
    VariableAccess,
    VariableReference,
)
from griptape_nodes.retained_mode.events.node_events import (
    GetFlowForNodeRequest,
    GetFlowForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
    RenameVariableRequest,
    RenameVariableResultSuccess,
    SetVariableTypeRequest,
    SetVariableTypeResultSuccess,
    SetVariableValueRequest,
    SetVariableValueResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

from griptape_nodes_library.variables.variable_utils import (
    _get_flow_for_node,
    create_advanced_parameter_group,
    get_variable,
    has_variable,
    scope_string_to_variable_scope,
)

logger = logging.getLogger("griptape_nodes")


class SetVariable(ControlNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.variable_name_param = Parameter(
            name="variable_name",
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="Name of the variable to set. The variable is created if it does not exist.",
        )
        self.add_parameter(self.variable_name_param)

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ANY.value,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="The new value to set for the workflow variable",
        )
        self.add_parameter(self.value_param)

        # Advanced parameters group (collapsed by default)
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

    def after_incoming_connection(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Infer the variable's type from the source parameter when something connects to ``value``."""
        if target_parameter is not self.value_param:
            return

        detected_type = source_parameter.output_type
        self.value_param.type = detected_type
        self.value_param.output_type = detected_type

        # If the variable is already registered in the engine, propagate the type change.
        self._try_sync_variable_type(detected_type)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Reset ``value``'s declared type when a connection is removed."""
        super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)
        if target_parameter is not self.value_param:
            return

        self.value_param.type = ParameterTypeBuiltin.ANY.value
        self.value_param.output_type = ParameterTypeBuiltin.ANY.value
        # Intentionally do not re-emit SetVariableTypeRequest: leaving the engine-side variable
        # at its last inferred type is less disruptive than clobbering to 'any' on disconnect.

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        """Eagerly register/rename the variable in the engine as the user edits ``variable_name``.

        This allows downstream nodes (``GetVariable``, ``HasVariable``) to see the variable
        at graph-edit time instead of only after this node has run.

        If the node isn't attached to a flow yet (e.g. during deserialization), eager action
        is silently skipped; ``process()`` will create the variable when the flow runs.
        """
        if parameter is not self.variable_name_param:
            return value

        old_name = self.get_parameter_value(self.variable_name_param.name)
        new_name = value

        # No-op transitions we can short-circuit without touching the engine.
        if old_name == new_name:
            return value
        if not new_name:
            # Clearing the field: leave any existing variable alone. Users often blank the field
            # mid-retype; auto-deleting would destroy data and confuse other nodes that reference it.
            return value

        try:
            if not old_name:
                self._eager_create_variable(new_name)
            else:
                self._eager_rename_variable(old_name=old_name, new_name=new_name)
        except (RuntimeError, ValueError, LookupError) as exc:
            # Node may not be attached to a flow yet, or the engine rejected the op (e.g. rename
            # collision). process() will reconcile on run.
            logger.debug("SetVariable '%s' skipped eager registration: %s", self.name, exc)

        return value

    def _eager_create_variable(self, variable_name: str) -> None:
        """Create the variable in the engine if it does not already exist in this flow."""
        current_flow_name = _get_flow_for_node(self.name)

        if has_variable(node_name=self.name, variable_name=variable_name, scope=VariableScope.CURRENT_FLOW_ONLY):
            # Already exists in this flow; adopt it silently.
            return

        initial_value = self.get_parameter_value(self.value_param.name)
        variable_type = self.value_param.output_type or ParameterTypeBuiltin.ANY.value

        create_request = CreateVariableRequest(
            name=variable_name,
            type=variable_type,
            is_global=False,
            value=initial_value,
            owning_flow=current_flow_name,
        )
        create_result = GriptapeNodes.handle_request(create_request)
        if not isinstance(create_result, CreateVariableResultSuccess):
            msg = f"Eager create for variable '{variable_name}' failed: {create_result.result_details}"
            raise RuntimeError(msg)  # noqa: TRY004

    def _eager_rename_variable(self, old_name: str, new_name: str) -> None:
        """Rename the variable in the engine. Skips silently if old doesn't exist; raises on collision."""
        current_flow_name = _get_flow_for_node(self.name)

        # If the old variable isn't registered (node was never eagerly registered, or user
        # edited before the flow existed), fall through to a create for the new name.
        if not has_variable(node_name=self.name, variable_name=old_name, scope=VariableScope.CURRENT_FLOW_ONLY):
            self._eager_create_variable(new_name)
            return

        rename_request = RenameVariableRequest(
            name=old_name,
            new_name=new_name,
            lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
            starting_flow=current_flow_name,
        )
        rename_result = GriptapeNodes.handle_request(rename_request)
        if not isinstance(rename_result, RenameVariableResultSuccess):
            msg = f"Eager rename from '{old_name}' to '{new_name}' failed: {rename_result.result_details}"
            raise RuntimeError(msg)  # noqa: TRY004

    def _try_sync_variable_type(self, new_type: str) -> None:
        """Best-effort update of the engine-side variable's type; silent on any failure."""
        try:
            variable_name = self.get_parameter_value(self.variable_name_param.name)
            if not variable_name:
                return
            current_flow_name = _get_flow_for_node(self.name)
            if not has_variable(
                node_name=self.name, variable_name=variable_name, scope=VariableScope.CURRENT_FLOW_ONLY
            ):
                return
            type_request = SetVariableTypeRequest(
                name=variable_name,
                type=new_type,
                lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
                starting_flow=current_flow_name,
            )
            type_result = GriptapeNodes.handle_request(type_request)
            if not isinstance(type_result, SetVariableTypeResultSuccess):
                logger.debug(
                    "SetVariable '%s' could not update variable type: %s", self.name, type_result.result_details
                )
        except (RuntimeError, ValueError, LookupError) as exc:
            logger.debug("SetVariable '%s' skipped variable type sync: %s", self.name, exc)

    async def aprocess(self) -> None:
        variable_name = self.get_parameter_value(self.variable_name_param.name)
        if not variable_name:
            msg = f"SetVariable node '{self.name}' requires a non-empty variable_name."
            raise ValueError(msg)

        value = self.get_parameter_value(self.value_param.name)
        scope_str = self.get_parameter_value(self.scope_param.name)
        scope = scope_string_to_variable_scope(scope_str)

        flow_request = GetFlowForNodeRequest(node_name=self.name)
        flow_result = await GriptapeNodes.ahandle_request(flow_request)
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            msg = f"Failed to get flow for node '{self.name}': {flow_result.result_details}"
            raise TypeError(msg)
        current_flow_name = flow_result.flow_name

        has_request = HasVariableRequest(
            name=variable_name,
            lookup_scope=scope,
            starting_flow=current_flow_name,
        )
        has_result = await GriptapeNodes.ahandle_request(has_request)
        if not isinstance(has_result, HasVariableResultSuccess):
            msg = f"Failed to check if variable '{variable_name}' exists: {has_result.result_details}"
            raise TypeError(msg)

        if has_result.exists:
            set_request = SetVariableValueRequest(
                value=value,
                name=variable_name,
                lookup_scope=scope,
                starting_flow=current_flow_name,
            )
            set_result = await GriptapeNodes.ahandle_request(set_request)
            if not isinstance(set_result, SetVariableValueResultSuccess):
                msg = f"Failed to set variable '{variable_name}': {set_result.result_details}"
                raise TypeError(msg)
        else:
            variable_type = self.value_param.output_type or ParameterTypeBuiltin.ANY.value
            create_request = CreateVariableRequest(
                name=variable_name,
                type=variable_type,
                is_global=False,
                value=value,
                owning_flow=current_flow_name,
            )
            create_result = await GriptapeNodes.ahandle_request(create_request)
            if not isinstance(create_result, CreateVariableResultSuccess):
                msg = f"Failed to create variable '{variable_name}': {create_result.result_details}"
                raise TypeError(msg)

        self.parameter_output_values[self.variable_name_param.name] = variable_name

    def get_node_dependencies(self) -> NodeDependencies | None:
        """Declare the variable this node reads/writes so it survives serialization.

        Access is READ_WRITE: ``process()`` calls ``HasVariableRequest`` before deciding whether to
        ``SetVariableValueRequest`` or ``CreateVariableRequest``, so the node both reads and writes
        the variable's state.

        Reads the current value of ``variable_name`` via ``get_parameter_value`` — if the parameter
        is driven by an incoming connection, this returns the last propagated value (or ``None`` if
        nothing has propagated yet). No declaration is emitted for empty/None names.
        """
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()

        variable_name = self.get_parameter_value(self.variable_name_param.name)
        if isinstance(variable_name, str) and variable_name:
            scope_str = self.get_parameter_value(self.scope_param.name)
            scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL
            deps.variable_references.add(
                VariableReference(name=variable_name, scope=scope, access=VariableAccess.READ_WRITE)
            )

        return deps

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to treat it as volatile (if the value has changed, mark as unresolved)."""
        if self._state == NodeResolutionState.RESOLVED:
            variable_name = self.get_parameter_value(self.variable_name_param.name)
            scope_str = self.get_parameter_value(self.scope_param.name)

            # Convert scope string to VariableScope enum
            scope = scope_string_to_variable_scope(scope_str)

            # This can throw if the variable doesn't exist.
            try:
                variable = get_variable(node_name=self.name, variable_name=variable_name, scope=scope)

                var_value = variable.value
                our_value = self.get_parameter_value(self.value_param.name)
                if var_value != our_value:
                    return NodeResolutionState.UNRESOLVED
            except LookupError:
                # Variable may not have been created yet; assume unresolved.
                return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        # Have to override the setter if we override the getter.
        self._state = new_state
