import logging
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import BaseNode, VariableAccess
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.node_events import (
    GetFlowForNodeRequest,
    GetFlowForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
    SetVariableTypeRequest,
    SetVariableTypeResultSuccess,
    SetVariableValueRequest,
    SetVariableValueResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

from griptape_nodes_library.variables.base_variable_node import BaseVariableNode
from griptape_nodes_library.variables.variable_utils import (
    _get_flow_for_node,
    get_variable,
    has_variable,
)

logger = logging.getLogger("griptape_nodes")

CREATE_NEW_SENTINEL = "Create new variable"


class SetVariable(BaseVariableNode):
    # process() calls HasVariableRequest before deciding whether to SetVariableValueRequest or
    # CreateVariableRequest, so the node both reads and writes the variable's state.
    VARIABLE_ACCESS = VariableAccess.READ_WRITE

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self._add_variable_name_parameter(
            ParameterString(
                name="variable_name",
                placeholder_text="Select an existing variable or create a new one",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                ui_options={"dropdown_row_icons": True},
                tooltip="Name of the variable to set. The variable is created if it does not exist.",
            )
        )
        self.variable_name_param.update_ui_options(
            {
                "data": self._build_variable_data(self._get_variable_names()),
                "dropdown_row_icons": True,
            }
        )

        self.new_variable_name_param = ParameterString(
            name="new_variable_name",
            allow_input=True,
            allow_property=True,
            placeholder_text="Enter new variable name",
            tooltip="Name for the new variable to create.",
        )
        self.new_variable_name_param.hide = True
        self.add_parameter(self.new_variable_name_param)

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ANY.value,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="The new value to set for the workflow variable",
        )
        self.add_parameter(self.value_param)

        self._add_scope_parameter()

    def _variable_name_choices(self, names: list[str]) -> list[str]:
        return [*names, CREATE_NEW_SENTINEL]

    def _build_variable_data(self, names: list[str]) -> list[dict]:
        return [{"name": name} for name in names] + [{"name": CREATE_NEW_SENTINEL, "icon": "circle-plus"}]

    def _resolve_variable_name(self) -> str:
        """Return the effective variable name, resolving the 'create new' sentinel if selected."""
        name = self.get_parameter_value(self.variable_name_param.name)
        if name == CREATE_NEW_SENTINEL:
            name = self.get_parameter_value(self.new_variable_name_param.name)
        return name or ""

    def _resolve_variable_names(self) -> list[str]:
        name = self._resolve_variable_name()
        return [name] if name else []

    def _refresh_variable_names(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:
        super()._refresh_variable_names(button, button_details)
        self.variable_name_param.update_ui_options({"data": self._build_variable_data(self._get_variable_names())})
        return None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self.variable_name_param:
            self.new_variable_name_param.hide = value != CREATE_NEW_SENTINEL

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
        """Eagerly ensure the variable named in ``variable_name`` exists in the engine.

        Letting the variable exist at graph-edit time means downstream nodes (``GetVariable``,
        ``HasVariable``) can see it before this node has run.

        We deliberately do NOT rename the prior variable when ``variable_name`` changes.
        There is no UI signal distinguishing "rename the variable I own" from "point this node
        at a different variable," so editing the name is always treated as the latter. Any
        engine-side variable left behind by a re-point is filtered out at save time by
        ``NodeDependencies``-driven serialization.

        If the node isn't attached to a flow yet (e.g. during deserialization), eager action
        is silently skipped; ``process()`` will create the variable when the flow runs.
        """
        if parameter is not self.variable_name_param:
            return value

        new_name = value
        if not new_name or new_name == CREATE_NEW_SENTINEL:
            return value

        try:
            self._eager_create_variable(new_name)
        except (RuntimeError, ValueError, LookupError) as exc:
            # Node may not be attached to a flow yet, or the engine rejected the op.
            # process() will reconcile on run.
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

    def _try_sync_variable_type(self, new_type: str) -> None:
        """Best-effort update of the engine-side variable's type; silent on any failure."""
        try:
            variable_name = self._resolve_variable_name()
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
        variable_name = self._resolve_variable_name()
        if not variable_name:
            msg = "A variable name is required. Pick one in 'variable_name' or enter one in 'new_variable_name'."
            raise ValueError(msg)

        value = self.get_parameter_value(self.value_param.name)
        scope = self._get_scope()

        flow_request = GetFlowForNodeRequest(node_name=self.name)
        flow_result = await GriptapeNodes.ahandle_request(flow_request)
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            msg = f"Failed to get the flow that contains this node: {flow_result.result_details}"
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

        # Only propagate the output value when the dropdown itself holds the name.
        # When the sentinel is active, the resolved name came from new_variable_name and
        # writing it back through parameter_output_values would trigger the Options converter,
        # snapping the dropdown away from the sentinel.
        if self.get_parameter_value(self.variable_name_param.name) != CREATE_NEW_SENTINEL:
            self.parameter_output_values[self.variable_name_param.name] = variable_name

    def _is_stale(self) -> bool:
        # Stale if the variable's value differs from what we last wrote.
        variable = get_variable(
            node_name=self.name, variable_name=self._resolve_variable_name(), scope=self._get_scope()
        )
        return variable.value != self.get_parameter_value(self.value_param.name)
