import logging
import os
import re
from enum import StrEnum
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
from griptape_nodes.retained_mode.events.connection_events import (
    DeleteConnectionRequest,
    DeleteConnectionResultSuccess,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")


class CaseStyle(StrEnum):
    UPPER = "UPPER CASE"
    UPPER_SNAKE = "UPPER_SNAKE_CASE"
    TITLE = "Title Case"
    SNAKE = "snake_case"
    KEBAB = "kebab-case"
    PASCAL = "PascalCase"
    CAMEL = "camelCase"
    AS_IS = "as is"


class CreateVariable(ControlNode):
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
            tooltip="The name of the variable to create",
        )
        self.add_parameter(self.variable_name_param)

        self.auto_name_param = Parameter(
            name="auto_name",
            type="bool",
            default_value=False,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Automatically derive the variable name from the connected value (file basename, artifact name, or source node name)",
        )
        self.add_parameter(self.auto_name_param)

        self.auto_name_case_param = Parameter(
            name="auto_name_case",
            type="str",
            default_value=CaseStyle.SNAKE,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Case style used when auto_name is on",
        )
        self.auto_name_case_param.add_trait(Options(choices=list(CaseStyle)))
        self.add_parameter(self.auto_name_case_param)
        self.hide_parameter_by_name(self.auto_name_case_param.name)

        self.variable_type_param = Parameter(
            name="variable_type",
            type=ParameterTypeBuiltin.STR.value,
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="The user-defined type of the variable (e.g., 'JSON', 'str', 'int')",
        )
        self.add_parameter(self.variable_type_param)

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ANY.value,
            input_types=[ParameterTypeBuiltin.ANY.value],
            output_type=ParameterTypeBuiltin.ALL.value,
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="The initial value of the variable",
        )
        self.add_parameter(self.value_param)

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Handle incoming connections, especially to the value parameter for auto-type detection."""
        if target_parameter.name == self.value_param.name:
            # Auto-name: set immediately from source node name; after_value_set will refine
            # once the actual value propagates.
            if self.get_parameter_value(self.auto_name_param.name):
                derived = self._derive_variable_name(None, source_node.name)
                if derived:
                    self.set_parameter_value(self.variable_name_param.name, derived)

            detected_type = source_parameter.output_type

            # Lock down the variable_type parameter since it's now controlled by the incoming connection
            # Remove INPUT mode so users can't manually edit it while a connection exists
            self.variable_type_param.allowed_modes = self.variable_type_param.allowed_modes - {ParameterMode.INPUT}
            # Make it non-settable programmatically to prevent external interference
            self.variable_type_param.settable = False

            # Clean up any existing incoming connections to variable_type since we're now the authority
            # This prevents conflicts between manual type setting and auto-detected type
            self._delete_incoming_connections_to_parameter(self.variable_type_param.name)

            # Set the detected type as the new value
            # Note: this call will trigger the before_value_set callback,
            # which will setup the value_param's types properly.
            self.set_parameter_value(self.variable_type_param.name, detected_type)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,  # noqa: ARG002
        target_parameter: Parameter,
    ) -> None:
        """Handle removal of incoming connections, especially from the value parameter."""
        if target_parameter.name == self.value_param.name:
            # Restore INPUT mode to variable_type parameter since auto-detection is no longer active
            self.variable_type_param.allowed_modes = self.variable_type_param.allowed_modes | {ParameterMode.INPUT}
            # Make it settable again for manual editing
            self.variable_type_param.settable = True

    def _delete_incoming_connections_to_parameter(self, parameter_name: str) -> None:
        """Helper to delete all incoming connections to a specific parameter."""
        connections_request = ListConnectionsForNodeRequest(node_name=self.name)
        connections_result = GriptapeNodes.handle_request(connections_request)

        if not isinstance(connections_result, ListConnectionsForNodeResultSuccess):
            error_msg = f"Failed to list connections for node '{self.name}': {connections_result.result_details}"
            raise TypeError(error_msg)

        for connection in connections_result.incoming_connections:
            if connection.target_parameter_name == parameter_name:
                delete_request = DeleteConnectionRequest(
                    source_parameter_name=connection.source_parameter_name,
                    target_parameter_name=connection.target_parameter_name,
                    source_node_name=connection.source_node_name,
                    target_node_name=self.name,
                )
                delete_result = GriptapeNodes.handle_request(delete_request)
                if not isinstance(delete_result, DeleteConnectionResultSuccess):
                    error_msg = f"Failed to delete connection from {connection.source_node_name}.{connection.source_parameter_name} to {self.name}.{parameter_name}: {delete_result.result_details}"
                    raise TypeError(error_msg)

    def _cleanup_incompatible_value_connections(self) -> None:
        """Remove all connections to/from value parameter that are incompatible with its current type."""
        connections_request = ListConnectionsForNodeRequest(node_name=self.name)
        connections_result = GriptapeNodes.handle_request(connections_request)

        if isinstance(connections_result, ListConnectionsForNodeResultSuccess):
            # Check incoming connections - we are the target
            for connection in connections_result.incoming_connections:
                if connection.target_parameter_name == self.value_param.name:
                    source_node = GriptapeNodes.NodeManager().get_node_by_name(connection.source_node_name)
                    source_parameter = source_node.get_parameter_by_name(connection.source_parameter_name)

                    # Ask if we (target) accept the source parameter's output_type
                    if source_parameter and not self.value_param.is_incoming_type_allowed(source_parameter.output_type):
                        logger.debug(
                            "Deleting incompatible incoming connection: %s.%s (%s) -> %s.%s (%s)",
                            connection.source_node_name,
                            connection.source_parameter_name,
                            source_parameter.output_type,
                            self.name,
                            connection.target_parameter_name,
                            self.value_param.type,
                        )
                        delete_request = DeleteConnectionRequest(
                            source_node_name=connection.source_node_name,
                            source_parameter_name=connection.source_parameter_name,
                            target_node_name=self.name,
                            target_parameter_name=connection.target_parameter_name,
                        )
                        delete_result = GriptapeNodes.handle_request(delete_request)
                        if not isinstance(delete_result, DeleteConnectionResultSuccess):
                            error_msg = (
                                f"Failed to delete incompatible incoming connection: {delete_result.result_details}"
                            )
                            raise TypeError(error_msg)

            # Check outgoing connections - we are the source
            for connection in connections_result.outgoing_connections:
                if connection.source_parameter_name == self.value_param.name:
                    target_node = GriptapeNodes.NodeManager().get_node_by_name(connection.target_node_name)
                    target_parameter = target_node.get_parameter_by_name(connection.target_parameter_name)

                    # Ask if the target accepts our output_type
                    if target_parameter and not target_parameter.is_incoming_type_allowed(self.value_param.output_type):
                        logger.debug(
                            "Deleting incompatible outgoing connection: %s.%s (%s) -> %s.%s (%s)",
                            self.name,
                            connection.source_parameter_name,
                            self.value_param.output_type,
                            connection.target_node_name,
                            connection.target_parameter_name,
                            target_parameter.type,
                        )
                        delete_request = DeleteConnectionRequest(
                            source_node_name=self.name,
                            source_parameter_name=connection.source_parameter_name,
                            target_node_name=connection.target_node_name,
                            target_parameter_name=connection.target_parameter_name,
                        )
                        delete_result = GriptapeNodes.handle_request(delete_request)
                        if not isinstance(delete_result, DeleteConnectionResultSuccess):
                            error_msg = (
                                f"Failed to delete incompatible outgoing connection: {delete_result.result_details}"
                            )
                            raise TypeError(error_msg)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self.auto_name_param:
            if value:
                self.show_parameter_by_name(self.auto_name_case_param.name)
            else:
                self.hide_parameter_by_name(self.auto_name_case_param.name)

        if parameter is self.value_param and self.get_parameter_value(self.auto_name_param.name):
            source_node_name = self._get_value_source_node_name()
            derived = self._derive_variable_name(value, source_node_name)
            if derived:
                self.set_parameter_value(self.variable_name_param.name, derived)

        super().after_value_set(parameter, value)

    def _get_value_source_node_name(self) -> str | None:
        """Return the name of the node wired into the value parameter, or None."""
        try:
            result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
            if isinstance(result, ListConnectionsForNodeResultSuccess):
                for conn in result.incoming_connections:
                    if conn.target_parameter_name == self.value_param.name:
                        return conn.source_node_name
        except Exception:
            pass
        return None

    def _derive_variable_name(self, value: Any, source_node_name: str | None) -> str | None:
        """Derive a variable name from a value and/or source node name, styled per auto_name_case."""
        raw = self._get_raw_name(value, source_node_name)
        if not raw:
            return None
        case_style = self.get_parameter_value(self.auto_name_case_param.name) or CaseStyle.SNAKE
        if case_style == CaseStyle.AS_IS:
            return raw
        words = self._tokenize(raw)
        if not words:
            return None
        return self._apply_case(words, case_style)

    def _get_raw_name(self, value: Any, source_node_name: str | None) -> str | None:
        """Extract an unstyled base name from a value or source node name.

        Priority:
        1. Artifact with a filename (ImageArtifact, AudioArtifact, VideoArtifact, BlobArtifact)
        2. String that looks like a file path → basename without extension
        3. Source node name (engine _N suffix stripped)
        """
        # Any artifact type: try .name first (filename-bearing artifacts like ImageArtifact),
        # then .value (URL-bearing artifacts like ImageUrlArtifact). Guards on class name so we
        # don't accidentally match plain dicts or strings that happen to have a .name attr.
        if "Artifact" in type(value).__name__:
            artifact_name = getattr(value, "name", None)
            if artifact_name and self._has_file_extension(str(artifact_name)):
                return self._strip_version_suffix(os.path.splitext(str(artifact_name))[0])

            artifact_value = getattr(value, "value", None)
            if isinstance(artifact_value, str):
                basename = os.path.basename(artifact_value.rstrip("/").split("?")[0])
                stem = os.path.splitext(basename)[0]
                if stem and self._has_file_extension(basename):
                    return self._strip_version_suffix(stem)

        # String that looks like a file path
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and ("/" in stripped or "\\" in stripped or self._has_file_extension(stripped)):
                return self._strip_version_suffix(os.path.splitext(os.path.basename(stripped))[0])

        # Fall back to source node name (strip engine _N suffix)
        if source_node_name:
            return re.sub(r"_\d+$", "", source_node_name)

        return None

    @staticmethod
    def _tokenize(name: str) -> list[str]:
        """Split a name into word tokens, handling snake_case, kebab-case, and CamelCase."""
        # Split CamelCase/PascalCase boundaries before any other splitting
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        tokens = re.split(r"[\s_\-\.]+", name)
        return [t for t in tokens if t]

    @staticmethod
    def _apply_case(words: list[str], case_style: str) -> str:
        """Join tokens using the requested case convention."""
        match case_style:
            case CaseStyle.UPPER:
                return " ".join(w.upper() for w in words)
            case CaseStyle.UPPER_SNAKE:
                return "_".join(w.upper() for w in words)
            case CaseStyle.TITLE:
                return " ".join(w.capitalize() for w in words)
            case CaseStyle.SNAKE:
                return "_".join(w.lower() for w in words)
            case CaseStyle.KEBAB:
                return "-".join(w.lower() for w in words)
            case CaseStyle.PASCAL:
                return "".join(w.capitalize() for w in words)
            case CaseStyle.CAMEL:
                return words[0].lower() + "".join(w.capitalize() for w in words[1:])
            case _:
                msg = f"Unknown case style: {case_style!r}"
                raise ValueError(msg)

    @staticmethod
    def _strip_version_suffix(stem: str) -> str:
        """Strip VFX-convention version/frame suffixes from a filename stem.

        character_v001 → character, yellow_house_v001 → yellow_house,
        yellow_house_001 → yellow_house. Single trailing digits (texture2d) are left alone.
        """
        stem = re.sub(r"[_\-]v\d+$", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"[_\-]\d{2,}$", "", stem)
        return stem

    @staticmethod
    def _has_file_extension(s: str) -> bool:
        _EXTENSIONS = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".mp4",
            ".mov",
            ".avi",
            ".mp3",
            ".wav",
            ".ogg",
            ".json",
            ".txt",
            ".csv",
            ".pdf",
        }
        _, ext = os.path.splitext(s)
        return ext.lower() in _EXTENSIONS

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        """Handle changes to the variable_type parameter."""
        if parameter == self.variable_type_param:
            # Step 1: If variable_type_param is set to None or "", assign it to None
            if value is None or value == "":
                value = None

            # Step 2: If variable_type_param is being set to None, reset value_param to defaults
            if value is None:
                # Leave all outgoing connections from value_param intact
                # Change value_param's type to ANY and output_type to ALL
                self.value_param.type = ParameterTypeBuiltin.ANY.value
                self.value_param.output_type = ParameterTypeBuiltin.ALL.value
            else:
                # Step 3: If variable_type_param is NOT being set to None, delete incompatible connections
                # Update value_param type information first
                self.value_param.type = value
                self.value_param.output_type = value

                # Clean up incompatible connections
                self._cleanup_incompatible_value_connections()

        return value

    def process(self) -> None:
        # Lazy imports to avoid circular import issues
        from griptape_nodes.retained_mode.events.node_events import (
            GetFlowForNodeRequest,
            GetFlowForNodeResultSuccess,
        )
        from griptape_nodes.retained_mode.events.variable_events import (
            CreateVariableRequest,
            CreateVariableResultSuccess,
            GetVariableDetailsRequest,
            GetVariableDetailsResultSuccess,
            HasVariableRequest,
            HasVariableResultSuccess,
            SetVariableTypeRequest,
            SetVariableTypeResultSuccess,
            SetVariableValueRequest,
            SetVariableValueResultSuccess,
        )
        from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
        from griptape_nodes.retained_mode.variable_types import VariableScope

        variable_name = self.get_parameter_value("variable_name")
        variable_type = self.get_parameter_value("variable_type")
        value = self.get_parameter_value("value")

        # Get the flow that owns this node
        flow_request = GetFlowForNodeRequest(node_name=self.name)
        flow_result = GriptapeNodes.handle_request(flow_request)

        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            error_msg = f"Failed to get flow for node '{self.name}': {flow_result.result_details}"
            raise TypeError(error_msg)

        current_flow_name = flow_result.flow_name

        # Step 1: Check if the variable already exists in the current flow
        has_request = HasVariableRequest(
            name=variable_name,
            lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
            starting_flow=current_flow_name,
        )
        has_result = GriptapeNodes.handle_request(has_request)

        if not isinstance(has_result, HasVariableResultSuccess):
            error_msg = f"Failed to check if variable '{variable_name}' exists: {has_result.result_details}"
            raise TypeError(error_msg)

        if has_result.exists:
            # Variable exists - check if type needs updating
            # Step 2a: Get variable details to check type
            details_request = GetVariableDetailsRequest(
                name=variable_name,
                lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
                starting_flow=current_flow_name,
            )
            details_result = GriptapeNodes.handle_request(details_request)

            if not isinstance(details_result, GetVariableDetailsResultSuccess):
                error_msg = (
                    f"Failed to get details for existing variable '{variable_name}': {details_result.result_details}"
                )
                raise TypeError(error_msg)

            # Step 2b: Update type if it doesn't match
            if details_result.details.type != variable_type:
                type_request = SetVariableTypeRequest(
                    name=variable_name,
                    type=variable_type,
                    lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
                    starting_flow=current_flow_name,
                )
                type_result = GriptapeNodes.handle_request(type_request)

                if not isinstance(type_result, SetVariableTypeResultSuccess):
                    error_msg = f"Failed to update type for variable '{variable_name}': {type_result.result_details}"
                    raise TypeError(error_msg)

            # Step 3: Update the value for existing variable
            value_request = SetVariableValueRequest(
                name=variable_name,
                value=value,
                lookup_scope=VariableScope.CURRENT_FLOW_ONLY,
                starting_flow=current_flow_name,
            )
            value_result = GriptapeNodes.handle_request(value_request)

            if not isinstance(value_result, SetVariableValueResultSuccess):
                error_msg = f"Failed to set value for variable '{variable_name}': {value_result.result_details}"
                raise TypeError(error_msg)
        else:
            # Variable doesn't exist - create it (creation includes setting the initial value)
            create_request = CreateVariableRequest(
                name=variable_name,
                type=variable_type,
                is_global=False,  # Always create flow-scoped variables
                value=value,
                owning_flow=current_flow_name,
            )
            create_result = GriptapeNodes.handle_request(create_request)

            if not isinstance(create_result, CreateVariableResultSuccess):
                error_msg = f"Failed to create variable '{variable_name}': {create_result.result_details}"
                raise TypeError(error_msg)

        # Set output values
        self.parameter_output_values["variable_name"] = variable_name
        self.parameter_output_values["variable_type"] = variable_type
        self.parameter_output_values["value"] = value

    def get_node_dependencies(self) -> NodeDependencies | None:
        """Declare the variable this node creates or updates so it survives serialization.

        Access is READ_WRITE: ``process()`` calls ``HasVariableRequest`` + ``GetVariableDetailsRequest``
        to decide whether to update an existing variable or create a new one, so the node both
        reads and writes the variable's state.

        The node has no ``scope`` parameter — it unconditionally creates flow-scoped variables,
        so the reference is declared at ``CURRENT_FLOW_ONLY``.

        Reads the current value of ``variable_name`` via ``get_parameter_value`` — if the parameter
        is driven by an incoming connection, this returns the last propagated value (or ``None`` if
        nothing has propagated yet). No declaration is emitted for empty/None names.
        """
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()

        variable_name = self.get_parameter_value(self.variable_name_param.name)
        if isinstance(variable_name, str) and variable_name:
            deps.variable_references.add(
                VariableReference(
                    name=variable_name,
                    scope=VariableScope.CURRENT_FLOW_ONLY,
                    access=VariableAccess.READ_WRITE,
                )
            )

        return deps

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Variable nodes have side effects and need to execute every workflow run."""
        self.make_node_unresolved(
            current_states_to_trigger_change_event={NodeResolutionState.RESOLVED, NodeResolutionState.RESOLVING}
        )
        return None

    def validate_before_node_run(self) -> list[Exception] | None:
        """Variable nodes have side effects and need to execute every time they run."""
        self.make_node_unresolved(
            current_states_to_trigger_change_event={NodeResolutionState.RESOLVED, NodeResolutionState.RESOLVING}
        )
        return None
