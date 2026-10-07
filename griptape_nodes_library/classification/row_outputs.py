"""Give each row of a ParameterList its own flow output, kept in step with the list."""

from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import ControlParameterOutput, Parameter, ParameterList
from griptape_nodes.exe_types.node_types import BaseNode, NodeResolutionState
from griptape_nodes.retained_mode.events.connection_events import (
    DeleteConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)


def parse_row(text: str | None) -> tuple[str, str | None]:
    """Split 'Label: description' at the first colon. A row without a colon is just a label."""
    label, _, description = (text or "").partition(":")
    return label.strip(), description.strip() or None


class RowOutputsMixin(BaseNode):
    """Adds a flow output for each non-empty row of the ParameterList named ROWS_PARAM.

    Outputs are named OUTPUT_PREFIX plus the row's unique ID rather than its position,
    so reordering, renaming, and deleting rows keep every wire on the row it was
    connected to.

    Any parameter whose name starts with OUTPUT_PREFIX is treated as a row output,
    so no other parameter on the node may share the prefix.
    """

    ROWS_PARAM: ClassVar[str]
    OUTPUT_PREFIX: ClassVar[str]
    _syncing = False

    def _row_output_label(self, index: int, text: str) -> str:
        """The label for a row's output. index counts non-empty rows from 0."""
        raise NotImplementedError

    def _row_output_tooltip(self, label: str) -> str:
        return f"Taken when JEV picks {label}."

    # Adding, deleting, or reordering a row fires no value hook, but each one marks the node
    # unresolved. Catch that here to keep the flow outputs in step with the list.
    @property
    def state(self) -> NodeResolutionState:
        return self._state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        BaseNode.state.__set__(self, new_state)
        if new_state == NodeResolutionState.UNRESOLVED:
            self._sync_row_outputs()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == self.ROWS_PARAM:
            self._sync_row_outputs()
        return super().after_value_set(parameter, value)

    def add_parameter(self, param: Parameter) -> None:
        # Loading a saved workflow recreates added parameters as plain Parameters, which loses
        # the flow output class. Rebuild row outputs as real flow outputs, keeping the saved label.
        if param.name.startswith(self.OUTPUT_PREFIX) and not isinstance(param, ControlParameterOutput):
            param = ControlParameterOutput(name=param.name, display_name=param.display_name, tooltip=param.tooltip)
        super().add_parameter(param)

    def _row_output_params(self) -> list[Parameter]:
        return [p for p in self.parameters if p.name.startswith(self.OUTPUT_PREFIX)]

    def _row_output_name(self, row: Parameter) -> str:
        return self.OUTPUT_PREFIX + row.name.rsplit("_", 1)[-1]

    def _rows(self) -> list[tuple[str, str]]:
        """The non-empty rows in order, as (output name, row text)."""
        rows_param = self.get_parameter_by_name(self.ROWS_PARAM)
        if not isinstance(rows_param, ParameterList):
            return []
        rows = []
        for row in rows_param.get_child_parameters():
            text = (self.get_parameter_value(row.name) or "").strip()
            if text:
                rows.append((self._row_output_name(row), text))
        return rows

    def _sync_row_outputs(self) -> None:
        """Give each non-empty row a flow output, labeled and ordered to match the list."""
        rows_param = self.get_parameter_by_name(self.ROWS_PARAM) if hasattr(self, "root_ui_element") else None
        if self._syncing or not isinstance(rows_param, ParameterList):
            return
        self._syncing = True
        try:
            wanted: list[tuple[str, str]] = []
            for row in rows_param.get_child_parameters():
                output_name = self._row_output_name(row)
                if row.name not in self.parameter_values:
                    existing = self.get_parameter_by_name(output_name)
                    if existing is not None:
                        wanted.append((output_name, existing.display_name or ""))
                    continue
                text = (self.parameter_values[row.name] or "").strip()
                if text:
                    label = self._row_output_label(len(wanted), text)
                    if label:
                        wanted.append((output_name, label))

            wanted_names = [name for name, _ in wanted]
            for param in self._row_output_params():
                if param.name not in wanted_names:
                    self._delete_output_connections(param.name)
                    self.remove_parameter_element(param)

            for name, label in wanted:
                param = self.get_parameter_by_name(name)
                if param is None:
                    self.add_parameter(
                        ControlParameterOutput(name=name, display_name=label, tooltip=self._row_output_tooltip(label))
                    )
                elif param.display_name != label:
                    param.display_name = label
                    param.tooltip = self._row_output_tooltip(label)

            if [p.name for p in self._row_output_params()] != wanted_names:
                for name in wanted_names:
                    param = self.get_parameter_by_name(name)
                    if param is not None:
                        self.root_ui_element.remove_child(param)
                        self.root_ui_element.add_child(param)

            # New row outputs are appended, so move Failed back below them.
            failure = self.get_parameter_by_name("failure")
            if failure is not None and self.root_ui_element.children[-1] is not failure:
                self.root_ui_element.remove_child(failure)
                self.root_ui_element.add_child(failure)
        finally:
            self._syncing = False

    def _delete_output_connections(self, parameter_name: str) -> None:
        """Delete the wires leaving an output before removing it."""
        result = self.engine.handle_request(ListConnectionsForNodeRequest(node_name=self.name, broadcast_result=False))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            return
        for connection in result.outgoing_connections:
            if connection.source_parameter_name == parameter_name:
                self.engine.handle_request(
                    DeleteConnectionRequest(
                        source_node_name=self.name,
                        source_parameter_name=parameter_name,
                        target_node_name=connection.target_node_name,
                        target_parameter_name=connection.target_parameter_name,
                    )
                )
