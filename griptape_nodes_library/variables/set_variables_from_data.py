import json
import logging
import re
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import (
    NodeDependencies,
    NodeResolutionState,
    SuccessFailureNode,
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
    SetVariableValueRequest,
    SetVariableValueResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

from griptape_nodes_library.variables.variable_utils import (
    create_advanced_parameter_group,
    scope_string_to_variable_scope,
)

logger = logging.getLogger("griptape_nodes")


class SetVariablesFromData(SuccessFailureNode):
    """Turn a dict / JSON object / list of key-value pairs into workflow variables in one step.

    The ``source`` input is intentionally permissive. The node sniffs the runtime shape rather
    than making the user pick a mode:

    - ``dict`` -> used directly (the common case).
    - ``str`` -> parsed as JSON, then treated as the resulting dict or list.
    - ``list`` -> treated as an ordered sequence of key/value pairs. Two item forms are accepted:
      ``[{"key": ..., "value": ...}, ...]`` and ``[["NAME", "Jason"], ...]``.

    ``dict`` and JSON-object inputs silently collapse duplicate keys (Python dict semantics). The
    list form can carry duplicate keys; the defined rule is **last write wins** — the final value
    for a repeated key is its last occurrence in the list.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.source_param = Parameter(
            name="source",
            type=ParameterTypeBuiltin.ANY.value,
            input_types=["dict", "json", "str", "list", ParameterTypeBuiltin.ANY.value],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip=(
                "A dict, a JSON string, or a list of key/value pairs. "
                "Accepted list-item forms: "
                '[{"key": "NAME", "value": "Jason"}], '
                '[["NAME", "Jason"]], or '
                '[{"NAME": "Jason"}] (single-entry dict). '
                "Each key becomes a workflow variable."
            ),
        )
        self.add_parameter(self.source_param)

        self.sanitize_names_param = Parameter(
            name="sanitize_names",
            type=ParameterTypeBuiltin.BOOL.value,
            default_value=True,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip=(
                "Convert keys that aren't valid variable names (spaces, punctuation) into safe "
                "names. When off, keys are used verbatim and an invalid key raises an error."
            ),
        )
        self.add_parameter(self.sanitize_names_param)

        self.overwrite_existing_param = Parameter(
            name="overwrite_existing",
            type=ParameterTypeBuiltin.BOOL.value,
            default_value=True,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="If a variable with the same name already exists, overwrite its value. When off, existing variables are left untouched.",
        )
        self.add_parameter(self.overwrite_existing_param)

        self.created_names_param = Parameter(
            name="variable_names",
            type="list",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The names of the variables created or updated, in order.",
        )
        self.add_parameter(self.created_names_param)

        # Advanced parameters group (collapsed by default) — shares the scope control with the
        # other variable nodes.
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

        self._create_status_parameters(
            result_details_tooltip="Details about the variables created or updated.",
            result_details_placeholder="Variable creation details will appear here after the node runs.",
        )

    def _resolve_pairs(self) -> list[tuple[str, Any]]:
        """Normalize whatever is on ``source`` into an ordered list of (key, value) pairs."""
        source = self.get_parameter_value(self.source_param.name)
        return _source_to_pairs(source)

    def _build_variable_map(self, raw_pairs: list[tuple[str, Any]], sanitize: bool) -> tuple[list[str], dict[str, Any]]:
        """Validate and sanitize raw pairs, collapse duplicates on the sanitized name.

        Dedup happens after sanitization so that e.g. "Full Name" and "Full_Name" correctly
        merge rather than producing a duplicate "Full_Name" entry. Last-write-wins; first-seen
        ordering is preserved.

        Returns (seen_order, final_values).
        """
        seen_order: list[str] = []
        final_values: dict[str, Any] = {}
        for raw_key, value in raw_pairs:
            variable_name = _sanitize_name(raw_key) if sanitize else str(raw_key)
            if not _is_valid_name(variable_name):
                if sanitize:
                    msg = (
                        f"Key {raw_key!r} cannot be made into a valid variable name even after "
                        f"sanitization (result: {variable_name!r}). Provide a key with at least "
                        f"one alphanumeric character."
                    )
                else:
                    msg = (
                        f"Key {raw_key!r} is not a valid variable name. Enable 'sanitize_names' or "
                        f"provide keys without spaces/punctuation."
                    )
                raise ValueError(msg)
            if variable_name not in final_values:
                seen_order.append(variable_name)
            final_values[variable_name] = value
        return seen_order, final_values

    async def _write_variable(
        self,
        variable_name: str,
        value: Any,
        overwrite: bool,
        scope: VariableScope,
        flow_name: str,
    ) -> bool:
        """Create or update a single variable. Returns True if written, False if skipped."""
        has_result = await GriptapeNodes.ahandle_request(
            HasVariableRequest(name=variable_name, lookup_scope=scope, starting_flow=flow_name)
        )
        if not isinstance(has_result, HasVariableResultSuccess):
            msg = f"Failed to check if variable '{variable_name}' exists: {has_result.result_details}"
            raise TypeError(msg)

        if has_result.exists:
            if not overwrite:
                logger.debug(
                    "SetVariablesFromData '%s' skipped existing variable '%s' (overwrite_existing is off)",
                    self.name,
                    variable_name,
                )
                return False
            set_result = await GriptapeNodes.ahandle_request(
                SetVariableValueRequest(value=value, name=variable_name, lookup_scope=scope, starting_flow=flow_name)
            )
            if not isinstance(set_result, SetVariableValueResultSuccess):
                msg = f"Failed to set variable '{variable_name}': {set_result.result_details}"
                raise TypeError(msg)
        else:
            create_result = await GriptapeNodes.ahandle_request(
                CreateVariableRequest(
                    name=variable_name,
                    type=_infer_type(value),
                    is_global=False,
                    value=value,
                    owning_flow=flow_name,
                )
            )
            if not isinstance(create_result, CreateVariableResultSuccess):
                msg = f"Failed to create variable '{variable_name}': {create_result.result_details}"
                raise TypeError(msg)
        return True

    async def aprocess(self) -> None:
        self._clear_execution_status()
        try:
            sanitize = bool(self.get_parameter_value(self.sanitize_names_param.name))
            overwrite = bool(self.get_parameter_value(self.overwrite_existing_param.name))
            scope_str = self.get_parameter_value(self.scope_param.name)
            scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL

            seen_order, final_values = self._build_variable_map(self._resolve_pairs(), sanitize)

            flow_result = await GriptapeNodes.ahandle_request(GetFlowForNodeRequest(node_name=self.name))
            if not isinstance(flow_result, GetFlowForNodeResultSuccess):
                msg = f"Failed to get flow for node '{self.name}': {flow_result.result_details}"
                raise TypeError(msg)
            flow_name = flow_result.flow_name

            created_names = [
                variable_name
                for variable_name in seen_order
                if await self._write_variable(variable_name, final_values[variable_name], overwrite, scope, flow_name)
            ]
            self.parameter_output_values[self.created_names_param.name] = created_names

            detail = (
                f"Set {len(created_names)} variable(s): {', '.join(created_names)}"
                if created_names
                else "No variables written (all skipped or source was empty)."
            )
            self._set_status_results(was_successful=True, result_details=detail)
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self._handle_failure_exception(exc)

    def get_node_dependencies(self) -> NodeDependencies | None:
        """Declare every variable this node creates/updates so they survive serialization.

        Access is READ_WRITE: ``aprocess()`` checks existence before deciding to set or create,
        so the node both reads and writes each variable's state. Names are resolved from the
        current ``source`` value; if ``source`` is driven by a connection that hasn't propagated
        yet, no references are emitted (they'll be created when the flow runs).
        """
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()

        scope_str = self.get_parameter_value(self.scope_param.name)
        scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL
        sanitize = bool(self.get_parameter_value(self.sanitize_names_param.name))

        try:
            seen_order, _ = self._build_variable_map(self._resolve_pairs(), sanitize)
        except (ValueError, TypeError):
            # Source isn't in a usable shape yet, or contains invalid names — nothing to declare.
            return deps

        for variable_name in seen_order:
            deps.variable_references.add(
                VariableReference(name=variable_name, scope=scope, access=VariableAccess.READ_WRITE)
            )

        return deps

    def _reset_resolution_state(self) -> None:
        self.make_node_unresolved(
            current_states_to_trigger_change_event={NodeResolutionState.RESOLVED, NodeResolutionState.RESOLVING}
        )

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Variable nodes have side effects and need to execute every workflow run."""
        self._reset_resolution_state()
        return None

    def validate_before_node_run(self) -> list[Exception] | None:
        """Variable nodes have side effects and need to execute every time they run."""
        self._reset_resolution_state()
        return None


def _source_to_pairs(source: Any) -> list[tuple[str, Any]]:
    """Normalize a dict / JSON string / list-of-pairs into an ordered list of (key, value) tuples.

    Raises:
        ValueError: if ``source`` (or a list item) isn't a recognizable key/value shape.
    """
    if source is None:
        msg = "SetVariablesFromData requires a non-empty 'source' (dict, JSON string, or list of key/value pairs)."
        raise ValueError(msg)

    # JSON string -> parse, then fall through to dict/list handling.
    if isinstance(source, str):
        text = source.strip()
        if not text:
            msg = "SetVariablesFromData received an empty string for 'source'."
            raise ValueError(msg)
        try:
            source = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"'source' is a string but not valid JSON: {exc}"
            raise ValueError(msg) from exc

    if isinstance(source, dict):
        return [(str(key), value) for key, value in source.items()]

    if isinstance(source, list):
        return [_list_item_to_pair(item, index) for index, item in enumerate(source)]

    msg = f"'source' must be a dict, JSON object/array string, or list of key/value pairs; got {type(source).__name__}."
    raise ValueError(msg)


def _list_item_to_pair(item: Any, index: int) -> tuple[str, Any]:
    """Turn a single list item into a (key, value) pair.

    Accepts ``{"key": ..., "value": ...}``, a single-entry ``{name: value}`` dict, or a
    two-element ``[key, value]`` sequence.
    """
    if isinstance(item, dict):
        if "key" in item and "value" in item:
            extra = set(item) - {"key", "value"}
            if extra:
                logger.debug(
                    "_list_item_to_pair: item %d has extra keys %r — only 'key' and 'value' are used.",
                    index,
                    sorted(extra),
                )
            return (str(item["key"]), item["value"])
        if len(item) == 1:
            key, value = next(iter(item.items()))
            return (str(key), value)
        msg = (
            f"List item {index} is a dict but not a key/value pair. Use "
            f'{{"key": ..., "value": ...}} or a single-entry {{name: value}} dict.'
        )
        raise ValueError(msg)

    if isinstance(item, (list, tuple)):
        if len(item) == 2:  # noqa: PLR2004
            key, value = item
            return (str(key), value)
        msg = f"List item {index} must have exactly two elements [key, value]; got {len(item)}."
        raise ValueError(msg)

    msg = f"List item {index} must be a [key, value] pair or a key/value dict; got {type(item).__name__}."
    raise ValueError(msg)


def _sanitize_name(raw_key: Any) -> str:
    """Coerce a key into a safe variable name: trim, then replace runs of invalid chars with '_'.

    If the key is already a valid identifier after trimming (e.g. "_config"), it is returned
    as-is so that intentional leading underscores are not silently dropped.
    """
    text = str(raw_key).strip()
    if _is_valid_name(text):
        return text
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _is_valid_name(name: str) -> bool:
    """A usable variable name is a non-empty Python-identifier-like string."""
    return bool(name) and name.isidentifier()


def _infer_type(value: Any) -> str:
    """Map a Python value to the library's variable type string.

    bool is checked before int because ``bool`` is a subclass of ``int`` in Python.
    """
    match value:
        case bool():
            return ParameterTypeBuiltin.BOOL.value
        case int():
            return ParameterTypeBuiltin.INT.value
        case float():
            return ParameterTypeBuiltin.FLOAT.value
        case str():
            return ParameterTypeBuiltin.STR.value
        case dict() | list():
            return "json"
        case _:
            return ParameterTypeBuiltin.ANY.value
