"""Load a node the way a saved workflow (or a worker's copy) does, and record what it asks while loading.

A load creates the node with ``CreateNodeRequest(initial_setup=True)``, replays its structure
commands, then sets each saved value with ``SetParameterValueRequest(initial_setup=True)``. The
copy lands in a scratch flow holding nothing but itself, so it has no connections and no
neighbours. While that runs, node code must not read or change connections, other nodes, or the
contents and variables of its flow (strict-mode rule ``graph-access-while-loading``).
"""

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.base_events import RequestPayload
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    DeleteConnectionRequest,
    ListConnectionsForNodeRequest,
)
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    ListNodesInFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeResultSuccess,
    GetFlowForNodeRequest,
    SerializeNodeToCommandsRequest,
    SerializeNodeToCommandsResultSuccess,
)
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterGroupToNodeRequest,
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    AlterParameterGroupDetailsRequest,
    GetConnectionsForParameterRequest,
    RemoveParameterFromNodeRequest,
)
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    DeleteVariableRequest,
    GetVariableDetailsRequest,
    GetVariableRequest,
    HasVariableRequest,
    ListVariablesRequest,
    SetVariableTypeRequest,
    SetVariableValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.event_manager import EventManager
from griptape_nodes.serialization.values import decode_value

FORBIDDEN_WHILE_LOADING: tuple[type[RequestPayload], ...] = (
    ListConnectionsForNodeRequest,
    GetConnectionsForParameterRequest,
    CreateConnectionRequest,
    DeleteConnectionRequest,
    GetFlowForNodeRequest,
    ListNodesInFlowRequest,
    HasVariableRequest,
    GetVariableRequest,
    GetVariableDetailsRequest,
    ListVariablesRequest,
    CreateVariableRequest,
    SetVariableValueRequest,
    SetVariableTypeRequest,
    DeleteVariableRequest,
    RemoveParameterFromNodeRequest,
)

# Element commands carry the source node's name and must be pointed at the copy.
_ELEMENT_COMMAND_TYPES = (
    AlterParameterDetailsRequest,
    AddParameterToNodeRequest,
    AddParameterGroupToNodeRequest,
    AlterParameterGroupDetailsRequest,
)


# The engine loads node modules from the library's files, so match on this checkout's source path.
LIBRARY_ROOT = str(Path(__file__).parents[3] / "griptape_nodes_library")


def _sent_by_library_code() -> bool:
    """Whether library code is on the stack. The engine's own requests during a load are not the node's."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.startswith(LIBRARY_ROOT):
            return True
        frame = frame.f_back
    return False


@dataclass
class RequestRecorder:
    """Every request the engine dispatched while recording, in order, and which ones library code sent."""

    requests: list[RequestPayload] = field(default_factory=list)
    library_requests: list[RequestPayload] = field(default_factory=list)

    def forbidden(self) -> list[str]:
        """Names of the forbidden requests library code sent."""
        return [
            type(request).__name__ for request in self.library_requests if isinstance(request, FORBIDDEN_WHILE_LOADING)
        ]

    def of_type(self, request_type: type[RequestPayload]) -> list[Any]:
        return [request for request in self.requests if isinstance(request, request_type)]

    def _record(self, request: RequestPayload) -> None:
        self.requests.append(request)
        if _sent_by_library_code():
            self.library_requests.append(request)


@contextmanager
def record_requests(monkeypatch: pytest.MonkeyPatch) -> Generator[RequestRecorder, None, None]:
    """Record each request at the engine's single dispatch point, sync and async."""
    recorder = RequestRecorder()
    original_handle = EventManager.handle_request
    original_ahandle = EventManager.ahandle_request

    def handle_request(self: EventManager, request: RequestPayload, **kwargs: Any) -> Any:
        recorder._record(request)
        return original_handle(self, request, **kwargs)

    async def ahandle_request(self: EventManager, request: RequestPayload, **kwargs: Any) -> Any:
        recorder._record(request)
        return await original_ahandle(self, request, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(EventManager, "handle_request", handle_request)
        patch.setattr(EventManager, "ahandle_request", ahandle_request)
        yield recorder


def create_scratch_flow(flow_name: str) -> str:
    """A top-level flow that will hold only the loaded copy, like a worker's."""
    result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=flow_name))
    assert isinstance(result, CreateFlowResultSuccess), result.result_details
    return result.flow_name


def load_copy(
    source_node_name: str, flow_name: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[BaseNode, RequestRecorder]:
    """Serialize ``source_node_name`` and load it into ``flow_name`` as a saved workflow would.

    Returns the loaded copy and every request dispatched while it loaded.
    """
    values: dict = {}
    serialized = GriptapeNodes.handle_request(
        SerializeNodeToCommandsRequest(node_name=source_node_name, unique_parameter_uuid_to_values=values)
    )
    assert isinstance(serialized, SerializeNodeToCommandsResultSuccess), serialized.result_details
    commands = serialized.serialized_node_commands
    create_request = commands.create_node_command
    assert create_request.initial_setup
    create_request.node_name = f"{source_node_name}_copy"
    create_request.override_parent_flow_name = flow_name

    with record_requests(monkeypatch) as recorder:
        created = GriptapeNodes.handle_request(create_request)
        assert isinstance(created, CreateNodeResultSuccess), created.result_details
        copy_name = created.node_name
        for element_command in commands.element_modification_commands:
            if isinstance(element_command, _ELEMENT_COMMAND_TYPES):
                element_command.node_name = copy_name
            result = GriptapeNodes.handle_request(element_command)
            assert result.succeeded(), result.result_details
        for indirect in serialized.set_parameter_value_commands:
            set_request = indirect.set_parameter_value_command
            assert set_request.initial_setup
            set_request.node_name = copy_name
            set_request.value = decode_value(values[indirect.unique_value_uuid])
            result = GriptapeNodes.handle_request(set_request)
            assert result.succeeded(), result.result_details

    return GriptapeNodes.NodeManager().get_node_by_name(copy_name), recorder
