import logging

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload
from griptape_nodes.retained_mode.events.workflow_events import (
    PublishWorkflowRegisteredEventData,
    PublishWorkflowRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes")


def _publish_workflow_request_handler(request: RequestPayload) -> ResultPayload:
    if not isinstance(request, PublishWorkflowRequest):
        msg = f"Expected PublishWorkflowRequest, got {type(request).__name__}"
        raise TypeError(msg)
    from griptape_nodes_library.publish_workflow.local_publisher import LocalPublisher

    publisher = LocalPublisher(
        workflow_name=request.workflow_name,
        pickle_control_flow_result=request.pickle_control_flow_result,
    )
    return publisher.publish_workflow()


class GriptapeNodesLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for the default Griptape Nodes Library."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        """Called after all nodes have been loaded from the library."""
        GriptapeNodes.LibraryManager().on_register_event_handler(
            request_type=PublishWorkflowRequest,
            handler=_publish_workflow_request_handler,
            library_data=library_data,
            event_data=PublishWorkflowRegisteredEventData(
                start_flow_node_type="StartFlow",
                start_flow_node_library_name=library_data.name,
                end_flow_node_type="EndFlow",
                end_flow_node_library_name=library_data.name,
            ),
        )
