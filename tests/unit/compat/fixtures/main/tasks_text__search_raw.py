# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__search_raw"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "SearchWeb"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:39.462310Z
# last_modified_date = 2026-10-07T00:30:39.463220Z
# 
# ///

import pickle
from griptape_nodes.node_library.library_registry import IconVariant, NodeDeprecationMetadata, NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, AlterParameterDetailsRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(RegisterLibraryFromFileRequest(library_name='Griptape Nodes Library', perform_discovery_if_not_found=True))
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {'124399c7-1ad1-4faa-97ef-b62c6d446a69': pickle.loads(b'\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0bpydantic ai\x94.'), '2d65676c-dc2b-4ad0-b1f2-9d6628ab5a93': pickle.loads(b'\x80\x04\x89.'), '18678e9c-b357-40a5-99fc-de928f5f9e16': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94.'), '7e6076b7-0ee6-458c-8d01-cbed7033d72e': pickle.loads(b'\x80\x04\x95\x0e\x00\x00\x00\x00\x00\x00\x00\x8c\nDuckDuckGo\x94.'), 'ef0aea93-3ac1-4971-84e0-75ffa2890b3d': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), '35ffc987-d434-4cfb-8af3-f344fb698091': pickle.loads(b'\x80\x04\x95\x11\x08\x00\x00\x00\x00\x00\x00X\n\x08\x00\x00{"title": "Pydantic AI | Pydantic Docs", "url": "https://pydantic.dev/docs/ai/overview/", "description": "Pydantic Logfire is the AI observability platform that sees your whole app, not just the LLM calls, and the Pydantic AI Gateway is one key for every model with real-time cost monitoring and budget control; the Gateway self-hosts if you would rather, and our instrumentation is plain OpenTelemetry, so any backend you already run works."}\n\n{"title": "Pydantic AI: Type-Safe Python Framework for AI Agents & LLM Applications", "url": "https://pydantic.dev/pydantic-ai", "description": "Build production-grade AI applications with Pydantic AI - a model-agnostic Python framework featuring type safety, structured outputs, validation, and seamless observability. Supports OpenAI, Anthropic, Google, and more."}\n\n{"title": "GitHub - pydantic/pydantic-ai: How Python does AI. Agents, realtime ...", "url": "https://github.com/pydantic/pydantic-ai", "description": "Pydantic AI is the Python AI SDK: a typed, extensible agent loop with every model a string swap away. The same agent runs everywhere you need it: behind a web frontend, in the terminal, on a voice call, on a durable background queue, in GitHub Actions, or as a plain object you call run () on. Image generation and embeddings come in the same box; Pydantic Graph and Pydantic Evals are separate ..."}\n\n{"title": "Pydantic AI: A Beginner\'s Guide With Practical Examples", "url": "https://www.datacamp.com/tutorial/pydantic-ai-guide", "description": "Learn how to build reliable AI agents with Pydantic AI in Python. Validate outputs, use tools, and stream insights with practical code examples."}\n\n{"title": "pydantic-ai \\u00b7 PyPI", "url": "https://pypi.org/project/pydantic-ai/", "description": "Pydantic AI is the Python AI SDK: a typed, extensible agent loop with every model a string swap away. The same agent runs everywhere you need it: behind a web frontend, in the terminal, on a voice call, on a durable background queue, in GitHub Actions, or as a plain object you call run () on."}\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='SearchWeb', specific_library_name='Griptape Nodes Library', node_name='search_raw', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Search the web for information', 'display_name': 'Search Web', 'tags': ['text', 'web', 'api', 'search'], 'icon': 'binoculars', 'color': None, 'group': 'Input/Output', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'SearchWeb'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['124399c7-1ad1-4faa-97ef-b62c6d446a69'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='summarize', node_name=node0_name, value=top_level_unique_values_dict['2d65676c-dc2b-4ad0-b1f2-9d6628ab5a93'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['18678e9c-b357-40a5-99fc-de928f5f9e16'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='search_engine', node_name=node0_name, value=top_level_unique_values_dict['7e6076b7-0ee6-458c-8d01-cbed7033d72e'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['ef0aea93-3ac1-4971-84e0-75ffa2890b3d'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['35ffc987-d434-4cfb-8af3-f344fb698091'], initial_setup=True, is_output=True))
