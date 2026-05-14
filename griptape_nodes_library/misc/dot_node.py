from griptape_nodes_library.execution.base_pass_through import BasePassThroughNode


class DotNode(BasePassThroughNode):
    pass_thru_parameter_name = "value"
    pass_thru_parameter_tooltip = "Pass-through value"
