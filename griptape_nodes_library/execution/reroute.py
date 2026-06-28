from griptape_nodes_library.execution.base_pass_through import BasePassThroughNode


class Reroute(BasePassThroughNode):
    pass_thru_parameter_name = "passThru"

    def process(self) -> None:
        pass
