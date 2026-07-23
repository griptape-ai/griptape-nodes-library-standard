from typing import Literal

from griptape.artifacts import BaseArtifact
from griptape.common import Message, PromptStack
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.tools import BaseTool as GtBaseTool
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

from griptape_nodes_library.utils.model_invocation import declare_model_invocation_sync

VariantType = Literal["info", "warning", "error", "success", "tip", "note", "help", "docs", "link", "cloud-upload"]

# Marks a driver whose `run` has already been wrapped by `_gate_prompt_driver`, so a
# driver reused across multiple `process()` calls (e.g. a connected upstream driver
# instance) is not wrapped in a growing chain of redundant gates.
_GATED_RUN_MARKER = "_declare_model_invocation_gated"


class BaseTool(DataNode):
    """Base tool node for creating Griptape tools.

    This node provides a generic implementation for initializing Griptape tools with configurable parameters.

    Attributes:
        off_prompt (bool): Indicates whether the tool should operate in off-prompt mode.
        tool (BaseTool): A dictionary representation of the created tool.
    """

    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name, metadata)

        tool_param = Parameter(
            name="tool",
            input_types=["Tool"],
            type="Tool",
            output_type="Tool",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Connect this to an Agent's tools input.",
        )
        tool_param.set_badge(
            variant="info",
            title="Tool",
            message="This tool can be provided to an agent to help it perform tasks.",
            hide_clear_button=True,
        )
        self.add_parameter(tool_param)

        self.add_parameter(
            Parameter(
                name="off_prompt",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=False,
                tooltip="",
            )
        )

    def update_tool_info(self, value: str = "", title: str = "", variant: VariantType = "info") -> None:
        tool_param = self.get_parameter_by_name("tool")
        if tool_param is None:
            return
        tool_param.set_badge(
            variant=variant,
            title=title or None,
            message=value or None,
            hide_clear_button=True,
        )

    def process(self) -> None:
        off_prompt = self.parameter_values.get("off_prompt", False)

        # Create the tool
        tool = GtBaseTool(off_prompt=off_prompt)

        # Set the output
        self.parameter_output_values["tool"] = tool

    def _gate_prompt_driver(self, driver: BasePromptDriver) -> BasePromptDriver:
        """Wrap `driver.run` so the permission layer gates every actual model call.

        Tool nodes that build an engine around a prompt driver (extraction,
        summarization) hand the driver to griptape's engine code, which calls
        `driver.run(...)` itself once the agent invokes the tool -- after this
        node's own `process()` has already returned, and potentially many times
        over the tool's lifetime. The driver is not a node and has no way to reach
        `declare_model_invocation` on its own, so the owning node wraps the bound
        `run` here, closing over itself, before the driver is handed off. That
        makes the declaration fire at the moment each real call actually happens,
        rather than once (and only once) back when the node ran.

        Mutates and returns `driver`. Idempotent: a driver already gated by a
        previous `process()` call (e.g. one connected in from an upstream node and
        reused across runs) is returned unchanged rather than wrapped again.
        Fails closed -- a denied declaration raises instead of letting the
        wrapped call reach the provider.
        """
        if getattr(driver, _GATED_RUN_MARKER, False):
            return driver

        original_run = driver.run

        def _gated_run(prompt_input: PromptStack | BaseArtifact) -> Message:
            declaration = declare_model_invocation_sync(self, driver.model)
            if declaration.failed():
                details = str(declaration.result_details or f"{self.name}: model invocation was not permitted.")
                raise RuntimeError(details)
            return original_run(prompt_input)

        driver.run = _gated_run  # type: ignore[method-assign]
        setattr(driver, _GATED_RUN_MARKER, True)
        return driver
