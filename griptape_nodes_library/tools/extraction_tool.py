from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.engines import CsvExtractionEngine, JsonExtractionEngine
from griptape.rules import Rule
from griptape.tools import ExtractionTool as GtExtractionTool

from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.cloud_driver_auth import cloud_driver_auth

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"


class StructuredDataExtractor(BaseTool):
    def process(self) -> None:
        prompt_driver = self.parameter_values.get("prompt_driver", None)
        extraction_type = self.parameter_values.get("extraction_type", "json")
        column_names_string = self.parameter_values.get("column_names", "")
        column_names = (
            [column_name.strip() for column_name in column_names_string.split(",")] if column_names_string else []
        )
        template_schema = self.parameter_values.get("template_schema", "")

        # Set default prompt driver if none provided
        if not prompt_driver:
            # Without an explicit credential attrs falls back to os.environ["GT_CLOUD_API_KEY"],
            # which the engine plants as "" -- so a license-only user passed validation below
            # and then got a 401 from an empty bearer.
            prompt_driver = GriptapeCloudPromptDriver(model="gpt-4o", **cloud_driver_auth())

        # Create the appropriate extraction engine based on type
        engine = None
        if extraction_type == "csv":
            engine = CsvExtractionEngine(prompt_driver=prompt_driver, column_names=column_names)
        elif extraction_type == "json":
            engine = JsonExtractionEngine(prompt_driver=prompt_driver, template_schema=template_schema)

        # Create the tool with parameters
        params: dict = {"extraction_engine": engine}
        tool = GtExtractionTool(**params, rules=[Rule("Raw output please")])

        # Set the output
        self.parameter_output_values["tool"] = tool

    def validate_before_workflow_run(self) -> list[Exception] | None:
        exceptions = []
        if self.parameter_values.get("prompt_driver", None):
            return exceptions
        api_key = resolve_cloud_api_key()
        if not api_key:
            msg = missing_credential_message("create the Extraction tool")
            exceptions.append(KeyError(msg))
            return exceptions
        return exceptions if exceptions else None
