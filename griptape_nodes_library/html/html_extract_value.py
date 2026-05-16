from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_html import ParameterHtml
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from lxml import etree, html


class HtmlExtractValue(DataNode):
    """Extract values from HTML using XPath expressions."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterHtml(
                name="html",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Input HTML data to extract from",
            )
        )

        path_param = ParameterHtml(
            name="path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="",
            tooltip="XPath expression to extract data (e.g., '//h1', '//a/@href', '//div[@class=\"main\"]')",
            placeholder_text='ex: //h1, //a/@href, //div[@class="main"]',
        )
        path_param.set_badge(
            variant="help",
            title="XPath Syntax",
            message='`//h1` — all h1 elements\n`//a/@href` — link hrefs\n`//div[@class="main"]` — by attribute\n\n[XPath Docs](https://www.w3schools.com/xml/xpath_syntax.asp)',
        )
        self.add_parameter(path_param)

        self.add_parameter(
            ParameterHtml(
                name="output",
                tooltip="The extracted value(s)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _serialize_result(self, result: Any) -> str:
        if isinstance(result, etree._Element):
            return html.tostring(result, encoding="unicode").strip()
        return str(result) if result is not None else ""

    def _perform_extraction(self) -> None:
        html_str = self.get_parameter_value("html")
        path = self.get_parameter_value("path")

        if not html_str:
            result_str = ""
        else:
            root = html.fromstring(html_str)

            if not path:
                result_str = html.tostring(root, encoding="unicode").strip()
            else:
                try:
                    results = root.xpath(path)
                except etree.XPathEvalError as e:
                    msg = f"{self.name}: Invalid XPath expression '{path}': {e}"
                    raise ValueError(msg) from e

                if not results:
                    result_str = ""
                elif len(results) == 1:
                    result_str = self._serialize_result(results[0])
                else:
                    result_str = "\n".join(self._serialize_result(r) for r in results)

        GriptapeNodes.handle_request(
            SetParameterValueRequest(parameter_name="output", value=result_str, node_name=self.name)
        )
        self.publish_update_to_parameter("output", result_str)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["html", "path"]:
            self._perform_extraction()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_extraction()
