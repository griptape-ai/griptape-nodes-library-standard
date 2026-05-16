from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_xml import ParameterXml
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from lxml import etree


class XmlExtractValue(DataNode):                                                                                                                                                                                                                    
    """Extract values from XML using XPath expressions."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterXml(
                name="xml",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Input XML data to extract from",
            )
        )

        path_param = ParameterString(
            name="path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="",
            tooltip="XPath expression to extract data (e.g., '//book/title', '//item/@id', '//book[1]')",
            placeholder_text="ex: //book/title, //item/@id, //book[1]",
        )
        path_param.set_badge(
            variant="help",
            title="XPath Syntax",
            message="`//book/title` — nested element\n`//item/@id` — attribute value\n`//book[1]` — first match\n\n[XPath Docs](https://www.w3schools.com/xml/xpath_syntax.asp)",
        )
        self.add_parameter(path_param)

        self.add_parameter(
            ParameterXml(
                name="output",
                tooltip="The extracted value(s)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _serialize_result(self, result: Any) -> str:
        if isinstance(result, etree._Element):
            etree.indent(result, space="    ")
            return etree.tostring(result, encoding="unicode").strip()
        return str(result) if result is not None else ""

    def _perform_extraction(self) -> None:
        xml_str = self.get_parameter_value("xml")
        path = self.get_parameter_value("path")

        if not xml_str:
            result_str = ""
        else:
            try:
                root = etree.fromstring(xml_str.encode() if isinstance(xml_str, str) else xml_str)
            except etree.XMLSyntaxError as e:
                msg = f"{self.name}: Invalid XML provided. Failed to parse: {e}. Input was: {str(xml_str)[:200]!r}"
                raise ValueError(msg) from e

            if not path:
                result_str = etree.tostring(root, encoding="unicode", pretty_print=True).strip()
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
        if parameter.name in ["xml", "path"]:
            self._perform_extraction()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_extraction()
