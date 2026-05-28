from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_xml import ParameterXml
from lxml import etree


class XmlReplace(ControlNode):
    """Replace values in XML using XPath expressions."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterXml(
                name="xml",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Input XML data to modify",
            )
        )

        self.add_parameter(
            ParameterString(
                name="path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="XPath expression pointing to the element or attribute to replace (e.g., '//book/title', '//item/@id')",
                placeholder_text="ex: //book/title, //item/@id",
            )
        )

        self.add_parameter(
            ParameterString(
                name="replacement_value",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="The new value to set at the specified path",
            )
        )

        self.add_parameter(
            ParameterXml(
                name="output",
                tooltip="The modified XML with the replacement applied",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _perform_replacement(self) -> None:
        xml_str = self.get_parameter_value("xml")
        path = self.get_parameter_value("path")
        replacement = self.get_parameter_value("replacement_value") or ""

        if not xml_str:
            result_str = ""
        else:
            try:
                root = etree.fromstring(xml_str.encode() if isinstance(xml_str, str) else xml_str)
            except etree.XMLSyntaxError as e:
                msg = f"{self.name}: Invalid XML provided. Failed to parse: {e}."
                raise ValueError(msg) from e

            if path:
                try:
                    targets = root.xpath(path)
                except etree.XPathEvalError as e:
                    msg = f"{self.name}: Invalid XPath expression '{path}': {e}"
                    raise ValueError(msg) from e

                if not isinstance(targets, list):
                    targets = [targets]
                for target in targets:
                    if isinstance(target, etree._Element):
                        for child in list(target):
                            target.remove(child)
                        target.text = replacement
                    elif isinstance(target, str) and hasattr(target, "attrname"):
                        parent = target.getparent()  # type: ignore[union-attr]
                        attrname = target.attrname  # type: ignore[union-attr]
                        if parent is not None and attrname is not None:
                            parent.set(attrname, replacement)

            result_str = etree.tostring(root, encoding="unicode", pretty_print=True).strip()

        self.set_parameter_value("output", result_str)
        self.publish_update_to_parameter("output", result_str)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["xml", "path", "replacement_value"]:
            self._perform_replacement()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_replacement()
