import csv
import io
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class CsvToListOfDicts(SuccessFailureNode):
    """Parse CSV text into a list of dicts using the header row as keys."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: str = "",
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterString(
                name="csv",
                default_value=value,
                input_types=["str"],
                tooltip="CSV text to parse. The first row must be the header row.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="name,age,city\nAlice,30,NYC\nBob,25,LA",
            )
        )
        self.add_parameter(
            ParameterString(
                name="delimiter",
                default_value=",",
                input_types=["str"],
                tooltip="Column delimiter character (default: comma)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterString(
                name="results",
                tooltip="Summary of the parsed CSV",
                allow_input=False,
                allow_output=False,
                allow_property=True,
                multiline=True,
                markdown=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output",
                default_value=[],
                output_type="list",
                type="list",
                tooltip="List of dicts, one per CSV row, keyed by the header row",
                hide_property=True,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            )
        )
        self._create_status_parameters(
            result_details_tooltip="Details about the CSV parse operation",
            result_details_placeholder="Parse details will appear here after execution.",
        )

    def _parse(self) -> list[dict[str, str]]:
        csv_text = self.get_parameter_value("csv") or ""
        delimiter = self.get_parameter_value("delimiter") or ","
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
        return [dict(row) for row in reader]

    def _build_results(self, rows: list[dict[str, str]]) -> str:
        if not rows:
            return "_No data parsed yet._"
        keys = list(rows[0].keys())
        headers_md = " | ".join(f"`{k}`" for k in keys)
        example = ", ".join(f"`{k}`: {v!r}" for k, v in list(rows[0].items())[:3])
        if len(rows[0]) > 3:
            example += ", ..."
        return f"**{len(rows)} row{'s' if len(rows) != 1 else ''}** · {len(keys)} column{'s' if len(keys) != 1 else ''}\n\n**Columns:** {headers_md}\n\n**First row:** {example}"

    def _apply(self, rows: list[dict[str, str]]) -> None:
        results = self._build_results(rows)
        self.set_parameter_value("output", rows)
        self.publish_update_to_parameter("output", rows)
        self.set_parameter_value("results", results)
        self.publish_update_to_parameter("results", results)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ("csv", "delimiter"):
            self._apply(self._parse())
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._clear_execution_status()
        try:
            rows = self._parse()
            self._apply(rows)
            self._set_status_results(
                was_successful=True,
                result_details=f"Parsed {len(rows)} row{'s' if len(rows) != 1 else ''} successfully.",
            )
        except Exception as e:
            self._set_status_results(was_successful=False, result_details=f"Parse failed: {e}")
            self._handle_failure_exception(e)
