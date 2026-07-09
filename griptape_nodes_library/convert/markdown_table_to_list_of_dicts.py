import re
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class MarkdownTableToListOfDicts(SuccessFailureNode):
    """Parse a Markdown table into a list of dicts using the header row as keys."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: str = "",
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterString(
                name="markdown",
                default_value=value,
                input_types=["str"],
                tooltip="Markdown table text to parse. The first non-empty row is the header row.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="| name | age | city |\n|------|-----|------|\n| Alice | 30 | NYC |\n| Bob | 25 | LA |",
            )
        )
        self.add_parameter(
            ParameterString(
                name="results",
                tooltip="Summary of the parsed table",
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
                tooltip="List of dicts, one per table row, keyed by the header row",
                hide_property=True,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            )
        )
        self._create_status_parameters(
            result_details_tooltip="Details about the Markdown table parse operation",
            result_details_placeholder="Parse details will appear here after execution.",
        )

    @staticmethod
    def _split_row(line: str) -> list[str]:
        stripped = line.strip()
        # Drop the leading/trailing pipe so we don't produce empty edge cells.
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        # Split on unescaped pipes only, then unescape any "\|" inside each cell.
        cells = re.split(r"(?<!\\)\|", stripped)
        return [cell.strip().replace("\\|", "|") for cell in cells]

    @staticmethod
    def _is_separator(cells: list[str]) -> bool:
        # A separator row looks like |---|:--:|---:| — only dashes, colons, spaces.
        return bool(cells) and all(set(cell) <= {"-", ":", " "} and "-" in cell for cell in cells)

    def _parse(self) -> list[dict[str, str]]:
        markdown_text = self.get_parameter_value("markdown") or ""
        lines = [line for line in markdown_text.splitlines() if line.strip()]
        if not lines:
            return []

        header = self._split_row(lines[0])
        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            cells = self._split_row(line)
            if self._is_separator(cells):
                continue
            # Pad or truncate to the header width so zip stays aligned.
            cells = (cells + [""] * len(header))[: len(header)]
            rows.append(dict(zip(header, cells, strict=False)))
        return rows

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
        if parameter.name == "markdown":
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
