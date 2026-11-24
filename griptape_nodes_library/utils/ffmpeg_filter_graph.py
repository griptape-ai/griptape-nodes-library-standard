"""Build an FFmpeg ``-filter_complex`` graph one chain at a time, by stream instead of by label string."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Stream:
    """A stream in a filter graph: an input like ``[1]``, or the labelled output of a chain."""

    label: str

    @property
    def pad(self) -> str:
        """The label in brackets, as FFmpeg writes it in a graph and in ``-map``."""
        return f"[{self.label}]"


class FilterGraph:
    """Collects filter chains and joins them into one ``-filter_complex`` value.

    Each chain takes streams in, applies filters in order, and returns its labelled output streams.
    FFmpeg lets an input be read more than once but each chain output only once, so reusing an output
    label or consuming the same output twice raises here instead of failing inside FFmpeg.
    """

    def __init__(self) -> None:
        self._chains: list[str] = []
        self._labels: set[str] = set()
        self._consumed: set[str] = set()

    @staticmethod
    def input(index: int) -> Stream:
        """The video stream of the FFmpeg input at this ``-i`` position."""
        return Stream(str(index))

    def chain(self, inputs: Stream | Sequence[Stream], filters: str | Sequence[str], output: str) -> Stream:
        """Apply filters to the inputs and return the result as the stream labelled ``output``."""
        return self._add(inputs, filters, [output])[0]

    def split(self, source: Stream, *outputs: str, filters: Sequence[str] = ()) -> tuple[Stream, ...]:
        """Apply filters to source, then split it into one stream per output label."""
        min_outputs = 2
        if len(outputs) < min_outputs:
            msg = f"A split needs at least {min_outputs} outputs, got {len(outputs)}."
            raise ValueError(msg)
        split_filter = "split" if len(outputs) == min_outputs else f"split={len(outputs)}"
        return self._add(source, [*filters, split_filter], outputs)

    def render(self) -> str:
        """The graph as one ``-filter_complex`` value."""
        return ";".join(self._chains)

    def _add(
        self, inputs: Stream | Sequence[Stream], filters: str | Sequence[str], outputs: Sequence[str]
    ) -> tuple[Stream, ...]:
        input_streams = [inputs] if isinstance(inputs, Stream) else list(inputs)
        filter_list = [filters] if isinstance(filters, str) else list(filters)
        if not filter_list:
            msg = "A filter chain needs at least one filter."
            raise ValueError(msg)

        consumed = [s.label for s in input_streams if s.label in self._labels]
        for label in consumed:
            if label in self._consumed or consumed.count(label) > 1:
                msg = f"Stream [{label}] is already consumed; split it to use it twice."
                raise ValueError(msg)
        if len(set(outputs)) != len(outputs):
            msg = f"Stream labels {list(outputs)} repeat within one chain."
            raise ValueError(msg)
        for label in outputs:
            if label in self._labels:
                msg = f"Stream label [{label}] is already in use."
                raise ValueError(msg)
            if label.isdigit():
                msg = f"Stream label [{label}] would read as an input index."
                raise ValueError(msg)

        self._consumed.update(consumed)
        self._labels.update(outputs)
        output_streams = tuple(Stream(label) for label in outputs)
        self._chains.append(
            "".join(s.pad for s in input_streams) + ",".join(filter_list) + "".join(s.pad for s in output_streams)
        )
        return output_streams
