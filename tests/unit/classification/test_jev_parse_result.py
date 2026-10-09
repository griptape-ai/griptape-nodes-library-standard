"""A JEV response without the answer the node asked for fails the node with the response attached."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.classification.jev_ask_yes_no import JevAskYesNo
from griptape_nodes_library.classification.jev_pick_one import JevPickOne
from griptape_nodes_library.classification.jev_rate import JevRate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_class", "missing"),
    [(JevAskYesNo, "an answer"), (JevPickOne, "a choice"), (JevRate, "a score")],
)
async def test_a_response_without_the_answer_raises_node_error(node_class: type, missing: str) -> None:
    node = node_class(name="jev")
    response = {"answers": {}}

    with pytest.raises(NodeError, match=f"JEV's response didn't include {missing}") as raised:
        await node._parse_result(response, "gen-1")

    assert raised.value.fields == {"generation_id": "gen-1"}
    assert raised.value.response == response
