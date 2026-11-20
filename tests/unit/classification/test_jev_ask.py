"""The question builder nodes hand specs to JEV Ask, which asks them in one call and reads each answer."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.classification.jev_ask import JevAsk
from griptape_nodes_library.classification.jev_pick_one_question import JevPickOneQuestion
from griptape_nodes_library.classification.jev_rate_question import JevRateQuestion
from griptape_nodes_library.classification.jev_yes_no_question import JevYesNoQuestion


def _spec(node: object) -> dict:
    node.process()
    return node.parameter_output_values["question"]


def _yes_no(key: str = "truck") -> dict:
    node = JevYesNoQuestion(name="yes_no")
    node.set_parameter_value("key", key)
    node.set_parameter_value("question_text", "Is this a truck?")
    node.set_parameter_value("threshold", 0.7)
    return _spec(node)


def _pick_one() -> dict:
    node = JevPickOneQuestion(name="pick")
    node.set_parameter_value("options", ["Realistic", "Cartoon: drawn, not photographed"])
    return _spec(node)


def _rate() -> dict:
    node = JevRateQuestion(name="rate")
    node.set_parameter_value("key", "sharp")
    node.set_parameter_value("levels", ["Soft: blurry", "Sharp: crisp"])
    return _spec(node)


def test_builders_name_the_answer_after_the_key_or_the_node() -> None:
    assert _yes_no()["key"] == "truck"
    assert _pick_one()["key"] == "pick"


def test_builders_carry_what_the_ask_node_needs_to_read_the_answer() -> None:
    assert _yes_no()["threshold"] == 0.7
    assert _rate()["labels"] == ["Soft", "Sharp"]
    assert _pick_one()["question"]["criteria"] == {
        "Realistic": "Realistic",
        "Cartoon": "drawn, not photographed",
    }


def test_builders_reject_incomplete_questions() -> None:
    with pytest.raises(ValueError, match="'Question' is empty"):
        JevYesNoQuestion(name="empty").process()
    with pytest.raises(ValueError, match="at least two"):
        JevPickOneQuestion(name="empty").process()
    with pytest.raises(ValueError, match="at least two"):
        JevRateQuestion(name="empty").process()


def _ask(*specs: dict) -> JevAsk:
    node = JevAsk(name="ask")
    node.set_parameter_value("questions", list(specs))
    return node


@pytest.mark.asyncio
async def test_ask_sends_every_question_in_one_request() -> None:
    node = _ask(_yes_no(), _pick_one())
    node.set_parameter_value("context", "A red pickup on a road.")

    payload = await node._build_payload()

    assert payload["state"] == "A red pickup on a road."
    assert list(payload["questions"]) == ["truck", "pick"]
    assert payload["questions"]["truck"]["type"] == "noul"
    assert payload["questions"]["pick"]["type"] == "choice"


@pytest.mark.asyncio
async def test_ask_rejects_duplicate_names_and_no_questions() -> None:
    node = _ask(_yes_no("same"), _yes_no("same"))
    node.set_parameter_value("context", "x")
    with pytest.raises(ValueError, match="unique Name"):
        await node._build_payload()

    empty = _ask()
    empty.set_parameter_value("context", "x")
    with pytest.raises(ValueError, match="'Questions' is empty"):
        await empty._build_payload()


@pytest.mark.asyncio
async def test_ask_reads_each_answer() -> None:
    node = _ask(_yes_no(), _pick_one(), _rate())
    response = {
        "answers": {
            "truck": {"noul": 0.9},
            "pick": {"choice": "Cartoon", "confidence": 0.8, "probabilities": {"Cartoon": 0.8, "Realistic": 0.2}},
            "sharp": {"score": 0.2, "confidence": 0.7, "probabilities": {"0": 0.8, "1": 0.2}},
        }
    }

    await node._parse_result(response, "gen-1")

    assert node.parameter_output_values["answers"] == {"truck": True, "pick": "Cartoon", "sharp": 1}
    assert node.parameter_output_values["details"]["sharp"]["label"] == "Soft"


@pytest.mark.asyncio
async def test_ask_fails_with_the_response_when_an_answer_is_missing() -> None:
    node = _ask(_yes_no(), _pick_one())
    response = {"answers": {"truck": {"noul": 0.9}}}

    with pytest.raises(NodeError, match="didn't include an answer for 'pick'") as raised:
        await node._parse_result(response, "gen-1")

    assert raised.value.fields == {"generation_id": "gen-1"}
    assert raised.value.response == response
    assert "answers" not in node.parameter_output_values
