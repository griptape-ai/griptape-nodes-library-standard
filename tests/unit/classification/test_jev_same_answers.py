"""A single-question node and the Questions node read the same JEV response the same way."""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import ParameterList

from griptape_nodes_library.classification.jev_ask_yes_no import JevAskYesNo
from griptape_nodes_library.classification.jev_common import QUESTION_KEY
from griptape_nodes_library.classification.jev_pick_one import JevPickOne
from griptape_nodes_library.classification.jev_pick_one_question import JevPickOneQuestion
from griptape_nodes_library.classification.jev_questions import read_answer
from griptape_nodes_library.classification.jev_rate import JevRate
from griptape_nodes_library.classification.jev_rate_question import JevRateQuestion
from griptape_nodes_library.classification.jev_yes_no_question import JevYesNoQuestion


def _fill(node: Any, rows_param: str, rows: list[str]) -> None:
    rows_list = node.get_parameter_by_name(rows_param)
    assert isinstance(rows_list, ParameterList)
    for text in rows:
        node.set_parameter_value(rows_list.add_child_parameter().name, text)


def _spec(question_node: Any) -> dict[str, Any]:
    question_node.process()
    return question_node.parameter_output_values["question"]


def _details(spec: dict[str, Any], answer_data: dict[str, Any]) -> dict[str, Any]:
    read = read_answer(spec, answer_data)
    assert read is not None
    return read[1]


async def _single_outputs(node: Any, answer_data: dict[str, Any], names: list[str]) -> dict[str, Any]:
    await node._parse_result({"answers": {QUESTION_KEY: answer_data}}, "gen-1")
    return {name: node.parameter_output_values[name] for name in names}


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [0.2, 0.5, 0.51, 0.99])
@pytest.mark.parametrize("threshold", [None, 0.7])
async def test_yes_no(probability: float, threshold: float | None) -> None:
    single = JevAskYesNo(name="single")
    single.set_parameter_value("threshold", threshold)
    builder = JevYesNoQuestion(name="builder")
    builder.set_parameter_value("question_text", "Is it a truck?")
    builder.set_parameter_value("threshold", threshold)

    answer = {"noul": probability}
    outputs = await _single_outputs(single, answer, ["answer", "probability"])
    details = _details(_spec(builder), answer)

    assert outputs == {"answer": details["answer"], "probability": details["probability"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("picked", ["Lighting", "Layout", "Animation"])
async def test_pick_one(picked: str) -> None:
    options = ["Lighting: rim and shadow", "Layout: camera and staging", "Animation"]
    single = JevPickOne(name="single")
    _fill(single, "options", options)
    builder = JevPickOneQuestion(name="builder")
    _fill(builder, "options", options)

    answer = {"choice": picked, "confidence": 0.8, "probabilities": {"Lighting": 0.1, "Layout": 0.1, "Animation": 0.8}}
    outputs = await _single_outputs(single, answer, ["label", "description", "confidence", "probabilities"])
    details = _details(_spec(builder), answer)

    assert outputs == {key: details[key] for key in outputs}


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [0.0, 0.49, 0.5, 1.2, 2.0, 7.0])
async def test_rate(score: float) -> None:
    levels = ["Soft: blurry", "crisp enough", "Sharp: very crisp"]
    single = JevRate(name="single")
    _fill(single, "levels", levels)
    builder = JevRateQuestion(name="builder")
    _fill(builder, "levels", levels)

    answer = {"score": score, "confidence": 0.6, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}
    outputs = await _single_outputs(
        single, answer, ["score", "level", "label", "description", "confidence", "probabilities"]
    )
    details = _details(_spec(builder), answer)

    assert outputs == {key: details[key] for key in outputs}
    exit_ = single.get_next_control_output()
    assert exit_ is not None
    assert exit_.display_name == details["label"]
