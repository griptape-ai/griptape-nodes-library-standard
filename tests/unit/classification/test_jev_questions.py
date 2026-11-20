import pytest

from griptape_nodes_library.classification.jev_common import QUESTION_KEY, build_batch_payload, build_payload
from griptape_nodes_library.classification.jev_questions import (
    MAX_LEVELS,
    choice_question,
    score_question,
    yes_no_question,
)


def test_yes_no_question_with_criteria() -> None:
    assert yes_no_question(" Is it a truck? ", "has wheels", "") == {
        "type": "noul",
        "instructions": "Is it a truck?",
        "criteria": {"true": "has wheels"},
    }


def test_yes_no_question_without_criteria() -> None:
    assert yes_no_question("Is it a truck?") == {"type": "noul", "instructions": "Is it a truck?"}


def test_yes_no_question_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="'Question' is empty"):
        yes_no_question("  ")


def test_choice_question_uses_label_for_missing_description() -> None:
    assert choice_question("Which?", {"A": "first", "B": None}) == {
        "type": "choice",
        "criteria": {"A": "first", "B": "B"},
        "instructions": "Which?",
    }


def test_choice_question_omits_blank_instructions() -> None:
    assert "instructions" not in choice_question("", {"A": None, "B": None})


def test_score_question() -> None:
    assert score_question("", ["bad", "good"]) == {"type": "score", "criteria": ["bad", "good"]}


@pytest.mark.parametrize("levels", [["only one"], ["x"] * (MAX_LEVELS + 1)])
def test_score_question_rejects_bad_level_counts(levels: list[str]) -> None:
    with pytest.raises(ValueError, match="Levels"):
        score_question("", levels)


def test_build_payload_wraps_the_question_under_the_shared_key() -> None:
    question = yes_no_question("Is it a truck?")
    assert build_payload("ctx", question, {}) == {"state": "ctx", "questions": {QUESTION_KEY: question}}


def test_build_batch_payload_sends_every_named_question() -> None:
    questions = {"truck": yes_no_question("Truck?"), "dept": choice_question("", {"A": None, "B": None})}
    assert build_batch_payload("ctx", questions, {}) == {"state": "ctx", "questions": questions}


def test_build_batch_payload_lists_unreferenced_images() -> None:
    assets = {"[image 1]": {"type": "image", "format": "jpg", "data_base64": "AAAA"}}
    questions = {"truck": yes_no_question("Is [image 1] a truck?"), "crayon": yes_no_question("A drawing?")}
    payload = build_batch_payload(None, questions, assets)
    assert payload["questions"] == questions
    assert payload["state"] == {"images": []}
    assert payload["assets"] == assets
