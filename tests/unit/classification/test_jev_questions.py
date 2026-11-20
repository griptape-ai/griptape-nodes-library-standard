import pytest

from griptape_nodes_library.classification.jev_common import QUESTION_KEY, build_batch_payload, build_payload
from griptape_nodes_library.classification.jev_questions import (
    MAX_LEVELS,
    QuestionKind,
    choice_question,
    make_spec,
    parse_levels,
    parse_options,
    read_answer,
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


def test_parse_options_keeps_descriptions_and_skips_empty_rows() -> None:
    assert parse_options(["Lighting: rim and shadow", "", "Layout"]) == {"Lighting": "rim and shadow", "Layout": None}


def test_parse_options_rejects_duplicates_and_too_few() -> None:
    with pytest.raises(ValueError, match="more than once"):
        parse_options(["A", "A"])
    with pytest.raises(ValueError, match="at least two"):
        parse_options(["A", " "])


def test_parse_levels_names_unlabeled_rows_by_position() -> None:
    assert parse_levels(["Minor: workaround exists", "", "it is broken"]) == (
        ["workaround exists", "it is broken"],
        ["Minor", "Level 2"],
    )


def test_read_yes_no_applies_the_threshold() -> None:
    spec = make_spec(QuestionKind.YES_NO, "truck", {}, threshold=0.7)
    assert read_answer(spec, {"noul": 0.6}) == (False, {"answer": False, "probability": 0.6})
    assert read_answer(spec, {"noul": 0.7}) == (True, {"answer": True, "probability": 0.7})


def test_read_choice_returns_the_label() -> None:
    spec = make_spec(QuestionKind.CHOICE, "dept", {})
    answer = {"choice": "Lighting", "confidence": 0.9, "probabilities": {"Lighting": 0.9, "Layout": 0.1}}
    assert read_answer(spec, answer) == (
        "Lighting",
        {"label": "Lighting", "confidence": 0.9, "probabilities": {"Lighting": 0.9, "Layout": 0.1}},
    )


def test_read_score_rounds_half_up_and_numbers_levels_from_one() -> None:
    spec = make_spec(QuestionKind.SCORE, "sharp", {}, labels=["Soft", "OK", "Sharp"])
    value, details = read_answer(spec, {"score": 1.5, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}})
    assert value == "Sharp"
    assert details["level"] == 3
    assert details["label"] == "Sharp"
    assert details["score"] == 2.5
    assert details["probabilities"] == {"1": 0.1, "2": 0.4, "3": 0.5}
    assert read_answer(spec, {"score": 9})[0] == "Sharp"


@pytest.mark.parametrize("kind", list(QuestionKind))
def test_read_answer_is_none_when_jev_left_the_answer_out(kind: QuestionKind) -> None:
    assert read_answer(make_spec(kind, "q", {}, threshold=0.5, labels=["a", "b"]), {}) is None


def test_read_answer_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown question kind"):
        read_answer({"kind": "other"}, {})
