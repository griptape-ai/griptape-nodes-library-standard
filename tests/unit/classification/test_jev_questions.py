import pytest

from griptape_nodes_library.classification.jev_common import QUESTION_KEY, build_batch_payload, build_payload
from griptape_nodes_library.classification.jev_questions import (
    DEFAULT_THRESHOLD,
    MAX_LEVELS,
    QuestionKind,
    choice_question,
    level_description,
    level_label,
    make_spec,
    parse_levels,
    parse_options,
    read_answer,
    read_choice,
    read_yes_no,
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


IMAGE_ASSETS = {"[image 1]": {"type": "image", "format": "jpg", "data_base64": "AAAA"}}


def test_an_image_only_one_question_names_is_still_listed_for_the_others() -> None:
    questions = {"truck": yes_no_question("Is [image 1] a truck?"), "crayon": yes_no_question("A drawing?")}
    payload = build_batch_payload(None, questions, IMAGE_ASSETS)
    assert payload["questions"] == questions
    assert payload["state"] == {"images": ["[image 1]"]}
    assert payload["assets"] == IMAGE_ASSETS


def test_an_image_every_question_names_is_not_listed() -> None:
    questions = {
        "truck": yes_no_question("Is [image 1] a truck?"),
        "crayon": yes_no_question("Is [image 1] a drawing?"),
    }
    assert build_batch_payload(None, questions, IMAGE_ASSETS)["state"] == {"images": []}


def test_an_image_the_context_names_is_not_listed() -> None:
    questions = {"truck": yes_no_question("A truck?"), "crayon": yes_no_question("A drawing?")}
    payload = build_batch_payload("Look at [image 1].", questions, IMAGE_ASSETS)
    assert payload["state"] == {"context": "Look at [image 1].", "images": []}


def test_a_single_question_naming_its_image_does_not_list_it() -> None:
    payload = build_payload(None, yes_no_question("Is [image 1] a truck?"), IMAGE_ASSETS)
    assert payload["state"] == {"images": []}


def test_the_size_limit_counts_an_image_once_per_use_plus_once_for_the_list() -> None:
    big = {"[image 1]": {"type": "image", "format": "jpg", "data_base64": "A" * (16 * 1024 * 1024)}}
    one_question_each = {"a": yes_no_question("Is [image 1] a truck?"), "b": yes_no_question("A drawing?")}
    # Named once and listed once: two copies, 32 MiB of base64, over the 30 MiB limit.
    with pytest.raises(ValueError, match="more than JEV accepts"):
        build_batch_payload(None, one_question_each, big)
    # Named by every question and not listed: two copies as well.
    both_name_it = {"a": yes_no_question("Is [image 1] a truck?"), "b": yes_no_question("Is [image 1] a drawing?")}
    with pytest.raises(ValueError, match="more than JEV accepts"):
        build_batch_payload(None, both_name_it, big)
    single = build_payload(None, yes_no_question("Is [image 1] a truck?"), big)
    assert single["assets"] == big


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
    spec = make_spec(QuestionKind.CHOICE, "dept", {}, descriptions={"Lighting": "rim and shadow", "Layout": None})
    answer = {"choice": "Lighting", "confidence": 0.9, "probabilities": {"Lighting": 0.9, "Layout": 0.1}}
    assert read_answer(spec, answer) == (
        "Lighting",
        {
            "label": "Lighting",
            "description": "rim and shadow",
            "confidence": 0.9,
            "probabilities": {"Lighting": 0.9, "Layout": 0.1},
        },
    )
    layout = read_answer(spec, {"choice": "Layout"})
    assert layout is not None
    assert layout[1]["description"] == ""


def test_read_score_rounds_half_up_and_numbers_levels_from_one() -> None:
    spec = make_spec(
        QuestionKind.SCORE, "sharp", {}, labels=["Soft", "OK", "Sharp"], descriptions=["blurry", "fine", "crisp"]
    )
    answer = read_answer(spec, {"score": 1.5, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}})
    assert answer is not None
    value, details = answer
    assert value == "Sharp"
    assert details["level"] == 3
    assert details["label"] == "Sharp"
    assert details["description"] == "crisp"
    assert details["score"] == 2.5
    assert details["probabilities"] == {"1": 0.1, "2": 0.4, "3": 0.5}
    clamped = read_answer(spec, {"score": 9})
    assert clamped is not None
    assert clamped[0] == "Sharp"


@pytest.mark.parametrize("kind", list(QuestionKind))
def test_read_answer_is_none_when_jev_left_the_answer_out(kind: QuestionKind) -> None:
    spec = make_spec(
        kind, "q", {}, threshold=0.5, labels=["a", "b"], descriptions={} if kind == QuestionKind.CHOICE else []
    )
    assert read_answer(spec, {}) is None


def test_read_yes_no_without_a_threshold_uses_the_default() -> None:
    at_default = read_yes_no({"noul": DEFAULT_THRESHOLD}, None)
    below_default = read_yes_no({"noul": DEFAULT_THRESHOLD - 0.01}, None)
    assert at_default is not None
    assert below_default is not None
    assert at_default["answer"] is True
    assert below_default["answer"] is False


def test_read_choice_only_looks_up_descriptions_when_jev_picked_something() -> None:
    def descriptions() -> dict[str, str | None]:
        raise AssertionError("looked up descriptions with nothing picked")

    assert read_choice({}, descriptions) is None


def test_level_labels_and_descriptions() -> None:
    assert level_label(1, "Minor: a workaround exists") == "Minor"
    assert level_label(1, "a workaround exists") == "Level 2"
    assert level_description("Minor: a workaround exists") == "a workaround exists"
    assert level_description("a workaround exists") == "a workaround exists"


def test_read_answer_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown question kind"):
        read_answer({"kind": "other"}, {})
