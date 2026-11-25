from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from griptape_nodes_library.classification.row_outputs import parse_row

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_THRESHOLD",
    "MAX_LEVELS",
    "QUESTION_TYPE",
    "QuestionKind",
    "choice_question",
    "level_description",
    "level_label",
    "make_spec",
    "parse_levels",
    "parse_options",
    "read_answer",
    "read_choice",
    "read_score",
    "read_yes_no",
    "score_question",
    "structured",
    "yes_no_question",
]

MAX_LEVELS = 10
# The probability at or above which a yes/no question answers Yes, when no threshold is set.
DEFAULT_THRESHOLD = 0.5
# The wire type the question builder nodes output and the Questions node accepts.
QUESTION_TYPE = "JevQuestion"


class QuestionKind(StrEnum):
    YES_NO = "yes_no"
    CHOICE = "choice"
    SCORE = "score"


def structured(text: str) -> str | dict | list:
    """Parse text that holds a JSON object or array, so JEV reads it as structured. Anything else stays text."""
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return text


def yes_no_question(question: str, yes_means: str = "", no_means: str = "") -> dict[str, Any]:
    """Build a JEV yes/no ("noul") question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("'Question' is empty. Write a yes/no question about the context.")

    noul: dict[str, Any] = {"type": "noul", "instructions": structured(question)}
    criteria: dict[str, Any] = {}
    if yes_means := (yes_means or "").strip():
        criteria["true"] = structured(yes_means)
    if no_means := (no_means or "").strip():
        criteria["false"] = structured(no_means)
    if criteria:
        noul["criteria"] = criteria
    return noul


def choice_question(question: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    """Build a JEV pick-one ("choice") question. A label with no description stands in for its own."""
    choice: dict[str, Any] = {
        "type": "choice",
        "criteria": {k: structured(v) if v else k for k, v in criteria.items()},
    }
    if question := (question or "").strip():
        choice["instructions"] = structured(question)
    return choice


def score_question(question: str, descriptions: list[str]) -> dict[str, Any]:
    """Build a JEV rating ("score") question. Descriptions run lowest level first."""
    if len(descriptions) < 2:  # noqa: PLR2004
        raise ValueError("'Levels' needs at least two rows for JEV to rate against. Add another level.")
    if len(descriptions) > MAX_LEVELS:
        raise ValueError(f"'Levels' has {len(descriptions)} rows. JEV accepts up to {MAX_LEVELS}; remove some.")

    score: dict[str, Any] = {"type": "score", "criteria": [structured(d) for d in descriptions]}
    if question := (question or "").strip():
        score["instructions"] = structured(question)
    return score


def parse_options(rows: list[str]) -> dict[str, str | None]:
    """Turn 'Label: description' rows into {label: description}, skipping empty rows."""
    criteria: dict[str, str | None] = {}
    for row in rows:
        if not isinstance(structured(row), str):
            raise ValueError(f"An option in 'Options' is all JSON. Give it a label: 'Label: {row.strip()}'.")
        label, description = parse_row(row)
        if not label:
            continue
        if label in criteria:
            raise ValueError(f"'{label}' appears more than once in 'Options'. Give each option a unique label.")
        criteria[label] = description
    if len(criteria) < 2:  # noqa: PLR2004
        raise ValueError("'Options' needs at least two rows for JEV to pick from. Add another option.")
    return criteria


def level_label(index: int, text: str) -> str:
    """The label of a rating level: the text before the colon, or 'Level N' (from 1) for a level with none."""
    if not isinstance(structured(text), str):
        return f"Level {index + 1}"
    label, description = parse_row(text)
    return label if description else f"Level {index + 1}"


def level_description(text: str) -> str:
    """The description of a rating level: the text after the colon, or the whole row if there is no label."""
    if not isinstance(structured(text), str):
        return text
    return parse_row(text)[1] or text


def parse_levels(rows: list[str]) -> tuple[list[str], list[str]]:
    """Turn level rows into (descriptions, labels), skipping empty rows."""
    rows = [row for row in rows if (row or "").strip()]
    return [level_description(row) for row in rows], [level_label(index, row) for index, row in enumerate(rows)]


def make_spec(kind: QuestionKind, key: str, question: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Bundle a JEV question with what the Questions node needs to name and read its answer."""
    return {"key": key, "kind": str(kind), "question": question, **extra}


def read_yes_no(answer_data: dict[str, Any], threshold: float | None) -> dict[str, Any] | None:
    """Read JEV's answer to a yes/no question, or None if JEV left it out. No threshold means the default."""
    raw = answer_data.get("noul")
    if raw is None:
        return None
    probability = float(raw)
    limit = DEFAULT_THRESHOLD if threshold is None else threshold
    return {"answer": probability >= limit, "probability": probability}


def read_choice(
    answer_data: dict[str, Any], descriptions: Callable[[], dict[str, str | None]]
) -> dict[str, Any] | None:
    """Read JEV's answer to a pick-one question, or None if JEV left it out.

    `descriptions` maps each label to its description. It is a function so it is only called once JEV has
    picked something, as the single-question node reads its options only then.
    """
    picked = answer_data.get("choice")
    if picked is None:
        return None
    return {
        "label": picked,
        "description": descriptions().get(picked) or "",
        "confidence": float(answer_data.get("confidence", 0.0)),
        "probabilities": dict(answer_data.get("probabilities") or {}),
    }


def read_score(answer_data: dict[str, Any], labels: list[str], descriptions: list[str]) -> dict[str, Any] | None:
    """Read JEV's answer to a rating question, or None if JEV left it out.

    JEV numbers levels from 0 and the score is a weighted average, so it is rounded half-up to the nearest
    level. Levels are shown from 1, so the score, the level and the probability keys all shift by one.
    """
    raw = answer_data.get("score")
    if raw is None:
        return None
    score = float(raw)
    index = min(int(score + 0.5), len(labels) - 1)
    probabilities = answer_data.get("probabilities") or {}
    return {
        "level": index + 1,
        "score": score + 1,
        "label": labels[index],
        "description": descriptions[index],
        "confidence": float(answer_data.get("confidence", 0.0)),
        "probabilities": {str(int(key) + 1): value for key, value in probabilities.items()},
    }


def read_answer(spec: dict[str, Any], answer_data: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    """Read JEV's answer to one question as (simple value, details), or None if JEV left it out.

    The simple value is a bool for yes/no and the label for pick one and rating. The details are what the
    single-question nodes put on their outputs, read by the same functions.
    """
    match spec["kind"]:
        case QuestionKind.YES_NO:
            details = read_yes_no(answer_data, spec["threshold"])
            return None if details is None else (details["answer"], details)
        case QuestionKind.CHOICE:
            details = read_choice(answer_data, lambda: spec["descriptions"])
            return None if details is None else (details["label"], details)
        case QuestionKind.SCORE:
            details = read_score(answer_data, spec["labels"], spec["descriptions"])
            return None if details is None else (details["label"], details)
        case _:
            msg = f"Unknown question kind: {spec['kind']!r}"
            raise ValueError(msg)
