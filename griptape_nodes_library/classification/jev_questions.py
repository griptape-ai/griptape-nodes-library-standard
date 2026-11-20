from __future__ import annotations

from enum import StrEnum
from typing import Any

from griptape_nodes_library.classification.row_outputs import parse_row

__all__ = [
    "MAX_LEVELS",
    "QUESTION_TYPE",
    "QuestionKind",
    "choice_question",
    "make_spec",
    "parse_levels",
    "parse_options",
    "read_answer",
    "score_question",
    "yes_no_question",
]

MAX_LEVELS = 10
# The wire type the question builder nodes output and the JEV Ask node accepts.
QUESTION_TYPE = "JevQuestion"


class QuestionKind(StrEnum):
    YES_NO = "yes_no"
    CHOICE = "choice"
    SCORE = "score"


def yes_no_question(question: str, yes_means: str = "", no_means: str = "") -> dict[str, Any]:
    """Build a JEV yes/no ("noul") question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("'Question' is empty. Write a yes/no question about the context.")

    noul: dict[str, Any] = {"type": "noul", "instructions": question}
    criteria: dict[str, str] = {}
    if yes_means := (yes_means or "").strip():
        criteria["true"] = yes_means
    if no_means := (no_means or "").strip():
        criteria["false"] = no_means
    if criteria:
        noul["criteria"] = criteria
    return noul


def choice_question(question: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    """Build a JEV pick-one ("choice") question. A label with no description stands in for its own."""
    choice: dict[str, Any] = {"type": "choice", "criteria": {k: v or k for k, v in criteria.items()}}
    if question := (question or "").strip():
        choice["instructions"] = question
    return choice


def score_question(question: str, descriptions: list[str]) -> dict[str, Any]:
    """Build a JEV rating ("score") question. Descriptions run lowest level first."""
    if len(descriptions) < 2:  # noqa: PLR2004
        raise ValueError("'Levels' needs at least two rows for JEV to rate against. Add another level.")
    if len(descriptions) > MAX_LEVELS:
        raise ValueError(f"'Levels' has {len(descriptions)} rows. JEV accepts up to {MAX_LEVELS}; remove some.")

    score: dict[str, Any] = {"type": "score", "criteria": descriptions}
    if question := (question or "").strip():
        score["instructions"] = question
    return score


def parse_options(rows: list[str]) -> dict[str, str | None]:
    """Turn 'Label: description' rows into {label: description}, skipping empty rows."""
    criteria: dict[str, str | None] = {}
    for row in rows:
        label, description = parse_row(row)
        if not label:
            continue
        if label in criteria:
            raise ValueError(f"'{label}' appears more than once in 'Options'. Give each option a unique label.")
        criteria[label] = description
    if len(criteria) < 2:  # noqa: PLR2004
        raise ValueError("'Options' needs at least two rows for JEV to pick from. Add another option.")
    return criteria


def parse_levels(rows: list[str]) -> tuple[list[str], list[str]]:
    """Turn level rows into (descriptions, labels). A row with no label is named 'Level N'."""
    rows = [row for row in rows if (row or "").strip()]
    descriptions = [parse_row(row)[1] or row for row in rows]
    labels = [parse_row(row)[0] if parse_row(row)[1] else f"Level {index + 1}" for index, row in enumerate(rows)]
    return descriptions, labels


def make_spec(kind: QuestionKind, key: str, question: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Bundle a JEV question with what the Ask node needs to name and read its answer."""
    return {"key": key, "kind": str(kind), "question": question, **extra}


def read_answer(spec: dict[str, Any], answer_data: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    """Read JEV's answer to one question as (simple value, details), or None if JEV left it out.

    The simple value is a bool for yes/no, the label for pick one and the level number for rating.
    """
    confidence = float(answer_data.get("confidence", 0.0))
    probabilities = dict(answer_data.get("probabilities") or {})
    match spec["kind"]:
        case QuestionKind.YES_NO:
            raw = answer_data.get("noul")
            if raw is None:
                return None
            probability = float(raw)
            answer = probability >= spec["threshold"]
            return answer, {"answer": answer, "probability": probability}
        case QuestionKind.CHOICE:
            picked = answer_data.get("choice")
            if picked is None:
                return None
            return picked, {"label": picked, "confidence": confidence, "probabilities": probabilities}
        case QuestionKind.SCORE:
            raw = answer_data.get("score")
            if raw is None:
                return None
            labels = spec["labels"]
            # Round half-up; JEV numbers from 0, display from 1.
            index = min(int(float(raw) + 0.5), len(labels) - 1)
            details = {
                "level": index + 1,
                "score": float(raw) + 1,
                "label": labels[index],
                "confidence": confidence,
                "probabilities": {str(int(k) + 1): v for k, v in probabilities.items()},
            }
            return index + 1, details
        case _:
            msg = f"Unknown question kind: {spec['kind']!r}"
            raise ValueError(msg)
