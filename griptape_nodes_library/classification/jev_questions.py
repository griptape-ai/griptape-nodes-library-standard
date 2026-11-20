from __future__ import annotations

from typing import Any

__all__ = ["MAX_LEVELS", "choice_question", "score_question", "yes_no_question"]

MAX_LEVELS = 10


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
