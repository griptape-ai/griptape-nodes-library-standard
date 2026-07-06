# Griptape Nodes Library — Standard

## Python conventions

### Hoist imports to the top level unless they cause circular imports

Keep all imports at the top of the file. Only use lazy (function-level) imports when necessary to break a circular dependency — in that case add a `# avoid circular import` comment so the reason is clear.

```python
# BAD — lazy import with no circular-import justification
def _my_method(self) -> None:
    from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
    GriptapeNodes.handle_request(...)

# GOOD
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

def _my_method(self) -> None:
    GriptapeNodes.handle_request(...)

# ACCEPTABLE — circular import that can't be avoided
def process(self) -> None:
    from griptape_nodes.retained_mode.events.variable_events import CreateVariableRequest  # avoid circular import
    ...
```

### Use `StrEnum` for repeated string constants in logic

Use `StrEnum` when the same string values appear repeatedly across logic (comparisons, branches, return values) and a typo would cause a silent bug. The canonical case is a value that drives behaviour in multiple methods — not just a list of choices passed to a single `Options(choices=...)` call.

```python
from enum import StrEnum

class CaseStyle(StrEnum):
    SNAKE = "snake_case"
    CAMEL = "camelCase"
    PASCAL = "PascalCase"

# In logic across multiple methods:
if case_style == CaseStyle.SNAKE:
    ...
```

Use `StrEnum` (not plain `Enum`) so values compare equal to their string equivalents — parameter values coming back from the UI are plain strings, and `StrEnum` members match them without explicit `.value` lookups.

For **dropdown-only option lists** that are passed straight to `Options(choices=...)` and never repeated in logic, a plain module-level list is fine and less ceremony:

```python
# Fine for a simple dropdown — no need for StrEnum
ORDERING_MODES = ["Sequential", "Respect frame numbers"]
traits={Options(choices=ORDERING_MODES)}
```

### Use `match`/`case` instead of `if`/`elif` chains

Whenever you're branching on a single value across three or more cases — whether it's a `StrEnum`, a string, an int, or a type tag — use `match`/`case` instead of a chain of `if` comparisons. Always end with a wildcard `case _:` that raises `ValueError`; this surfaces unhandled values immediately rather than silently falling through to a wrong default.

```python
# BAD — if/elif chain
if case_style == CaseStyle.SNAKE:
    return "_".join(w.lower() for w in words)
elif case_style == CaseStyle.CAMEL:
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])
# ... six more elifs, then a silent default or nothing

# GOOD
match case_style:
    case CaseStyle.SNAKE:
        return "_".join(w.lower() for w in words)
    case CaseStyle.CAMEL:
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])
    case _:
        msg = f"Unknown case style: {case_style!r}"
        raise ValueError(msg)
```

The wildcard `case _: raise ValueError(...)` is not optional — it ensures that adding a new enum member (or passing an unexpected value) is caught immediately rather than producing a silent wrong result.
