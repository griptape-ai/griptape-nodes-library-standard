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

### Use `StrEnum` for string option sets

When a parameter has a fixed set of string choices (e.g. a dropdown), declare them as a `StrEnum` rather than bare module-level constants. This keeps the options co-located, makes comparisons type-safe, and lets you pass `list(MyEnum)` directly to `Options(choices=...)`.

```python
from enum import StrEnum

class CaseStyle(StrEnum):
    SNAKE = "snake_case"
    CAMEL = "camelCase"
    PASCAL = "PascalCase"

# In __init__:
param.add_trait(Options(choices=list(CaseStyle)))

# In logic:
if case_style == CaseStyle.SNAKE:
    ...
```

Use `StrEnum` (not plain `Enum`) so values compare equal to their string equivalents — parameter values coming back from the UI are plain strings, and `StrEnum` members match them without explicit `.value` lookups.
