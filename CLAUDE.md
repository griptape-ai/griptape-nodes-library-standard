# Griptape Nodes Library — Standard

## Validate inputs before the run

When you add or edit a node whose provider documents limits on its inputs, declare them on the parameter with `pre_run_validators=`. The engine checks them at the node's turn, before it runs, so a bad input fails before anything is uploaded or sent, and the error says which limit it broke.

- **Use the engine's validators**, all imported from `griptape_nodes.exe_types.validators`. For any value: `MaxCount` for a list's length, and `IfParameter` for limits that change with another parameter, such as the selected model (`IfParameter("model", {"pro": [...]}, default=[...])`). For image, video and audio files: `AllowedFormats`, `MaxFileSize`, `SideRange`, `PixelArea`, `AspectRatioRange`, `ExactAspectRatios`, `MinDuration`, `MaxDuration`, `FrameRate`, `FrameCount`, `VideoCodecs` and `AudioCodecs`. The media ones read each file once and skip anything they can't read locally. `.first(n)` narrows any validator to a list's first `n` items, and `.when(condition)`/`.unless(condition)` narrow a media one by what was read from the file. Mark the condition with `@reads(MediaNeed...)` for what it reads, or it sees only what its validator reads and can silently never fire. Don't write your own readers or limit checks in the library. See [Checking inputs before a run](https://docs.griptapenodes.com/en/stable/development/custom_nodes/error_handling/#checking-inputs-before-a-run).
- **Rules across inputs go in `validate_before_node_run`.** A limit that involves more than one input, such as a total length across several clips, is checked in the node's `validate_before_node_run`, which the engine runs right after the declared validators. A limit on one input that changes with a setting is an `IfParameter` on that input, even when it depends on two settings. For media, the readers in `griptape_nodes.utils.media_probe` (`local_video_probe` and the rest) return what the validators already read. A rule no built-in validator can express on one input is a small subclass of the engine's base: `PreRunValidator` for any value, `MediaValidator` for one judged on the file.
- **Keep only the provider's specifics in the node:** the limits, each with a comment quoting the provider's docs and its URL, and `input_limits_url()`, which returns the page the errors link to (the selected model's page, where the provider has one). The errors name each input by its parameter's `display_name`. Lists and validators shared by one provider's nodes go in that provider's existing common module. Keep `validate_before_node_run` for checks across several inputs, and for checks that aren't limits on a value, such as a missing API key.
- **Move a check into the engine only when two or more nodes need it now.** If a validator or check isn't in the engine yet and two or more nodes from different providers need it, add it to the engine, not to a library module. Nodes from one provider that already share a module (such as `seedance_common.py`) don't count, and neither does a node that might need it later: keep that logic in the library until a second real user exists.

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

## Changelog

`CHANGELOG.md` follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/). Each version is the `library_version` in `griptape_nodes_library.json`. Its readers are artists using these nodes and the agents that build workflows with them.

**Add an entry for every user-facing change** - In the same PR, add a file to `changelog.d/` holding the entry. User-facing means anything a user notices after updating the library: a node added, removed, or renamed; a parameter, default, or output that differs; a provider model added or retired; a saved workflow that loads or runs differently. After adding one, tell the user so they can review the wording.

**One file per entry, named for its type** - Never edit `## [Unreleased]` in `CHANGELOG.md` directly; `make check/changelog` rejects it. Create `changelog.d/<type>-<slug>.md`, where `<type>` is one of `added`, `changed`, `deprecated`, `removed`, `fixed`, or `security`, and `<slug>` is a few lowercase words joined by hyphens. The whole file is the bullet, with no heading. `make changelog` prints `[Unreleased]` with these files folded in. `changelog.d/README.md` is the same contract for humans.

**Skip what users never see** - Refactors, tests, CI, docs-only changes, and dev dependency bumps get no entry. Neither does a fix for a bug that never shipped in a version: edit or delete the entry that introduced it.

**Describe the result, not the code** - Name the node and parameter in backticks or by their editor label, say what the user sees now, present tense, one or two sentences. `Fixed` if the old behavior was a bug, `Changed` if it was intentional. Link the GitHub issue on its own last line. Leave out PR numbers and `@handles`. Wrap at about 100 characters, continuation lines indented two spaces.

**Mark breaking changes** - A change that makes a saved workflow stop working, or give a different result, without edits starts with `**Breaking:**`. Say what breaks and what to do about it.

**Leave `CHANGELOG.md` to the version bump** - Do not add version headings or link definitions. `make version/patch`, `make version/minor`, `make version/major`, and `make version/set` fold the `changelog.d/` entries into the new version section and delete them. Fixing a mistake in a released entry is fine.
