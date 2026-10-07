# Saved-workflow compatibility fixtures

`main/*.py` are workflows the engine built, ran, and saved with the griptape-era library
(the merge-base, `d19c834`). The compat tests load them on this branch and run them with a
fake model. `generation_status.json` records which ones ran on main; the rest were saved
unrun because the provider needed a key, and still carry main's parameter values.

`main_parameter_shapes.json` is every touched node's parameters as main declared them.

Regenerate with main checked out at `MAIN` (spends a little Griptape Cloud credit):

```bash
uv run --project MAIN python tests/unit/compat/generate/generate.py MAIN [spec ...]
```

Specs live in `generate/specs.py`. Paths in saved values become `__COMPAT_MEDIA__`,
`__COMPAT_PYTHON__`, and `__COMPAT_MCP_SERVER__`; `harness.materialize` puts real paths back.
