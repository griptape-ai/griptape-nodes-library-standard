"""Driver presets and per-family UI data for Griptape Cloud-backed chat models.

The roster of available models -- which models exist, their provider model ids,
and their family -- lives in the library's `model_catalog` declaration in
`griptape_nodes_library.json`. That declaration is the single source of truth;
this module never restates the model list.

What stays here is the library-specific data the catalog schema does not model:
the Griptape prompt-driver arg presets and the provider logos. Both are keyed by
the catalog `family` tag, so they are a small, stable mapping rather than a
per-model list. The helpers join a node's declared catalog models
(`get_declared_models`) to those presets, producing the `{name, icon, args}`
rows the nodes feed to their model dropdowns.

When Cloud's catalog changes (new model, deprecated model), update the
`model_catalog` declaration; new families additionally need an entry here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from griptape_nodes.node_library.library_registry import get_declared_models

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

# --- Per-family arg presets ---

_CLAUDE_ARGS = {"stream": True, "structured_output_strategy": "tool", "max_tokens": 64000}
_DEEPSEEK_R1_ARGS = {"stream": False, "structured_output_strategy": "tool", "top_p": None}
_DEEPSEEK_V3_ARGS = {"stream": True, "structured_output_strategy": "tool"}
_LLAMA_ARGS = {"stream": True, "structured_output_strategy": "tool"}
_GEMINI_ARGS = {"stream": True}
_OPENAI_ARGS = {"stream": True}

_DEFAULT_ICON = "logos/griptape.svg"

# Catalog `family` tag -> per-family driver args and provider logo. Keys must
# match the `family` values declared on the Griptape Cloud models in the
# `model_catalog`; a family with no entry falls back to no args and the default
# icon.
_FAMILY_PRESETS: dict[str, dict[str, Any]] = {
    "Anthropic (via Griptape Cloud)": {"args": _CLAUDE_ARGS, "icon": "logos/anthropic.svg"},
    "DeepSeek V3 (via Griptape Cloud)": {"args": _DEEPSEEK_V3_ARGS, "icon": "logos/deepseek.svg"},
    "DeepSeek R1 (via Griptape Cloud)": {"args": _DEEPSEEK_R1_ARGS, "icon": "logos/deepseek.svg"},
    "Meta Llama (via Griptape Cloud)": {"args": _LLAMA_ARGS, "icon": "logos/meta.svg"},
    "Google Gemini (via Griptape Cloud)": {"args": _GEMINI_ARGS, "icon": "logos/google.svg"},
    "OpenAI (via Griptape Cloud)": {"args": _OPENAI_ARGS, "icon": "logos/openai.svg"},
    "OpenAI o-series (via Griptape Cloud)": {"args": _OPENAI_ARGS, "icon": "logos/openai.svg"},
}

# The catalog family whose models reject the `top_p` parameter (the OpenAI
# o-series). Drives whether `top_p` is forwarded to the driver.
_O_SERIES_FAMILY = "OpenAI o-series (via Griptape Cloud)"


# Maps deprecated model IDs that may appear in saved workflows to their live
# replacement. Consumers use this to rewrite the model on load and surface a
# deprecation notice to the user.
DEPRECATED_MODELS = {
    # Anthropic
    "claude-3-7-sonnet": "claude-sonnet-4-6",
    "claude-3-5-haiku": "claude-haiku-4-5",
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    # Bedrock
    "amazon.titan-text-premier-v1": "claude-sonnet-4-6",
    # Azure OpenAI
    "gpt-4.5-preview": "gpt-4.1",
    "o1-mini": "o3-mini",
    # Google
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-3-pro": "gemini-3.1-pro",
    "gemini-3-pro-preview": "gemini-3.1-pro",
}


def model_choices_args(node: BaseNode) -> list[dict[str, Any]]:
    """Per-model dropdown rows (`{name, icon, args}`) for `node`.

    Joins the catalog models the node declares (via its `model_usage` /
    `model_provider_usage` declarations) to their family presets, in the order
    the catalog declares them.
    """
    rows: list[dict[str, Any]] = []
    for resolved in get_declared_models(node):
        preset = _FAMILY_PRESETS.get(resolved.model.family or "", {})
        rows.append(
            {
                "name": resolved.model.provider_model_id,
                "icon": preset.get("icon", _DEFAULT_ICON),
                "args": preset.get("args", {}),
            }
        )
    return rows


def model_choices(node: BaseNode) -> list[str]:
    """Provider model ids `node` offers, sourced from the catalog roster."""
    return [row["name"] for row in model_choices_args(node)]


def args_for_model(node: BaseNode, model_id: str) -> dict[str, Any]:
    """Driver arg preset for a single model id, or an empty dict if unknown."""
    return next((row["args"] for row in model_choices_args(node) if row["name"] == model_id), {})


def o_series_model_ids(node: BaseNode) -> set[str]:
    """Model ids that reject `top_p`, derived from the catalog o-series family."""
    return {
        resolved.model.provider_model_id
        for resolved in get_declared_models(node)
        if resolved.model.family == _O_SERIES_FAMILY and resolved.model.provider_model_id is not None
    }
