"""Shared catalog of Griptape Cloud-backed chat models.

This module is the single source of truth for every node that offers a
Griptape Cloud model dropdown (e.g. the `Agent` node, the `GriptapeCloudPrompt`
node). It mirrors the active `model_type=chat` rows in Griptape Cloud's
ServiceModelConfig table.

When Cloud's catalog changes (new model added, deprecated model deactivated),
update this file and every consumer picks up the change.
"""

# --- Per-family arg presets ---

_CLAUDE_ARGS = {"stream": True, "structured_output_strategy": "tool", "max_tokens": 64000}
_DEEPSEEK_R1_ARGS = {"stream": False, "structured_output_strategy": "tool", "top_p": None}
_DEEPSEEK_V3_ARGS = {"stream": True, "structured_output_strategy": "tool"}
_LLAMA_ARGS = {"stream": True, "structured_output_strategy": "tool"}
_GEMINI_ARGS = {"stream": True}
_OPENAI_ARGS = {"stream": True}


MODEL_CHOICES_ARGS = [
    # Anthropic / Bedrock-Claude
    {"name": "claude-opus-4-7", "icon": "logos/anthropic.svg", "args": _CLAUDE_ARGS},
    {"name": "claude-sonnet-4-6", "icon": "logos/anthropic.svg", "args": _CLAUDE_ARGS},
    {"name": "claude-4-5-sonnet", "icon": "logos/anthropic.svg", "args": _CLAUDE_ARGS},
    {"name": "claude-haiku-4-5", "icon": "logos/anthropic.svg", "args": _CLAUDE_ARGS},
    # Bedrock non-Claude
    {"name": "deepseek-v3", "icon": "logos/deepseek.svg", "args": _DEEPSEEK_V3_ARGS},
    {"name": "deepseek.r1-v1", "icon": "logos/deepseek.svg", "args": _DEEPSEEK_R1_ARGS},
    {"name": "llama3-3-70b-instruct-v1", "icon": "logos/meta.svg", "args": _LLAMA_ARGS},
    {"name": "llama3-1-70b-instruct-v1", "icon": "logos/meta.svg", "args": _LLAMA_ARGS},
    # Google
    {"name": "gemini-3.1-pro", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    {"name": "gemini-3.1-flash-lite", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    {"name": "gemini-3-flash", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    {"name": "gemini-2.5-pro", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    {"name": "gemini-2.5-flash", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    {"name": "gemini-2.5-flash-lite", "icon": "logos/google.svg", "args": _GEMINI_ARGS},
    # Azure OpenAI
    {"name": "gpt-5.2", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-5.2-chat", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-5.1", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-5", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-5-mini", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-5-nano", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-4.1", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-4.1-mini", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-4.1-nano", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "gpt-4o", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "o4-mini", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "o3", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "o3-mini", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
    {"name": "o1", "icon": "logos/openai.svg", "args": _OPENAI_ARGS},
]

# Provider model ids for the same models, in the same order as MODEL_CHOICES_ARGS.
# Both Agent's and GriptapeCloudPrompt's dropdowns store these directly -- it is
# each model's own `name` from MODEL_CHOICES_ARGS, which is also what the
# license layer resolves the selection to a catalog id through.
PROVIDER_MODEL_CHOICES = [model["name"] for model in MODEL_CHOICES_ARGS]


# Model IDs whose backend does not accept top_p (the OpenAI o-series).
# Kept in sync with the o-entries in MODEL_CHOICES_ARGS.
O_SERIES_MODELS = {"o1", "o3", "o3-mini", "o4-mini"}
