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

# Catalog model keys for the same models, in the same order as MODEL_CHOICES_ARGS.
# GriptapeCloudPrompt's dropdown stores these -- the catalog id the license
# layer gates on -- rather than the provider's own model id; consumers that
# still key off the provider id (MODEL_CHOICES_ARGS, O_SERIES_MODELS) resolve
# one from the other via `_provider_model_id_for_selection`.
CATALOG_MODEL_CHOICES = [
    # Anthropic / Bedrock-Claude
    "gtc_claude_opus_4_7",
    "gtc_claude_sonnet_4_6",
    "gtc_claude_sonnet_4_5",
    "gtc_claude_haiku_4_5",
    # Bedrock non-Claude
    "gtc_deepseek_v3",
    "gtc_deepseek_r1",
    "gtc_llama_3_3_70b",
    "gtc_llama_3_1_70b",
    # Google
    "gtc_gemini_3_1_pro",
    "gtc_gemini_3_1_flash_lite",
    "gtc_gemini_3_flash",
    "gtc_gemini_2_5_pro",
    "gtc_gemini_2_5_flash",
    "gtc_gemini_2_5_flash_lite",
    # Azure OpenAI
    "gtc_gpt_5_2",
    "gtc_gpt_5_2_chat",
    "gtc_gpt_5_1",
    "gtc_gpt_5",
    "gtc_gpt_5_mini",
    "gtc_gpt_5_nano",
    "gtc_gpt_4_1",
    "gtc_gpt_4_1_mini",
    "gtc_gpt_4_1_nano",
    "gtc_gpt_4o",
    "gtc_o4_mini",
    "gtc_o3",
    "gtc_o3_mini",
    "gtc_o1",
]


# Model IDs whose backend does not accept top_p (the OpenAI o-series).
# Kept in sync with the o-entries in MODEL_CHOICES_ARGS.
O_SERIES_MODELS = {"o1", "o3", "o3-mini", "o4-mini"}
