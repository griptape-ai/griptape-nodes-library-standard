"""The one legacy-value table the Griptape Cloud chat-model dropdowns migrate through.

`Agent`, `GriptapeCloudPrompt`, and `DescribeImage` all offer Griptape Cloud's
chat models, so they all have to migrate the same historical stored values.
Defining the table once means retiring a model is a single edit: per-node copies
have to be updated in lockstep or the same saved workflow migrates differently
depending on which node held the value, and the per-node migration tests only
check each node's own table, so that divergence would pass green.
"""

from collections.abc import Sequence

# Migrates values saved before the dropdown stored the provider's own model id. Folds in
# both display labels once shown in the dropdown, catalog keys saved during the interval
# the dropdown stored those instead, and the nodes' former DEPRECATED_MODELS dicts
# (provider ids for models Griptape Cloud has fully retired, mapped to their replacement).
CLOUD_LEGACY_MODEL_VALUES: dict[str, str] = {
    "Claude Haiku 4.5": "claude-haiku-4-5",
    "Claude Opus 4.7": "claude-opus-5",
    "Claude Sonnet 4.5": "claude-sonnet-5",
    "Claude Sonnet 4.6": "claude-sonnet-5",
    "DeepSeek R1": "deepseek.r1-v1",
    "DeepSeek V3": "deepseek-v3",
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "GPT-4o": "gpt-4o",
    "GPT-5": "gpt-5",
    "GPT-5 mini": "gpt-5-mini",
    "GPT-5 nano": "gpt-5-nano",
    "GPT-5.1": "gpt-5.1",
    "GPT-5.2": "gpt-5.2",
    "GPT-5.2 Chat": "gpt-5.2-chat",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 3 Flash": "gemini-3-flash",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini 3.1 Pro": "gemini-3.1-pro",
    "Llama 3.1 70B Instruct": "llama3-1-70b-instruct-v1",
    "Llama 3.3 70B Instruct": "llama3-3-70b-instruct-v1",
    "amazon.titan-text-premier-v1": "claude-sonnet-5",
    "claude-3-5-haiku": "claude-haiku-4-5",
    "claude-3-7-sonnet": "claude-sonnet-5",
    "claude-4-5-sonnet": "claude-sonnet-5",
    "claude-opus-4-7": "claude-opus-5",
    "claude-sonnet-4-20250514": "claude-sonnet-5",
    "claude-sonnet-4-6": "claude-sonnet-5",
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-3-pro": "gemini-3.1-pro",
    "gemini-3-pro-preview": "gemini-3.1-pro",
    "gpt-4.5-preview": "gpt-4.1",
    "gtc_claude_haiku_4_5": "claude-haiku-4-5",
    "gtc_claude_opus_4_7": "claude-opus-5",
    "gtc_claude_sonnet_4_5": "claude-sonnet-5",
    "gtc_claude_sonnet_4_6": "claude-sonnet-5",
    "gtc_deepseek_r1": "deepseek.r1-v1",
    "gtc_deepseek_v3": "deepseek-v3",
    "gtc_gemini_2_5_flash": "gemini-2.5-flash",
    "gtc_gemini_2_5_flash_lite": "gemini-2.5-flash-lite",
    "gtc_gemini_2_5_pro": "gemini-2.5-pro",
    "gtc_gemini_3_1_flash_lite": "gemini-3.1-flash-lite",
    "gtc_gemini_3_1_pro": "gemini-3.1-pro",
    "gtc_gemini_3_flash": "gemini-3-flash",
    "gtc_gpt_4_1": "gpt-4.1",
    "gtc_gpt_4_1_mini": "gpt-4.1-mini",
    "gtc_gpt_4_1_nano": "gpt-4.1-nano",
    "gtc_gpt_4o": "gpt-4o",
    "gtc_gpt_5": "gpt-5",
    "gtc_gpt_5_1": "gpt-5.1",
    "gtc_gpt_5_2": "gpt-5.2",
    "gtc_gpt_5_2_chat": "gpt-5.2-chat",
    "gtc_gpt_5_mini": "gpt-5-mini",
    "gtc_gpt_5_nano": "gpt-5-nano",
    "gtc_llama_3_1_70b": "llama3-1-70b-instruct-v1",
    "gtc_llama_3_3_70b": "llama3-3-70b-instruct-v1",
    "gtc_o1": "o1",
    "gtc_o3": "o3",
    "gtc_o3_mini": "o3-mini",
    "gtc_o4_mini": "o4-mini",
    "o1-mini": "o3-mini",
    "o3 mini": "o3-mini",
    "o4 mini": "o4-mini",
}


def cloud_legacy_values_for(model_choices: Sequence[str]) -> dict[str, str]:
    """The entries of `CLOUD_LEGACY_MODEL_VALUES` a dropdown offering `model_choices` can migrate.

    `ModelAccessComponent` rejects a `deprecated_values` entry whose target is not
    one of the dropdown's choices, so a node that offers a slice of Griptape Cloud's
    chat models (`DescribeImage`, vision-capable only) subsets the shared table by
    target instead of keeping its own copy.
    """
    offered = set(model_choices)
    return {legacy: canonical for legacy, canonical in CLOUD_LEGACY_MODEL_VALUES.items() if canonical in offered}
