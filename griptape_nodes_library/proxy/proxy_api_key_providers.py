from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ProxyApiKeyProviderConfig",
    "get_proxy_api_key_provider_config",
    "is_proxy_api_key_provider_disabled",
]


@dataclass(frozen=True)
class ProxyApiKeyProviderConfig:
    # `provider_id` is the canonical, stable key a permission policy gates on,
    # and the identifier the model catalog uses for this provider. Keep it
    # lowercase/snake_case and in sync with the catalog's provider keys.
    provider_id: str
    api_key_name: str
    provider_name: str
    api_key_url: str
    secret_label: str = "API key"
    parameter_display_name: str = "API Key Provider"


BLACK_FOREST_LABS = ProxyApiKeyProviderConfig(
    provider_id="black_forest_labs",
    api_key_name="BFL_API_KEY",
    provider_name="BlackForest Labs",
    api_key_url="https://dashboard.bfl.ai/api/keys",
)
ELEVENLABS = ProxyApiKeyProviderConfig(
    provider_id="elevenlabs",
    api_key_name="ELEVENLABS_API_KEY",
    provider_name="ElevenLabs",
    api_key_url="https://elevenlabs.io/app/settings/api-keys",
)
GOOGLE = ProxyApiKeyProviderConfig(
    provider_id="google",
    api_key_name="GOOGLE_SERVICE_ACCOUNT_JSON",
    provider_name="Google Cloud",
    api_key_url="https://console.cloud.google.com/iam-admin/serviceaccounts",
    secret_label="service account JSON",
    parameter_display_name="Credentials Provider",
)
GROK = ProxyApiKeyProviderConfig(
    provider_id="xai",
    api_key_name="GROK_API_KEY",
    provider_name="xAI",
    api_key_url="https://console.x.ai",
)
DASHSCOPE = ProxyApiKeyProviderConfig(
    provider_id="dashscope",
    api_key_name="DASHSCOPE_API_KEY",
    provider_name="Alibaba Cloud Model Studio",
    api_key_url="https://bailian.console.aliyun.com/",
)
SEED = ProxyApiKeyProviderConfig(
    provider_id="bytedance_seed",
    api_key_name="SEED_API_KEY",
    provider_name="ByteDance Seed",
    api_key_url="https://seed.bytedance.com/",
)
TOPAZ = ProxyApiKeyProviderConfig(
    provider_id="topaz",
    api_key_name="TOPAZ_API_KEY",
    provider_name="Topaz Labs",
    api_key_url="https://developer.topazlabs.com/",
)
RODIN = ProxyApiKeyProviderConfig(
    provider_id="hyper3d",
    api_key_name="RODIN_API_KEY",
    provider_name="Hyper3D Rodin",
    api_key_url="https://hyper3d.ai/",
)
LTX = ProxyApiKeyProviderConfig(
    provider_id="ltx",
    api_key_name="LTX_API_KEY",
    provider_name="LTX Studio",
    api_key_url="https://docs.ltx.video/welcome",
)
KLING = ProxyApiKeyProviderConfig(
    provider_id="kling",
    api_key_name="KLING_API_KEY",
    provider_name="Kling AI",
    api_key_url="https://app.klingai.com/global/dev",
)
MINIMAX = ProxyApiKeyProviderConfig(
    provider_id="minimax",
    api_key_name="MINIMAX_API_KEY",
    provider_name="MiniMax",
    api_key_url="https://minimax.io/",
)
OPENAI = ProxyApiKeyProviderConfig(
    provider_id="openai",
    api_key_name="OPENAI_API_KEY",
    provider_name="OpenAI",
    api_key_url="https://platform.openai.com/api-keys",
)
WORLD_LABS = ProxyApiKeyProviderConfig(
    provider_id="world_labs",
    api_key_name="WORLD_LABS_API_KEY",
    provider_name="World Labs",
    api_key_url="https://platform.worldlabs.ai/api-keys",
)
TRIPO = ProxyApiKeyProviderConfig(
    provider_id="tripo",
    api_key_name="TRIPO_API_KEY",
    provider_name="Tripo 3D",
    api_key_url="https://platform.tripo3d.ai/api-keys",
)

_NODE_PROVIDER_CONFIGS = {
    "ElevenLabsMusicGeneration": ELEVENLABS,
    "ElevenLabsSoundEffectGeneration": ELEVENLABS,
    "ElevenLabsTextToSpeechGeneration": ELEVENLABS,
    "Flux2ImageGeneration": BLACK_FOREST_LABS,
    "FluxImageGeneration": BLACK_FOREST_LABS,
    "GeminiOmniFlashGeneration": GOOGLE,
    "GoogleImageGeneration": GOOGLE,
    "GrokImageEdit": GROK,
    "GrokImageGeneration": GROK,
    "GrokVideoEdit": GROK,
    "GrokVideoGeneration": GROK,
    "KlingImageToVideoGeneration": KLING,
    "KlingMotionControl": KLING,
    "KlingOmniVideoGeneration": KLING,
    "KlingTextToVideoGeneration": KLING,
    "KlingVideoExtension": KLING,
    "LTXAudioToVideoGeneration": LTX,
    "LTXImageToVideoGeneration": LTX,
    "LTXTextToVideoGeneration": LTX,
    "LTXVideoRetake": LTX,
    "MinimaxHailuoVideoGeneration": MINIMAX,
    "OmnihumanSubjectDetection": SEED,
    "OmnihumanSubjectRecognition": SEED,
    "OmnihumanVideoGeneration": SEED,
    "OpenAiImageGeneration": OPENAI,
    "QwenImageEdit": DASHSCOPE,
    "QwenImageGeneration": DASHSCOPE,
    "Rodin23DGeneration": RODIN,
    "SeedanceVideoGeneration": SEED,
    "Seedance20VideoGeneration": SEED,
    "SeedreamImageGeneration": SEED,
    "SoraVideoGeneration": OPENAI,
    "TopazImageEnhance": TOPAZ,
    "TranscribeAudio": OPENAI,
    "TripoImageTo3DGeneration": TRIPO,
    "TripoMultiviewTo3DGeneration": TRIPO,
    "TripoTextTo3DGeneration": TRIPO,
    "Veo3VideoGeneration": GOOGLE,
    "WanAnimateGeneration": DASHSCOPE,
    "WanImageGeneration": DASHSCOPE,
    "WanImageToVideoGeneration": DASHSCOPE,
    "WanReferenceToVideoGeneration": DASHSCOPE,
    "WanTextToVideoGeneration": DASHSCOPE,
    "WorldLabsWorldGeneration": WORLD_LABS,
}

_DISABLED_NODE_PROVIDER_CONFIGS: set[str] = {
    "GeminiOmniFlashGeneration",
    "GoogleImageGeneration",
    "Veo3VideoGeneration",
}


def get_proxy_api_key_provider_config(node_class_name: str) -> ProxyApiKeyProviderConfig | None:
    return _NODE_PROVIDER_CONFIGS.get(node_class_name)


def is_proxy_api_key_provider_disabled(node_class_name: str) -> bool:
    return node_class_name in _DISABLED_NODE_PROVIDER_CONFIGS
