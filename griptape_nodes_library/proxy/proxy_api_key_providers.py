from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ProxyApiKeyProviderConfig",
    "get_proxy_api_key_provider_config",
    "is_proxy_api_key_provider_disabled",
]


@dataclass(frozen=True)
class ProxyApiKeyProviderConfig:
    api_key_name: str
    provider_name: str
    api_key_url: str
    secret_label: str = "API key"
    parameter_display_name: str = "API Key Provider"


BLACK_FOREST_LABS = ProxyApiKeyProviderConfig(
    api_key_name="BFL_API_KEY",
    provider_name="BlackForest Labs",
    api_key_url="https://dashboard.bfl.ai/api/keys",
)
ELEVENLABS = ProxyApiKeyProviderConfig(
    api_key_name="ELEVENLABS_API_KEY",
    provider_name="ElevenLabs",
    api_key_url="https://elevenlabs.io/app/settings/api-keys",
)
GOOGLE = ProxyApiKeyProviderConfig(
    api_key_name="GOOGLE_SERVICE_ACCOUNT_JSON",
    provider_name="Google Cloud",
    api_key_url="https://console.cloud.google.com/iam-admin/serviceaccounts",
    secret_label="service account JSON",
    parameter_display_name="Credentials Provider",
)
GROK = ProxyApiKeyProviderConfig(
    api_key_name="GROK_API_KEY",
    provider_name="xAI",
    api_key_url="https://console.x.ai",
)
DASHSCOPE = ProxyApiKeyProviderConfig(
    api_key_name="DASHSCOPE_API_KEY",
    provider_name="Alibaba Cloud Model Studio",
    api_key_url="https://bailian.console.aliyun.com/",
)
SEED = ProxyApiKeyProviderConfig(
    api_key_name="SEED_API_KEY",
    provider_name="ByteDance Seed",
    api_key_url="https://seed.bytedance.com/",
)
TOPAZ = ProxyApiKeyProviderConfig(
    api_key_name="TOPAZ_API_KEY",
    provider_name="Topaz Labs",
    api_key_url="https://developer.topazlabs.com/",
)
RODIN = ProxyApiKeyProviderConfig(
    api_key_name="RODIN_API_KEY",
    provider_name="Hyper3D Rodin",
    api_key_url="https://hyper3d.ai/",
)
LTX = ProxyApiKeyProviderConfig(
    api_key_name="LTX_API_KEY",
    provider_name="LTX Studio",
    api_key_url="https://docs.ltx.video/welcome",
)
KLING = ProxyApiKeyProviderConfig(
    api_key_name="KLING_API_KEY",
    provider_name="Kling AI",
    api_key_url="https://app.klingai.com/global/dev",
)
MINIMAX = ProxyApiKeyProviderConfig(
    api_key_name="MINIMAX_API_KEY",
    provider_name="MiniMax",
    api_key_url="https://minimax.io/",
)
OPENAI = ProxyApiKeyProviderConfig(
    api_key_name="OPENAI_API_KEY",
    provider_name="OpenAI",
    api_key_url="https://platform.openai.com/api-keys",
)
WORLD_LABS = ProxyApiKeyProviderConfig(
    api_key_name="WORLD_LABS_API_KEY",
    provider_name="World Labs",
    api_key_url="https://platform.worldlabs.ai/api-keys",
)

_NODE_PROVIDER_CONFIGS = {
    "ElevenLabsMusicGeneration": ELEVENLABS,
    "ElevenLabsSoundEffectGeneration": ELEVENLABS,
    "ElevenLabsTextToSpeechGeneration": ELEVENLABS,
    "Flux2ImageGeneration": BLACK_FOREST_LABS,
    "FluxImageGeneration": BLACK_FOREST_LABS,
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
    "QwenImageEdit": DASHSCOPE,
    "QwenImageGeneration": DASHSCOPE,
    "Rodin23DGeneration": RODIN,
    "RodinBang3DEdit": RODIN,
    "SeedanceVideoGeneration": SEED,
    "SeedreamImageGeneration": SEED,
    "SeedVRImageUpscale": SEED,
    "SeedVRVideoUpscale": SEED,
    "SoraVideoGeneration": OPENAI,
    "TopazImageEnhance": TOPAZ,
    "Veo3VideoGeneration": GOOGLE,
    "WanAnimateGeneration": DASHSCOPE,
    "WanImageToVideoGeneration": DASHSCOPE,
    "WanReferenceToVideoGeneration": DASHSCOPE,
    "WanTextToVideoGeneration": DASHSCOPE,
    "WorldLabsWorldGeneration": WORLD_LABS,
}

_DISABLED_NODE_PROVIDER_CONFIGS: set[str] = {
    "GoogleImageGeneration",
    "Veo3VideoGeneration",
}


def get_proxy_api_key_provider_config(node_class_name: str) -> ProxyApiKeyProviderConfig | None:
    return _NODE_PROVIDER_CONFIGS.get(node_class_name)


def is_proxy_api_key_provider_disabled(node_class_name: str) -> bool:
    return node_class_name in _DISABLED_NODE_PROVIDER_CONFIGS
