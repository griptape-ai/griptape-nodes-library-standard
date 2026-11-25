"""Shared request and response handling for Kling 4.0 and Kling 4.0 Flash.

Kling 4.0 uses a different API from v1-v3: the request body is `contents[]` (prompt,
images, videos, elements, voices), `settings` (resolution, aspect ratio, duration,
audio) and `options`, and the completed task reports its videos in `outputs[]`.
https://kling.ai/document-api/api/video/4-0/video-omni
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from griptape_nodes_library import media

KLING_4_0 = "kling-4.0"
KLING_4_0_FLASH = "kling-4.0-flash"
KLING_4_MODELS = (KLING_4_0, KLING_4_0_FLASH)

RESOLUTION_720P = "720p"
RESOLUTION_1080P = "1080p"
RESOLUTION_4K = "4k"
DYNAMIC_RANGE_SDR = "sdr"
DYNAMIC_RANGE_HDR = "hdr"
DYNAMIC_RANGES = [DYNAMIC_RANGE_SDR, DYNAMIC_RANGE_HDR]
ASPECT_RATIO_ADAPTIVE = "adaptive"
GENERATED_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "21:9"]
AUDIO_MODES = ["native", "original"]

# Kling's prompt limit: English-only 26,000 characters, otherwise 11,000, and mixed
# text must satisfy english / 26000 + other / 11000 <= 1.
MAX_ENGLISH_PROMPT_CHARS = 26_000
MAX_OTHER_PROMPT_CHARS = 11_000

MAX_IMAGES = 10
MAX_ELEMENTS = 7
MAX_VOICES = 3
MAX_VIDEOS = 5
MAX_REFERENCES = 15

MIN_DURATION_SECONDS = 3
KLING_4_0_MAX_DURATION_SECONDS = 30
KLING_4_0_FLASH_MAX_DURATION_SECONDS = 20


@dataclass(frozen=True)
class Kling4Capabilities:
    resolutions: list[str]
    default_resolution: str
    max_duration: int
    supports_keyframes: bool
    supports_voices: bool
    # Resolutions Kling can render in HDR (`settings.dynamic_range: "hdr"`), for an extra per-second fee.
    hdr_resolutions: tuple[str, ...] = ()

    @property
    def durations(self) -> list[int]:
        return list(range(MIN_DURATION_SECONDS, self.max_duration + 1))


KLING_4_CAPABILITIES: dict[str, Kling4Capabilities] = {
    KLING_4_0: Kling4Capabilities(
        resolutions=[RESOLUTION_720P, RESOLUTION_1080P, RESOLUTION_4K],
        default_resolution=RESOLUTION_1080P,
        max_duration=KLING_4_0_MAX_DURATION_SECONDS,
        supports_keyframes=True,
        supports_voices=True,
        hdr_resolutions=(RESOLUTION_1080P, RESOLUTION_4K),
    ),
    KLING_4_0_FLASH: Kling4Capabilities(
        resolutions=[RESOLUTION_720P],
        default_resolution=RESOLUTION_720P,
        max_duration=KLING_4_0_FLASH_MAX_DURATION_SECONDS,
        supports_keyframes=False,
        supports_voices=False,
    ),
}


def is_kling_4(model_id: str) -> bool:
    return model_id in KLING_4_CAPABILITIES


def prompt_length_error(prompt: str) -> str | None:
    """Kling's mixed-language prompt limit, or None when the prompt fits."""
    english = sum(1 for char in prompt if char.isascii())
    other = len(prompt) - english
    if english / MAX_ENGLISH_PROMPT_CHARS + other / MAX_OTHER_PROMPT_CHARS <= 1:
        return None
    return (
        f"'prompt' is too long for Kling 4.0: up to {MAX_ENGLISH_PROMPT_CHARS:,} English characters or "
        f"{MAX_OTHER_PROMPT_CHARS:,} other characters, proportionally when mixed "
        f"(got {english:,} English and {other:,} other)."
    )


def parse_id_list(raw: str | None) -> list[str]:
    """Split a comma-separated list of ids, dropping blanks."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def build_settings(
    *,
    resolution: str,
    duration: int | None,
    aspect_ratio: str | None = None,
    audio: str | None = None,
    dynamic_range: str | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {"resolution": resolution}
    if duration:
        settings["duration"] = int(duration)
    if aspect_ratio:
        settings["aspect_ratio"] = aspect_ratio
    if audio:
        settings["audio"] = audio
    # sdr is Kling's default, so only an HDR request sends the field.
    if dynamic_range == DYNAMIC_RANGE_HDR:
        settings["dynamic_range"] = dynamic_range
    return settings


def image_item(item_type: str, url: str, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"type": item_type, "url": url}
    if item_id:
        item["id"] = item_id
    return item


def resolution_choice(model_id: str, current: Any) -> str:
    """``current`` when the model offers it, else the model's default resolution."""
    capabilities = KLING_4_CAPABILITIES[model_id]
    return current if current in capabilities.resolutions else capabilities.default_resolution


def supports_hdr(model_id: str, resolution: Any) -> bool:
    """Whether `model_id` can render HDR at `resolution`."""
    return resolution in KLING_4_CAPABILITIES[model_id].hdr_resolutions


def dynamic_range_choice(model_id: str, resolution: Any, current: Any) -> str:
    """`current` when HDR is offered at this model and resolution, else SDR."""
    if current == DYNAMIC_RANGE_HDR and supports_hdr(model_id, resolution):
        return DYNAMIC_RANGE_HDR
    return DYNAMIC_RANGE_SDR


def dynamic_range_error(model_id: str, resolution: Any, dynamic_range: Any) -> str | None:
    if dynamic_range in (None, "", DYNAMIC_RANGE_SDR):
        return None
    if dynamic_range != DYNAMIC_RANGE_HDR:
        return f"Unknown dynamic range '{dynamic_range}'. Use {', '.join(DYNAMIC_RANGES)}."
    if supports_hdr(model_id, resolution):
        return None
    offered = KLING_4_CAPABILITIES[model_id].hdr_resolutions
    if not offered:
        return f"{model_id} does not support HDR. Set 'dynamic_range' to {DYNAMIC_RANGE_SDR}."
    return f"{model_id} supports HDR only at {', '.join(offered)} (got '{resolution}'). Set 'dynamic_range' to {DYNAMIC_RANGE_SDR} or pick a higher resolution."


def resolution_error(model_id: str, resolution: Any) -> str | None:
    supported = KLING_4_CAPABILITIES[model_id].resolutions
    if resolution in supported:
        return None
    return f"{model_id} does not support resolution '{resolution}'. Use {', '.join(supported)}."


def validate_common(
    model_id: str, *, prompt: Any, resolution: Any, duration: Any, dynamic_range: Any = None
) -> list[ValueError]:
    """The checks every Kling 4.0 node shares: a prompt within Kling's limit, resolution, HDR and duration."""
    errors: list[ValueError] = []
    text = str(prompt or "").strip()
    if not text:
        errors.append(ValueError("A prompt is required to generate video. Enter one in 'prompt'."))
    length_error = prompt_length_error(text)
    if length_error:
        errors.append(ValueError(length_error))
    error = resolution_error(model_id, resolution)
    if error:
        errors.append(ValueError(error))
    hdr_error = dynamic_range_error(model_id, resolution, dynamic_range)
    if hdr_error:
        errors.append(ValueError(hdr_error))
    capabilities = KLING_4_CAPABILITIES[model_id]
    if duration not in capabilities.durations:
        errors.append(
            ValueError(
                f"{model_id} supports durations of {MIN_DURATION_SECONDS}-{capabilities.max_duration}s "
                f"(got {duration}s)."
            )
        )
    return errors


async def prepare_image(image_input: Any, *, label: str, node_name: str) -> str:
    """Raw base64 for a supplied image input; raises rather than dropping one that won't load.

    Dropping a frame would shift every reference id after it, so a prompt naming
    `@frame_3` would point at the wrong image while the generation still runs.
    """
    data_uri = await media.prepare_media_data_uri(image_input, kind="image", node_name=node_name)
    if not data_uri:
        msg = f"Couldn't load the image connected to '{label}'. Check that it's a valid image file or URL."
        raise ValueError(msg)
    return data_uri.split(";base64,", 1)[-1]


def kling_video_id(result_json: dict[str, Any]) -> str | None:
    """The first generated video's Kling id, from a 4.0 task (`data[].outputs[]`) or v1-v3 (`data.task_result`)."""
    data = result_json.get("data")
    if isinstance(data, dict):
        videos = (data.get("task_result") or {}).get("videos") or []
        first = videos[0] if videos else None
        return str(first["id"]) if isinstance(first, dict) and first.get("id") else None
    if not isinstance(data, list):
        return None
    for task in data:
        outputs = task.get("outputs") if isinstance(task, dict) else None
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if isinstance(output, dict) and output.get("type") == "video" and output.get("id"):
                return str(output["id"])
    return None
