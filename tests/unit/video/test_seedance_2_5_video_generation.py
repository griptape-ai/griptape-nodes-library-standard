from __future__ import annotations

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.traits.options import Options

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    create_provider_asset_reference,
)
from griptape_nodes_library.video.seedance_2_5_video_generation import (
    ADAPTIVE_ONLY_RATIO_CHOICES,
    ALL_DURATION_CHOICES,
    ALL_RATIO_CHOICES,
    MAX_REFERENCE_AUDIO,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MAX_TOTAL_REFERENCE_ASSETS,
    RESOLUTION_CHOICES,
    SEEDANCE_2_5_MODEL_ID,
    SMART_DURATION,
    TASK_CONSTRAINTS,
    OmniReferenceTaskType,
    Seedance25VideoGeneration,
    SeedanceTask,
)


def _parameter_list_by_name(node: Seedance25VideoGeneration, parameter_name: str) -> ParameterList:
    return next(
        parameter
        for parameter in node.parameters
        if isinstance(parameter, ParameterList) and parameter.name == parameter_name
    )


def _set_parameter_list_values(node: Seedance25VideoGeneration, parameter_name: str, values: list[object]) -> None:
    parameter_list = _parameter_list_by_name(node, parameter_name)
    parameter_list.clear_list()
    for value in values:
        child = parameter_list.add_child_parameter()
        node.set_parameter_value(child.name, value)


def _parameter_by_name(node: Seedance25VideoGeneration, parameter_name: str):
    return next(parameter for parameter in node.parameters if parameter.name == parameter_name)


def _option_choices(node: Seedance25VideoGeneration, parameter_name: str) -> list:
    parameter = _parameter_by_name(node, parameter_name)
    return parameter.find_elements_by_type(Options)[0].choices


def _editing_node(name: str = "Seedance25") -> Seedance25VideoGeneration:
    """A node configured for a valid Video Editing run."""
    node = Seedance25VideoGeneration(name=name)
    node.set_parameter_value("task", SeedanceTask.VIDEO_EDITING)
    node.set_parameter_value("prompt", "Remove everyone in @Video 1 except the protagonist")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])
    return node


# --- Model id and task selector --------------------------------------------------------------


def test_model_id_is_fixed_to_seedance_2_5() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert node._get_api_model_id() == SEEDANCE_2_5_MODEL_ID
    assert node._get_catalog_model_id() == SEEDANCE_2_5_MODEL_ID


def test_task_dropdown_offers_all_five_tasks() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert _option_choices(node, "task") == [task.value for task in SeedanceTask]


def test_frame_inputs_remain_input_only() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert _parameter_by_name(node, "first_frame").allowed_modes == {ParameterMode.INPUT}
    assert _parameter_by_name(node, "last_frame").allowed_modes == {ParameterMode.INPUT}


@pytest.mark.parametrize(
    ("parameter_name", "cap"),
    [
        ("reference_images", MAX_REFERENCE_IMAGES),
        ("reference_videos", MAX_REFERENCE_VIDEOS),
        ("reference_audio", MAX_REFERENCE_AUDIO),
    ],
)
def test_reference_list_caps_match_seedance_2_5_limits(parameter_name: str, cap: int) -> None:
    # ParameterList keeps its cap private, so assert it the way the editor sees it: the list accepts
    # exactly `cap` children and refuses the next one.
    node = Seedance25VideoGeneration(name="Seedance25")
    parameter_list = _parameter_list_by_name(node, parameter_name)
    for _ in range(cap):
        parameter_list.add_child_parameter()

    with pytest.raises(ValueError, match=f"Maximum {cap} items allowed"):
        parameter_list.add_child_parameter()


def test_reference_videos_carries_upload_badge() -> None:
    # Seedance cannot read video base64, so the node uploads each video for a public URL; the
    # badge is how the user learns that before running.
    node = Seedance25VideoGeneration(name="Seedance25")
    badge = _parameter_by_name(node, "reference_videos").get_badge()
    assert badge is not None
    assert badge.variant == "cloud-upload"


# --- Parameter visibility per task -----------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "expect_frames", "expect_references"),
    [
        (SeedanceTask.TEXT_TO_VIDEO, False, False),
        (SeedanceTask.FIRST_LAST_FRAME, True, False),
        (SeedanceTask.REFERENCE_TO_VIDEO, False, True),
        (SeedanceTask.VIDEO_EDITING, False, True),
        (SeedanceTask.VIDEO_EXTENSION, False, True),
    ],
)
def test_media_input_visibility_follows_task(task: SeedanceTask, expect_frames: bool, expect_references: bool) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)

    for name in ("first_frame", "last_frame"):
        assert _parameter_by_name(node, name).ui_options.get("hide", False) is not expect_frames
    for name in ("reference_images", "reference_videos", "reference_audio"):
        assert _parameter_by_name(node, name).ui_options.get("hide", False) is not expect_references


def test_last_frame_outputs_hidden_until_requested() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert _parameter_by_name(node, "last_frame_url").ui_options.get("hide") is True
    assert _parameter_by_name(node, "last_frame_file").ui_options.get("hide") is True

    node.set_parameter_value("return_last_frame", True)
    assert _parameter_by_name(node, "last_frame_url").ui_options.get("hide", False) is False
    assert _parameter_by_name(node, "last_frame_file").ui_options.get("hide", False) is False


# --- Ratio / duration narrowing --------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [SeedanceTask.FIRST_LAST_FRAME, SeedanceTask.VIDEO_EDITING, SeedanceTask.VIDEO_EXTENSION],
)
def test_ratio_narrows_to_adaptive_for_constrained_tasks(task: SeedanceTask) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    assert _option_choices(node, "ratio") == list(ADAPTIVE_ONLY_RATIO_CHOICES)


@pytest.mark.parametrize("task", [SeedanceTask.TEXT_TO_VIDEO, SeedanceTask.REFERENCE_TO_VIDEO])
def test_ratio_offers_all_choices_for_unconstrained_tasks(task: SeedanceTask) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    assert _option_choices(node, "ratio") == list(ALL_RATIO_CHOICES)


def test_duration_narrows_to_smart_only_for_video_editing() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.VIDEO_EDITING)
    assert _option_choices(node, "duration") == [SMART_DURATION]


@pytest.mark.parametrize(
    "task",
    [
        SeedanceTask.TEXT_TO_VIDEO,
        SeedanceTask.FIRST_LAST_FRAME,
        SeedanceTask.REFERENCE_TO_VIDEO,
        SeedanceTask.VIDEO_EXTENSION,
    ],
)
def test_duration_offers_full_range_for_other_tasks(task: SeedanceTask) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    assert _option_choices(node, "duration") == list(ALL_DURATION_CHOICES)


def test_switching_task_coerces_out_of_range_ratio_and_duration() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("ratio", "16:9")
    node.set_parameter_value("duration", 12)

    node.set_parameter_value("task", SeedanceTask.VIDEO_EDITING)

    assert node.get_parameter_value("ratio") == "adaptive"
    assert node.get_parameter_value("duration") == SMART_DURATION


# --- Resolution ------------------------------------------------------------------------------


def test_resolution_dropdown_offers_every_supported_tier() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert _option_choices(node, "resolution") == ["480p", "720p", "1080p"]


@pytest.mark.parametrize("task", list(SeedanceTask))
def test_resolution_choices_do_not_depend_on_the_task(task: SeedanceTask) -> None:
    # Unlike ratio and duration, the provider documents resolution with no per-task constraint, so
    # switching tasks must never narrow the tiers or coerce the selected one.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("resolution", "1080p")
    node.set_parameter_value("task", task)

    assert _option_choices(node, "resolution") == RESOLUTION_CHOICES
    assert node.get_parameter_value("resolution") == "1080p"


# --- Validation: media matches task ----------------------------------------------------------


def test_text_to_video_rejects_reference_inputs() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    with pytest.raises(ValueError, match="reference image/video/audio inputs are only used"):
        node._validate_parameters(node._get_parameters())


def test_reference_to_video_rejects_frame_inputs() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("first_frame", "data:image/png;base64,AAA")

    with pytest.raises(ValueError, match="only used by the First/Last Frame task"):
        node._validate_parameters(node._get_parameters())


@pytest.mark.parametrize("task", [SeedanceTask.VIDEO_EDITING, SeedanceTask.VIDEO_EXTENSION])
def test_editing_and_extension_require_a_reference_video(task: SeedanceTask) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    node.set_parameter_value("prompt", "Extend and edit the scene")

    with pytest.raises(ValueError, match="requires at least one reference video"):
        node._validate_parameters(node._get_parameters())


def test_first_last_frame_accepts_frames_only() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.FIRST_LAST_FRAME)
    node.set_parameter_value("prompt", "The girl turns to face the camera")
    node.set_parameter_value("first_frame", "data:image/png;base64,AAA")
    node.set_parameter_value("last_frame", "data:image/png;base64,BBB")

    node._validate_parameters(node._get_parameters())


def test_audio_only_reference_input_is_accepted() -> None:
    # Seedance 2.5 accepts audio with no image or video, unlike the 2.0 series.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "Animate a scene to match @Audio 1")
    _set_parameter_list_values(node, "reference_audio", ["data:audio/wav;base64,AAA"])

    node._validate_parameters(node._get_parameters())


def test_reference_to_video_requires_at_least_one_reference() -> None:
    # The provider defines a reference task by the presence of a reference asset, so with none the
    # request would contradict the task the node declares.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "A quiet street at dusk")

    with pytest.raises(ValueError, match="requires at least one reference image, video, or audio"):
        node._validate_parameters(node._get_parameters())


# --- Validation: trigger keywords ------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "prompt"),
    [
        (SeedanceTask.VIDEO_EDITING, "Remove the background music from @Video 1"),
        (SeedanceTask.VIDEO_EDITING, "Replace the character in @Video 1"),
        (SeedanceTask.VIDEO_EXTENSION, "Extend @Video 1 backward"),
        (SeedanceTask.VIDEO_EXTENSION, "Continue the story after @Video 1"),
    ],
)
def test_trigger_keyword_present_passes_validation(task: SeedanceTask, prompt: str) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    node.set_parameter_value("prompt", prompt)
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    node._validate_parameters(node._get_parameters())


@pytest.mark.parametrize("task", [SeedanceTask.VIDEO_EDITING, SeedanceTask.VIDEO_EXTENSION])
def test_missing_trigger_keyword_fails_before_submission(task: SeedanceTask) -> None:
    # The provider only reports a task misclassification asynchronously, after queueing, so this
    # check is what keeps the user from waiting on an avoidable failure.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    node.set_parameter_value("prompt", "A quiet street at dusk")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    with pytest.raises(ValueError, match="Include at least one of"):
        node._validate_parameters(node._get_parameters())


def test_reference_to_video_needs_no_trigger_keyword() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "A quiet street at dusk, styled after @Image 1")
    _set_parameter_list_values(node, "reference_images", [ImageUrlArtifact("https://public.example/style.png")])

    node._validate_parameters(node._get_parameters())


# --- Validation: reference counts ------------------------------------------------------------


def test_too_many_reference_images_is_rejected() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    params = node._get_parameters()
    params["reference_images"] = [f"data:image/png;base64,{index}" for index in range(MAX_REFERENCE_IMAGES + 1)]

    with pytest.raises(ValueError, match=f"up to {MAX_REFERENCE_IMAGES} reference images"):
        node._validate_parameters(params)


def test_too_many_reference_videos_is_rejected() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    params = node._get_parameters()
    params["reference_videos"] = [f"https://public.example/{index}.mp4" for index in range(MAX_REFERENCE_VIDEOS + 1)]

    with pytest.raises(ValueError, match=f"up to {MAX_REFERENCE_VIDEOS} reference videos"):
        node._validate_parameters(params)


def test_per_kind_caps_sum_to_the_documented_total_cap() -> None:
    # The provider documents a 50-asset total alongside the per-kind caps. Because the per-kind caps
    # sum to exactly that total, enforcing them is sufficient — this pins the arithmetic so a future
    # bump to any one cap fails here instead of silently letting the total be exceeded.
    assert MAX_REFERENCE_IMAGES + MAX_REFERENCE_VIDEOS + MAX_REFERENCE_AUDIO == MAX_TOTAL_REFERENCE_ASSETS


def test_reference_lists_at_every_cap_are_accepted() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    params = node._get_parameters()
    params["reference_images"] = [f"data:image/png;base64,{index}" for index in range(MAX_REFERENCE_IMAGES)]
    params["reference_videos"] = [f"https://public.example/{index}.mp4" for index in range(MAX_REFERENCE_VIDEOS)]
    params["reference_audio"] = [f"data:audio/wav;base64,{index}" for index in range(MAX_REFERENCE_AUDIO)]

    node._validate_parameters(params)


# --- Validation: ratio and duration ----------------------------------------------------------


def test_video_editing_rejects_specified_ratio_arriving_over_a_connection() -> None:
    # The dropdown is narrowed to adaptive, but a connected value bypasses the dropdown.
    node = _editing_node()
    params = node._get_parameters()
    params["ratio"] = "16:9"

    with pytest.raises(ValueError, match="only supports ratio adaptive"):
        node._validate_parameters(params)


def test_video_editing_rejects_specified_duration() -> None:
    node = _editing_node()
    params = node._get_parameters()
    params["duration"] = 10

    with pytest.raises(ValueError, match=f"only supports duration {SMART_DURATION}"):
        node._validate_parameters(params)


def test_out_of_range_duration_is_rejected() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    params = node._get_parameters()
    params["duration"] = 45

    with pytest.raises(ValueError, match="supports duration between 4-30 seconds"):
        node._validate_parameters(params)


def test_unknown_task_is_reported() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    params = node._get_parameters()
    params["task"] = "Motion Transfer"

    with pytest.raises(ValueError, match="unknown task"):
        node._validate_parameters(params)


# --- Validation: resolution ------------------------------------------------------------------


@pytest.mark.parametrize("resolution", ["4k", "1080P", "720", "", None])
def test_unsupported_resolution_arriving_over_a_connection_is_rejected(resolution: str | None) -> None:
    # The dropdown only offers supported tiers, but a connected value bypasses the dropdown.
    node = Seedance25VideoGeneration(name="Seedance25")
    params = node._get_parameters()
    params["resolution"] = resolution

    with pytest.raises(ValueError, match="supports resolution 480p, 720p, 1080p"):
        node._validate_parameters(params)


@pytest.mark.parametrize("resolution", RESOLUTION_CHOICES)
def test_every_offered_resolution_passes_validation(resolution: str) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("resolution", resolution)

    node._validate_parameters(node._get_parameters())


# --- Payload ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_to_video_payload_carries_all_settings() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("prompt", "  A fox runs through a forest  ")
    node.set_parameter_value("resolution", "480p")
    node.set_parameter_value("ratio", "16:9")
    node.set_parameter_value("duration", 8)
    node.set_parameter_value("generate_audio", True)
    node.set_parameter_value("output_format", "mov")
    node.set_parameter_value("watermark", True)
    node.set_parameter_value("return_last_frame", True)

    payload = await node._build_payload()

    assert payload == {
        "model": SEEDANCE_2_5_MODEL_ID,
        "resolution": "480p",
        "ratio": "16:9",
        "duration": 8,
        "generate_audio": True,
        "output_format": "mov",
        "watermark": True,
        "return_last_frame": True,
        "content": [{"type": "text", "text": "A fox runs through a forest"}],
    }


@pytest.mark.asyncio
async def test_1080p_reaches_the_payload_verbatim() -> None:
    # The tier drives the provider's encoding and our billing rate, so it must pass through as-is.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value("resolution", "1080p")

    payload = await node._build_payload()

    assert payload["resolution"] == "1080p"


@pytest.mark.asyncio
async def test_first_last_frame_payload_assigns_frame_roles() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.FIRST_LAST_FRAME)
    node.set_parameter_value("prompt", "The girl turns to face the camera")
    node.set_parameter_value("first_frame", ImageUrlArtifact("https://public.example/first.png"))
    node.set_parameter_value("last_frame", ImageUrlArtifact("https://public.example/last.png"))

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "The girl turns to face the camera"},
        {"type": "image_url", "image_url": {"url": "https://public.example/first.png"}, "role": "first_frame"},
        {"type": "image_url", "image_url": {"url": "https://public.example/last.png"}, "role": "last_frame"},
    ]


@pytest.mark.asyncio
async def test_reference_payload_preserves_list_order_per_kind() -> None:
    # List order is what @Image N / @Video N / @Audio N in the prompt resolve against.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "@Image 1 walks toward @Image 2 while @Audio 1 plays")
    _set_parameter_list_values(
        node,
        "reference_images",
        [
            ImageUrlArtifact("https://public.example/one.png"),
            ImageUrlArtifact("https://public.example/two.png"),
        ],
    )
    _set_parameter_list_values(
        node,
        "reference_videos",
        [
            VideoUrlArtifact("https://public.example/first.mp4"),
            VideoUrlArtifact("https://public.example/second.mp4"),
        ],
    )
    _set_parameter_list_values(node, "reference_audio", ["https://public.example/track.mp3"])

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "@Image 1 walks toward @Image 2 while @Audio 1 plays"},
        {
            "type": "image_url",
            "image_url": {"url": "https://public.example/one.png"},
            "role": "reference_image",
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://public.example/two.png"},
            "role": "reference_image",
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/first.mp4"},
            "role": "reference_video",
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/second.mp4"},
            "role": "reference_video",
        },
        {
            "type": "audio_url",
            "audio_url": {"url": "https://public.example/track.mp3"},
            "role": "reference_audio",
        },
    ]


@pytest.mark.asyncio
async def test_non_public_reference_video_is_uploaded_for_a_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Seedance rejects video base64, so a local/non-public video must be uploaded first.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "Match the motion in @Video 1")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("/tmp/local-clip.mp4")])

    monkeypatch.setattr(
        PublicArtifactUrlParameter,
        "get_public_url_for_parameter",
        lambda self: "https://public.example/uploaded.mp4",
    )

    payload = await node._build_payload()

    assert payload["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://public.example/uploaded.mp4"},
        "role": "reference_video",
    }


@pytest.mark.asyncio
async def test_build_payload_registers_private_asset_references(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "Animate the reference portrait")
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    registered: list[tuple[str, str]] = []

    async def fake_create_provider_asset(self, public_url: str, asset_kind: str, headers: dict[str, str]) -> str:
        registered.append((public_url, asset_kind))
        return "generated-asset-id"

    monkeypatch.setattr(Seedance25VideoGeneration, "_create_provider_asset", fake_create_provider_asset)
    monkeypatch.setattr(Seedance25VideoGeneration, "_validate_api_key", lambda self: "test-key")

    payload = await node._build_payload()

    assert registered == [("https://public.example/portrait.png", ASSET_KIND_IMAGE)]
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "asset://generated-asset-id"},
        "role": "reference_image",
    }


# --- Omni reference task type ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "prompt", "expected"),
    [
        (SeedanceTask.REFERENCE_TO_VIDEO, "A street at dusk, styled after @Image 1", "reference"),
        (SeedanceTask.VIDEO_EDITING, "Remove the background music from @Video 1", "edit"),
        (SeedanceTask.VIDEO_EXTENSION, "Extend @Video 1 backward", "extend"),
    ],
)
async def test_reference_subtasks_are_declared_to_the_provider(task: SeedanceTask, prompt: str, expected: str) -> None:
    # Declaring the subtask is what moves the provider's ratio/duration checks to submission time.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    node.set_parameter_value("prompt", prompt)
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    payload = await node._build_payload()

    assert payload["omni_reference_task_type"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("task", [SeedanceTask.TEXT_TO_VIDEO, SeedanceTask.FIRST_LAST_FRAME])
async def test_non_reference_tasks_omit_the_task_type(task: SeedanceTask) -> None:
    # Neither is an omni reference task, and omitting the field is what the provider treats as auto.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", task)
    node.set_parameter_value("prompt", "A fox runs through a forest")

    payload = await node._build_payload()

    assert "omni_reference_task_type" not in payload


def test_every_task_declares_a_value_the_provider_accepts() -> None:
    accepted = {None, *OmniReferenceTaskType}
    for task, constraints in TASK_CONSTRAINTS.items():
        assert constraints.omni_reference_task_type in accepted, task


# --- Private-asset gating --------------------------------------------------------------------


def test_private_assets_inactive_when_byok_enabled() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert node._private_assets_active() is True

    node.set_parameter_value("api_key_provider", True)
    assert node._is_byok_enabled() is True
    assert node._private_assets_active() is False


def test_byok_rejects_private_asset_reference() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("api_key_provider", True)
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    with pytest.raises(ValueError, match="require Griptape authentication"):
        node._validate_parameters(node._get_parameters())


def test_mismatched_private_asset_kind_is_rejected() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/clip.mp4", asset_kind=ASSET_KIND_VIDEO)],
    )

    with pytest.raises(ValueError, match=f"a {ASSET_KIND_VIDEO} private-asset reference is connected"):
        node._validate_parameters(node._get_parameters())


def test_audio_private_asset_reference_is_accepted_on_the_audio_input() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("task", SeedanceTask.REFERENCE_TO_VIDEO)
    node.set_parameter_value("prompt", "Match the voice in @Audio 1")
    _set_parameter_list_values(
        node,
        "reference_audio",
        [create_provider_asset_reference(value="https://public.example/voice.wav", asset_kind=ASSET_KIND_AUDIO)],
    )

    node._validate_parameters(node._get_parameters())


# --- Output format / filename ----------------------------------------------------------------


def test_output_filename_extension_follows_output_format() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    assert node.get_parameter_value("output_file") == "seedance_2_5_video.mp4"

    node.set_parameter_value("output_format", "mov")
    assert node.get_parameter_value("output_file") == "seedance_2_5_video.mov"

    node.set_parameter_value("output_format", "mp4")
    assert node.get_parameter_value("output_file") == "seedance_2_5_video.mp4"


def test_custom_output_filename_is_not_rewritten_by_output_format() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("output_file", "my_shot.mp4")

    node.set_parameter_value("output_format", "mov")

    assert node.get_parameter_value("output_file") == "my_shot.mp4"


# --- Result parsing --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_result_saves_last_frame_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("return_last_frame", True)

    downloaded: list[str] = []

    async def fake_download_and_save(self, url, output_param, artifact_factory, **kwargs) -> None:
        downloaded.append(url)
        self.parameter_output_values[output_param] = artifact_factory(url, "video.mp4")

    async def fake_download_bytes(url: str) -> bytes:
        downloaded.append(url)
        return b"last-frame-bytes"

    saved_bytes: list[bytes] = []

    class FakeSavedFile:
        location = "project://seedance_2_5_last_frame.png"
        name = "seedance_2_5_last_frame.png"

    class FakeDestination:
        async def awrite_bytes(self, data: bytes) -> FakeSavedFile:
            saved_bytes.append(data)
            return FakeSavedFile()

    monkeypatch.setattr(Seedance25VideoGeneration, "_download_and_save", fake_download_and_save)
    monkeypatch.setattr(Seedance25VideoGeneration, "_download_bytes_from_url", staticmethod(fake_download_bytes))
    monkeypatch.setattr(node._last_frame_file, "build_file", lambda **kwargs: FakeDestination())

    await node._parse_result(
        {
            "content": {
                "video_url": "https://public.example/output.mp4",
                "last_frame_url": "https://public.example/output_last.png",
            }
        },
        "generation-1",
    )

    assert downloaded == ["https://public.example/output.mp4", "https://public.example/output_last.png"]
    assert saved_bytes == [b"last-frame-bytes"]
    assert node.parameter_output_values["last_frame_url"].value == "project://seedance_2_5_last_frame.png"


@pytest.mark.asyncio
async def test_parse_result_skips_last_frame_when_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance25VideoGeneration(name="Seedance25")

    async def fake_download_and_save(self, url, output_param, artifact_factory, **kwargs) -> None:
        self.parameter_output_values[output_param] = artifact_factory(url, "video.mp4")

    def fail_if_called(url: str) -> bytes:
        raise AssertionError("the last frame must not be downloaded unless return_last_frame is set")

    monkeypatch.setattr(Seedance25VideoGeneration, "_download_and_save", fake_download_and_save)
    monkeypatch.setattr(Seedance25VideoGeneration, "_download_bytes_from_url", staticmethod(fail_if_called))

    await node._parse_result(
        {
            "content": {
                "video_url": "https://public.example/output.mp4",
                "last_frame_url": "https://public.example/output_last.png",
            }
        },
        "generation-1",
    )

    assert "last_frame_url" not in node.parameter_output_values


@pytest.mark.asyncio
async def test_failed_last_frame_download_does_not_fail_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # The video already succeeded (and was billed), so a missing last frame is logged, not fatal.
    node = Seedance25VideoGeneration(name="Seedance25")
    node.set_parameter_value("return_last_frame", True)

    async def fake_download_and_save(self, url, output_param, artifact_factory, **kwargs) -> None:
        self.parameter_output_values[output_param] = artifact_factory(url, "video.mp4")

    async def failing_download(url: str) -> bytes:
        msg = "410 Gone"
        raise RuntimeError(msg)

    monkeypatch.setattr(Seedance25VideoGeneration, "_download_and_save", fake_download_and_save)
    monkeypatch.setattr(Seedance25VideoGeneration, "_download_bytes_from_url", staticmethod(failing_download))

    await node._parse_result(
        {
            "content": {
                "video_url": "https://public.example/output.mp4",
                "last_frame_url": "https://public.example/output_last.png",
            }
        },
        "generation-1",
    )

    assert node.parameter_output_values["video_url"].value == "https://public.example/output.mp4"
    assert "last_frame_url" not in node.parameter_output_values


@pytest.mark.asyncio
async def test_parse_result_reports_failure_when_no_video_url() -> None:
    node = Seedance25VideoGeneration(name="Seedance25")

    await node._parse_result({"content": {}}, "generation-1")

    assert node.parameter_output_values["video_url"] is None
    assert node.parameter_output_values["was_successful"] is False
