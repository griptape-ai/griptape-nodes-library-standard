from types import SimpleNamespace
from typing import Any

import pytest
from griptape.artifacts import AudioArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.audio import trim_audio
from griptape_nodes_library.audio.trim_audio import TrimAudio, build_audio_trim_cmd, probe_audio_duration
from griptape_nodes_library.media import local_files


def _node(**values: Any) -> TrimAudio:
    node = TrimAudio.__new__(TrimAudio)
    node.name = "trim_audio"
    node.get_parameter_value = values.get  # type: ignore[method-assign]
    return node


def test_trim_cmd_with_end() -> None:
    cmd = build_audio_trim_cmd("ffmpeg", "in.mp3", 1.5, 4.0, "out.wav")
    assert cmd[cmd.index("-ss") + 1] == "1.500"
    assert cmd[cmd.index("-to") + 1] == "4.000"
    assert cmd[-1] == "out.wav"


def test_trim_cmd_zero_end_keeps_rest_of_clip() -> None:
    assert "-to" not in build_audio_trim_cmd("ffmpeg", "in.mp3", 1.5, 0.0, "out.wav")


def test_probe_duration_is_none_when_ffprobe_is_missing() -> None:
    assert probe_audio_duration("/no/such/ffprobe", "in.mp3") is None


def test_validate_requires_audio() -> None:
    errors = _node(audio=None).validate_before_node_run()
    assert errors
    assert isinstance(errors[0], NodeError)
    assert "Connect audio" in str(errors[0])


def test_validate_rejects_raw_audio_bytes() -> None:
    errors = _node(audio=AudioArtifact(value=b"RIFF", format="wav")).validate_before_node_run()
    assert errors
    assert "raw audio data" in str(errors[0])


def test_validate_rejects_end_before_start() -> None:
    errors = _node(audio=AudioUrlArtifact("http://x/a.wav"), start_time=5.0, end_time=2.0).validate_before_node_run()
    assert errors
    assert "'End Time' must be greater" in str(errors[0])


def test_validate_accepts_zero_end() -> None:
    assert (
        _node(audio=AudioUrlArtifact("http://x/a.wav"), start_time=5.0, end_time=0.0).validate_before_node_run() is None
    )


def test_widget_url_is_empty_without_audio() -> None:
    assert _node()._resolve_audio_url("") == ""


def test_widget_url_passes_http_through() -> None:
    assert _node()._resolve_audio_url("https://x/a.wav") == "https://x/a.wav"


def test_widget_url_is_empty_when_static_url_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_files.GriptapeNodes, "handle_request", lambda *_: SimpleNamespace(result_details=""))
    monkeypatch.setattr(local_files.File, "resolve", lambda self: "/abs/a.wav")
    assert _node()._resolve_audio_url("{outputs}/a.wav") == ""


def test_raw_audio_bytes_become_a_data_uri() -> None:
    assert _node()._audio_location(AudioArtifact(value=b"RIFF", format="wav")).startswith("data:audio/")


def test_audio_location_reads_artifacts_dicts_and_strings() -> None:
    node = _node()
    assert node._audio_location(AudioUrlArtifact("https://x/a.wav")) == "https://x/a.wav"
    assert node._audio_location({"type": "AudioUrlArtifact", "value": "{outputs}/a.wav"}) == "{outputs}/a.wav"
    assert node._audio_location("/abs/a.wav") == "/abs/a.wav"
    assert node._audio_location(None) == ""


def test_start_past_end_of_clip_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node._output_file = SimpleNamespace(build_file=lambda: SimpleNamespace(location="{outputs}/trimmed.wav"))  # type: ignore[assignment]
    monkeypatch.setattr(trim_audio, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(trim_audio, "probe_audio_duration", lambda *_, **__: 5.0)
    with pytest.raises(NodeError, match="only 5.000s long"):
        node._process("in.wav", 6.0, 0.0)


def _failing_node(monkeypatch: pytest.MonkeyPatch, *, failed_is_wired: bool) -> TrimAudio:
    node = _node(audio=AudioUrlArtifact("clip.wav"), start_time=0.0, end_time=0.0)
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda _: "not-a-location")
    node._clear_execution_status = lambda: None  # type: ignore[method-assign]
    node._set_status_results = lambda **_: None  # type: ignore[method-assign]
    node._has_outgoing_connections = lambda _param: failed_is_wired  # type: ignore[method-assign]
    node.failure_output = object()  # type: ignore[assignment]
    return node


@pytest.mark.asyncio
async def test_unopenable_audio_raises_when_failed_is_unwired(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _failing_node(monkeypatch, failed_is_wired=False)
    with pytest.raises(NodeError, match="can't be opened"):
        await node.aprocess()


@pytest.mark.asyncio
async def test_unopenable_audio_follows_failed_when_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _failing_node(monkeypatch, failed_is_wired=True)
    await node.aprocess()


def test_long_local_clip_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda raw: raw)
    monkeypatch.setattr(trim_audio, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(trim_audio, "probe_audio_duration", lambda *_, **__: trim_audio.MAX_WAVEFORM_SECONDS + 1)
    assert _node()._is_long_local_clip("/abs/long.wav") is True


def test_short_and_remote_clips_are_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda raw: raw)
    monkeypatch.setattr(trim_audio, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(trim_audio, "probe_audio_duration", lambda *_, **__: 10.0)
    node = _node()
    assert node._is_long_local_clip("/abs/short.wav") is False
    assert node._is_long_local_clip("https://x/long.wav") is False
    assert node._is_long_local_clip("") is False


def test_long_clip_sets_a_badge_and_a_short_one_clears_it() -> None:
    node = _node()
    badges: list[Any] = []
    editor = SimpleNamespace(
        set_badge=lambda **kwargs: badges.append(kwargs),
        clear_badge=lambda: badges.append(None),
    )
    node.get_parameter_by_name = lambda _name: editor  # type: ignore[method-assign]
    node._update_long_clip_badge(long_clip=True)
    node._update_long_clip_badge(long_clip=False)
    assert badges[0]["variant"] == "warning"
    assert "waveform is not drawn" in badges[0]["message"]
    assert badges[1] is None


def test_static_server_url_for_a_local_file_is_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda _: "/abs/long.wav")
    monkeypatch.setattr(trim_audio, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(trim_audio, "probe_audio_duration", lambda *_, **__: trim_audio.MAX_WAVEFORM_SECONDS + 1)
    assert _node()._is_long_local_clip("http://localhost:8124/static/long.wav") is True


def test_remote_url_is_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda raw: raw)

    def fail(*_: Any, **__: Any) -> float:
        msg = "a remote clip must not be probed"
        raise AssertionError(msg)

    monkeypatch.setattr(trim_audio, "probe_audio_duration", fail)
    assert _node()._is_long_local_clip("https://example.com/long.wav") is False


def test_push_sends_a_new_revision_each_time_and_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    pushed: list[dict[str, Any]] = []
    node.set_parameter_value = lambda _name, value: pushed.append(value)  # type: ignore[method-assign]
    node.publish_update_to_parameter = lambda *_: None  # type: ignore[method-assign]
    node.get_parameter_by_name = lambda _name: SimpleNamespace(set_badge=lambda **_: None, clear_badge=lambda: None)  # type: ignore[method-assign]
    monkeypatch.setattr(trim_audio, "resolve_media_location", lambda raw: raw)
    audio = AudioUrlArtifact("https://x/a.wav")
    node._push_audio_to_widget(audio)
    node._push_audio_to_widget(audio)
    assert pushed[0]["audio_url"] == pushed[1]["audio_url"] == "https://x/a.wav"
    assert pushed[0]["revision"] != pushed[1]["revision"]
    assert pushed[0]["max_waveform_seconds"] == trim_audio.MAX_WAVEFORM_SECONDS
    assert pushed[0]["long_clip"] is False


def test_failed_start_check_does_not_build_the_output_file(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    built: list[bool] = []
    node._output_file = SimpleNamespace(build_file=lambda: built.append(True))  # type: ignore[assignment]
    monkeypatch.setattr(trim_audio, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(trim_audio, "probe_audio_duration", lambda *_, **__: 5.0)
    with pytest.raises(NodeError):
        node._process("in.wav", 6.0, 0.0)
    assert built == []


@pytest.mark.parametrize(
    ("output_file", "expected"),
    [("mix.mp3", ".mp3"), ("trimmed.wav", ".wav"), ("no_extension", ".wav"), ("", ".wav"), (None, ".wav")],
)
def test_output_extension_follows_the_output_file_name(output_file: str | None, expected: str) -> None:
    assert _node(output_file=output_file)._output_extension() == expected
