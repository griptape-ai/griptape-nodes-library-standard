from griptape_nodes_library.audio.trim_audio import build_audio_trim_cmd


def test_trim_cmd_with_end() -> None:
    cmd = build_audio_trim_cmd("ffmpeg", "in.mp3", 1.5, 4.0, "out.wav")
    assert cmd[cmd.index("-ss") + 1] == "1.500"
    assert cmd[cmd.index("-to") + 1] == "4.000"
    assert cmd[-1] == "out.wav"


def test_trim_cmd_zero_end_keeps_rest_of_clip() -> None:
    assert "-to" not in build_audio_trim_cmd("ffmpeg", "in.mp3", 1.5, 0.0, "out.wav")
