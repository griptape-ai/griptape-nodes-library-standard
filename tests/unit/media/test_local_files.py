from types import SimpleNamespace

import pytest
from griptape_nodes.retained_mode.events.static_file_events import CreateStaticFileDownloadUrlFromPathResultSuccess

from griptape_nodes_library.media import browser_url, local_files, validate_url


@pytest.fixture
def resolve_to_abs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_files.File, "resolve", lambda self: "/abs/clip.wav")


@pytest.mark.parametrize("location", ["", "http://x/a.png", "https://x/a.png", "data:image/png;base64,AAAA"])
def test_browser_urls_pass_through(location: str) -> None:
    assert browser_url(location) == location


@pytest.mark.usefixtures("resolve_to_abs")
def test_local_file_gets_the_static_server_url(monkeypatch: pytest.MonkeyPatch) -> None:
    success = CreateStaticFileDownloadUrlFromPathResultSuccess.__new__(CreateStaticFileDownloadUrlFromPathResultSuccess)
    success.url = "http://localhost:8124/static/clip.wav"
    monkeypatch.setattr(local_files.GriptapeNodes, "handle_request", lambda *_: success)
    assert browser_url("{outputs}/clip.wav") == "http://localhost:8124/static/clip.wav"


@pytest.mark.usefixtures("resolve_to_abs")
def test_failed_lookup_returns_the_location_or_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_files.GriptapeNodes, "handle_request", lambda *_: SimpleNamespace(result_details="no"))
    assert browser_url("{outputs}/clip.wav") == "{outputs}/clip.wav"
    assert browser_url("{outputs}/clip.wav", fallback="") == ""


def test_lookup_error_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(_self: object) -> str:
        msg = "no project"
        raise RuntimeError(msg)

    monkeypatch.setattr(local_files.File, "resolve", boom)
    assert browser_url("{outputs}/clip.wav") == "{outputs}/clip.wav"
    assert "no project" in caplog.text


@pytest.mark.parametrize(
    ("url", "valid"),
    [
        ("https://x/a.mp4", True),
        ("file://host/a.mp4", True),
        ("/abs/a.mp4", True),
        ("relative/a.mp4", False),
        ("ftp://x/a.mp4", False),
    ],
)
def test_validate_url(url: str, *, valid: bool) -> None:
    assert validate_url(url) is valid
