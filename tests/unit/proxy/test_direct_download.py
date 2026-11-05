"""Direct download: opt-in header and parameter, provider-URL media loading, the size guard and Refresh."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.files.file import File, FileWriteError
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy, FileIOFailureReason

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy import griptape_proxy_node as proxy_module
from griptape_nodes_library.proxy.griptape_proxy_node import (
    DIRECT_DOWNLOAD_HEADER,
    DIRECT_DOWNLOAD_PARAMETER,
    DIRECT_DOWNLOAD_REQUIRED_BYTES,
    GriptapeProxyNode,
)
from griptape_nodes_library.proxy.hosted_artifacts import ArtifactKind, HostedArtifact, HostedArtifactError
from griptape_nodes_library.proxy.output_size import (
    BYTES_PER_GB,
    EXR_ZIP_COMPRESSION_FACTOR,
    estimate_exr_zip_bytes,
    estimate_prores_4444_bytes,
    format_gb,
)
from griptape_nodes_library.proxy.provider_asset_access import ProxyCredential

PROXY_BASE = "https://cloud.griptape.ai/api/proxy/v2/"
GENERATION_ID = "gen-1"
VIDEO_URL = "https://provider.example/signed/video.mp4?sig=secret"
EXR_URL = "https://provider.example/signed/frames.zip?sig=secret"
HOSTED_URL = "https://account.blob.core.windows.net/generations/gen-1/artifacts/0.mp4?sig=abc"
RESULT_JSON = {"id": "job", "status": "completed", "result": {"video": VIDEO_URL, "exr": EXR_URL}}


class DirectNode(Flux2ImageGeneration):
    SUPPORTS_DIRECT_DOWNLOAD = True

    def _direct_download_result_urls(self, result_json: dict[str, Any]) -> list[tuple[str, str]]:
        result = result_json["result"]
        return [(ArtifactKind.VIDEO, result["video"]), (ArtifactKind.IMAGE, result["exr"])]


class _FakeFile:
    location = "project/out.mp4"
    name = "out.mp4"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f"{proxy_module.__name__}.resolve_proxy_credential",
        lambda _name: ProxyCredential(value="test-key", source="GT_CLOUD_API_KEY"),
    )


def _node(cls: type[GriptapeProxyNode] = DirectNode, *, direct: bool = False) -> Any:
    node = cls(name="N")
    node._proxy_base = PROXY_BASE
    if cls.SUPPORTS_DIRECT_DOWNLOAD:
        node.set_parameter_value(DIRECT_DOWNLOAD_PARAMETER, direct)
    return node


# --- parameter and header ---------------------------------------------------------------------


def test_parameter_exists_only_on_opted_in_nodes() -> None:
    assert _node(DirectNode).get_parameter_by_name(DIRECT_DOWNLOAD_PARAMETER) is not None
    assert _node(Flux2ImageGeneration).get_parameter_by_name(DIRECT_DOWNLOAD_PARAMETER) is None
    assert _node(DirectNode).get_parameter_value(DIRECT_DOWNLOAD_PARAMETER) is False


class _PostRecorder:
    headers: dict[str, str] = {}

    async def __aenter__(self) -> _PostRecorder:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, _url: str, json: Any = None, headers: dict[str, str] | None = None, timeout: int = 0) -> Any:  # noqa: ARG002, ASYNC109
        type(self).headers = dict(headers or {})

        class _Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"generation_id": GENERATION_ID}

        return _Response()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "direct", "expect_header"),
    [(DirectNode, True, True), (DirectNode, False, False), (Flux2ImageGeneration, False, False)],
)
async def test_header_only_when_opted_in_and_enabled(
    monkeypatch: pytest.MonkeyPatch, cls: type[GriptapeProxyNode], direct: bool, expect_header: bool
) -> None:
    monkeypatch.setattr(f"{proxy_module.__name__}.httpx.AsyncClient", _PostRecorder)
    node = _node(cls, direct=direct)
    node._user_auth_info = "auth-info"

    await node._submit_generation({}, {"Authorization": "Bearer k"}, "model")

    assert (_PostRecorder.headers.get(DIRECT_DOWNLOAD_HEADER) == "true") is expect_header
    assert _PostRecorder.headers["X-GTC-PROXY-AUTH-INFO"] == "auth-info"
    assert _PostRecorder.headers["Authorization"] == "Bearer k"


# --- media loading ----------------------------------------------------------------------------


def _record_downloads(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict[str, str] | None]]) -> None:
    async def fake_download(
        url: str, *, headers: dict[str, str] | None = None, follow_redirects: bool = False
    ) -> bytes:
        assert follow_redirects or url == HOSTED_URL
        calls.append((url, headers))
        return b"bytes"

    monkeypatch.setattr(GriptapeProxyNode, "_download_bytes_from_url", staticmethod(fake_download))


@pytest.mark.asyncio
async def test_direct_mode_loads_from_result_urls_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []
    _record_downloads(monkeypatch, calls)
    listed = AsyncMock(return_value=[])
    monkeypatch.setattr(f"{proxy_module.__name__}.fetch_hosted_artifacts", listed)
    node = _node(direct=True)
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)

    assert await node._load_generated_media(GENERATION_ID, kind=ArtifactKind.IMAGE) == b"bytes"

    assert calls == [(EXR_URL, None)]
    listed.assert_not_called()


@pytest.mark.asyncio
async def test_direct_mode_reuses_the_cached_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return RESULT_JSON

    gets: list[str] = []

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, url: str, **_: Any) -> _Resp:
            gets.append(url)
            return _Resp()

    monkeypatch.setattr(f"{proxy_module.__name__}.httpx.AsyncClient", _Client)
    monkeypatch.setattr(f"{proxy_module.__name__}.build_griptape_cloud_headers_async", AsyncMock(return_value={}))
    calls: list[tuple[str, dict[str, str] | None]] = []
    _record_downloads(monkeypatch, calls)
    node = _node(direct=True)

    assert await node._fetch_generation_result(GENERATION_ID) == RESULT_JSON
    await node._load_generated_media(GENERATION_ID, kind=ArtifactKind.VIDEO)
    await node._load_generated_media(GENERATION_ID, kind=ArtifactKind.IMAGE)

    assert gets == [f"{PROXY_BASE}generations/{GENERATION_ID}/result"]
    assert [url for url, _ in calls] == [VIDEO_URL, EXR_URL]


@pytest.mark.asyncio
async def test_hosted_mode_is_unchanged_and_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []
    _record_downloads(monkeypatch, calls)
    hosted = [HostedArtifact(index=0, kind="video", url=HOSTED_URL)]
    monkeypatch.setattr(f"{proxy_module.__name__}.fetch_hosted_artifacts", AsyncMock(return_value=hosted))
    node = _node(direct=False)
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)

    await node._load_generated_media(GENERATION_ID, kind=ArtifactKind.VIDEO)

    assert calls == [(HOSTED_URL, {})]
    node._fetch_generation_result.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("hosted_count", [0, 1])
async def test_hosted_mode_falls_back_to_provider_url_when_artifact_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, hosted_count: int
) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []
    _record_downloads(monkeypatch, calls)
    hosted = [HostedArtifact(index=0, kind="video", url=HOSTED_URL)][:hosted_count]
    monkeypatch.setattr(f"{proxy_module.__name__}.fetch_hosted_artifacts", AsyncMock(return_value=hosted))
    node = _node(direct=False)
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)

    with caplog.at_level("WARNING", logger="griptape_nodes"):
        await node._load_generated_media(GENERATION_ID, kind=ArtifactKind.IMAGE)

    assert calls == [(EXR_URL, None)]
    assert any("straight from the provider" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_no_fallback_for_nodes_without_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{proxy_module.__name__}.fetch_hosted_artifacts", AsyncMock(return_value=[]))
    node = _node(Flux2ImageGeneration)
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)

    with pytest.raises(HostedArtifactError):
        await node._load_generated_media(GENERATION_ID)

    node._fetch_generation_result.assert_not_called()


# --- streaming --------------------------------------------------------------------------------


class _PolicyDest:
    """Mimics ``ProjectFileDestination.awrite_bytes``: policy-aware, returns a macro-form File."""

    def __init__(self, folder: Path, policy: str) -> None:
        self.folder = folder
        self.policy = policy
        self._existing_file_policy = ExistingFilePolicy(policy)

    def resolve(self) -> str:
        return str(self.folder / "out.mp4")

    def _map_to_macro_file(self, file: File) -> File:
        return File(f"{{outputs}}/{Path(file.resolve()).name}")

    async def awrite_bytes(self, content: bytes) -> File:
        path = self.folder / "out.mp4"
        match self.policy:
            case "create_new":
                index = 1
                while path.exists():
                    path = self.folder / f"out_{index:03}.mp4"
                    index += 1
            case "fail":
                if path.exists():
                    raise FileWriteError(
                        failure_reason=FileIOFailureReason.POLICY_NO_OVERWRITE, result_details="exists"
                    )
            case _:
                pass
        path.write_bytes(content)
        return File(
            f"{{outputs}}/{path.name}",
        )


def _direct_save_node(monkeypatch: pytest.MonkeyPatch, folder: Path, policy: str) -> Any:
    async def fake_stream(url: str, path: Path, *, headers: dict[str, str] | None = None) -> None:  # noqa: ARG001
        # The download must land beside the destination, never in the system temp dir.
        assert path.parent == folder
        assert path.name.startswith(".")
        assert path.name.endswith(".partial")
        path.write_bytes(b"big")

    monkeypatch.setattr(GriptapeProxyNode, "_download_url_to_file", staticmethod(fake_stream))
    monkeypatch.setattr(File, "resolve", lambda self: str(folder / self.location.removeprefix("{outputs}/")))
    node = _node(direct=True)
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)
    node._set_status_results = lambda **_: None  # type: ignore[method-assign]

    class _OutputFile:
        def build_file(self, **_: Any) -> _PolicyDest:
            return _PolicyDest(folder, policy)

    node._output_file = _OutputFile()  # type: ignore[assignment]
    return node


async def _save(node: Any) -> bool:
    return await node._save_generated_media(GENERATION_ID, "out", lambda value, name: (value, name), kind="video")


@pytest.mark.asyncio
async def test_direct_save_create_new_makes_distinct_files_in_macro_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _direct_save_node(monkeypatch, tmp_path, "create_new")

    assert await _save(node)
    first = node.parameter_output_values["out"]
    assert await _save(node)
    second = node.parameter_output_values["out"]

    assert first[0] == "{outputs}/out.mp4"
    assert second[0] == "{outputs}/out_001.mp4"
    assert (tmp_path / "out.mp4").read_bytes() == b"big"
    assert (tmp_path / "out_001.mp4").read_bytes() == b"big"


@pytest.mark.asyncio
async def test_direct_save_overwrite_policy_overwrites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "out.mp4").write_bytes(b"old")
    node = _direct_save_node(monkeypatch, tmp_path, "overwrite")

    assert await _save(node)

    assert (tmp_path / "out.mp4").read_bytes() == b"big"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]
    assert node.parameter_output_values["out"][0] == "{outputs}/out.mp4"


@pytest.mark.asyncio
async def test_overwrite_replace_failure_leaves_old_file_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "out.mp4").write_bytes(b"old")
    node = _direct_save_node(monkeypatch, tmp_path, "overwrite")

    def broken_replace(*_: Any) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{proxy_module.__name__}.os.replace", broken_replace)

    assert not await _save(node)

    assert (tmp_path / "out.mp4").read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]


@pytest.mark.asyncio
async def test_replace_failure_removes_the_reserved_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _direct_save_node(monkeypatch, tmp_path, "create_new")

    def broken_replace(*_: Any) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{proxy_module.__name__}.os.replace", broken_replace)

    assert not await _save(node)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_download_leaves_old_file_and_no_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "out.mp4").write_bytes(b"old")
    node = _direct_save_node(monkeypatch, tmp_path, "overwrite")

    async def failing_stream(url: str, path: Path, *, headers: dict[str, str] | None = None) -> None:  # noqa: ARG001
        path.write_bytes(b"half")
        msg = "connection lost"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(GriptapeProxyNode, "_download_url_to_file", staticmethod(failing_stream))

    assert not await _save(node)

    assert (tmp_path / "out.mp4").read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]


@pytest.mark.asyncio
async def test_hosted_mode_fallback_save_streams_and_never_buffers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _direct_save_node(monkeypatch, tmp_path, "create_new")
    node.set_parameter_value(DIRECT_DOWNLOAD_PARAMETER, False)
    monkeypatch.setattr(f"{proxy_module.__name__}.fetch_hosted_artifacts", AsyncMock(return_value=[]))
    buffered = AsyncMock()
    monkeypatch.setattr(GriptapeProxyNode, "_download_bytes_from_url", staticmethod(buffered))

    assert await _save(node)

    buffered.assert_not_called()
    assert (tmp_path / "out.mp4").read_bytes() == b"big"


@pytest.mark.asyncio
async def test_direct_save_fail_policy_fails_when_file_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "out.mp4").write_bytes(b"old")
    node = _direct_save_node(monkeypatch, tmp_path, "fail")

    assert not await _save(node)

    assert (tmp_path / "out.mp4").read_bytes() == b"old"
    assert node.parameter_output_values["out"] is None


# --- redirects --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redirect_is_followed_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "provider.example":
            return httpx.Response(302, headers={"location": "https://cdn.example/file.mp4"})
        return httpx.Response(200, content=b"cdn-bytes")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        f"{proxy_module.__name__}.httpx.AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    data = await GriptapeProxyNode._download_bytes_from_url(VIDEO_URL, follow_redirects=True)

    assert data == b"cdn-bytes"
    assert len(seen) == 2
    assert all("authorization" not in request.headers for request in seen)


@pytest.mark.asyncio
async def test_streaming_download_follows_redirects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "provider.example":
            return httpx.Response(302, headers={"location": "https://cdn.example/file.mp4"})
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"cdn-bytes")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        f"{proxy_module.__name__}.httpx.AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    target = tmp_path / "f.bin"

    await GriptapeProxyNode._download_url_to_file(VIDEO_URL, target)

    assert target.read_bytes() == b"cdn-bytes"


# --- size guard -------------------------------------------------------------------------------


async def _run_begin(monkeypatch: pytest.MonkeyPatch, node: Any) -> tuple[str | None, AsyncMock]:
    class _Permitted:
        result_details = None

        def failed(self) -> bool:
            return False

    monkeypatch.setattr(f"{proxy_module.__name__}.declare_model_invocation", AsyncMock(return_value=_Permitted()))
    node._model_access = None
    build = AsyncMock(return_value={})
    node._build_payload = build
    node._get_api_model_id = lambda: "model"
    node._submit_generation = AsyncMock(return_value=GENERATION_ID)
    node.failures = []
    node._handle_failure_exception = node.failures.append
    node._set_status_results = lambda **_: None
    node._set_safe_defaults = lambda: None
    return await node._begin_generation({}), build


@pytest.mark.asyncio
async def test_guard_blocks_above_threshold_without_building_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(direct=False)
    node._estimate_output_bytes = lambda: DIRECT_DOWNLOAD_REQUIRED_BYTES + 1

    generation_id, build = await _run_begin(monkeypatch, node)
    errors = node.failures

    assert generation_id is None
    build.assert_not_called()
    node._submit_generation.assert_not_called()
    assert isinstance(errors[0], NodeError)
    message = str(errors[0])
    assert format_gb(DIRECT_DOWNLOAD_REQUIRED_BYTES + 1) in message
    assert "Direct Download" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("estimate", "direct"),
    [
        (DIRECT_DOWNLOAD_REQUIRED_BYTES, False),
        (DIRECT_DOWNLOAD_REQUIRED_BYTES * 10, True),
        (None, False),
    ],
)
async def test_guard_allows(monkeypatch: pytest.MonkeyPatch, estimate: int | None, direct: bool) -> None:
    node = _node(direct=direct)
    node._estimate_output_bytes = lambda: estimate

    generation_id, build = await _run_begin(monkeypatch, node)

    assert generation_id == GENERATION_ID
    build.assert_called_once()


@pytest.mark.asyncio
async def test_guard_is_skipped_when_the_estimator_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(direct=False)

    def broken() -> int:
        msg = "probe failed"
        raise RuntimeError(msg)

    node._estimate_output_bytes = broken

    generation_id, build = await _run_begin(monkeypatch, node)

    assert generation_id == GENERATION_ID
    build.assert_called_once()


@pytest.mark.asyncio
async def test_guard_message_omits_direct_download_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(Flux2ImageGeneration)
    node._estimate_output_bytes = lambda: DIRECT_DOWNLOAD_REQUIRED_BYTES + 1

    generation_id, build = await _run_begin(monkeypatch, node)

    assert generation_id is None
    build.assert_not_called()
    message = str(node.failures[0])
    assert "Direct Download" not in message
    assert "Reduce the resolution" in message


def test_default_estimate_is_none() -> None:
    assert _node(Flux2ImageGeneration)._estimate_output_bytes() is None


# --- Refresh ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_in_direct_mode_downloads_from_result_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []
    _record_downloads(monkeypatch, calls)
    node = _node(direct=True)
    node.parameter_output_values["generation_id"] = GENERATION_ID
    node._fetch_status_for_refresh = AsyncMock(return_value={"status": "COMPLETED"})
    node._fetch_generation_result = AsyncMock(return_value=RESULT_JSON)
    node._set_status_results = lambda **_: None  # type: ignore[method-assign]
    monkeypatch.setattr(f"{proxy_module.__name__}.build_griptape_cloud_headers_async", AsyncMock(return_value={}))

    async def parse(result_json: dict[str, Any], generation_id: str) -> None:
        await node._load_generated_media(generation_id, kind=ArtifactKind.VIDEO)

    node._parse_result = parse  # type: ignore[method-assign]
    node._direct_download_results[GENERATION_ID] = {"stale": True}

    await node._refresh_async()

    assert calls == [(VIDEO_URL, None)]
    assert node._direct_download_results == {}


# --- estimators -------------------------------------------------------------------------------


def test_prores_estimate_scales_by_pixels_and_duration() -> None:
    assert estimate_prores_4444_bytes(1920, 1080, 10) == 420_000_000
    four_k = estimate_prores_4444_bytes(3840, 2160, 10)
    assert four_k == 4 * 420_000_000
    assert four_k == pytest.approx(1_720_000_000, rel=0.03)


def test_exr_estimate_uses_half_float_rgb() -> None:
    assert estimate_exr_zip_bytes(10, 100, 50) == int(10 * 100 * 50 * 6 * EXR_ZIP_COMPRESSION_FACTOR)


@pytest.mark.parametrize("args", [(0, 10, 10), (10, 0, 10), (10, 10, 0)])
def test_exr_estimate_is_zero_for_empty_input(args: tuple[int, int, int]) -> None:
    assert estimate_exr_zip_bytes(*args) == 0


def test_prores_estimate_is_zero_for_empty_input() -> None:
    assert estimate_prores_4444_bytes(1920, 1080, 0) == 0


def test_format_gb_is_decimal() -> None:
    assert format_gb(2.5 * BYTES_PER_GB) == "2.5 GB"
