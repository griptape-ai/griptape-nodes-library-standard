from __future__ import annotations

import base64

import pytest

from griptape_nodes_library.media import split_data_uri


@pytest.mark.parametrize(
    ("mime", "expected_extension"),
    [
        ("image/png", "png"),
        ("image/jpeg", "jpg"),
        ("image/webp", "webp"),
        ("image/svg+xml", "svg"),
        ("audio/mpeg", "mp3"),
        ("video/mp4", "mp4"),
    ],
)
def test_extension_comes_from_the_mime_subtype(mime: str, expected_extension: str) -> None:
    # The extension is the whole point: an uploader that names the stored object after the last path
    # segment of a data URI would otherwise use a slice of the base64 payload.
    parts = split_data_uri(f"data:{mime};base64,{base64.b64encode(b'payload').decode()}")

    assert parts is not None
    assert parts.extension == expected_extension
    assert parts.content == b"payload"


@pytest.mark.parametrize(
    "value",
    [
        "https://public.example/a.png",
        "/local/path/a.png",
        "",
        # Not base64-declared, so there are no bytes to decode.
        "data:image/png,notbase64",
        # Declared base64 but undecodable.
        "data:image/png;base64,!!!not-base64!!!",
        # No comma, so no payload at all.
        "data:image/png;base64",
    ],
)
def test_non_data_uris_and_malformed_ones_return_none(value: str) -> None:
    assert split_data_uri(value) is None


def test_line_wrapped_payload_still_decodes() -> None:
    # base64.encodebytes and MIME producers wrap at 76 columns, and the driver that reads these URIs
    # decodes that happily. Refusing it here would hand the URI to an uploader that names the stored
    # object after the payload, which is the failure this decoding exists to prevent.
    payload = base64.encodebytes(b"x" * 300).decode()
    assert "\n" in payload

    parts = split_data_uri(f"data:image/png;base64,{payload}")

    assert parts is not None
    assert parts.extension == "png"
    assert parts.content == b"x" * 300


@pytest.mark.parametrize("mime", ["image/png;charset=utf-8", "image/foo/bar", "image/x.custom", ""])
def test_extension_never_carries_path_or_parameter_punctuation(mime: str) -> None:
    # A subtype containing a separator would turn the temp file into a path into a missing directory;
    # anything non-alphanumeric ends up in the stored object's key.
    parts = split_data_uri(f"data:{mime};base64,{base64.b64encode(b'x').decode()}")

    assert parts is not None
    assert parts.extension.isalnum()
