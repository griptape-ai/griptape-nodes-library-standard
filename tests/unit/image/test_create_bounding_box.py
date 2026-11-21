"""Tests for the CreateBoundingBox node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
    CreateStaticFileDownloadUrlResultFailure,
)

from griptape_nodes_library.image.create_bounding_box import (
    ARRAY_ORDER_PRESETS,
    BOX_PARAM_NAMES,
    DICT_ORDER_PRESETS,
    EDITOR_PARAMETERS,
    BoundingBox,
    CanvasSize,
    CreateBoundingBox,
    ImageInfo,
    box_problems,
    build_array,
    build_dict,
    parse_order,
)
from griptape_nodes_library.json.json_replace import JsonReplace


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> CreateBoundingBox:
    created = CreateBoundingBox(name="test_create_bounding_box")
    monkeypatch.setattr(created, "publish_update_to_parameter", lambda *_args, **_kwargs: None)
    return created


def set_image(node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch, width: int, height: int) -> None:
    info = ImageInfo(path="image.png", width=width, height=height, url="http://localhost/image.png")
    monkeypatch.setattr(node, "_image_info", lambda: info)


def connect_image(node: CreateBoundingBox) -> None:
    node.after_incoming_connection(MagicMock(), MagicMock(), node._get_parameter("input_image"))


def set_box(node: CreateBoundingBox, left: int, top: int, width: int, height: int) -> None:
    for name, value in (("left", left), ("top", top), ("width", width), ("height", height)):
        node.set_parameter_value(name, value)


class TestParseOrder:
    def test_default_orders_parse(self) -> None:
        assert parse_order("x1, y1, x2, y2", "array") == ["x1", "y1", "x2", "y2"]

    def test_tokens_are_trimmed_and_lowercased(self) -> None:
        assert parse_order(" X ,Y,  Width , HEIGHT ", "dict") == ["x", "y", "width", "height"]

    @pytest.mark.parametrize("text", ["", "   ", None, 5])
    def test_empty_or_non_string_is_rejected(self, text: object) -> None:
        with pytest.raises(ValueError, match="at least one token is required"):
            parse_order(text, "array")

    def test_unknown_token_is_rejected_with_accepted_tokens(self) -> None:
        with pytest.raises(ValueError, match="'depth' is not a recognised token") as excinfo:
            parse_order("x1, depth", "array")
        assert "width" in str(excinfo.value)

    def test_duplicate_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="'x1' appears more than once"):
            parse_order("x1, y1, x1", "array")

    @pytest.mark.parametrize("text", ["x1,,y1", "x1, y1,"])
    def test_empty_entry_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError, match="empty entry"):
            parse_order(text, "array")

    def test_every_preset_parses(self) -> None:
        for preset in [*ARRAY_ORDER_PRESETS, *DICT_ORDER_PRESETS]:
            assert parse_order(preset, "preset")


class TestBuilders:
    box = BoundingBox(x1=10, y1=20, x2=110, y2=70)

    def test_default_array_is_corners(self) -> None:
        assert build_array(self.box, ["x1", "y1", "x2", "y2"]) == [10, 20, 110, 70]

    def test_array_follows_custom_order(self) -> None:
        assert build_array(self.box, ["y2", "x1", "height", "width"]) == [70, 10, 50, 100]

    def test_default_dict_matches_add_bounding_boxes(self) -> None:
        result = build_dict(self.box, ["x", "y", "width", "height"])

        assert result == {"x": 10, "y": 20, "width": 100, "height": 50}

    def test_dict_keys_use_tokens_as_typed_and_in_order(self) -> None:
        result = build_dict(self.box, ["right", "left", "h"])

        assert list(result.items()) == [("right", 110), ("left", 10), ("h", 50)]


class TestBoxMath:
    canvas = CanvasSize(width=100, height=80)

    def test_valid_box_has_no_problems(self) -> None:
        assert box_problems(BoundingBox(0, 0, 100, 80), self.canvas) == []

    @pytest.mark.parametrize(
        ("box", "fragment"),
        [
            (BoundingBox(50, 0, 50, 10), "width (0) must be greater than zero"),
            (BoundingBox(0, 20, 10, 10), "height (-10) must be greater than zero"),
            (BoundingBox(-1, 0, 10, 10), "must not be negative"),
            (BoundingBox(0, 0, 101, 10), "must not exceed the canvas width (100)"),
            (BoundingBox(0, 0, 10, 81), "must not exceed the canvas height (80)"),
        ],
    )
    def test_invalid_boxes_are_described(self, box: BoundingBox, fragment: str) -> None:
        assert any(fragment in problem for problem in box_problems(box, self.canvas))


class TestDefaults:
    def test_default_canvas_and_box(self, node: CreateBoundingBox) -> None:
        assert node.get_parameter_value("canvas_width") == 1000
        assert node.get_parameter_value("canvas_height") == 1000
        assert node._read_box() == BoundingBox(250, 250, 750, 750)
        assert [node.get_parameter_value(name) for name in ("left", "top", "width", "height")] == [250, 250, 500, 500]
        coordinates = node.get_group_by_name_or_element_id("bounding_box_coordinates")
        assert coordinates is not None
        assert coordinates.ui_options["collapsed"] is False

    def test_default_orders(self, node: CreateBoundingBox) -> None:
        assert node.get_parameter_value("array_order") == "x1, y1, x2, y2"
        assert node.get_parameter_value("dict_order") == "x, y, width, height"
        assert node.get_parameter_value("array_order") in ARRAY_ORDER_PRESETS
        assert node.get_parameter_value("dict_order") in DICT_ORDER_PRESETS

    def test_default_outputs(self, node: CreateBoundingBox) -> None:
        assert node.get_parameter_value("bbox_array") == [250, 250, 750, 750]
        assert node.get_parameter_value("bbox_dict") == {"x": 250, "y": 250, "width": 500, "height": 500}
        assert node.parameter_output_values["bbox_array"] == [250, 250, 750, 750]
        assert node.parameter_output_values["bbox_dict"] == {"x": 250, "y": 250, "width": 500, "height": 500}
        assert node._get_parameter("bbox_array").type == "json"
        assert node._get_parameter("bbox_array").output_type == "json"
        assert node._get_parameter("bbox_dict").hide_property

    def test_bbox_array_is_accepted_by_json_replace(self, node: CreateBoundingBox) -> None:
        replacement_node = JsonReplace(name="test_json_replace")
        replacement_value = replacement_node.get_parameter_by_name("replacement_value")
        bbox_array = node._get_parameter("bbox_array")

        assert replacement_value is not None
        assert bbox_array.output_type in replacement_value.input_types

    def test_image_size_toggle_starts_hidden(self, node: CreateBoundingBox) -> None:
        toggle = node._get_parameter("use_image_size")

        assert toggle is not None
        assert toggle.ui_options.get("hide") is True

    @pytest.mark.parametrize(
        "name",
        ["input_image", "use_image_size", "canvas_width", "canvas_height", "left", "top", "width", "height"],
    )
    def test_requested_parameters_support_input_and_output_modes(self, node: CreateBoundingBox, name: str) -> None:
        parameter = node._get_parameter(name)

        assert ParameterMode.INPUT in parameter.allowed_modes
        assert ParameterMode.OUTPUT in parameter.allowed_modes


class TestLiveOutputs:
    def test_coordinate_edit_updates_coordinate_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("left", 100)

        assert node.parameter_output_values["left"] == 100

    def test_editing_a_coordinate_updates_both_outputs(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("left", 100)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 600, 750]
        assert node.parameter_output_values["bbox_dict"] == {"x": 100, "y": 250, "width": 500, "height": 500}

    def test_returning_to_a_valid_box_resumes_output_updates(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("width", 600)
        node.set_parameter_value("left", 800)
        node.set_parameter_value("left", 100)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 700, 750]

    def test_editor_is_synced_with_the_box_and_canvas_parameters(self, node: CreateBoundingBox) -> None:
        editor = node._get_parameter("bbox_editor")

        assert editor.ui_options["synced_parameters"] == EDITOR_PARAMETERS
        assert set(EDITOR_PARAMETERS) == {"left", "top", "width", "height", "canvas_width", "canvas_height"}
        for name in EDITOR_PARAMETERS:
            assert node._get_parameter(name) is not None

    def test_box_and_canvas_edits_leave_the_editor_value_alone(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("width", 650)
        node.set_parameter_value("canvas_width", 640)

        assert node.get_parameter_value("bbox_editor") == {"image_url": "", "image_revision": 0}

    def test_changing_array_order_updates_array_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("array_order", "width, height, x1, y1")

        assert node.parameter_output_values["bbox_array"] == [500, 500, 250, 250]

    def test_changing_dict_order_updates_dict_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("dict_order", "left, top, right, bottom")

        assert node.parameter_output_values["bbox_dict"] == {"left": 250, "top": 250, "right": 750, "bottom": 750}

    def test_invalid_order_keeps_last_good_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("left", 100)
        node.set_parameter_value("array_order", "x1, nope")
        node.set_parameter_value("left", 5)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 600, 750]
        assert node.parameter_output_values["bbox_dict"]["x"] == 5

    def test_invalid_box_keeps_last_good_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("width", 600)
        node.set_parameter_value("left", 800)

        assert node.parameter_output_values["bbox_array"] == [250, 250, 850, 750]


class TestOrderPresets:
    @pytest.mark.parametrize(
        ("name", "presets"),
        [("array_order", ARRAY_ORDER_PRESETS), ("dict_order", DICT_ORDER_PRESETS)],
    )
    def test_order_fields_offer_presets_and_custom_text(
        self, node: CreateBoundingBox, name: str, presets: list[str]
    ) -> None:
        parameter = node._get_parameter(name)

        assert parameter.ui_options["simple_dropdown"] == presets
        assert parameter.ui_options["allow_custom"] is True

        node.set_parameter_value(name, "height, width")
        assert node.get_parameter_value(name) == "height, width"


class TestCanvas:
    def test_canvas_size_must_be_positive(self, node: CreateBoundingBox) -> None:
        with pytest.raises(ValueError):
            node.set_parameter_value("canvas_width", 0)

    def test_shrinking_the_canvas_keeps_the_box_and_the_last_valid_outputs(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("canvas_width", 600)
        node.set_parameter_value("canvas_height", 700)

        assert node._read_box() == BoundingBox(250, 250, 750, 750)
        assert node.parameter_output_values["bbox_array"] == [250, 250, 750, 750]

    def test_canvas_size_limits_the_coordinate_sliders(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("canvas_width", 640)
        node.set_parameter_value("canvas_height", 480)

        limits = {name: node._get_parameter(name).ui_options["slider"]["max_val"] for name in BOX_PARAM_NAMES}
        assert limits == {"left": 639, "top": 479, "width": 640, "height": 480}


class TestImageSize:
    def test_toggle_is_ignored_without_an_image(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("use_image_size", True)

        assert node._read_canvas() == CanvasSize(1000, 1000)
        assert node.get_parameter_value("bbox_editor")["image_url"] == ""

    def test_toggle_is_shown_when_an_image_is_connected(self, node: CreateBoundingBox) -> None:
        connect_image(node)

        assert node._get_parameter("use_image_size").ui_options.get("hide") is False

    def test_toggle_ignores_an_image_that_is_not_connected(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)

        node._refresh_image_state()

        assert node._get_parameter("use_image_size").ui_options.get("hide") is True

    @pytest.mark.parametrize("dimension", ["canvas_width", "canvas_height"])
    def test_toggle_stays_visible_but_locked_while_a_canvas_dimension_is_connected(
        self, node: CreateBoundingBox, dimension: str
    ) -> None:
        connect_image(node)
        toggle = node._get_parameter("use_image_size")
        canvas_parameter = node._get_parameter(dimension)

        node.after_incoming_connection(MagicMock(), MagicMock(), canvas_parameter)
        assert toggle.ui_options.get("hide") is False
        assert not toggle.settable

        node.after_incoming_connection_removed(MagicMock(), MagicMock(), canvas_parameter)
        assert toggle.ui_options.get("hide") is False
        assert toggle.settable

    def test_toggle_is_locked_when_the_canvas_is_connected_before_the_image(self, node: CreateBoundingBox) -> None:
        node.after_incoming_connection(MagicMock(), MagicMock(), node._get_parameter("canvas_width"))

        connect_image(node)

        toggle = node._get_parameter("use_image_size")
        assert toggle.ui_options.get("hide") is False
        assert not toggle.settable

    def test_toggle_stays_locked_until_every_canvas_dimension_disconnects(self, node: CreateBoundingBox) -> None:
        connect_image(node)
        width = node._get_parameter("canvas_width")
        height = node._get_parameter("canvas_height")
        node.after_incoming_connection(MagicMock(), MagicMock(), width)
        node.after_incoming_connection(MagicMock(), MagicMock(), height)

        node.after_incoming_connection_removed(MagicMock(), MagicMock(), width)

        assert not node._get_parameter("use_image_size").settable

    def test_image_does_not_change_the_canvas_while_the_toggle_is_off(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)

        node._refresh_image_state()

        assert node._read_canvas() == CanvasSize(1000, 1000)
        assert node.get_parameter_value("bbox_editor")["image_url"] == "http://localhost/image.png"

    def test_toggle_on_uses_the_image_size_and_shows_the_image(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node._refresh_image_state()

        node.set_parameter_value("use_image_size", True)

        assert node._read_canvas() == CanvasSize(800, 600)
        assert node.get_parameter_value("bbox_editor")["image_url"] == "http://localhost/image.png"
        assert node.get_parameter_value("canvas_width") == 800
        assert node.get_parameter_value("canvas_height") == 600
        assert not node._get_parameter("canvas_width").settable
        assert not node._get_parameter("canvas_height").settable

    def test_toggle_on_keeps_the_box_and_the_last_valid_outputs(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 500, 400)

        node.set_parameter_value("use_image_size", True)

        assert node._read_box() == BoundingBox(250, 250, 750, 750)
        assert node.parameter_output_values["bbox_array"] == [250, 250, 750, 750]

    def test_toggle_off_makes_the_canvas_editable_again(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node._refresh_image_state()
        node.set_parameter_value("use_image_size", True)

        node.set_parameter_value("use_image_size", False)

        assert node._get_parameter("canvas_width").settable
        assert node._read_canvas() == CanvasSize(800, 600)
        assert node.get_parameter_value("bbox_editor")["image_url"] == "http://localhost/image.png"

    def test_connected_canvas_locks_the_toggle_and_the_dimension(self, node: CreateBoundingBox) -> None:
        node.after_incoming_connection(MagicMock(), MagicMock(), node._get_parameter("canvas_width"))

        assert not node._get_parameter("canvas_width").settable
        assert node._get_parameter("canvas_height").settable
        assert not node._get_parameter("use_image_size").settable

    def test_connecting_a_canvas_dimension_turns_the_toggle_off(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node._refresh_image_state()
        node.set_parameter_value("use_image_size", True)

        node.after_incoming_connection(MagicMock(), MagicMock(), node._get_parameter("canvas_height"))

        assert node.get_parameter_value("use_image_size") is False
        assert not node._get_parameter("canvas_height").settable

    def test_disconnecting_a_canvas_dimension_frees_it(self, node: CreateBoundingBox) -> None:
        node.after_incoming_connection(MagicMock(), MagicMock(), node._get_parameter("canvas_width"))

        node.after_incoming_connection_removed(MagicMock(), MagicMock(), node._get_parameter("canvas_width"))

        assert node._get_parameter("canvas_width").settable
        assert node._get_parameter("use_image_size").settable

    def test_removing_the_image_frees_the_canvas(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node.set_parameter_value("use_image_size", True)
        monkeypatch.setattr(node, "_image_info", lambda: None)

        node._refresh_image_state()

        assert node._get_parameter("canvas_width").settable
        assert node.get_parameter_value("bbox_editor")["image_url"] == ""


class TestImageLoading:
    @pytest.fixture
    def loads(self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        class FakeImage:
            width = 800
            height = 600

        def fake_load(path: str) -> FakeImage:
            calls.append(path)
            return FakeImage()

        monkeypatch.setattr("griptape_nodes_library.image.create_bounding_box.load_pil_from_url", fake_load)
        monkeypatch.setattr(node, "_resolve_image_url", lambda path: f"http://localhost/{path}")
        return calls

    def test_image_is_loaded_once_per_change(self, node: CreateBoundingBox, loads: list[str]) -> None:
        node.set_parameter_value("input_image", "a.png")
        node.set_parameter_value("use_image_size", True)
        node.set_parameter_value("left", 10)

        assert loads == ["a.png"]

    def test_connection_shows_the_toggle_before_the_image_loads(
        self, node: CreateBoundingBox, loads: list[str]
    ) -> None:
        connect_image(node)

        assert node._get_parameter("use_image_size").ui_options.get("hide") is False
        assert loads == []

    def test_disconnecting_hides_the_toggle(self, node: CreateBoundingBox) -> None:
        connect_image(node)
        input_image = node._get_parameter("input_image")

        node.after_incoming_connection_removed(MagicMock(), MagicMock(), input_image)

        assert node._get_parameter("use_image_size").ui_options.get("hide") is True

    def test_image_given_as_a_dict_is_loaded(self, node: CreateBoundingBox, loads: list[str]) -> None:
        node.set_parameter_value("input_image", {"type": "ImageUrlArtifact", "value": "a.png"})

        assert loads == ["a.png"]

    def test_setting_the_same_path_again_reloads_the_image(self, node: CreateBoundingBox, loads: list[str]) -> None:
        node.set_parameter_value("input_image", "a.png")
        first_revision = node.get_parameter_value("bbox_editor")["image_revision"]
        node.set_parameter_value("input_image", "a.png")

        assert loads == ["a.png", "a.png"]
        assert node.get_parameter_value("bbox_editor")["image_revision"] > first_revision

    def test_failed_load_is_not_retried(self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def failing_load(path: str) -> None:
            calls.append(path)
            msg = "unreadable"
            raise ValueError(msg)

        monkeypatch.setattr("griptape_nodes_library.image.create_bounding_box.load_pil_from_url", failing_load)

        node.set_parameter_value("input_image", "broken.png")
        node.set_parameter_value("left", 10)
        node.set_parameter_value("use_image_size", True)

        assert calls == ["broken.png"]

    def test_reading_the_canvas_does_not_load_the_image(self, node: CreateBoundingBox, loads: list[str]) -> None:
        node.parameter_values["input_image"] = "a.png"

        node._read_canvas()
        node.validate_before_node_run()

        assert loads == []

    def test_oversized_image_is_ignored(self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch) -> None:
        class HugeImage:
            width = 70_000
            height = 10

        monkeypatch.setattr(
            "griptape_nodes_library.image.create_bounding_box.load_pil_from_url", lambda _path: HugeImage()
        )

        node.set_parameter_value("input_image", "huge.png")

        assert node._image_info() is None

    def test_disconnecting_the_image_clears_it_and_frees_the_canvas(
        self, node: CreateBoundingBox, loads: list[str]
    ) -> None:
        connect_image(node)
        node.set_parameter_value("input_image", "a.png")
        node.set_parameter_value("use_image_size", True)
        assert node._get_parameter("use_image_size").ui_options.get("hide") is False

        input_image = node._get_parameter("input_image")
        node.after_incoming_connection_removed(node, input_image, input_image)

        assert node.get_parameter_value("input_image") is None
        assert node._get_parameter("use_image_size").ui_options.get("hide") is True
        assert node._get_parameter("canvas_width").settable
        assert node.get_parameter_value("bbox_editor")["image_url"] == ""


class TestResolveImageUrl:
    @pytest.fixture
    def requests(self, monkeypatch: pytest.MonkeyPatch) -> list[CreateStaticFileDownloadUrlFromPathRequest]:
        sent: list[CreateStaticFileDownloadUrlFromPathRequest] = []

        def fake_handle_request(request: CreateStaticFileDownloadUrlFromPathRequest) -> object:
            sent.append(request)
            if request.file_path.startswith("missing"):
                return CreateStaticFileDownloadUrlResultFailure(error="not found", result_details="not found")
            return CreateStaticFileDownloadUrlFromPathResultSuccess(
                url=f"http://localhost/static/{request.file_path}", result_details="ok"
            )

        monkeypatch.setattr(
            "griptape_nodes_library.image.create_bounding_box.GriptapeNodes.handle_request", fake_handle_request
        )
        return sent

    @pytest.mark.parametrize(
        "url", ["http://example.com/a.png", "https://example.com/a.png", "data:image/png;base64,AA"]
    )
    def test_urls_are_returned_without_asking_the_engine(
        self, node: CreateBoundingBox, requests: list[CreateStaticFileDownloadUrlFromPathRequest], url: str
    ) -> None:
        assert node._resolve_image_url(url) == url
        assert requests == []

    def test_path_is_handed_to_the_engine_as_given(
        self, node: CreateBoundingBox, requests: list[CreateStaticFileDownloadUrlFromPathRequest]
    ) -> None:
        assert node._resolve_image_url("{outputs}/a.png") == "http://localhost/static/{outputs}/a.png"
        assert [request.file_path for request in requests] == ["{outputs}/a.png"]

    def test_failure_falls_back_to_the_path(
        self, node: CreateBoundingBox, requests: list[CreateStaticFileDownloadUrlFromPathRequest]
    ) -> None:
        assert node._resolve_image_url("missing.png") == "missing.png"


class TestValidation:
    def validate(self, node: CreateBoundingBox) -> list[Exception]:
        return node.validate_before_node_run() or []

    def test_defaults_are_valid(self, node: CreateBoundingBox) -> None:
        assert node.validate_before_node_run() is None

    def test_negative_width_fails(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("width", -1)

        errors = self.validate(node)

        assert any("width (-1) must be greater than zero" in str(error) for error in errors)

    def test_box_outside_canvas_fails(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("width", 2000)

        errors = self.validate(node)

        assert any("must not exceed the canvas width" in str(error) for error in errors)

    @pytest.mark.parametrize("name", ["array_order", "dict_order"])
    def test_bad_order_fails(self, node: CreateBoundingBox, name: str) -> None:
        node.set_parameter_value(name, "x1, bogus")

        errors = self.validate(node)

        assert any("'bogus' is not a recognised token" in str(error) for error in errors)

    def test_process_writes_outputs(self, node: CreateBoundingBox) -> None:
        node.parameter_output_values.clear()

        node.process()

        assert node.parameter_output_values["bbox_array"] == [250, 250, 750, 750]


def test_node_metadata_retains_output_configuration(node: CreateBoundingBox) -> None:
    assert node.get_parameter_value("array_order") == "x1, y1, x2, y2"
    assert node.get_parameter_value("dict_order") == "x, y, width, height"
    assert node._get_parameter("bbox_array").output_type == "json"
    assert node._get_parameter("bbox_dict").allowed_modes == {ParameterMode.OUTPUT}


def test_dict_output_satisfies_add_bounding_boxes(node: CreateBoundingBox) -> None:
    node.process()

    assert {"x", "y", "width", "height"} <= node.parameter_output_values["bbox_dict"].keys()
