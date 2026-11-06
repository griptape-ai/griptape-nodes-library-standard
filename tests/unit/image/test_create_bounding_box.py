"""Tests for the CreateBoundingBox node."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode

from griptape_nodes_library.image.create_bounding_box import (
    ARRAY_ORDER_PRESETS,
    DICT_ORDER_PRESETS,
    BoundingBox,
    CanvasSize,
    CreateBoundingBox,
    ImageInfo,
    box_problems,
    build_array,
    build_dict,
    fit_box_to_canvas,
    parse_order,
)
from griptape_nodes_library.json.json_replace import JsonReplace


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> CreateBoundingBox:
    created = CreateBoundingBox(name="test_create_bounding_box")
    monkeypatch.setattr(created, "_get_locked_params", lambda: [])
    monkeypatch.setattr(created, "publish_update_to_parameter", lambda *_args, **_kwargs: None)
    return created


def set_image(node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch, width: int, height: int) -> None:
    info = ImageInfo(path="image.png", width=width, height=height, url="http://localhost/image.png")
    monkeypatch.setattr(node, "_image_info", lambda: info)


def set_box(node: CreateBoundingBox, x1: int, y1: int, x2: int, y2: int) -> None:
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
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
            (BoundingBox(50, 0, 50, 10), "x2 (50) must be greater than x1 (50)"),
            (BoundingBox(0, 20, 10, 10), "y2 (10) must be greater than y1 (20)"),
            (BoundingBox(-1, 0, 10, 10), "must not be negative"),
            (BoundingBox(0, 0, 101, 10), "must not exceed the canvas width (100)"),
            (BoundingBox(0, 0, 10, 81), "must not exceed the canvas height (80)"),
        ],
    )
    def test_invalid_boxes_are_described(self, box: BoundingBox, fragment: str) -> None:
        assert any(fragment in problem for problem in box_problems(box, self.canvas))

    def test_fit_pulls_an_overhanging_box_inside(self) -> None:
        fitted = fit_box_to_canvas(BoundingBox(-5, 10, 500, 500), self.canvas)

        assert fitted == BoundingBox(0, 10, 100, 80)

    def test_fit_repairs_an_inverted_box(self) -> None:
        fitted = fit_box_to_canvas(BoundingBox(60, 60, 20, 20), self.canvas)

        assert box_problems(fitted, self.canvas) == []


class TestDefaults:
    def test_default_canvas_and_box(self, node: CreateBoundingBox) -> None:
        assert node.get_parameter_value("canvas_width") == 1000
        assert node.get_parameter_value("canvas_height") == 1000
        assert node._read_box() == BoundingBox(250, 250, 750, 750)

    def test_default_orders(self, node: CreateBoundingBox) -> None:
        assert node.get_parameter_value("array_order") == "x1, y1, x2, y2"
        assert node.get_parameter_value("dict_order") == "x, y, width, height"

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
        ["input_image", "use_image_size", "canvas_width", "canvas_height", "x1", "y1", "x2", "y2"],
    )
    def test_requested_parameters_support_input_and_output_modes(self, node: CreateBoundingBox, name: str) -> None:
        parameter = node._get_parameter(name)

        assert ParameterMode.INPUT in parameter.allowed_modes
        assert ParameterMode.OUTPUT in parameter.allowed_modes


class TestLiveOutputs:
    def test_coordinate_edit_updates_coordinate_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x1", 100)

        assert node.parameter_output_values["x1"] == 100

    def test_editing_a_coordinate_updates_both_outputs(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x1", 100)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 750, 750]
        assert node.parameter_output_values["bbox_dict"] == {"x": 100, "y": 250, "width": 650, "height": 500}

    def test_widget_edit_updates_params_and_outputs(self, node: CreateBoundingBox) -> None:
        widget = {"left": 10, "top": 20, "width": 100, "height": 50, "locked": []}

        node.set_parameter_value("bbox_editor", widget)

        assert node._read_box() == BoundingBox(10, 20, 110, 70)
        assert node.parameter_output_values["bbox_array"] == [10, 20, 110, 70]

    def test_param_edit_pushes_the_box_to_the_widget(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x2", 900)

        widget = node.get_parameter_value("bbox_editor")
        assert (widget["left"], widget["width"]) == (250, 650)

    def test_changing_array_order_updates_array_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("array_order", "width, height, x1, y1")

        assert node.parameter_output_values["bbox_array"] == [500, 500, 250, 250]

    def test_changing_dict_order_updates_dict_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("dict_order", "left, top, right, bottom")

        assert node.parameter_output_values["bbox_dict"] == {"left": 250, "top": 250, "right": 750, "bottom": 750}

    def test_invalid_order_keeps_last_good_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x1", 100)
        node.set_parameter_value("array_order", "x1, nope")
        node.set_parameter_value("x1", 5)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 750, 750]
        assert node.parameter_output_values["bbox_dict"]["x"] == 5

    def test_mid_edit_inverted_box_keeps_last_good_output(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x1", 100)
        node.set_parameter_value("x1", 800)

        assert node.parameter_output_values["bbox_array"] == [100, 250, 750, 750]


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

    def test_shrinking_the_canvas_pulls_the_box_inside(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("canvas_width", 400)
        node.set_parameter_value("canvas_height", 300)

        assert node._read_box() == BoundingBox(250, 250, 400, 300)
        assert node.parameter_output_values["bbox_array"] == [250, 250, 400, 300]

    def test_widget_receives_the_canvas_size(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("canvas_width", 640)

        widget = node.get_parameter_value("bbox_editor")
        assert (widget["img_width"], widget["img_height"]) == (640, 1000)
        assert widget["image_url"] == ""


class TestImageSize:
    def test_toggle_is_ignored_without_an_image(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("use_image_size", True)

        assert node._read_canvas() == CanvasSize(1000, 1000)
        assert node.get_parameter_value("bbox_editor")["image_url"] == ""

    def test_toggle_is_hidden_without_an_image(self, node: CreateBoundingBox) -> None:
        node._refresh_image_state()

        assert node._get_parameter("use_image_size").ui_options.get("hide") is True

    def test_toggle_is_shown_with_an_image(self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch) -> None:
        set_image(node, monkeypatch, 800, 600)

        node._refresh_image_state()

        assert node._get_parameter("use_image_size").ui_options.get("hide") is False

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

        node.set_parameter_value("use_image_size", True)

        widget = node.get_parameter_value("bbox_editor")
        assert node._read_canvas() == CanvasSize(800, 600)
        assert widget["image_url"] == "http://localhost/image.png"
        assert (widget["img_width"], widget["img_height"]) == (800, 600)
        assert node.get_parameter_value("canvas_width") == 1000
        assert node.get_parameter_value("canvas_height") == 1000
        assert not node._get_parameter("canvas_width").settable
        assert not node._get_parameter("canvas_height").settable

    def test_toggle_on_clamps_the_box_to_the_image(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 500, 400)

        node.set_parameter_value("use_image_size", True)

        assert node.parameter_output_values["bbox_array"] == [250, 250, 500, 400]

    def test_toggle_off_makes_the_canvas_editable_again(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node.set_parameter_value("use_image_size", True)

        node.set_parameter_value("use_image_size", False)

        assert node._get_parameter("canvas_width").settable
        assert node._read_canvas() == CanvasSize(1000, 1000)
        widget = node.get_parameter_value("bbox_editor")
        assert (widget["img_width"], widget["img_height"]) == (1000, 1000)
        assert widget["image_url"] == "http://localhost/image.png"

    def test_removing_the_image_frees_the_canvas(
        self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_image(node, monkeypatch, 800, 600)
        node.set_parameter_value("use_image_size", True)
        monkeypatch.setattr(node, "_image_info", lambda: None)

        node._refresh_image_state()

        assert node._get_parameter("canvas_width").settable
        assert node._get_parameter("use_image_size").ui_options.get("hide") is True
        assert node.get_parameter_value("bbox_editor")["image_url"] == ""


class TestLockedCoordinates:
    def test_widget_edit_skips_connected_params(self, node: CreateBoundingBox, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(node, "_get_locked_params", lambda: ["x1"])

        node.set_parameter_value("bbox_editor", {"left": 10, "top": 20, "width": 100, "height": 50, "locked": ["x1"]})

        assert node.get_parameter_value("x1") == 250
        assert node.get_parameter_value("y1") == 20


class TestValidation:
    def validate(self, node: CreateBoundingBox) -> list[Exception]:
        return node.validate_before_node_run() or []

    def test_defaults_are_valid(self, node: CreateBoundingBox) -> None:
        assert node.validate_before_node_run() is None

    def test_inverted_box_fails(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x1", 800)

        errors = self.validate(node)

        assert len(errors) == 1
        assert "x2 (750) must be greater than x1 (800)" in str(errors[0])

    def test_box_outside_canvas_fails(self, node: CreateBoundingBox) -> None:
        node.set_parameter_value("x2", 2000)

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


def test_dict_output_satisfies_add_bounding_boxes(node: CreateBoundingBox) -> None:
    node.process()

    assert {"x", "y", "width", "height"} <= node.parameter_output_values["bbox_dict"].keys()
