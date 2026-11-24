"""FilterGraph joins chains into a ``-filter_complex`` value and catches label mistakes before FFmpeg does."""

import pytest

from griptape_nodes_library.utils.ffmpeg_filter_graph import FilterGraph, Stream


def test_chain_applies_filters_in_order_and_labels_the_output() -> None:
    graph = FilterGraph()
    stream = graph.chain(graph.input(0), ["format=rgba", "setsar=1"], "src")

    assert stream == Stream("src")
    assert graph.render() == "[0]format=rgba,setsar=1[src]"


def test_chains_follow_the_current_stream() -> None:
    graph = FilterGraph()
    stream = graph.chain(graph.input(0), "format=rgba", "src")
    stream = graph.chain(stream, "scale=64:64", "sized")

    assert graph.render() == "[0]format=rgba[src];[src]scale=64:64[sized]"
    assert stream.pad == "[sized]"


def test_chain_takes_several_inputs() -> None:
    graph = FilterGraph()
    graph.chain([graph.input(1), graph.input(0)], "overlay=0:0", "out")

    assert graph.render() == "[1][0]overlay=0:0[out]"


def test_split_names_each_side_branch() -> None:
    graph = FilterGraph()
    color, alpha = graph.split(graph.input(0), "color", "alpha_src", filters=["format=rgba"])
    alpha = graph.chain(alpha, "alphaextract", "alpha")
    graph.chain([color, alpha], "alphamerge", "out")

    assert graph.render() == (
        "[0]format=rgba,split[color][alpha_src];[alpha_src]alphaextract[alpha];[color][alpha]alphamerge[out]"
    )


def test_split_into_more_than_two_counts_its_outputs() -> None:
    graph = FilterGraph()
    graph.split(graph.input(0), "a", "b", "c")

    assert graph.render() == "[0]split=3[a][b][c]"


def test_split_needs_two_outputs() -> None:
    with pytest.raises(ValueError, match="at least 2 outputs"):
        FilterGraph().split(FilterGraph.input(0), "only")


def test_an_input_can_be_read_more_than_once() -> None:
    graph = FilterGraph()
    graph.chain(graph.input(1), "format=gbrp", "a")
    graph.chain(graph.input(1), "format=gray", "b")

    assert graph.render() == "[1]format=gbrp[a];[1]format=gray[b]"


def test_reusing_an_output_label_raises() -> None:
    graph = FilterGraph()
    graph.chain(graph.input(0), "format=rgba", "src")

    with pytest.raises(ValueError, match=r"\[src\] is already in use"):
        graph.chain(graph.input(1), "format=rgba", "src")


def test_repeating_a_label_within_one_split_raises() -> None:
    with pytest.raises(ValueError, match="repeat within one chain"):
        FilterGraph().split(FilterGraph.input(0), "a", "a")


def test_a_numeric_label_raises() -> None:
    with pytest.raises(ValueError, match="input index"):
        FilterGraph().chain(FilterGraph.input(0), "format=rgba", "2")


def test_consuming_an_output_twice_raises() -> None:
    graph = FilterGraph()
    stream = graph.chain(graph.input(0), "format=rgba", "src")
    graph.chain(stream, "scale=64:64", "a")

    with pytest.raises(ValueError, match=r"\[src\] is already consumed"):
        graph.chain(stream, "scale=32:32", "b")


def test_consuming_an_output_twice_in_one_chain_raises() -> None:
    graph = FilterGraph()
    stream = graph.chain(graph.input(0), "format=rgba", "src")

    with pytest.raises(ValueError, match=r"\[src\] is already consumed"):
        graph.chain([stream, stream], "blend=all_mode=screen", "out")


def test_a_rejected_chain_leaves_the_graph_unchanged() -> None:
    graph = FilterGraph()
    stream = graph.chain(graph.input(0), "format=rgba", "src")

    with pytest.raises(ValueError, match="already in use"):
        graph.chain(stream, "scale=64:64", "src")
    graph.chain(stream, "scale=64:64", "sized")

    assert graph.render() == "[0]format=rgba[src];[src]scale=64:64[sized]"


def test_a_chain_needs_a_filter() -> None:
    with pytest.raises(ValueError, match="at least one filter"):
        FilterGraph().chain(FilterGraph.input(0), [], "out")
