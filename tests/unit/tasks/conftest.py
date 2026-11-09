import pytest


@pytest.fixture(autouse=True)
def _open_workflow(open_workflow: None) -> None:  # noqa: ARG001
    """Every task test runs a node built straight from the library."""
