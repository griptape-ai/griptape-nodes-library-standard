from __future__ import annotations

import pytest

import griptape_nodes_library.video.seedance_human_reference_asset as asset_module
from griptape_nodes_library.proxy.provider_asset_access import ProviderAssetAccess, ProviderAssetAccessOutcome
from griptape_nodes_library.video.seedance_human_reference_asset import SeedanceHumanReferenceAsset


@pytest.fixture
def granted(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Stub the access probe to GRANTED and count how many times it is called."""
    calls = {"n": 0}

    def _probe() -> ProviderAssetAccess:
        calls["n"] += 1
        return ProviderAssetAccess(outcome=ProviderAssetAccessOutcome.GRANTED, detail="ok")

    monkeypatch.setattr(asset_module, "check_provider_asset_access", _probe)
    return calls


def test_access_not_probed_during_construction(granted: dict[str, int]) -> None:
    node = SeedanceHumanReferenceAsset(name="HRA")
    assert granted["n"] == 0, "construction must not perform the access probe (no network I/O on load)"
    assert node._access is None
    assert node._access_probed is False


def test_access_probed_lazily_on_first_interaction_then_cached(granted: dict[str, int]) -> None:
    node = SeedanceHumanReferenceAsset(name="HRA")

    node.set_parameter_value("asset_kind", "Image")
    assert granted["n"] == 1, "first interaction should probe access once"

    node.set_parameter_value("asset_kind", "Video")
    assert granted["n"] == 1, "subsequent interactions should use the cached result"


def test_validate_before_run_blocks_on_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        asset_module,
        "check_provider_asset_access",
        lambda: ProviderAssetAccess(outcome=ProviderAssetAccessOutcome.DENIED, detail="request access from Foundry"),
    )
    node = SeedanceHumanReferenceAsset(name="HRA")

    exceptions = node.validate_before_node_run()
    assert exceptions, "a denied org must block the run"
    assert "request access from Foundry" in str(exceptions[0])


@pytest.mark.parametrize(
    "outcome",
    [ProviderAssetAccessOutcome.GRANTED, ProviderAssetAccessOutcome.INDETERMINATE],
)
def test_validate_before_run_does_not_block_unless_denied(
    monkeypatch: pytest.MonkeyPatch, outcome: ProviderAssetAccessOutcome
) -> None:
    monkeypatch.setattr(
        asset_module,
        "check_provider_asset_access",
        lambda: ProviderAssetAccess(outcome=outcome, detail="..."),
    )
    node = SeedanceHumanReferenceAsset(name="HRA")
    assert node.validate_before_node_run() is None
