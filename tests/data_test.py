"""Test data objects."""


def test_asset_data(asset_data):
    """Test AssetData initialization."""
    assert not asset_data.all_modelable_generators().empty


def test_ba_data(profile_data):
    """Test BA data setup."""
    ba_data = profile_data.get_ba_data("569")
    for df in ("profiles", "plant_data", "cost_data", "re_plant_specs", "fuel_curve"):
        assert not ba_data[df].empty, f"{df} should not be empty"
