import pytest

from patio.data.cems import CEMS, _ba_cems_mp_helper, new_crosswalk


@pytest.mark.skip(reason="debug only")
class TestCEMS:
    def test_eia(self):
        self_ = CEMS("al")
        a, b = self_.apply_crosswalk()
        raise AssertionError()

    def test_new_crosswalk(self):
        x = new_crosswalk()
        raise AssertionError()

    def test_ba_cems(self, asset_data):
        bas = asset_data.all_modelable_generators().final_ba_code.unique()
        _ba_cems_mp_helper((134, bas[134]))
        raise AssertionError()

    def test_cems_fuel_split(self, asset_data):
        al = CEMS.eia_from_p("al")
        f = al.fuel_by_type()
        s = al.count_unit_starts(freq="MS")
        raise AssertionError()

    def test_cems_fuel_by_type_and_starts(self):
        al = CEMS.eia_from_p("al")
        out = al.fuel_by_type_and_starts(freq="MS")
        raise AssertionError()
