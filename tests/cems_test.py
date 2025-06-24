from patio.data.cems import CEMS, _ba_cems_mp_helper, new_crosswalk


def test_eia():
    self = CEMS("al")
    a, b = self.apply_crosswalk()
    raise AssertionError()


def test_new_crosswalk():
    x = new_crosswalk()
    raise AssertionError()


def test_ba_cems(asset_data):
    bas = asset_data.all_modelable_generators().final_ba_code.unique()
    _ba_cems_mp_helper((134, bas[134]))
    raise AssertionError()


def test_cems_fuel_split(asset_data):
    al = CEMS.eia_from_p("al")
    f = al.fuel_by_type()
    s = al.count_unit_starts(freq="MS")
    raise AssertionError()


def test_cems_fuel_by_type_and_starts():
    al = CEMS.eia_from_p("al")
    out = al.fuel_by_type_and_starts(freq="MS")
    raise AssertionError()
