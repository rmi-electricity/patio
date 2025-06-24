# import hypothesis
import numpy as np
import pytest

from patio import ROOT_PATH, Plant, Plants

# from hypothesis import given
# from hypothesis import strategies as st
#
# # from hypothesis.extra.pandas import data_frames, column, range_indexes
# from hypothesis.extra.numpy import arrays
from patio.helpers import idfn
from patio.model.ba_level import (
    BA,
    BAs,
)
from patio.model.base import (
    ScenarioConfig,
    equal_capacity,
    equal_energy,
)
from tests.const_test import TRICKY_PLANTS


# float_wo_nan = st.floats(min_value=1, max_value=4000, allow_nan=False)
# float_w_nan = st.floats(min_value=0, max_value=1, allow_nan=False)
#
#
# @given(
#     ar1=arrays(float, (5000, 1), elements=st.floats(allow_nan=False)),
#     ar2=arrays(
#         float,
#         (5000, 3),
#         elements=st.floats(min_value=0, max_value=1, allow_nan=False),
#     ),
# )
def test_energy(ar1, ar2):
    result = equal_energy(ar1, ar2)
    assert result


class TestPlants:
    @pytest.mark.parametrize(("pid", "pm", "fuel", "state", "cap"), TRICKY_PLANTS, ids=idfn)
    def test_walk_through(self, plants, pid, pm, fuel, state, cap):
        x = Plant(**plants.pd.get_dailies(pid, pm, fuel, state, capacity=cap))
        cap, *args = x.for_cap_func
        a = equal_energy.py_func(*args)
        b = equal_capacity.py_func(cap, *args)
        re = x.re_profiles.sum()
        if np.sum(a) >= cap:
            assert True
        assert x.avoided_test(b) > x.avoided_test(a)

    def test_compare_plant_years(self):
        plants = Plants.from_pkl("2022070809")
        df = plants.compare_plant_years()
        raise AssertionError()

    def test_plant(self, plants):
        modelable = plants.ad.all_modelable
        self = Plant(**plants.pd.get_dailies(**modelable[0]))
        self.optimize()
        df = self._output_incr_freq(0.5)
        raise AssertionError()


# @given(
#     ar1=arrays(float, (5000, 1), elements=st.floats(allow_nan=False)),
#     ar2=arrays(
#         float,
#         (5000, 3),
#         elements=st.floats(min_value=0, max_value=1, allow_nan=False),
#     ),
# )
# def test_zero_offshore(ar1, ar2):
#     pass


cems_extent_errors = [
    "129",  # Old Dominion Electric Coop.
    "164",  # Southwestern Electric Power Co.
    "531",  # Basin Electric Power Coop.
    "582",  # American Mun Power-Ohio, Inc
    "6",  # Appalachian Power Co.
    "74",  # Indianapolis Power & Light Co.
    "Alaska",
    "IPCO",
    "LGEE",
    "NWMT",
    "SEPA",
]
setup_errors = ["164", "Alaska", "FPC", "HECO", "SEC", "SEPA"]


@pytest.mark.parametrize("ba", setup_errors, ids=idfn)
def test_ba_setup(profile_data, ba):
    d = profile_data.get_ba_data(ba)
    config = ScenarioConfig(0.5, 0, 0.1, 4)
    self = BA(**d, scenario_configs=[config])
    # self.run_all_scens()
    assert not self.profiles.empty


run_errors = [
    "120",
    "454",
    "531",
    "556",
    "CISO",
    "CPLE",
    "ERCO",
    "FMPP",
    "GVL",
    "HGMA",
    "ISNE",
    "PJM",
    "SWPP",
]


@pytest.mark.parametrize("ba", run_errors, ids=idfn)
def test_ba(profile_data, ba):
    d = profile_data.get_ba_data(ba)
    config = ScenarioConfig(0.5, 0, 0.1, 4)
    self = BA(**d, scenario_configs=[config])
    self.run_all_scens()
    assert not self.profiles.empty


@pytest.mark.parametrize(
    "ba",
    [
        "101",
        "12",
        "129",
        "134",
        "163",
        "178",
        "186",
        "2",
        "531",
        "544",
        "554",
        "560",
        "562",
        "567",
        "579",
        "58",
        "582",
        "AEC",
        "ERCO",
        "HGMA",
        "IPCO",
        "ISNE",
        "NYIS",
        "PGE",
        "PJM",
        "PSEI",
        "SPA",
        "TEC",
        "WACM",
    ],
    ids=idfn,
)
def test_bad_re_metrics(profile_data, ba):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    self = BA(
        **d,
        scenario_configs=[
            ScenarioConfig(0.1, 0, 0, 0),
            # ScenarioConfig(0.1, 0, 0.05, 4),
            # ScenarioConfig(0.2, 0, 0, 0),
            # ScenarioConfig(0.2, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.25, 4),
            ScenarioConfig(0.5, 1, 0.1, 4),
            # ScenarioConfig(0.5, 1, 0.25, 4),
            # ScenarioConfig(0.75, 1, 0.25, 4),
            ScenarioConfig(1.0, 1, 0.25, 4),
        ],
    )
    self.run_all_scens()
    df = self.allocated_output()
    assert df[df.redispatch_mwh.isna()].empty
    assert df[df.capacity_mw.isna()].empty


def test_system_metrics(profile_data, ba="TAL"):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    self = BA(
        **d,
        scenario_configs=[
            # ScenarioConfig(0.1, 0, 0, 0),
            ScenarioConfig(0.1, 0, 0.05, 4),
            # ScenarioConfig(0.2, 0, 0, 0),
            # ScenarioConfig(0.2, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.25, 4),
            ScenarioConfig(0.5, 1, 0.1, 4),
            # ScenarioConfig(0.5, 1, 0.25, 4),
            # ScenarioConfig(0.75, 1, 0.25, 4),
            # ScenarioConfig(1.0, 1, 0.25, 4),
        ],
    )
    self.run_all_scens()
    df = self.system_output()
    assert True


def test_system_metrics2(profile_data, ba="8"):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    self = BA(
        **d,
        scenario_configs=[
            # ScenarioConfig(0.1, 0, 0, 0),
            ScenarioConfig(0.1, 0, 0.05, 4),
            # ScenarioConfig(0.2, 0, 0, 0),
            # ScenarioConfig(0.2, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.1, 4),
            # ScenarioConfig(0.5, 0, 0.25, 4),
            ScenarioConfig(0.5, 1, 0.1, 4),
            # ScenarioConfig(0.5, 1, 0.25, 4),
            # ScenarioConfig(0.75, 1, 0.25, 4),
            # ScenarioConfig(1.0, 1, 0.25, 4),
        ],
    )
    self.run_all_scens()
    df = self.system_output()
    assert True


@pytest.mark.parametrize("ba", run_errors, ids=idfn)
def test_ba2(profile_data, ba):
    d = profile_data.get_ba_data(ba)
    config = ScenarioConfig(0.5, 0, 0.5, 8)
    self = BA(**d, scenario_configs=[config])
    self.run_all_scens()
    assert not self.profiles.empty


def test_ba_debug(profile_data, ba="EVRG"):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    config = ScenarioConfig(0.5, 1, 0.5, 8, 1)
    self = BA(
        **d,
        scenario_configs=[
            ScenarioConfig(0.25, 1, 0.5, 8, 0),
            ScenarioConfig(0.25, 0, 0.5, 8, 1),
        ],
    )
    self.run_all_scens()
    df = self.full_output()
    assert not df.empty


def test_ba_io(profile_data, test_dir, ba="120"):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    config = ScenarioConfig(0.5, 1, 0.5, 8)
    self = BA(**d, scenario_configs=[config])
    file0 = test_dir / "ba0.zip"
    file1 = test_dir / "ba1.zip"
    try:
        self.to_file(file0, include_output=False)
        self1 = BA.from_file(file0)
        assert isinstance(self1, BA)
        self1.run_all_scens()
        self1.to_file(file1)
        self2 = BA.from_file(file1)
    except Exception:
        raise
    else:
        assert True
    finally:
        file0.unlink(missing_ok=True)
        file1.unlink(missing_ok=True)
    assert not self.profiles.empty


def test_ba_profile_check(profile_data, ba="134"):
    d = profile_data.get_ba_data(ba)
    self = BA(**d)
    df = self.profile_check()
    df1 = self.profile_check(["state", "prime_with_cc"], multiindex=True)
    assert not self.profiles.empty


def test_ba_re(profile_data):
    self = BAs()
    df = self.close_re(export=False)
    assert not self.req_profile.empty


def test_BAs(profile_data):
    import warnings

    name = "test"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=FutureWarning)
            self = BAs(
                name=name,
                profile_data=profile_data,
                bas=[
                    # "7",
                    # "CISO",
                    # "132",
                    # "134",
                    # "144",
                    # "148",
                    # "163",
                    # "CPLE",
                    # "DUK",
                    # "EPE",
                    "TAL"
                ],
                solar_ilr=1.3,
                data_kwargs={"re_by_plant": True},
            )
            self.run_all()
            fos, re, sys, *_ = self.output()
    except Exception:
        raise
    else:
        assert not fos.empty
        assert not re.empty
    finally:
        (ROOT_PATH / f"{name}.zip").unlink(missing_ok=True)


def test_ba_dz():
    self = BAs(
        solar_ilr=1.3,
        data_kwargs={"re_by_plant": True},
        bas=[
            "ETR",
            # "APS", "EPE"
        ],
    )
    self.run_all()
