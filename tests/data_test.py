import numpy as np
import pandas as pd
import polars as pl
import pytest
from etoolbox.utils.testing import idfn

from patio.helpers import generate_projection_from_historical

ALL_BAS = [
    pytest.param("130"),
    pytest.param("177"),
    pytest.param("186"),
    pytest.param("195"),
    pytest.param("2"),
    pytest.param("210"),
    pytest.param("22"),
    pytest.param("531"),
    pytest.param("552"),
    pytest.param("556"),
    pytest.param("569"),
    pytest.param("57"),
    pytest.param("58"),
    pytest.param("656"),
    pytest.param("658"),
    pytest.param("AEC"),
    pytest.param("AECI"),
    pytest.param("APS"),
    pytest.param("Alaska", marks=pytest.mark.xfail),
    pytest.param("CAISO"),
    pytest.param("DUKE"),
    pytest.param("EPE"),
    pytest.param("ERCO"),
    pytest.param("ETR"),
    pytest.param("EVRG"),
    pytest.param("FMPP"),
    pytest.param("FPC"),
    pytest.param("FPL"),
    pytest.param("HECO", marks=pytest.mark.xfail),
    pytest.param("ISNE"),
    pytest.param("JEA"),
    pytest.param("LDWP"),
    pytest.param("LGEE"),
    # pytest.param("LNT"),
    pytest.param("MISO"),
    pytest.param("NBSO", marks=pytest.mark.xfail),
    pytest.param("NEVP"),
    pytest.param("NYIS"),
    pytest.param("PAC"),
    pytest.param("PJM"),
    pytest.param("PNM"),
    pytest.param("PNW"),
    pytest.param("PSCO"),
    pytest.param("SC"),
    pytest.param("SCEG"),
    pytest.param("SEC"),
    pytest.param("SOCO"),
    pytest.param("SRP"),
    pytest.param("SWPP"),
    pytest.param("TEC"),
    pytest.param("TEPC"),
    pytest.param("TVA"),
    pytest.param("WACM"),
    pytest.param("WALC"),
    pytest.param("WUAW", marks=pytest.mark.xfail),
]


class TestAssetData:
    # def test_re_dist_cols(self, asset_data):
    #     re_dist = asset_data.re_dist()
    #     missing_col = [
    #         x
    #         for x in [
    #             "plant_id_eia_l",
    #             "latitude_l",
    #             "longitude_l",
    #             "plant_id_eia_r",
    #             "latitude_r",
    #             "longitude_r",
    #             "technology_description_r",
    #             "capacity_mw_r",
    #             "distance",
    #         ]
    #         if x not in re_dist
    #     ]
    #     assert missing_col == []

    # def test_re_dist_rows(self, asset_data):
    #     assert asset_data.re_dist().shape[0] > 530_000

    def test_re_curve(self, asset_data):
        x = asset_data.re_curve(2025, 1.25)
        assert False

    def test_proposed_atb_class(self, asset_data):
        x = asset_data.proposed_atb_class(self.gens)
        assert not x.is_empty()

    def test_curve_for_site_selection(self, asset_data):
        x = asset_data.curve_for_site_selection(2025, 1.34)
        assert not x.is_empty()

    def test_re_sites_for_encom(self, asset_data):
        x = asset_data.re_sites_for_encom()
        assert not x.is_empty()

    @pytest.mark.parametrize(
        "ba_code,",
        ALL_BAS,
        ids=idfn,
    )
    def test_all_modelable_bas(self, asset_data, ba_code):
        ba_gens = asset_data.all_modelable_generators().query("final_ba_code == @ba_code")
        assert isinstance(ba_gens, pd.DataFrame)
        assert not ba_gens.empty

    @pytest.mark.parametrize(
        "ba_code,",
        ALL_BAS,
        ids=idfn,
    )
    def test_get_generator_data(self, asset_data, ba_code):
        plants, costs, bats = asset_data.get_generator_data(ba_code, "counterfactual", 1.3)
        assert not plants.empty
        assert not costs.empty

    @pytest.mark.parametrize(
        ("cost_type", "ba_code"),
        [
            (
                ("historical", *x.values)
                if x.values[0] not in ("Alaska", "HECO", "NBSO")
                else pytest.param("historical", *x.values, marks=pytest.mark.xfail)
            )
            for x in ALL_BAS
        ]
        + [
            (
                ("counterfactual", *x.values)
                if x.values[0] not in ("Alaska", "HECO", "NBSO")
                else pytest.param("historical", *x.values, marks=pytest.mark.xfail)
            )
            for x in ALL_BAS
        ],
        ids=idfn,
    )
    def test_cost_data(self, asset_data, cost_type, ba_code):
        if cost_type == "historical":
            cost_data = asset_data.historical_cost().query("final_ba_code == @ba_code")
        elif cost_type == "counterfactual":
            cost_data = asset_data.counterfactual_cost().query("final_ba_code == @ba_code")
        else:
            raise AssertionError()
        assert isinstance(cost_data, pd.DataFrame)
        assert not cost_data.empty

    @pytest.mark.parametrize(
        "ba_code",
        ALL_BAS,
        ids=idfn,
    )
    def test_ad_close_re_ba(self, asset_data, ba_code):
        """Test that close RE data is available for all BAs.

        Also make sure that the individual weights for each technology sum to one.
        """
        d = asset_data.close_re_ba(ba_code)
        assert np.all(np.isclose(d.groupby(["technology_description"]).ba_weight.sum(), 1))

    # @pytest.mark.parametrize(
    #     "ba_code",
    #     ALL_BAS,
    #     ids=idfn,
    # )
    # def test_close_re_ba_by_gen(self, asset_data, ba_code):
    #     """Test that we can generatre close RE data by generator."""
    #     df = asset_data.close_re_ba_by_gen(ba_code, drop_lcoe_na=True)
    #     assert df.query("latitude.isna()").empty

    # @pytest.mark.skip(reason="for debug")
    # def test_close_re_ba_by_gen2(self, asset_data, ba_code="NEVP"):
    #     """Test that we can generatre close RE data by generator."""
    #     df_clean = asset_data.close_re_ba_by_gen(ba_code, drop_lcoe_na=True)
    #     df_full = asset_data.close_re_ba_by_gen(ba_code, drop_lcoe_na=False)
    #     assert df_full.query("latitude.isna()").empty

    # @pytest.mark.parametrize(
    #     "pid",
    #     [
    #         2801,
    #         8222,
    #     ],
    #     ids=idfn,
    # )
    # def test_close_re(self, asset_data, pid):
    #     df = asset_data.close_re(pid)
    #     assert df.query("latitude.isna()").empty


no_counterfactual_re = (
    "166",
    "177",
    "22",
    "529",
    "552",
    "554",
    "569",
    "656",
    "658",
    "99",
    "AEC",
    "Alaska",
    "ETR",
    "HECO",
    "NBSO",
    "SEC",
)


class TestProfileData:
    ba_data_types = {
        "ba_code": str,
        "profiles": pd.DataFrame,
        "plant_data": pd.DataFrame,
        "cost_data": pd.DataFrame,
        "re_profiles": pd.DataFrame,
        "re_plant_specs": pd.DataFrame,
        "old_re_specs": pd.DataFrame,
        "old_re_profs": pd.DataFrame,
    }

    # @pytest.mark.skip
    @pytest.mark.parametrize("ba_code", ALL_BAS, ids=idfn)
    def test_get_ba_data(self, profile_data, ba_code):
        """We test this in parts to see what's wrong."""
        r = profile_data.get_ba_data(
            ba_code,
            re_by_plant=True,
        )
        for k, v in self.ba_data_types.items():
            assert k in r
            assert isinstance(r[k], v)

    def test_get_ba_data_debug(self, profile_data, profile_data_limited, ba_code="DUKE"):
        """We test this in parts to see what's wrong."""
        r = profile_data.get_ba_data(
            ba_code,
            re_by_plant=True,
        )
        r2 = profile_data_limited.get_ba_data(
            ba_code,
            re_by_plant=True,
        )
        for k, v in self.ba_data_types.items():
            assert k in r
            assert isinstance(r[k], v)

    @pytest.mark.parametrize(
        "ba_code",
        [
            "DUKE",
            "ISNE",
            "NYIS",
            "656",
        ],
        ids=idfn,
    )
    def test_get_ba_data_problem_cases(self, profile_data, ba_code):
        r = profile_data.get_ba_data(ba_code, extend_cems=True)
        for k, v in self.ba_data_types.items():
            assert k in r
            assert isinstance(r[k], v)
            if v is pd.DataFrame:
                assert not r[k].empty

    @pytest.mark.parametrize("ba_code", ALL_BAS, ids=idfn)
    def test_extended_cems(self, profile_data, ba_code):
        cems = profile_data.load_cems(ba_code, True)
        assert isinstance(cems, pd.DataFrame)
        assert not cems.empty

    def test_subplant_cems(self, profile_data):
        cems = profile_data.subplant_cems()
        assert isinstance(cems, pl.LazyFrame)
        # assert not cems.empty

    def test_ba_cems_maker(self, profile_data):
        cems = profile_data.ba_cems_maker(test=True)
        g = profile_data.ad.gens
        # assert isinstance(cems, pd.DataFrame)
        # assert not cems.empty

    def test_ba_cems_maker_extended(self, profile_data):
        cems = profile_data.ba_cems_maker(extend_cems=True, test=True)
        g = profile_data.ad.gens

    def test_cems_extender(self, profile_data):
        ba_code = "LGEE"
        plant_data = profile_data.ad.all_modelable_generators().query(
            "final_ba_code == @ba_code"
        )
        cems = profile_data.cems_extender(ba_code, plant_data)
        g = profile_data.ad.gens

    # def test_hourly_re(self, profile_data):
    #     re_meta = profile_data.ad.close_re_meta
    #     all_profs = pl.read_parquet(
    #         ROOT_PATH / "all_re.parquet",
    #         # parallel="row_groups",
    #         use_statistics=True,
    #         use_pyarrow=True,
    #         memory_map=True,
    #     ).lazy()
    #
    #     ba_code = "ISNE"
    #     ba_re_meta, ba_re_prof = profile_data._hourly_re_by_plant(
    #         ba_code, re_meta, all_profs
    #     )
    #     assert True

    def test_ba_re_maker(self, profile_data):
        cems = profile_data.ba_re_maker(test=True)
        g = profile_data.ad.gens
        # assert isinstance(cems, pd.DataFrame)
        # assert not cems.empty

    @pytest.mark.parametrize("ba_code", ALL_BAS, ids=idfn)
    def test_cems(self, profile_data, ba_code):
        cems = profile_data.load_cems(ba_code, False)
        assert isinstance(cems, pd.DataFrame)
        assert not cems.empty

    @pytest.mark.parametrize("ba_code", ALL_BAS, ids=idfn)
    def test_hourly_re_by_plant(self, profile_data, ba_code):
        specs, prof = profile_data.hourly_re_by_plant(
            ba_code,
        )
        assert isinstance(prof, pd.DataFrame)
        assert not prof.empty
        assert isinstance(specs, pd.DataFrame)
        assert not specs.empty

    def test_hourly_re_by_plant_debug(self, profile_data):
        specs, prof = profile_data.hourly_re_by_plant(
            "PSCO",
        )
        assert isinstance(prof, pd.DataFrame)
        assert not prof.empty
        assert isinstance(specs, pd.DataFrame)
        assert not specs.empty

    @pytest.mark.parametrize(
        "ba_code",
        [
            (
                pytest.param(*x.values, marks=pytest.mark.xfail)
                if x.values[0] in no_counterfactual_re
                else x
            )
            for x in ALL_BAS
        ],
        ids=idfn,
    )
    def test_counterfactual_re(self, profile_data, ba_code):
        """Test counterfactual RE data."""
        prof, specs = profile_data.counterfactual_re(
            ba_code,
            profile_data.hourly_re_by_plant(
                ba_code,
            )[1],
        )
        assert isinstance(prof, pd.DataFrame)
        assert not prof.empty
        assert isinstance(specs, pd.DataFrame)
        assert not specs.empty

    def test_really_modelable_plants(self, profile_data):
        x = profile_data.really_modelable_plants()

    # def test_remap(self, profile_data):
    #     x = profile_data.hyperlocal_re_maker()
    #     raise AssertionError()

    def test_ba_debug(self, profile_data, ba="PJM"):
        d = profile_data.get_ba_data(ba, re_by_plant=True)
        profile_data.ad.get_missing_generators(ba, d["plant_data"])
        tbls = ["re_profiles", "profiles", "cost_data", "old_re_profs"]
        d2 = {}
        for tbl in tbls:
            try:
                d2[tbl] = generate_projection_from_historical(d[tbl])
            except Exception as exc:
                raise ValueError(tbl) from exc
        raise AssertionError()

    def test_get_re_for_dl2(self, profile_data):
        x = profile_data.get_re_for_dl2()

    def test_get_re_for_dl(self, profile_data):
        x = profile_data.get_re_for_dl()

    # def test_get_hyperlocal_re_for_dl(self, profile_data):
    #     x = profile_data.get_hyperlocal_re_for_dl()

    def test_re_to_parquet2(self, profile_data):
        x = profile_data.re_to_parquet2()
