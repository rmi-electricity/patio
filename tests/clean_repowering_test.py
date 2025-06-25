import pytest

from patio.model import BAs


@pytest.mark.script_launch_mode("inprocess")
def test_patio_entry_point_i(script_runner):
    """Test ``cepm`` entry point."""
    ret = script_runner.run(["patio", "-b", "57", "-l"], print_result=True)
    assert ret.success


@pytest.mark.skip(reason="not implemented")
class PostProcessingTest:
    def test_plant_list(self):
        bas = BAs.from_cloud("BAs_202311111854.zip")
        df = bas.plant_list()
        assert not df.empty()

    def test_plant_list_2025(self):
        """Test ``cepm`` entry point."""
        bas = BAs.from_file("BAs_202406121704")
        bas.plant_list()

    def test_gen_list(self):
        bas = BAs.from_file("BAs_202311111854.zip")
        df = bas.gen_list()
        assert not df.empty()

    def test_new_maps(self):
        bas = BAs.from_file("BAs_202311111854.zip")
        df = bas.make_potential_selected_maps()
        assert df

    def test_site_list(self):
        bas = BAs.from_file("BAs_202405011751.zip")
        df = bas.cr_site_list()
        assert df

    def test_unused_clean_repowering(self):
        bas = BAs.from_file("BAs_202406121704.zip")
        df = bas.unused_clean_repowering()
        assert df

    def test_econ_selected_full(self):
        bas = BAs.from_file("BAs_202309062254.zip")
        z = bas.econ_selected_full()
        f = bas.make_potential_selected_maps(owners=["SO"])

    def test_results_by_lse(self):
        # self = BAs.from_file("BAs_202311111854.zip")
        bas = BAs.from_cloud("BAs_202504270143")
        df_lc = bas.by_lse(least_cost=False)
        # df_me = bas.by_lse(least_cost=False)
        assert not df_lc.empty()
        # a = bas.econ_results()
        # f, r, *others = bas.output()
        # r1 = (
        #     r.reset_index()
        #     .groupby("re_plant_id", as_index=False)[
        #         ["latitude_nrel_site", "longitude_nrel_site"]
        #     ]
        #     .first()
        # )
        # re_sites = (
        #     a.query(
        #         "selected & re_plant_id.notna() & operating_year == 2054 & category == 'patio_clean'"
        #     )
        #     .groupby(
        #         [
        #             "ba_code",
        #             "plant_id_eia",
        #             "generator_id",
        #             "re_plant_id",
        #             "least_cost",
        #             "technology_description",
        #         ],
        #         observed=True,
        #         as_index=False,
        #     )
        #     .MW.sum()
        #     .astype({"re_plant_id": int})
        # )
        # re_sites2 = re_sites.merge(r1, on="re_plant_id", validate="m:1", how="left").merge(
        #     bas.ad.gens[
        #         [
        #             "plant_id_eia",
        #             "generator_id",
        #             "plant_name_eia",
        #             "technology_description",
        #         ]
        #     ].rename(columns={"technology_description": "fossil_technology"}),
        #     on=["plant_id_eia", "generator_id"],
        #     validate="m:1",
        #     how="left",
        # )
