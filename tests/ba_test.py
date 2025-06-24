import json
import shutil
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from patio.constants import ROOT_PATH
from patio.data.asset_data import irp_data
from patio.model.ba_level import BA, BAs
from patio.model.base import ScenarioConfig, equal_energy
from patio.model.colo_core import (
    combine_runs,
    model_colo_config,
    set_timeout,
    set_workers,
)
from patio.model.colo_lp import Info, Model


def test_equal_energy(profile_data, ba="57"):
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    d_ = profile_data.one_plant(6052, d)
    x = equal_energy.py_func(
        d_["profiles"].sum(axis=1),
        d_["re_profiles"].groupby(level=1, axis=1).mean(),
        max_re_cap=np.array([500.0, 200.0]),
    )
    assert x.shape == (2,)


def test_ba_debug_iter(
    profile_data,
    # ba="22",
    ba="57",
):
    # [
    #     x.casefold().replace(" ", "_")
    #     for x in profile_data.ad.gens.technology_description.unique()
    # ]
    scenarios = [
        # ScenarioConfig(re_energy=0.4, storage_li_pct=0.0),
        # ScenarioConfig(re_energy=0.2, storage_li_pct=0.25),
        ScenarioConfig(re_energy=0.8, storage_li_pct=0.25),
    ]
    d = profile_data.get_ba_data(
        ba,
        re_by_plant=True,
        re_limits_dispatch=True,
        max_re_distance=45.0,
        colo_techs=["Natural Gas Fired Combustion Turbine"],
    )
    # d = DataZip(Path.home() / "patio_data/BAs_202310120010_data.zip")[ba]
    d["re_limits_dispatch"] = True
    self = BA(
        **d,
        colo_method="iter",
        scenario_configs=scenarios,
    )
    # self.make_dm_mini(BytesIO())
    self.run_all_scens()


def test_ba_debug(
    profile_data,
    # ba="22",
    ba="57",
):
    # [
    #     x.casefold().replace(" ", "_")
    #     for x in profile_data.ad.gens.technology_description.unique()
    # ]
    from patio.model.ba_scenario import set_workers

    set_workers(1)
    scenarios = [
        # ScenarioConfig(re_energy=0.4, storage_li_pct=0.0),
        # ScenarioConfig(re_energy=0.2, storage_li_pct=0.25),
        ScenarioConfig(re_energy=0.6, storage_li_pct=0.25),
    ]
    d = profile_data.get_ba_data(
        ba,
        re_by_plant=True,
        re_limits_dispatch=True,
        max_re_distance=45.0,
        colo_techs=["Natural Gas Fired Combustion Turbine"],
    )
    # d = DataZip(Path.home() / "patio_data/BAs_202310120010_data.zip")[ba]
    d["re_limits_dispatch"] = True
    self = BA(
        **d,
        colo_method="direct",
        scenario_configs=[],
    )
    # self.make_dm_mini(BytesIO())
    self.run_all_scens()
    df = self.full_output()
    sys_df = self.system_output()
    re_df = self.allocated_output()
    assert not df.empty
    to_test = (
        re_df.reset_index()
        .query("datetime.dt.year == 2030")
        .pivot_table(
            index=["scenario"],
            columns=["ba_code", "re_plant_id", "re_type"],
            values="capacity_mw",
            aggfunc="sum",
        )
        .fillna(0.0)
    )
    assert all(to_test[x].is_monotonic_increasing for x in to_test)
    # profile_data.ad.temp_ccs_conversion()


def test_dumb():
    o = BAs.from_file("BAs_202407301648")
    q = o.output()


def test_ba_by_plant_debug(profile_data, ba="529"):
    scenarios = [
        ScenarioConfig(re_energy=0.05, storage_li_pct=0.4),
        ScenarioConfig(
            re_energy=0.5,
            storage_li_pct=0.4,
        ),
        ScenarioConfig(
            re_energy=1.5,
            storage_li_pct=0.4,
        ),
    ]
    d = profile_data.get_ba_data(ba, re_by_plant=True)
    d_ = profile_data.one_plant(
        d["plant_data"]
        .query(
            "technology_description == 'Conventional Steam Coal' & category == 'existing_fossil'"
        )
        .index.get_level_values("plant_id_eia")
        .unique()[0],
        d,
    )
    self = BA(
        **d_,
        scenario_configs=scenarios,
    )
    self.run_all_scens()
    df = self.full_output()
    sys_df = self.system_output()
    re_df = self.allocated_output()
    assert not df.empty


@pytest.mark.script_launch_mode("inprocess")
def test_patio_entry_point_colo_data(script_runner):
    """Test ``cepm`` entry point."""
    ret = script_runner.run(["patio -b 57 -c NGCT -l"], print_result=True)
    assert ret.success


@pytest.mark.script_launch_mode("inprocess")
def test_patio_colo_entry_point(script_runner):
    """Test ``cepm`` entry point."""
    ret = script_runner.run(
        ["patio-colo", "-l", "-k", "-w", "1", "-p", "56502"], print_result=True
    )
    assert ret.success


@pytest.mark.script_launch_mode("inprocess")
def test_patio_colo_entry_point_saved(script_runner):
    """Test ``cepm`` entry point."""
    ret = script_runner.run(
        ["patio-colo", "-l", "-k", "-w", "1", "-p", "56502", "-S", "colo_202505202334"],
        print_result=True,
    )
    assert ret.success


def test_bas_colo(bas=("57",)):
    # scenarios = [
    #     ScenarioConfig(re_energy=0.05, storage_li_pct=0.4),
    #     ScenarioConfig(
    #         re_energy=0.5,
    #         storage_li_pct=0.4,
    #     ),
    #     ScenarioConfig(
    #         re_energy=1.5,
    #         storage_li_pct=0.4,
    #     ),
    # ]
    set_workers(1)
    set_timeout(480)
    patio = BAs(
        solar_ilr=1.34,
        data_kwargs={
            "re_by_plant": True,
            "include_figs": False,
            "re_limits_dispatch": True,
            "colo_techs": [
                "Natural Gas Fired Combustion Turbine",
                "Natural Gas Fired Combined Cycle",
                "Nuclear",
                "Solar Photovoltaic",
            ],
        },
        bas=list(bas),
        scenario_configs=[],
        regime="reference",
        # by_plant=True,
    )
    patio.run_all()
    a, b, c, *_ = patio.output()
    patio.write_output()

    # df = patio.fuel_curve_comparison()
    assert True


def test_bas(bas=("58",)):
    # scenarios = [
    #     ScenarioConfig(re_energy=0.05, storage_li_pct=0.4),
    #     ScenarioConfig(
    #         re_energy=0.5,
    #         storage_li_pct=0.4,
    #     ),
    #     ScenarioConfig(
    #         re_energy=1.5,
    #         storage_li_pct=0.4,
    #     ),
    # ]
    set_workers(1)
    set_timeout(6000)
    patio = BAs(
        solar_ilr=1.34,
        data_kwargs={
            "re_by_plant": True,
            "include_figs": False,
            "re_limits_dispatch": True,
            # "colo_techs": ["Natural Gas Fired Combustion Turbine"],
        },
        bas=list(bas),
        # scenario_configs=[],
        pudl_release="v2025.2.0",
        regime="reference",
        # by_plant=True,
    )
    patio.run_all()
    a, b, c, *_ = patio.output()
    patio.write_output()

    # df = patio.fuel_curve_comparison()
    assert True


def test_bas_by_plant(bas=("529",)):
    scenarios = [
        ScenarioConfig(re_energy=0.05, storage_li_pct=0.4),
        ScenarioConfig(
            re_energy=0.5,
            storage_li_pct=0.4,
        ),
        ScenarioConfig(
            re_energy=1.5,
            storage_li_pct=0.4,
        ),
    ]
    patio = BAs(
        solar_ilr=1.3,
        data_kwargs={"re_by_plant": True, "include_figs": False},
        bas=list(bas),
        scenario_configs=scenarios,
        by_plant=True,
    )
    patio.run_all()
    a, b, c, *_ = patio.output()
    assert True


@pytest.mark.script_launch_mode("inprocess")
def test_patio_entry_point(script_runner):
    """Test ``cepm`` entry point."""
    ret = script_runner.run(
        ["patio", "-b", "57", "-c", "NGCT", "--colo-only"], print_result=True
    )
    assert ret.success


PDATA = {
    55061: {
        "ba": "57",
        "pid": 55061,
        "gens": ["GTG1", "GTG2", "GTG3", "GTG4", "GTG5", "GTG6"],
        "tech": "nggt",
        "status": "existing",
        "cap": 1099.1999816894531,
        # "years": [2031, 2034],
    },
    7838: {
        "ba": "186",
        "pid": 7838,
        "gens": ["1", "2", "3", "4"],
        "tech": "nggt",
        "status": "existing",
        "cap": 705.5,
        # "years": [2033, 2037],
    },
    7315: {
        "ba": "CAISO",
        "pid": 7315,
        "gens": ["2", "3", "4"],
        "tech": "nggt",
        "status": "existing",
        "cap": 174.0,
        # "years": [2033, 2029],
    },
    634: {
        "ba": "FPC",
        "pid": 634,
        "gens": ["4AGT", "4BGT", "4CGT", "4DGT", "4ST"],
        "tech": "ngcc",
        "status": "existing",
        "cap": 1254.0,
        # "years": [2038, 2031],
    },
    56241: {
        "ba": "MISO",
        "pid": 56241,
        "gens": ["UNT1", "UNT2"],
        "tech": "nggt",
        "status": "existing",
        "cap": 346.79998779296875,
        # "years": [2030, 2035],
    },
    7845: {
        "ba": "TVA",
        "pid": 7845,
        "gens": [f"GT{x}" for x in range(1, 13)],
        "tech": "nggt",
        "status": "existing",
        "cap": 1020.7999877929688,
        # "years": [2037, 2028]
    },
    55241: {
        "ba": "2",
        "pid": 55241,
        "gens": ["CT01", "ST01"],
        "tech": "ngcc",
        "status": "existing",
        "cap": 280.0,
        # "years": [2031, 2034]
    },
    7964: {
        "ba": "177",
        "pid": 7964,
        "gens": ["GT1", "GT2", "GT3", "GT4"],
        "tech": "nggt",
        "status": "existing",
        "cap": 240.0,
    },
    7829: {
        "ba": "569",
        "pid": 7829,
        "gens": ["1", "2"],
        "tech": "nggt",
        "status": "existing",
        "cap": 242.0,
    },
    55411: {
        "ba": "2",
        "pid": 55411,
        "gens": ["HEC1", "HEC2", "HECS"],
        "tech": "ngcc",
        "status": "existing",
        "cap": 822.7999877929688,
    },
    7813: {
        "ba": "569",
        "pid": 7813,
        "gens": ["1", "2", "3", "4"],
        "tech": "nggt",
        "status": "existing",
        "cap": 570.0,
    },
    56807: {
        "ba": "186",
        "pid": 56807,
        "gens": ["1A", "1B", "1C"],
        "tech": "ngcc",
        "status": "existing",
        "cap": 559.0,
    },
    55234: {
        "ba": "177",
        "pid": 55234,
        "gens": ["GT1", "GT2", "GT3", "GT4", "GT5", "GT6", "GT7", "GT8"],
        "tech": "nggt",
        "status": "existing",
        "cap": 814.4000244140625,
    },
    60264: {
        "ba": "ERCO",
        "pid": 60264,
        "gens": ["BCGT1", "BCGT2", "BCGT3", "BCGT4", "BCGT5", "BCGT6"],
        "tech": "nggt",
        "status": "existing",
        "cap": 427.1999816894531,
    },
    10: {
        "ba": "2",
        "pid": 10,
        "gens": ["GT10", "GT2", "GT3", "GT4", "GT5", "GT6", "GT7", "GT8", "GT9"],
        "tech": "nggt",
        "status": "existing",
        "cap": 720.0,
    },
}


@pytest.mark.parametrize(
    "ix,pid, regime",
    [
        # (10, 55061, "limited"),
        # (11, 55061, "limited"),
        # (0, 55061, "limited"),
        # (9, 55061, "limited"),
        # (8, 55061, "limited"),
        # (0, 7838, "limited"),
        # (13, 7838, "limited"),
        # (0, 7315, "limited"),
        # (0, 634, "limited"),
        # (0, 56241, "limited"),  # still need to examine
        # (2, 7845, "limited"),
        # (2, 55241, "limited")
        # (0, 7964, "limited"),
        # (8, 7829, "limited"),
        # (0, 7813, "limited"),   # user limit
        # (14, 55411, "limited"),
        # (14, 10, "limited"),
        # (11, 56502, "limited"),
        # (0, 55411, "limited"),
        # (11, 56807, "limited"),
        # (10, 56807, "reference"),
        # (10, 55234, "limited"),
        # ("pure_surplus", 7838, "reference")
        # ("moderate", 7838, "reference"),
        ("pure_surplus", 10, "limited"),
        # ("pure_surplus", 10, "reference"),
        # ('form', 56807, 'reference',)
        ("pure_surplus", 55029, "reference"),
    ],
)
def test_model_colo_config(
    copt_license_etc, test_dir, temp_dir, ix, pid, regime: Literal["reference", "limited"]
):
    if not (temp_dir / "colo_test" / "data").exists():
        shutil.copytree(test_dir / "colo_test_data" / "data", temp_dir / "colo_test" / "data")
        (temp_dir / "colo_test" / "results").mkdir()

    with open(ROOT_PATH / "colo.toml", "rb") as f:
        configs = tomllib.load(f)

    with open(Path.home() / "patio_data/colo_data/colo.json", "rb") as f:
        PDATA = {v["pid"]: v for v in json.load(f).get("plants")}

    if ix in configs["scenario"]:
        config = configs["scenario"][ix] | {"name": ix}
    else:
        config = [{"name": k} | v for k, v in configs["scenario"].items() if v["ix"] == ix][0]

    x = model_colo_config(
        config,
        colo_dir=temp_dir / "colo_test",
        data_dir=Path.home() / "patio_data/colo_data",
        info=Info(**PDATA[pid]),
        regime=regime,
    )
    with open(test_dir / f"{pid}_{ix}_{regime}.json", "w") as output:
        json.dump(x, output, indent=4)

    assert x["run_status"] == "SUCCESS"


@pytest.mark.parametrize(
    "name,pid, regime, colo_dir",
    [
        # (10, 55061, "limited"),
        # (11, 55061, "limited"),
        # (0, 55061, "limited"),
        # (9, 55061, "limited"),
        # (8, 55061, "limited"),
        # (0, 7838, "limited"),
        # (13, 7838, "limited"),
        # (0, 7315, "limited"),
        # (0, 634, "limited"),
        # (0, 56241, "limited"),  # still need to examine
        # (2, 7845, "limited"),
        # (2, 55241, "limited")
        # (0, 7964, "limited"),
        # (8, 7829, "limited"),
        # (0, 7813, "limited"),   # user limit+
        # (14, 55411, "limited"),
        # (14, 10, "limited"),
        # (11, 56502, "limited"),
        # (0, 55411, "limited"),
        # (11, 56807, "limited"),
        # (10, 56807, "reference"),
        # (10, 55234, "limited"),
        # ("pure_surplus", 7838, "reference", "colo_202504152353"),
        ("form", 56807, "reference", "colo_202504152353")
    ],
)
def test_from_run(copt_license_etc, name, pid, regime, colo_dir):
    with open(Path.home() / "patio_data/colo_data/colo.json", "rb") as f:
        PDATA = {v["pid"]: v for v in json.load(f).get("plants")}

    info = Info(**PDATA[pid])
    model = Model.from_run(info, name, regime, Path.home() / "patio_data" / colo_dir)
    model.dispatch_all(150)
    model.check_service()
    assert False


@pytest.mark.parametrize(
    "ix,pid, regime",
    [
        # (10, 55061, "limited"),
        # (11, 55061, "limited"),
        # (0, 55061, "limited"),
        # (9, 55061, "limited"),
        # (8, 55061, "limited"),
        (0, 7838, "limited"),
        # (0, 7315, "limited"),
        # (0, 634, "limited"),
        # (0, 56241, "limited"),  # still need to examine
        # (2, 7845, "limited"),
        # (2, 55241, "limited")
        # (0, 7964, "limited"),
        # (8, 7829, "limited"),
        # (0, 7813, "limited"),   # user limit
        (0, 55411, "limited"),
        (11, 56807, "limited"),
        (10, 56807, "reference"),
    ],
)
def test_model_colo_config2(copt_license_etc, test_dir, temp_dir, ix, pid, regime):
    if not (temp_dir / "colo_test" / "data").exists():
        shutil.copytree(test_dir / "colo_test_data" / "data", temp_dir / "colo_test" / "data")
        (temp_dir / "colo_test" / "results").mkdir()

    with open(ROOT_PATH / "colo.toml", "rb") as f:
        config = tomllib.load(f)

    config = [{"name": k} | v for k, v in config["scenario"].items() if v["ix"] == ix][0]
    x = model_colo_config(
        config,
        colo_dir=temp_dir / "colo_test",
        info=Info(**PDATA[pid]),
        regime=regime,
    )
    with open(test_dir / f"{pid}_{ix}_{regime}.json", "w") as output:
        json.dump(x, output, indent=4)

    assert "ppa" in x


def test_patio_colo_entry_point2(copt_license_etc):
    """Test ``cepm`` entry point."""
    from patio.model.colo_core import main as colo_main

    set_timeout(6000)

    colo_main(
        colo_dir="/Users/aengel/patio_data/tester",
        # colo_dir="/Users/aengel/patio_data/colo_202501171818",
        test=True,
    )


def test_plant_list_2025():
    """Test ``cepm`` entry point."""
    self = BAs.from_file("BAs_202406121704")
    self.plant_list()


def test_toy():
    self = BAs.from_file("BAs_202309062254.zip")
    z = self.emission_cost_reduction("ba_code")
    f = self.make_potential_selected_maps(owners=["SO"])
    df = self.for_toy()


def test_combine_runs():
    combine_runs("colo_202412011258")


def test_econ_selected_full():
    self = BAs.from_file("BAs_202309062254.zip")
    z = self.econ_selected_full()
    f = self.make_potential_selected_maps(owners=["SO"])


def compare(a, b, attr):
    a = getattr(a, attr)
    b = getattr(b, attr)
    return a.compare(b)


def test_results_by_lse():
    # self = BAs.from_file("BAs_202311111854.zip")
    self = BAs.from_cloud("BAs_202504270143")
    # a = self.econ_results()
    # f, r, *others = self.output()
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
    #     self.ad.gens[
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
    df_lc = self.by_lse(least_cost=False)
    # df_me = self.by_lse(least_cost=False)
    assert not df_lc.empty()


def test_plant_list():
    self = BAs.from_cloud("BAs_202311111854.zip")
    df = self.plant_list()
    assert not df.empty()


def test_gen_list():
    self = BAs.from_file("BAs_202311111854.zip")
    df = self.gen_list()
    assert not df.empty()


def test_new_maps():
    self = BAs.from_file("BAs_202311111854.zip")
    df = self.make_potential_selected_maps()
    assert df


def test_site_list():
    self = BAs.from_file("BAs_202405011751.zip")
    df = self.cr_site_list()
    assert df


def test_unused_clean_repowering():
    self = BAs.from_file("BAs_202406121704.zip")
    df = self.unused_clean_repowering()
    assert df


def test_irp(asset_data):
    df = irp_data(asset_data)


def test_split_job_json():
    split_job_json(Path.home() / "patio_data/colo_202501240728")  # noqa: F821
