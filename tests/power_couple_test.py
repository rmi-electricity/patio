import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
from etoolbox.utils.testing import idfn

from patio.constants import ROOT_PATH
from patio.helpers import make_core_lhs_rhs, solver
from patio.model.colo_common import get_flows, get_summary, hstack, make_case_sheets, vstack
from patio.model.colo_core import model_colo_config, set_timeout, setup_plants_configs
from patio.model.colo_lp import Data, Info, Model
from patio.model.colo_resources import (
    CleanExport,
    ConstantLoad,
    Curtailment,
    EndogenousDurationStorage,
    EndogenousLoad,
    ExportOnlyIncumbentFossil,
    FeStorage,
    FixedDurationStorage,
    FlexLoad,
    IncumbentFossil,
    LoadOnlyFossil,
    Renewables,
)


def test_os_solver(os_solver):
    assert solver() == "HIGHS"


def test_default_solver():
    assert solver() == "COPT"


class TestPowerCouple:
    test_keys = (
        "load_mw",
        "solar_mw",
        "onshore_wind_mw",
        "li_mw",
        "li_duration",
        "fe_mw",
        "new_fossil_mw",
    )

    @pytest.mark.parametrize(
        "pid,scenario,expected",
        [
            pytest.param(
                55234,
                "moderate",
                (600.0, 1550.0, 1350.0, 600.0, 10.0, None, None),
                marks=pytest.mark.skip,
            ),
            (55234, "form", (600.0, 2750.0, 2120.0, 0.0, 0.0, 1000.0, None)),
            (55234, "pure_surplus", (600.0, 1720.0, 1690.0, 600.0, 10.0, None, 400.0)),
        ],
        ids=idfn,
    )
    def test_colo_config(self, os_solver, test_dir, temp_dir, pid, scenario, expected):
        """Validate colo model on select plants / configs."""
        set_timeout(1800)

        expected = dict(zip(self.test_keys, expected, strict=False))
        with open(test_dir.parent / "patio.toml", "rb") as f:
            config = tomllib.load(f)["colo"]
        config["project"]["plant_ids"] = [pid]
        config["project"]["scenarios"] = [scenario]
        config["scenario"]["default"]["setting"]["saved_select"] = ""
        with open(test_dir / "colo_test_data/colo.json") as f:
            plant_json = json.load(f)

        configs, plants_data = setup_plants_configs(config, **plant_json)

        result = model_colo_config(
            configs[0],
            colo_dir=temp_dir,
            data_dir=test_dir / "colo_test_data",
            info=plants_data[0],
            regime="reference",
        )
        if result["run_status"] != "SUCCESS":
            raise AssertionError(f"run failed with {result['run_status']}")
        bad = []
        for name in self.test_keys:
            if result.get(name, None) != expected[name]:
                bad.append(f"{result.get(name, None)=} != {expected[name]=}")
        assert not bad, "\n".join(bad)


@pytest.mark.skip(reason="test_colo_config approach is easier to target.")
@pytest.mark.script_launch_mode("inprocess")
def test_patio_colo_entry_point(script_runner, test_dir, temp_dir):
    """Test ``patio-colo`` entry point function."""
    ret = script_runner.run(
        [
            "patio-colo",
            "-l",
            "-k",
            "-w",
            "1",
            "-p",
            # "6195",
            # "55343",
            "55234",
            "-D",
            str(temp_dir.relative_to(Path.home())),
            "-L",
            str((test_dir / "colo_test_data").relative_to(Path.home())),
        ],
        print_result=True,
    )
    assert ret.success


@pytest.mark.skip(reason="debug only.")
def test_case_sheets(asset_data, run="colo_202507070053"):
    subplots = make_case_sheets(
        get_summary(run=run, ad=asset_data).replace(
            {"name": {"form_fos": "form_new_fossil", "pure_surplus": "new_fossil"}}
        ),
        get_flows(run).with_columns(
            name=pl.col("name").replace(
                {"form_fos": "form_new_fossil", "pure_surplus": "new_fossil"}
            )
        ),
        test=False,
        land_screen=False,
    )
    assert subplots


@pytest.mark.skip
class OLD:
    SOLAR = np.sin(np.pi * np.arange(4000) / 24) ** 8
    WIND = np.cos(np.pi * np.arange(4000) / 24) ** 2
    DT_RNG = pd.to_datetime(
        [
            *pd.date_range(start="2025-01-01", periods=2000, freq="h"),
            *pd.date_range(start="2028-01-01", periods=2000, freq="h"),
        ]
    )
    RE_SPECS = pd.DataFrame(
        {
            "re_site_id": [1, 2],
            "re_type": ["solar", "onshore_wind"],
            "generator_id": ["solar", "onshore_wind"],
            "icx_capacity": [50, 50],
            "area_per_mw": [0.031, 0.056],
            "area_sq_km": [100.0, 100.0],
            "icx_genid": [82, 82],
            "combi_id": ["82_1_solar", "82_2_onshore_wind"],
            "capacity_mw_nrel_site": [3225.0, 1785.0],
            "ones": [1.0, 1.0],
            "distance": [9.24, 8.01],
            "reg_mult": [1.13, 1.15],
        },
    )
    LOAD = (
        550
        + 40 * (np.sin(np.pi * np.arange(4000) / 24 - np.pi / 6)) ** 2
        + 20 * (np.sin(np.pi * np.arange(4000) / 12)) ** 2
        + 250 * (np.cos(np.pi * np.arange(4000) / 4392) ** 2)
        + 200 * (np.sin(np.pi * np.arange(4000) / 8784) ** 2)
    )
    BA_LOAD = pd.DataFrame(
        vstack(LOAD, LOAD - 100 * WIND - 150 * SOLAR).T, index=DT_RNG, columns=["l", "nl"]
    )
    POI = 300
    PDATA = {
        55061: {
            "ba": "57",
            "pid": 55061,
            "gens": ["GTG1", "GTG2", "GTG3", "GTG4", "GTG5", "GTG6"],
            "tech": "nggt",
            "status": "existing",
            "cap": 1099.1999816894531,
        },
        7838: {
            "ba": "186",
            "pid": 7838,
            "gens": ["1", "2", "3", "4"],
            "tech": "nggt",
            "status": "existing",
            "cap": 705.5,
        },
        7315: {
            "ba": "CAISO",
            "pid": 7315,
            "gens": ["2", "3", "4"],
            "tech": "nggt",
            "status": "existing",
            "cap": 174.0,
        },
        634: {
            "ba": "FPC",
            "pid": 634,
            "gens": ["4AGT", "4BGT", "4CGT", "4DGT", "4ST"],
            "tech": "ngcc",
            "status": "existing",
            "cap": 1254.0,
        },
        56241: {
            "ba": "MISO",
            "pid": 56241,
            "gens": ["UNT1", "UNT2"],
            "tech": "nggt",
            "status": "existing",
            "cap": 346.79998779296875,
        },
        7845: {
            "ba": "TVA",
            "pid": 7845,
            "gens": [f"GT{x}" for x in range(1, 13)],
            "tech": "nggt",
            "status": "existing",
            "cap": 1020.7999877929688,
        },
        55241: {
            "ba": "2",
            "pid": 55241,
            "gens": ["CT01", "ST01"],
            "tech": "ngcc",
            "status": "existing",
            "cap": 280.0,
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
            "gens": ("1", "2", "3", "4"),
            "tech": "nggt",
            "status": "existing",
            "cap": 570.0,
        },
    }

    def model(self, test_dir, temp_dir, ix, pid, pad=""):
        shutil.copytree(
            test_dir / "colo_test_data" / "data", temp_dir / "colo_test" / f"data{pad}"
        )
        (temp_dir / "colo_test" / "results").mkdir()

        with open(ROOT_PATH / "patio.toml", "rb") as f:
            config = tomllib.load(f)["colo"]

        config = [{"name": k} | v for k, v in config["scenario"].items() if v["ix"] == ix][0]
        config.pop("ix")
        config.pop("pre_charge")
        i = Info(**self.PDATA[pid], years=(), max_re=5.0)

        pc = Model(
            pad,
            i,
            Data.from_dz(test_dir / "colo_test_data" / "data", i),
            **config,
            # num_crit_hrs=25,
        )
        # pc.update_years()
        return pc

    def test_from_file(self):
        mdl = Model.from_file(
            "/Users/aengel/patio_data/colo_202504230013/results/PSCO_6112_nggt_existing_reference_2.zip"
        )
        hourly = mdl.hourly()
        hourly = mdl.add_mapped_yrs(redispatch(mdl, hourly))  # noqa: F821
        f, e = econ_and_flows(mdl, hourly, True)  # noqa: F821
        assert False

    @pytest.mark.parametrize(
        "ix,pid",
        [
            # (0, 55061),
            # (9, 55061),
            # (8, 55061),
            # (0, 7838),
            # (0, 7315),
            # (0, 634),
            # (0, 56241),  # still need to examine
            # (2, 7845),
            # (2, 55241)
            # (0, 7964),
            # (8, 7829),
            # (0, 7813),   # user limit
            (0, 56502)
        ],
    )
    def test_pc(self, test_dir, temp_dir, ix, pid):
        pc = self.model(test_dir, temp_dir, ix, pid)
        ConstantLoad(pc, ld_value=2000, x_cap=100.0, sqkm_per_mw=1 / 247)
        # FlexLoad(pc, ld_value=2000)
        Renewables(pc)
        # FixedDurationStorage(
        #     pc,
        #     8,
        # )
        EndogenousDurationStorage(pc)
        CleanExport(pc)
        Curtailment(pc)
        IncumbentFossil(pc, mcoe=pc.d.mcoe)
        o = pc.pre_check(100.0)
        pc.select_resources(240)
        pc["storage"].cost_cap[pc.i.years]
        df = pc.select_cost
        pc.shourly()
        pc.round()
        pc.dispatch_all()
        out = pc.out_result_dict
        _cc = pc.a.clean_cost()
        _ptc = pc.a.full_ptc()
        coef = pc.a.c_mw_all()
        # clean = pc.a.new_clean
        pc.cost_detail()
        pc.d.load_ba_data(test_dir / "colo_test_data" / "data" / "57.zip")
        print(pc)

    @pytest.mark.parametrize(
        "ix,pid",
        [
            # (0, 55061),
            # (9, 55061),
            # (8, 55061),
            (0, 7838),
            # (0, 7315),
            # (0, 634),
            # (0, 56241),  # still need to examine
            # (2, 7845),
            # (2, 55241)
            # (0, 7964),
            # (8, 7829),
            # (0, 7813),   # user limit
            # (0, 55411)
        ],
    )
    def test_pc_pure_surplus_form(self, copt_license_etc, test_dir, temp_dir, ix, pid):
        pc = self.model(test_dir, temp_dir, ix, pid, "_formsis")
        FlexLoad(pc, uptime=0.5)
        Renewables(pc)
        EndogenousDurationStorage(pc)
        CleanExport(pc)
        Curtailment(pc)
        ExportOnlyIncumbentFossil(pc, mcoe=pc.d.mcoe)
        FeStorage(pc)
        o = pc.pre_check(100.0)
        pc.select_resources(480)
        pc.round()
        pc.dispatch_all(480)
        print(pc)

    @pytest.mark.parametrize(
        "ix,pid",
        [
            (0, 55061),
            # (9, 55061),
            # (8, 55061),
            (0, 7838),
            # (0, 7315),
            # (0, 634),
            # (0, 56241),  # still need to examine
            # (2, 7845),
            # (2, 55241)
            # (0, 7964),
            # (8, 7829),
            # (0, 7813),   # user limit
            # (0, 55411)
        ],
    )
    def test_pc_pure_surplus(self, copt_license_etc, test_dir, temp_dir, ix, pid):
        pc = self.model(test_dir, temp_dir, ix, pid, "_fossis")
        FlexLoad(pc, uptime=0.5)
        Renewables(pc)
        EndogenousDurationStorage(pc)
        CleanExport(pc)
        Curtailment(pc)
        ExportOnlyIncumbentFossil(pc, mcoe=pc.d.mcoe)
        LoadOnlyFossil(pc, mcoe=pc.d.mcoe.with_columns(pl.col("total_var_mwh") * 1.25))
        o = pc.pre_check(100.0)
        pc.select_resources(480)
        pc.round()
        pc.dispatch_all(480)
        print(pc)

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            (
                "load",
                ((2028,), 48, [0, 3]),
                hstack(np.zeros((48, 1)), np.eye(48), -np.eye(48), np.zeros((48, 48))),
            ),
            (
                "critical",
                ((2028,), 48, [1, 2, 3]),
                hstack(np.zeros((48, 1)), -np.eye(48), np.eye(48), np.zeros((48, 48)))[
                    [1, 2, 3], :
                ],
            ),
            (
                "export_req",
                ((2028,), 48, [0, 3]),
                None,
            ),
            (
                "clean_export",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 1)), -np.eye(48), np.eye(48), np.zeros((48, 48))),
                    np.zeros((48, 145)),
                ),
            ),
            ("fossil_ops", ((2028,), 48, [0, 3]), None),
            ("icx_ops", ((2028,), 48, [0, 3]), None),
            ("land", ((2028,), 48, [0, 3]), None),
            (
                "ub",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.full((48, 1), -8.0), np.zeros((48, 96)), np.eye(48)),
                    hstack(np.full((48, 1), -8.0), np.zeros((48, 48)), np.eye(48), np.eye(48)),
                    hstack(
                        np.full((48, 1), -1.0),
                        np.zeros((48, 48)),
                        np.eye(48),
                        np.zeros((48, 48)),
                    ),
                    hstack(np.full((48, 1), -1.0), np.eye(48), np.zeros((48, 96))),
                    hstack(np.full((48, 1), 0.0), np.eye(48), np.zeros((48, 48)), -np.eye(48)),
                    hstack(np.full((48, 1), -1.0), np.eye(48), np.eye(48), np.zeros((48, 48))),
                ),
            ),
            ("n_ub", ((2028,), 48, {}), 6 * 48),
            (
                "eq",
                ((2028,), 48, [0, 5]),
                vstack(
                    hstack(
                        np.zeros((47, 1)),
                        -np.eye(47, 48, k=1) / 0.99,
                        np.eye(47, 48, k=1) * 0.99,
                        np.eye(47, 48) - 1.01 * np.eye(47, 48, k=1),
                    ),
                    hstack(np.full((2, 1), -4), np.zeros((2, 96)), np.eye(48)[[0, 5], :]),
                ),
            ),
            ("locked", ((2028,), 48, {}), hstack(np.ones((1, 1)), np.zeros((1, 48 * 3)))),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), hstack([172826], np.zeros(48 * 3))),
            (
                "bounds",
                ((2028,), 48, {}),
                vstack(
                    np.zeros(48 * 3 + 1),
                    hstack(np.full(48 * 2 + 1, 2e4), np.full(48, 2e4 * 8)),
                ),
            ),
        ],
        ids=idfn,
    )
    def test_storage(self, pc, meth, args, expected):
        s = FixedDurationStorage(pc, 8, 0.99, 0.99, 0.01)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), vstack(SOLAR, WIND).T[:48, :]),
            ("critical", ((2028,), 48, [1, 2, 3]), -vstack(SOLAR, WIND).T[[1, 2, 3], :]),
            ("export_req", ((2028,), 48, [0, 3]), None),
            (
                "clean_export",
                ((2028,), 48, [0, 3]),
                vstack(-vstack(SOLAR, WIND).T[:48, :], np.zeros((48, 2))),
            ),
            ("fossil_ops", ((2028,), 48, [0, 3]), None),
            ("icx_ops", ((2028,), 48, [0, 3]), None),
            (
                "land",
                ((2028,), 48, [0, 3]),
                make_core_lhs_rhs(RE_SPECS).query("cons_set != 'fossil'").to_numpy()[:, :-1],
            ),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48), np.eye(2)),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), np.array([129506, 149000])),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(2), np.full(2, 2e4))),
        ],
        ids=idfn,
    )
    def test_renewables(self, pc, meth, args, expected):
        s = Renewables(pc)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), -np.ones((48, 1))),
            ("critical", ((2028,), 48, [1, 2, 3]), np.ones((3, 1))),
            ("export_req", ((2028,), 48, [0, 3]), None),
            ("clean_export", ((2028,), 48, [0, 3]), None),
            (
                "fossil_ops",
                ((2028,), 48, [0, 3]),
                vstack([-0.2 * 48], np.zeros((2, 1)), -np.ones((48, 1))),
            ),
            ("icx_ops", ((2028,), 48, [0, 3]), None),
            ("land", ((2028,), 48, [0, 3]), vstack([1 / 247], np.zeros((1, 1)))),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), np.ones((1, 1))),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), [-48000]),
            ("bounds", ((2028,), 48, {}), vstack([0], [2e4])),
        ],
        ids=idfn,
    )
    def test_load(self, pc, meth, args, expected):
        s = EndogenousLoad(pc)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), -np.eye(48)),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            ("export_req", ((2028,), 48, [0, 3]), None),
            ("clean_export", ((2028,), 48, [0, 3]), None),
            ("fossil_ops", ((2028,), 48, [0, 3]), None),
            ("icx_ops", ((2028,), 48, [0, 3]), None),
            ("land", ((2028,), 48, [0, 3]), None),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), None),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), np.full(48, 18.7)),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(48), np.full(48, 1e5))),
        ],
        ids=idfn,
    )
    def test_curtailment(self, pc, meth, args, expected):
        s = Curtailment(pc)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), -np.eye(48)),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            ("export_req", ((2028,), 48, [0, 3]), -np.eye(48)),
            ("clean_export", ((2028,), 48, [0, 3]), vstack(np.eye(48), np.eye(48))),
            ("fossil_ops", ((2028,), 48, [0, 3]), None),
            ("icx_ops", ((2028,), 48, [0, 3]), vstack(np.eye(48), np.zeros((48, 48)))),
            ("land", ((2028,), 48, [0, 3]), None),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), None),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), np.full(48, -17.5)),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(48), np.full(48, POI))),
        ],
        ids=idfn,
    )
    def test_export(self, pc, meth, args, expected):
        s = CleanExport(pc)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), hstack(np.eye(48), np.zeros((48, 48)))),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            ("export_req", ((2028,), 48, [0, 3]), hstack(np.zeros((48, 48)), -np.eye(48))),
            (
                "clean_export",
                ((2028,), 48, [0, 3]),
                vstack(np.zeros((48, 96)), hstack(np.eye(48), np.zeros((48, 48)))),
            ),
            (
                "fossil_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.ones((1, 48)), np.zeros((1, 48))),
                    hstack(np.zeros((1, 48)), np.ones((1, 48))),
                    np.ones((1, 96)),
                    hstack(np.eye(48), np.zeros((48, 48))),
                ),
            ),
            (
                "icx_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 48)), np.eye(48)),
                    hstack(np.eye(48), np.eye(48)),
                ),
            ),
            ("land", ((2028,), 48, [0, 3]), None),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), None),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), np.full(96, 25)),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(96), np.full(96, POI))),
        ],
        ids=idfn,
    )
    def test_fossil_export(self, pc, meth, args, expected):
        s = IncumbentFossil(pc, pd.Series(25.0, index=self.DT_RNG))

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            ("load", ((2028,), 48, [0, 3]), np.eye(48)),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            ("export_req", ((2028,), 48, [0, 3]), np.zeros((48, 48))),
            ("clean_export", ((2028,), 48, [0, 3]), vstack(np.zeros((48, 48)), np.eye(48))),
            (
                "fossil_ops",
                ((2028,), 48, [0, 3]),
                vstack(np.ones((1, 48)), np.zeros((1, 48)), np.ones((1, 48)), np.eye(48)),
            ),
            (
                "icx_ops",
                ((2028,), 48, [0, 3]),
                vstack(np.zeros((48, 48)), np.eye(48)),
            ),
            ("land", ((2028,), 48, [0, 3]), None),
            ("ub", ((2028,), 48, [0, 3]), None),
            ("n_ub", ((2028,), 48, {}), 0),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), None),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), np.full(48, 25)),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(48), np.full(48, POI))),
        ],
        ids=idfn,
    )
    def test_fossil_no_export(self, pc, meth, args, expected):
        s = IncumbentFossil(pc, pd.Series(25.0, index=self.DT_RNG), exportable=False)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            (
                "load",
                ((2028,), 48, [0, 3]),
                hstack(np.zeros((48, 1)), np.eye(48), np.zeros((48, 48))),
            ),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            (
                "export_req",
                ((2028,), 48, [0, 3]),
                hstack(np.zeros((48, 1)), np.zeros((48, 48)), -np.eye(48)),
            ),
            (
                "clean_export",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 1)), np.zeros((48, 96))),
                    hstack(np.zeros((48, 1)), np.eye(48), np.zeros((48, 48))),
                ),
            ),
            (
                "fossil_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack([[0]], np.ones((1, 48)), np.zeros((1, 48))),
                    hstack([[0]], np.zeros((1, 48)), np.ones((1, 48))),
                    hstack(np.zeros((1, 1)), np.ones((1, 96))),
                    hstack(np.zeros((48, 1)), np.eye(48), np.zeros((48, 48))),
                ),
            ),
            (
                "icx_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 49)), np.eye(48)),
                    hstack(np.zeros((48, 1)), np.eye(48), np.eye(48)),
                ),
            ),
            ("land", ((2028,), 48, [0, 3]), None),
            (
                "ub",
                ((2028,), 48, [0, 3]),
                hstack(-np.ones((48, 1)), np.eye(48), np.eye(48)),
            ),
            ("n_ub", ((2028,), 48, {}), 48),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), hstack([[1]], np.zeros((1, 96)))),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), hstack([1.24594384e05], np.full(96, 25))),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(97), np.full(97, 2e4))),
        ],
        ids=idfn,
    )
    def test_new_fossil(self, pc, meth, args, expected):
        s = LoadOnlyFossil(pc, pd.Series(25.0, index=self.DT_RNG), exportable=True)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))

    @pytest.mark.parametrize(
        "meth, args, expected",
        [
            (
                "load",
                ((2028,), 48, [0, 3]),
                hstack(np.zeros((48, 1)), np.eye(48)),
            ),
            ("critical", ((2028,), 48, [1, 2, 3]), None),
            (
                "export_req",
                ((2028,), 48, [0, 3]),
                hstack(np.zeros((48, 1)), np.zeros((48, 48))),
            ),
            (
                "clean_export",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 1)), np.zeros((48, 48))),
                    hstack(np.zeros((48, 1)), np.eye(48)),
                ),
            ),
            (
                "fossil_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack([[0]], np.ones((1, 48))),
                    hstack([[0]], np.zeros((1, 48))),
                    hstack(np.zeros((1, 1)), np.ones((1, 48))),
                    hstack(np.zeros((48, 1)), np.eye(48)),
                ),
            ),
            (
                "icx_ops",
                ((2028,), 48, [0, 3]),
                vstack(
                    hstack(np.zeros((48, 49))),
                    hstack(np.zeros((48, 1)), np.eye(48)),
                ),
            ),
            ("land", ((2028,), 48, [0, 3]), None),
            (
                "ub",
                ((2028,), 48, [0, 3]),
                hstack(-np.ones((48, 1)), np.eye(48)),
            ),
            ("n_ub", ((2028,), 48, {}), 48),
            ("eq", ((2028,), 48, [0, 5]), None),
            ("locked", ((2028,), 48, {}), hstack([[1]], np.zeros((1, 48)))),
            ("b_locked", (), None),
            ("objective", ((2028,), 48, {}), hstack([1.24594384e05], np.full(48, 25))),
            ("bounds", ((2028,), 48, {}), vstack(np.zeros(49), np.full(49, 2e4))),
        ],
        ids=idfn,
    )
    def test_new_fossil_no_export(self, pc, meth, args, expected):
        s = LoadOnlyFossil(pc, pd.Series(25.0, index=self.DT_RNG), exportable=False)

        method = getattr(s, meth)
        if hasattr(method, "_pre_cvx"):
            method = method._pre_cvx
            result = method(s, *args)
        else:
            result = method(*args)
        if expected is None:
            assert result is None
        elif isinstance(result, sp.csc_array | sp.coo_array):
            assert np.all(np.isclose(result.toarray(), expected))
        else:
            assert np.all(np.isclose(result, expected))
