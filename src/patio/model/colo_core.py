from __future__ import annotations

import gc
import io
import json
import logging
import logging.handlers
import multiprocessing
import os
import platform
import shutil
import threading
import tomllib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from dispatch.constants import COLOR_MAP
from etoolbox.datazip import DataZip
from etoolbox.utils.cloud import rmi_cloud_fs
from etoolbox.utils.misc import all_logging_disabled
from etoolbox.utils.pudl import pl_scan_pudl
from plotly.subplots import make_subplots
from pypdf import PdfReader, PdfWriter
from tqdm.asyncio import tqdm

from patio.constants import PATIO_PUDL_RELEASE, REGION_MAP, ROOT_PATH, TECH_CODES
from patio.data.asset_data import AssetData, clean_atb
from patio.helpers import (
    _git_commit_info,
)
from patio.model.colo_common import (
    AEO_MAP,
    COSTS,
    FAIL_DISPATCH,
    FAIL_SELECT,
    FAIL_SERVICE,
    FAIL_SMALL,
    FANCY_COLS,
    LAND,
    STATUS,
    SUCCESS,
    SUM_COL_ORDER,
    f_pmt,
    order_columns,
    pl_filter,
    sankey,
    text_position,
    timer,
)
from patio.model.colo_lp import Data, Info, Model

LOGGER = logging.getLogger("patio")

BAD_COLO_BAS = ("130",)
if platform.system() == "Darwin" and platform.machine() == "arm64":
    if os.cpu_count() >= 16:
        WORKERS = os.cpu_count() // 2
        RLP_TL = 480
        DLP_TL = 120
    else:
        WORKERS = os.cpu_count() // 2
        RLP_TL = 480
        DLP_TL = 120
else:
    WORKERS = os.cpu_count() // 3
    RLP_TL = 2400
    DLP_TL = 240


def set_workers(workers):
    global WORKERS

    WORKERS = workers


def set_timeout(seconds):
    global RLP_TL

    RLP_TL = seconds


def mround(x, base=50):
    return base * round(x / base)


def run_colo_mp(config, queue, *, mp=True):
    d_dir = Path.home() / config["project"]["data_path"]
    if not d_dir.exists():
        raise FileNotFoundError(f"data directory {d_dir} does not exist")

    c_dir = Path.home() / config["project"]["run_dir_path"]
    if not (result_dir := c_dir.joinpath("results")).exists():
        result_dir.mkdir()

    colo_json = Path.home() / config["project"]["run_dir_path"] / "colo.json"
    if not colo_json.exists():
        raise FileNotFoundError(colo_json)
    with open(colo_json) as f:
        plant_json = json.load(f)

    configs, plants_data = setup_plants_configs(config, **plant_json)

    make_all_re_sites(c_dir, d_dir, plants_data)

    # plants_data = plant_json.get("plants")
    # plants_data = [Info(**(v | {"gens": tuple(v.pop("gens"))})) for v in plants_data]
    # if pids := config["project"].get("plant_ids", None):
    #     plants_data = [p for p in plants_data if p.pid in pids]
    #
    # config["scenario"]["default"]["param"]["pudl_release"] = plant_json.get(
    #     "pudl_release", PATIO_PUDL_RELEASE
    # )
    # configs = [
    #     {"name": k, "ix": config["scenario"][k]["ix"]}
    #     | {
    #         p: config["scenario"].get("default", {}).get(p, {})
    #            | config["scenario"].get(k, {}).get(p, {})
    #         for p in ("setting", "param", "dv")
    #     }
    #     for k in config["project"]["scenarios"]
    # ]
    def rn(x):
        if x is None:
            return 0.0
        return x

    workers = config["project"]["workers"]
    LOGGER.warning("%s workers, %s time limit", workers, RLP_TL)

    if mp and workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=worker_logger, initargs=(queue,)
        ) as pool:
            fs = []
            for info in plants_data:
                for conf in configs:
                    if (
                        conf["techs"]
                        and TECH_CODES.get(info.tech, info.tech) not in conf["techs"]
                    ):
                        continue
                    xt = info.extra(name=conf["name"])
                    conf = deepcopy(conf)

                    if not (result_dir / info.file("limited", **conf)).exists():
                        fs.append(
                            (xt, pool.submit(model_colo_config, conf, c_dir, d_dir, info))
                        )
                        continue
                    if (result_dir / info.file("reference", **conf)).exists():
                        LOGGER.info("skipping, reference version exists", extra=xt)
                        continue
                    with open(result_dir / info.file("limited", **conf)) as f:
                        result = json.load(f)
                    ll = result.get("load_mw", 0)
                    if ll is None:
                        ll = 0
                    if "ppa" in result and rn(result.get("load_mw", 0)) >= max(
                        50.0, mround(info.cap * 0.5, 25)
                    ):
                        LOGGER.info("skipping, good limited version exists", extra=xt)
                        continue
                    fs.append((xt, pool.submit(model_colo_config, conf, c_dir, d_dir, info)))

            for xt, f in tqdm(fs, total=len(fs), desc="Running colos", position=0):
                try:
                    f.result()
                except BrokenProcessPool:
                    pass
                except Exception as exc:
                    LOGGER.info("%r", exc, extra=xt, exc_info=exc)
        return None
    for info, conf in tqdm(
        [(da, con) for da in plants_data for con in configs],
        total=len(plants_data) * len(configs),
        desc="Running colos",
        position=0,
    ):
        if conf["techs"] and TECH_CODES.get(info.tech, info.tech) not in conf["techs"]:
            continue
        xt = info.extra(name=conf["name"])
        conf = deepcopy(conf)

        if not (result_dir / info.file("limited", **conf)).exists():
            model_colo_config(conf, c_dir, d_dir, info)
            continue
        if (result_dir / info.file("reference", **conf)).exists():
            LOGGER.info("skipping, reference version exists", extra=xt)
            continue
        with open(result_dir / info.file("limited", **conf)) as f:
            result = json.load(f)
        if "ppa" in result and rn(result.get("load_mw", 0)) >= max(
            50.0, mround(info.cap * 0.5, 25)
        ):
            LOGGER.info("skipping, good limited version exists", extra=xt)
            continue
        model_colo_config(conf, c_dir, d_dir, info)
    return None


def make_all_re_sites(c_dir, d_dir, plants_data):
    re_all = {}
    for pdat in tqdm(plants_data):
        with DataZip(d_dir / pdat.file(suffix=".zip")) as z:
            if pdat.pid not in re_all:
                re_all[pdat.pid] = z["re"].with_columns(icx_id=pl.lit(pdat.pid))
    pl.concat(re_all.values()).write_parquet(c_dir / "all_re_sites.parquet")


def setup_plants_configs(
    config, plants, pudl_release=PATIO_PUDL_RELEASE, **kwargs
) -> tuple[list[dict], list[Info]]:
    """Combine plant_data and data config from colo_json with scenarios from toml."""
    plants_data = [Info(**(v | {"gens": tuple(v.pop("gens"))})) for v in plants]
    if pids := config["project"].get("plant_ids", None):
        plants_data = [p for p in plants_data if p.pid in pids]
    config["scenario"]["default"]["param"]["pudl_release"] = pudl_release
    configs = [
        {
            "name": k,
            "ix": config["scenario"][k]["ix"],
            "techs": config["scenario"][k].get("techs", []),
        }
        | {
            p: config["scenario"].get("default", {}).get(p, {})
            | config["scenario"].get(k, {}).get(p, {})
            for p in ("setting", "param", "dv")
        }
        for k in config["project"]["scenarios"]
    ]
    return configs, plants_data


def make_configs_from_defaults(config):
    return [
        {"name": k, "ix": config["scenario"][k]["ix"]}
        | {
            p: config["scenario"].get("default", {}).get(p, {})
            | config["scenario"].get(k, {}).get(p, {})
            for p in ("setting", "param", "dv")
        }
        for k in config["project"]["scenarios"]
    ]


# noinspection PyUnreachableCode
def model_colo_config(
    config,
    colo_dir,
    data_dir,
    info,
    regime: Literal["reference", "limited"] = "limited",
):
    logger = logging.getLogger("patio")
    result_dir = colo_dir / "results"
    setting = config.get("setting", {})

    file = result_dir / info.file(regime, **config)
    if file.exists():
        xt = info.extra(regime, **config["param"])
        if (result_dir / info.file("reference", **config)).exists():
            logger.info("Skipping in full because limited and reference exist", extra=xt)
            return True
        with open(file) as f:
            result = json.load(f)
        if "ppa" in result and result.get("load_mw", 0) >= max(
            50.0, mround(info.cap * 0.5, 25)
        ):
            logger.info("Skipping in full because limited was successful", extra=xt)
            return True
        logger.info("Limited exists but was not sucessful so running reference", extra=xt)
        regime: Literal["reference", "limited"] = "reference"

    model = Model(
        config["ix"],
        config["name"],
        info,
        Data.from_dz(
            data_dir,
            info,
            re_filter=(pl.col("re_type") == "solar")
            | ((pl.col("re_type") == "onshore_wind") & (pl.col("distance") <= 20)),
        ),
        **config["param"],
        regime=regime,
        dvs=config["dv"],
        logger=logger,
        result_dir=result_dir,
    )
    if len(model.es_types) > 2:
        raise RuntimeError(f"can only have two storage types, this case has {model.es_types}")
    del regime  # to prevent this var from being out of step with model state
    file = result_dir / info.file(model.regime, **config)

    start = perf_counter()
    try:
        if model.i.max_re < 150:
            raise AssertionError(f"{model.i.max_re=:.2f} < 150")
        model = model_colo_load(model, **setting)
    except AssertionError as exc:
        logger.info("%r", exc, extra=model.extra)
        model.errors.append(repr(exc))
        # model.to_file()
    except pl.exceptions.PolarsError as exc:
        e_str = repr(exc).split("Resolved plan until failure")[0] + "')"
        logger.error("%s", e_str, extra=model.extra)
        logger.info("error ", exc_info=exc, extra=model.extra)
        model.errors.append(e_str)
        # model.to_file()
    except Exception as exc:
        logger.error("%r", exc, extra=model.extra)
        logger.info("%r", exc, exc_info=exc, extra=model.extra)
        model.errors.append(repr(exc))
        # model.to_file()
    finally:
        elapsed = perf_counter() - start
        logger.info("%s seconds=%d", STATUS[model.status], elapsed, extra=model.extra)
        # pixs = info.pixs(model.regime, **config)
        pixs = info.pixs(model.regime, config["name"], config["ix"], {})
        to_json = (
            info.pixs(model.regime, **config) | model.out_result_dict | {"total_time": elapsed}
        )
        with open(file, "w") as output:
            json.dump(to_json, output, indent=4)
        for name, df in model.dfs.items():
            try:
                df_path = result_dir / name / f"{file.stem}.parquet"
                if not df_path.parent.exists():
                    df_path.parent.mkdir(parents=True)
                df = df.lazy()
                df.with_columns(**{k: pl.lit(v) for k, v in pixs.items()}).select(
                    *pixs, *df.collect_schema().names()
                ).collect().write_parquet(df_path)
                del df
            except Exception as exc:
                logger.error("%s %r", name, exc, extra=model.extra)
                logger.info("%s %r", name, exc, exc_info=True, extra=model.extra)
        load = to_json["load_mw"]
        served_pct = to_json.get("served_pct", 0.0)
        ppa = to_json.get("ppa", 0.0)
        load_icx = max(50.0, mround(info.cap * 0.5, 25))
        if model.regime == "limited" and (
            model.status != SUCCESS
            or load is None
            or load < load_icx
            or served_pct < 0.95
            or ppa > 100
        ):
            msg = (
                "failed with limited regime"
                if model.status != SUCCESS
                else f"({load=} < {load_icx=}) or ({served_pct=} < 0.95) or ({ppa=} > 100)"
            )
            logger.info("%s, retrying with reference regime", msg, extra=model.extra)
            gc.collect()
            return model_colo_config(  # noqa: B012
                config,
                colo_dir,
                data_dir,
                info,
                "reference",
            )
        else:
            del model
            gc.collect()
            return to_json


def model_colo_load(model: Model, run_econ, saved_select, **kwargs):
    if kwargs:
        model.logger.warning("%s settings will be ignored", list(kwargs), extra=model.extra)
    if model.pre_check(50.0).status == FAIL_SMALL:
        return model

    if saved_select:
        model.load_saved_x_cap(saved_select)
        if model.status in (FAIL_SELECT, FAIL_SMALL):
            return model
    else:
        with timer() as t:
            model.select_resources(RLP_TL)
        model.add_to_result_dict(select_time=t())
        if model.status in (FAIL_SELECT, FAIL_SMALL):
            return model
        load_ = model.load_mw
        model.round()
        load = model.load_mw
        model.logger.info(
            "Sizing LP complete in %d s, load DV %s, using %s",
            *(t(), load_, load),
            extra=model.extra,
        )

    with timer() as t:
        model.dispatch_all(DLP_TL)
    model.add_to_result_dict(dispatch_time=t())
    if model.status == FAIL_DISPATCH:
        return model

    model.check_service()
    if model.status == FAIL_SERVICE:
        model.add_df(hourly=model.hourly().collect())
        raise AssertionError(model.errors[-1])
    model.logger.info("Dispatch completed successfully in %d s", t(), extra=model.extra)

    hourly = model.hourly()
    # try:
    #     path = model.to_file(BytesIO())
    #     other = Model.from_file(path)
    #     hourly0 = other.hourly()
    #     cols = sorted(
    #         set(hourly.collect_schema().names()).intersection(
    #             set(hourly0.collect_schema().names())
    #         )
    #     )
    #     assert_frame_equal(
    #         hourly.sort("datetime").select(*cols),
    #         hourly0.sort("datetime").select(*cols),
    #         atol=1,
    #     )
    # except Exception as exc:
    #     exc_repr, *_ = repr(exc).partition("')")
    #     model.logger.error("unable to serialize model %s')", exc_repr, extra=model.extra)
    #     model.logger.debug(
    #         "unable to serialize model %s')", exc_repr, exc_info=True, extra=model.extra
    #     )

    with timer() as t:
        hourly = model.add_mapped_yrs(model.redispatch(hourly))
    model.add_to_result_dict(redispatch_time=t())
    model.logger.info("Redispatch completed successfully in %d s", t(), extra=model.extra)

    model.verify_hourly_df(hourly)

    hourly = hourly.select(~cs.starts_with("_"))
    econ_df, flows = model.econ_and_flows(hourly, run_econ)

    model.d.del_ba_data()
    gc.collect()
    model.add_df(
        hourly=hourly.with_columns(cs.by_dtype(pl.Float64).cast(pl.Float32)).collect(),
        full=econ_df,
        flows=flows,
    )

    model.logger.debug("finished successfully", extra=model.extra)
    return model


# def econ_and_flows(model, hourly, run_econ):
#     with timer() as t0:
#         flows = (
#             hourly.lazy()
#             .rename({"load_fossil": "load__fossil"})
#             .with_columns(
#                 re=pl.sum_horizontal(*model.re_types),
#                 charge=-pl.min_horizontal(
#                     0.0,
#                     pl.sum_horizontal(cs.contains("discharge"))
#                     - pl.sum_horizontal(cs.contains("_charge")),
#                 ),
#                 discharge=pl.max_horizontal(
#                     0.0,
#                     pl.sum_horizontal(cs.contains("discharge"))
#                     - pl.sum_horizontal(cs.contains("_charge")),
#                 ),
#                 load__clean=c.load - c.load__fossil,
#                 load__re=pl.min_horizontal(
#                     c.load - c.load__fossil, pl.sum_horizontal(*model.re_types)
#                 ),
#                 export_req__clean=pl.min_horizontal(c.export_requirement, c.export_clean),
#                 export_addl__clean=c.export - c.export_requirement,
#             )
#             .with_columns(
#                 export_addl__re=pl.min_horizontal(c.export_addl__clean, c.re - c.load__re),
#             )
#             .with_columns(
#                 export_req__re=c.re
#                                - pl.sum_horizontal("export_addl__re", "load__re", "charge", "curtailment"),
#             )
#             .with_columns(
#                 export_req__storage=c.export_req__clean - c.export_req__re,
#                 export_req__fossil=c.export_requirement - c.export_req__clean,
#                 export_addl__storage=c.export_addl__clean - c.export_addl__re,
#                 load__storage=pl.min_horizontal(
#                     c.load - c.load__fossil - c.load__re, c.discharge
#                 ),
#                 storage__re=pl.min_horizontal(c.re, c.charge),
#                 curtailment__re=c.curtailment,
#             )
#         )
#         re_ = flows.select(model.re_types).collect()
#         alloc = (re_ / re_.sum_horizontal()).fill_nan(0.0)
#         to_repl = ("load", "export_req", "export_addl", "storage", "curtailment")
#         flows = flows.with_columns(
#             **{
#                 f"{part}__{t}": alloc[t] * pl.col(f"{part}__re")
#                 for part in to_repl
#                 for t in model.re_types
#             }
#         )
#
#         flows = flows.select(
#             ~cs.ends_with("_re")
#             & ~cs.ends_with("_discharge")
#             & ~cs.ends_with("_charge")
#             & ~cs.ends_with("_clean")
#             & ~cs.ends_with("_soc")
#             & ~cs.starts_with("redispatch_sys_")
#             & (~cs.starts_with("baseline_sys_") | cs.contains("baseline_sys_lambda"))
#         )
#         check = (
#             flows.with_columns(
#                 load_check=(
#                         c.load
#                         - pl.sum_horizontal(
#                     f"load__{t}" for t in ("fossil", "storage", *model.re_types)
#                 )
#                 ).abs(),
#                 export_check=(
#                         c.export
#                         - pl.sum_horizontal(cs.matches("^export_req__.*|^export_addl__.*"))
#                 ).abs(),
#                 export_req_check=(
#                         c.export_requirement - pl.sum_horizontal(cs.matches("^export_req__.*"))
#                 ).abs(),
#                 fossil_check=(c.fossil - c.export_req__fossil - c.load__fossil).abs(),
#                 discharge_check=(
#                         c.discharge - pl.sum_horizontal(cs.matches(".*__storage$"))
#                 ).abs(),
#                 charge_check=(c.charge - pl.sum_horizontal(cs.matches("^storage__.*"))).abs(),
#                 curtailment_check=(
#                         c.curtailment - pl.sum_horizontal(cs.matches("^curtailment__.*"))
#                 ).abs(),
#             )
#             .filter(
#                 (c.load_check > 0.1)
#                 | (c.export_check > 0.1)
#                 | (c.export_req_check > 0.1)
#                 | (c.fossil_check > 0.1)
#                 | (c.discharge_check > 0.1)
#                 | (c.charge_check > 0.1)
#                 | (c.curtailment_check > 0.1)
#             )
#             .collect()
#         )
#     econ_df = pl.DataFrame()
#     if not check.is_empty():
#         model.add_to_result_dict(flows_time=t0())
#         model.logger.error("check of flows failed", extra=model.extra)
#         flows = pl.DataFrame()
#     else:
#         with timer() as t1:
#             flows_hourly = flows.select(
#                 cs.by_name("datetime", "starts", "baseline_sys_lambda") | cs.contains("__")
#             )
#             flows = (
#                 flows_hourly.group_by_dynamic("datetime", every="1y")
#                 .agg(cs.contains("__").sum())
#                 .collect()
#             )
#         model.add_to_result_dict(flows_time=t0() + t1())
#         if run_econ:
#             with timer() as t:
#                 try:
#                     econ_df = model.create_df_for_econ_model(
#                         flows_hourly, hourly, model.i, model.d
#                     ).collect()
#                 except Exception as exc:
#                     model.errors.append(f"could not create econ_df, {exc!r}")
#                     model.logger.error(model.errors[-1], extra=model.extra)
#             model.add_to_result_dict(econ_df_time=t())
#         else:
#             model.to_file()
#         del flows_hourly
#     return econ_df, flows


# def redispatch(model, hourly):
#     model.d.load_ba_data()
#     dmd = model.d["ba_dm_data"]
#     pd_dt_idx = pd.Index(hourly.select("datetime").collect().to_pandas().squeeze())
#     new_profs = dmd["dispatchable_profiles"].loc[pd_dt_idx, :]
#     # breaks for nuclear because not in new profs (also no export capacity)
#     if (
#         TECH_CODES.get(model.i.tech, model.i.tech) in OTHER_TD_MAP
#         and "fossil" in model.dvs
#         and isinstance(model["fossil"], IncumbentFossil)
#     ):
#         icx_ixs = [(model.i.pid, g) for g in model.i.gens]
#         colo_load_fossil = (
#             hourly.select("datetime", "load_fossil")
#             .collect()
#             .to_pandas()
#             .set_index("datetime")
#             .fillna(0.0)
#         ).squeeze()
#
#         new_profs.loc[:, icx_ixs] = adjust_profiles(
#             new_profs.loc[:, icx_ixs].to_numpy(),
#             colo_load_fossil.to_numpy(),
#             dmd["dispatchable_specs"].loc[icx_ixs, "capacity_mw"].to_numpy(),
#         )
#         max_check = (new_profs.loc[:, icx_ixs].sum(axis=1) + colo_load_fossil).max()
#         assert max_check < model.i.cap or np.isclose(max_check, model.i.cap), (
#             "fossil availability adjustment for load use of generators failed"
#         )
#     else:
#         icx_ixs = []
#     dm = DispatchModel(
#         load_profile=hourly.select("datetime", "baseline_sys_load_net_of_clean_export")
#         .collect()
#         .to_pandas()
#         .set_index("datetime")
#         .fillna(0.0)
#         .squeeze(),
#         dispatchable_specs=dmd["dispatchable_specs"],
#         dispatchable_profiles=new_profs,
#         dispatchable_cost=dmd["dispatchable_cost"]
#         .filter(c.datetime.dt.year() >= model.d.opt_years[0])
#         .collect()
#         .to_pandas()
#         .set_index(["plant_id_eia", "generator_id", "datetime"]),
#         storage_specs=dmd["storage_specs"],
#         re_profiles=dmd["re_profiles"].loc[pd_dt_idx, :],
#         re_plant_specs=dmd["re_plant_specs"],
#     )()
#     hly = (
#         hourly.select("datetime", "export_clean", "load_fossil")
#         .collect()
#         .to_pandas()
#         .set_index("datetime")
#     )
#     redispatch = pd.concat(
#         [
#             dm.redispatch.loc[hly.index, :].sum(axis=1).rename("redispatch_sys_dispatchable"),
#             # rather than trying to break out colo contribution to both storage and
#             # renewable all clean exports here are categorized as renewable because
#             # fossil generally can't charge the battery
#             (dm.re_profiles_ac.loc[hly.index, :].sum(axis=1) + hly.export_clean).rename(
#                 "redispatch_sys_renewable"
#             ),
#             dm.storage_dispatch.loc[hly.index, :]
#             .T.groupby(level=0)
#             .sum()
#             .T.assign(redispatch_sys_storage=lambda x: x.discharge - x.gridcharge)[
#                 "redispatch_sys_storage"
#             ],
#         ],
#         axis=1,
#     )
#     factors = (
#         (dm.dispatchable_cost.heat_rate * dm.dispatchable_cost.co2_factor)
#         .reset_index()
#         .pivot(index="datetime", columns=["plant_id_eia", "generator_id"], values=0)
#         .reindex_like(dm.redispatch, method="ffill")
#         .loc[hly.index, :]
#     )
#     summary = pd.concat(
#         [
#             dm.system_summary_core(freq="h")
#             .loc[hly.index, ["load_mwh", "deficit_mwh", "curtailment_mwh", "curtailment_pct"]]
#             .rename(
#                 columns={
#                     "load_mwh": "redispatch_sys_load",
#                     "deficit_mwh": "redispatch_sys_deficit",
#                     "curtailment_mwh": "redispatch_sys_curtailment",
#                     "curtailment_pct": "redispatch_sys_curtailment_pct",
#                 }
#             ),
#             (
#                 (dm.redispatch.loc[hly.index, :] * factors).sum(axis=1)
#                 # needs to never be nans even if no fossil incumbent, so above remains
#                 + factors.loc[:, icx_ixs].mean(axis=1).fillna(0.0) * hly.load_fossil
#             ).rename("redispatch_sys_co2"),
#             calc_redispatch_cost(dm).loc[hly.index].rename("redispatch_sys_cost"),
#             redispatch,
#         ],
#         axis=1,
#     )
#     hourly = (
#         pl.concat(
#             [
#                 hourly,
#                 pl.from_pandas(
#                     (factors.loc[:, icx_ixs].mean(axis=1) * hly.load_fossil).rename(
#                         "load_co2"
#                     ),
#                     schema_overrides={"datetime": pl.Datetime()},
#                     include_index=True,
#                 ).lazy(),
#                 pl.from_pandas(
#                     dm.redispatch.loc[hly.index, icx_ixs]
#                     .sum(axis=1)
#                     .rename("redispatch_export_fossil"),
#                     schema_overrides={"datetime": pl.Datetime()},
#                     include_index=True,
#                 ).lazy(),
#                 pl.from_pandas(
#                     dm.redispatch_lambda().loc[hly.index].rename("redispatch_sys_lambda"),
#                     schema_overrides={"datetime": pl.Datetime()},
#                     include_index=True,
#                 ).lazy(),
#                 pl.from_pandas(
#                     summary, schema_overrides={"datetime": pl.Datetime()}, include_index=True
#                 ).lazy(),
#             ],
#             how="align",
#         )
#         .with_columns(_rcurt=np.maximum(np.minimum(1, c.redispatch_sys_curtailment_pct), 0))
#         .with_columns(
#             _hr_mkt_rev_red=c.redispatch_sys_lambda
#             * model.mkt_rev_mult.value
#             * (1 - c._rcurt),
#         )
#         .with_columns(
#             redispatch_rev_clean=c.export_clean * c._hr_mkt_rev_red,
#             redispatch_rev_fossil=c.redispatch_export_fossil * c._hr_mkt_rev_red,
#             redispatch_cost_fossil=(c.redispatch_export_fossil + c.load_fossil)
#             * c.c_export_fossil,
#             redispatch_cost_export_fossil=c.redispatch_export_fossil * c.c_export_fossil,
#             cost_curt=c.curtailment * c.c_curtailment
#             + c.export_clean * c.assumed_curtailment_pct * c._ptc,
#             redispatch_cost_curt=c.curtailment * c.c_curtailment
#             + c.export_clean * c._rcurt * c._ptc,
#             redispatch_system_cost=c.redispatch_sys_cost + c.export_clean * c._hr_mkt_rev_red,
#         )
#     )
#     if hourly.select("redispatch_export_fossil").sum().collect().item() == 0.0:
#         hourly = hourly.with_columns(redispatch_export_fossil=pl.col("export_fossil"))
#     return hourly


def assemble_results(colo_dir, *, keep=True):
    colo_dir = Path.home() / colo_dir
    result_dir = colo_dir / "results"
    sort_by = ["ba_code", "icx_id", "tech", "status", "name", "regime"]
    with tqdm("Assembling outputs", total=124) as pbar:
        pbar.update(1)
        if any(result_dir.glob("*.json")):
            sums = []
            for file in result_dir.glob("*.json"):
                try:
                    sums.append(pl.read_json(file))
                except pl.exceptions.ComputeError:
                    sums.append(pl.from_pandas(pd.read_json(file, typ="series").to_frame().T))
            out = pl.concat(sums, how="diagonal_relaxed")
            out_cols = list(
                dict.fromkeys([co for co in SUM_COL_ORDER if co in out.columns])
                | dict.fromkeys(out.columns)
            )
            out.sort(*sort_by).select(*out_cols).write_parquet(
                colo_dir / "colo_summary.parquet"
            )
            # if not keep:
            #     for file in result_dir.glob("*.json"):
            #         file.unlink()
        pbar.update(1)

        if any((flows_dir := result_dir / "flows").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    flows_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_flows.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in flows_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_flows.parquet")
            if not keep:
                shutil.rmtree(flows_dir)
        pbar.update(10)
        if any((full_dir := result_dir / "full").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    full_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_full.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in full_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_full.parquet")
            if not keep:
                shutil.rmtree(full_dir)
        pbar.update(10)
        if any((re_selected_dir := result_dir / "re_selected").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    re_selected_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_re_selected.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in re_selected_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_re_selected.parquet")
            if not keep:
                shutil.rmtree(re_selected_dir)
        pbar.update(10)
        if any((cost_dir := result_dir / "cost_detail").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    cost_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_cost_detail.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in cost_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_cost_detail.parquet")
            if not keep:
                shutil.rmtree(cost_dir)
        pbar.update(10)
        if any((annual_dir := result_dir / "annual").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    annual_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_annual.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in annual_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_annual.parquet")
            if not keep:
                shutil.rmtree(annual_dir)
        pbar.update(10)
        # if any((coef_ts_dir := result_dir / "coef_ts").glob("*.parquet")):
        #     pl.scan_parquet(
        #         coef_ts_dir / "*.parquet", allow_missing_columns=True
        #     ).sink_parquet(colo_dir / "colo_coef_ts.parquet")
        #     if not keep:
        #         shutil.rmtree(coef_ts_dir)
        # pbar.update(10)
        if any((coef_mw_dir := result_dir / "coef_mw").glob("*.parquet")):
            mw = []
            idx = [
                "ba_code",
                "icx_id",
                "icx_gen",
                "regime",
                "name",
                "ix",
                # "num_crit_hrs",
                # "max_pct_fos",
                # "max_pct_hist_fos",
                # "mkt_rev_mult",
                # "fos_load_cost_mult",
                # "life",
            ]
            mw = []
            for file in coef_mw_dir.glob("*.parquet"):
                try:
                    temp = pl.read_json(file)
                except pl.exceptions.ComputeError:
                    temp = pl.from_pandas(pd.read_parquet(file))
                temp = temp.unpivot(
                    on=set(temp.columns) - {*idx, "year"},
                    index=[*idx, "year"],
                    variable_name="resource",
                ).pivot(index=[*idx, "resource"], on="year", values="value")
                mw.append(temp)
            (
                pl.concat(
                    mw,
                    how="diagonal_relaxed",
                )
                .sort([x for x in sort_by if x in temp.columns])
                .write_parquet(colo_dir / "colo_coef_mw.parquet")
            )

            # pl.concat(
            #     (pl.scan_parquet(file) for file in coef_mw_dir.glob("*.parquet")),
            #     how="diagonal_relaxed",
            # ).sort(*sort_by).sink_parquet(colo_dir / "colo_coef_mw.parquet")
            if not keep:
                shutil.rmtree(coef_mw_dir)
        pbar.update(1)
        if any((hourly_rs_dir := result_dir / "resource_selection_hourly").glob("*.parquet")):
            try:
                pl.scan_parquet(
                    hourly_rs_dir / "*.parquet", allow_missing_columns=True
                ).sink_parquet(colo_dir / "colo_resource_selection_hourly.parquet")
            except pl.exceptions.SchemaError:
                pl.concat(
                    (pl.scan_parquet(file) for file in hourly_rs_dir.glob("*.parquet")),
                    how="diagonal_relaxed",
                ).sink_parquet(colo_dir / "colo_resource_selection_hourly.parquet")
            if not keep:
                shutil.rmtree(hourly_rs_dir)
        pbar.update(20)
        if any((hourly_dir := result_dir / "hourly").glob("*.parquet")):
            # dffs = list(hourly_dir.glob("*.parquet"))
            # for dff in tqdm(dffs, total=len(dffs)):
            #     df = pl.scan_parquet(dff)
            #     if pl.Float64 in {type(x) for x in df.collect_schema().values()}:
            #         try:
            #             df.with_columns(cs.by_dtype(pl.Float64).cast(pl.Float32)).sink_parquet(dff)
            #         except Exception as exc:
            #             print(f"unable to downcast {dff} {exc!r}")
            hourly_dir.rename(hourly_dir.parent.parent / "hourly")
            # try:
            #     pl.scan_parquet(
            #         hourly_dir / "*.parquet", allow_missing_columns=True
            #     ).sink_parquet(colo_dir / "colo_hourly.parquet")
            # except SchemaError:
            #     pl.concat(
            #         (pl.scan_parquet(file) for file in hourly_dir.glob("*.parquet")),
            #         how="diagonal_relaxed",
            #     ).sink_parquet(colo_dir / "colo_hourly.parquet")
            # if not keep:
            #     shutil.rmtree(hourly_dir)
        pbar.update(50)
        # if not keep:
        #     shutil.rmtree(result_dir)
        pbar.update(1)


def open_log(colo_dir=None, info=None, regime=None, name=None):
    if colo_dir is None:
        colo_dir = sorted(
            Path.home().glob("patio_data/*/log.jsonl"),
            key=lambda x: x.parent.stat().st_mtime,
            reverse=True,
        )[0]
    else:
        colo_dir = Path.home() / colo_dir / "log.jsonl"
    df = pl.read_ndjson(colo_dir)
    if info:
        extra = info.extra()
        df = df.filter(
            (pl.col("plant_id") == extra["plant_id"])
            & (pl.col("tech") == extra["tech"])
            & (pl.col("status") == extra["status"])
        )
    if regime:
        df = df.filter(pl.col("regime") == regime[:3])
    if name:
        df = df.filter(pl.col("config") == name)
    return df


def worker_logger(queue):
    qh = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(qh)
    root.setLevel(logging.DEBUG)


def setup_logger(log_path):
    import logging.config

    xtra_defs = {
        "ba_code": "",
        "plant_id": 0,
        "tech": "",
        "status": "",
        "regime": "",
        "config": "",
    }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "()": "etoolbox.utils.logging_utils.SafeFormatter",
                "format": "[%(levelname)8s|%(module)s|L%(lineno)4d|P%(process)5d||B%(ba_code)5s|P%(plant_id)5d|T%(tech)5s|S%(status)5s|C%(config)17s|R%(regime)4s]: %(message)s",
                "extra_defaults": xtra_defs,
            },
            "detailed": {
                "()": "etoolbox.utils.logging_utils.SafeFormatter",
                "format": "[%(levelname)8s|%(module)s|L%(lineno)4d|P%(process)5d||B%(ba_code)5s|P%(plant_id)5d|T%(tech)5s|S%(status)5s|C%(config)17s|R%(regime)4s] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                "extra_defaults": xtra_defs,
            },
            "json": {
                "()": "etoolbox.utils.logging_utils.JSONFormatter",
                "fmt_keys": {
                    "level": "levelname",
                    "message": "message",
                    "timestamp": "timestamp",
                    "logger": "name",
                    "module": "module",
                    "function": "funcName",
                    "line": "lineno",
                    "thread_name": "threadName",
                    "process": "process",
                    "ba_code": "ba_code",
                    "plant_id": "plant_id",
                    "tech": "tech",
                    "status": "status",
                    "regime": "regime",
                    "config": "config",
                },
            },
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "simple",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_path.with_suffix(".log"),
            },
            "file_json": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": log_path.with_suffix(".jsonl"),
            },
        },
        "loggers": {"root": {"level": "INFO", "handlers": ["stderr", "file", "file_json"]}},
        "root": {"level": "INFO", "handlers": ["stderr", "file", "file_json"]},
    }

    logging.config.dictConfig(config)
    # logging.setLoggerClass(CustomLogger)


def logging_process(queue):
    """alt: https://stackoverflow.com/questions/28050451/elegant-way-to-make-logging-loggeradapter-available-to-other-modules/28050837#28050837
    Args:
        queue:

    Returns:

    """  # noqa: D414
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            if (
                any(
                    (
                        "gurobipy" in record.name,
                        "botocore" in record.name,
                        "hooks" in record.name,
                    )
                )
                and record.levelno <= logging.INFO
            ):
                continue
            if "cvxpy" in record.name and record.levelno <= logging.INFO:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
            if "pool was terminated abruptly" in record.msg:
                break
            if "numba" not in record.name or record.levelno > 20:
                if not hasattr(record, "ba_code"):
                    record.__dict__.update(
                        {
                            "ba_code": "",
                            "config": "",
                            "tech": "",
                            "status": "",
                            "regime": "",
                            "plant_id": 0,
                        }
                    )

                logger = logging.getLogger(record.name)
                logger.handle(record)
        except Exception:
            import sys
            import traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def combine_runs(*runs):
    now = datetime.now().strftime("%Y%m%d%H%M")
    (colo_dir := Path.home() / f"patio_data/colo_{now}").mkdir()
    (data_dir := colo_dir / "data").mkdir()
    (results_dir := colo_dir / "results").mkdir()
    (results_dir / "hourly").mkdir()
    (results_dir / "coef_mw").mkdir()
    (results_dir / "coef_ts").mkdir()
    to_json = {}
    plants = []
    for run in tqdm(runs, desc="runs", total=len(runs), position=1):
        run_path = Path.home() / "patio_data" / run
        with tqdm(position=2, desc="run parts", leave=False, total=6) as pbar:
            with open(run_path / "colo.json") as cjson:
                json_dict = json.load(cjson)
            to_json.update({run: json_dict})
            plants_ = json_dict["plants"]
            pbar.update(1)
            plants.extend(plants_)
            for file in tqdm(
                (run_path / "data").glob("*.pkl"), desc="data", leave=False, position=3
            ):
                if not (new_file := data_dir / file.name).exists():
                    shutil.copy(file, new_file)
            pbar.update(1)
            for file in tqdm((run_path / "results").glob("*.json"), leave=False, position=3):
                shutil.copy(file, results_dir / file.name)
            pbar.update(1)
            for result_type in ("hourly", "coef_ts", "coef_mw"):
                for file in tqdm(
                    (run_path / "results" / result_type).glob("*.parquet"),
                    desc=result_type,
                    leave=False,
                    position=3,
                ):
                    shutil.copy(file, results_dir / result_type / file.name)
                pbar.update(1)
    to_json["plants"] = plants
    with open(colo_dir / "colo.json", "w") as cjson:
        json.dump(to_json, cjson, indent=4)


def main(
    *,
    test=False,
    colo_dir="",
    keep=True,
    local=True,
    rerun=False,
    workers=1,
    combine="",
    config="",
    plants="",
    source_data="",
):
    multiprocessing.set_start_method("spawn")

    from etoolbox.utils.cloud import put

    if test:
        args = SimpleNamespace(
            dir=colo_dir,
            keep=keep,
            local=local,
            rerun=rerun,
            dry_run=False,
            workers=workers,
            combine=combine,
            assemble=False,
            config=config,
            plants=plants,
            source_data=source_data,
            saved="",
        )
    else:
        import argparse

        parser = argparse.ArgumentParser(description="Run colo.")
        arg_config = [
            {
                "name": "-D, --dir",
                "type": str,
                "default": "",
                "help": "path from home, overrides `run_dir_path` in patio.toml",
            },
            {"name": "-r, --resume", "help": "resume most recent run"},
            {
                "name": "-C, --config",
                "type": str,
                "default": "",
                "help": "path to patio.toml config file relative to home",
            },
            {
                "name": "-L, --source-data",
                "type": str,
                "default": "",
                "help": "path from home, overrides `data_path` and `colo_json_path` in patio.toml",
            },
            {
                "name": "-p, --plants",
                "type": str,
                "default": "",
                "help": "override of plants to run",
            },
            {
                "name": "-w, --workers",
                "type": int,
                "help": "number of workers in colo analysis, default is core_count / 2 for Apple Silicon, otherwise core_count - 1",
                "default": -1,
            },
            {"name": "-k, --keep", "help": "keep intermediate files"},
            {"name": "-A, --assemble", "help": "assemble and upload finished results"},
            {"name": "-l, --local", "help": "do not upload to Azure"},
            {
                "name": "-S, --saved",
                "help": "re-run all plants/configs",
                "type": str,
                "default": "",
            },
            {"name": "-d, --dry-run", "help": "file deletion plan and exit"},
        ]
        for conf in arg_config:
            if "type" not in conf:
                conf.update({"action": "store_true", "default": False})
            parser.add_argument(*conf.pop("name").split(", "), **conf, required=False)
        args = parser.parse_args()
    print(args)
    if args.config != "":  # noqa: SIM108
        toml_path = Path.home() / args.config
    else:
        toml_path = ROOT_PATH / "patio.toml"

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)["colo"]
    if args.dir != "":
        config["project"]["run_dir_path"] = args.dir
    elif args.resume:
        config["project"]["run_dir_path"] = str(
            sorted(
                Path.home().glob(f"{Path(config['project']['run_dir_path']).parent}/colo_20*")
            )[-1].relative_to(Path.home())
        )
        print(config["project"]["run_dir_path"])
    else:
        config["project"]["run_dir_path"] = config["project"]["run_dir_path"].replace(
            "{NOW}", datetime.now().strftime("%Y%m%d%H%M")
        )

    if args.source_data != "":
        config["project"]["source_data_path"] = Path.home() / args.source_data
        config["project"]["colo_json_path"] = Path.home() / args.source_data / "colo.json"

    colo_dir = Path.home() / config["project"]["run_dir_path"]
    if not colo_dir.exists():
        if not colo_dir.parent.exists():
            raise FileNotFoundError(
                f"{colo_dir} and its parent directory do not exist, please create the parent directory"
            )
        colo_dir.mkdir()
    shutil.copy(toml_path, colo_dir / toml_path.name)

    if args.plants != "":
        config["project"]["plant_ids"] = [int(x.strip()) for x in args.plants.split(",")]

    if (workers := args.workers) != -1:
        config["workers"] = workers
    shutil.copy(Path.home() / config["project"]["colo_json_path"], colo_dir)
    _add_to_json(
        colo_dir / "colo.json",
        {"created": colo_dir.stem.replace("colo_", ""), "colo_run_commit": _git_commit_info()},
    )
    config["scenario"]["default"]["setting"]["saved_select"] = args.saved

    queue = multiprocessing.Manager().Queue(-1)
    lp = threading.Thread(target=logging_process, args=(queue,))
    setup_logger(colo_dir / "log")
    lp.start()

    try:
        if not args.assemble:
            run_colo_mp(config, queue)

        assemble_results(colo_dir, keep=args.keep)
        if not args.local:
            put(colo_dir, "patio-results/")
    finally:
        queue.put(None)
        lp.join()


def _add_to_json(json_path, add_to_json):
    with open(json_path) as cjson:
        json_plants = json.load(cjson)
    for k, v in add_to_json.items():
        if k not in json_plants:
            json_plants[k] = v
    with open(json_path, "w") as cjson:
        json.dump(json_plants, cjson, indent=4)


def break_summary(summary):
    for row in summary.iter_rows(named=True):
        file = f"{row['ba_code']}_{row['icx_id']}_{row['tech']}_{row['status']}_{row['regime']}_{row['ix']}.json"
        with open(f"/Users/alex/patio_data/colo_202507070053/results/{file}", "w") as f:
            json.dump(row, f, indent=4)


if __name__ == "__main__":
    main()


class Results:
    id_cols = ("ba_code", "icx_id", "icx_gen", "regime", "name")
    renamer = pl.col("name").replace(
        {"form_fos": "form_new_fossil", "pure_surplus": "new_fossil"}
    )
    colors = [
        "#003B63",
        "#45CFCC",
        "#58595B",
        "#92AEC5",
        "#135908",
        "#00CC66",
        "#FFCB00",
        "#E04D39",
        "#3DADF2",
        "#016A75",
        "#55A603",
        "#D9A91A",
        "#BF0436",
        "#0066B3",
        "#04A091",
        "#3B7302",
        "#EE8905",
        "#73020C",
    ]

    def __init__(
        self,
        *results,
        min_served=1,
        min_pct_load_clean=0.6,
        max_ppa=200,
        max_reg_rank=9,
        max_violation=0.25,
        max_attr_rev_export_clean=100,
        font="Gill Sans",
    ):
        self.fs = rmi_cloud_fs()
        self.min_served = min_served
        self.min_pct_load_clean = min_pct_load_clean
        self.max_ppa = max_ppa
        self.max_reg_rank = max_reg_rank
        self.max_violation = max_violation
        self.max_attr_rev_export_clean = max_attr_rev_export_clean
        self.font = font
        self.names = sorted(r.removeprefix("colo_") for r in results)
        self.name_indices = dict(enumerate(self.names)) | {int(r): r for r in self.names}
        with (
            all_logging_disabled(),
            self.fs.open(f"patio-results/colo_{self.last}/patio.toml") as f,
        ):
            self.config = tomllib.load(f)
        self.pudl_release = self.config["project"]["pudl_release"]
        with all_logging_disabled():
            self.ad = AssetData(pudl_release=self.pudl_release)
        self.summaries: dict[str, pl.DataFrame] = {}
        self.flows: dict[str, pl.DataFrame] = {}
        self.load()
        self.fig_path = Path.home() / "patio_data/figures"
        if not self.fig_path.exists():
            if not self.fig_path.parent.exists():
                self.fig_path.parent.mkdir()
            self.fig_path.mkdir()

    def update_params(self, **kwargs):
        update = False
        for k, v in kwargs.items():
            if k in (
                "min_served",
                "min_pct_load_clean",
                "max_ppa",
                "max_reg_rank",
                "max_violation",
                "max_attr_rev_export_clean",
            ):
                setattr(self, k, v)
                update = True
        if update:
            self.load()

    def load(self):
        for run in self.names:
            try:
                self.summaries[run] = self.get_summary(run)
            except Exception as exc:
                print(f"no summary for {run} {exc!r}")
            try:
                self.flows[run] = self.get_flows(run)
            except Exception as exc:
                print(f"no flows for {run} {exc!r}")

    def get_aligned(self, run=None, *, land_screen=False):
        if run is None:
            run = self.last
        run = self.norm_name(run)
        df = self.summaries[run]
        if land_screen:
            df = df.filter(pl.col("land_available"))
        return self.best_at_site(df)

    @property
    def last(self):
        return max(self.names)

    def good_screen[T: pl.DataFrame | pl.LazyFrame](self, df: T) -> T:
        return df.filter(
            (pl.col("run_status") == "SUCCESS")
            & (pl.col("served_pct") >= self.min_served)
            & (pl.col("pct_load_clean") >= self.min_pct_load_clean)
            & (pl.col("ppa_ex_fossil_export_profit") <= self.max_ppa)
            & (pl.col("reg_rank") <= self.max_reg_rank)
            & (pl.col("attr_rev_export_clean") < self.max_attr_rev_export_clean)
        ).filter(
            (pl.col("new_fossil_mw").fill_null(0.0) == 0.0)
            | (
                (pl.col("new_fossil_mw") > 0.0)
                & (pl.col("max_rolling_violation_pct") <= self.max_violation)
            )
        )

    def best_at_site[T: pl.DataFrame | pl.LazyFrame](self, df: T) -> T:
        colo_cols = df.lazy().collect_schema().names()
        return (
            df.pipe(self.good_screen)
            .sort("load_mw", descending=True)
            .group_by("icx_id", "icx_gen")
            .agg(pl.all().first())
            .select(*colo_cols)
            .sort("state", "utility_name_eia", "plant_name_eia")
        )

    def get_flows(self, run) -> pl.DataFrame:
        run = self.norm_name(run)
        with (
            all_logging_disabled(),
            self.fs.open(f"az://patio-results/colo_{run}/colo_flows.parquet") as f,
        ):
            flows_ = pl.read_parquet(f)
        repl = {"export_req": "Required Export", "export_addl": "Surplus Export"}

        out = (
            flows_.group_by(*self.id_cols)
            .agg(cs.contains("__").sum())
            .unpivot(on=cs.contains("__"), index=self.id_cols)
            .select(
                *self.id_cols,
                pl.col("variable")
                .str.split("__")
                .list.last()
                .replace(repl)
                .str.to_titlecase()
                .str.replace("_", " ")
                .alias("source_"),
                pl.col("variable")
                .str.split("__")
                .list.first()
                .replace(repl)
                .str.to_titlecase()
                .alias("target_"),
                "value",
            )
        )
        self.summaries[run] = (
            self.summaries[run]
            .join(
                out.select(
                    *self.id_cols, has_flows=pl.sum("value").over(*self.id_cols) > 0
                ).unique(),
                on=self.id_cols,
                how="left",
            )
            .with_columns(has_flows=pl.col("has_flows").fill_null(False))
        )
        return out

    def get_summary(self, run) -> pl.DataFrame:
        run = self.norm_name(run)
        with (
            all_logging_disabled(),
            self.fs.open(f"az://patio-results/colo_{run}/colo_summary.parquet") as f,
        ):
            colo_ = pl.read_parquet(f)

        if "fuel" not in colo_.columns:
            if run < "202507070053":
                colo_ = colo_.with_columns(fuel=pl.lit("natural_gas"))
            else:
                with (
                    all_logging_disabled(),
                    self.fs.open(f"az://patio-results/colo_{run}/colo.json") as f,
                ):
                    colo_ = colo_.join(
                        pl.from_dicts(json.load(f)["plants"])
                        .select("tech", "status", "fuel", icx_id="pid")
                        .unique(),
                        on=["tech", "status", "icx_id"],
                        how="left",
                        validate="m:1",
                    )

        try:
            with (
                all_logging_disabled(),
                self.fs.open(f"az://patio-results/colo_{run}/colo_re_selected.parquet") as f,
            ):
                re_select = (
                    pl.read_parquet(f)
                    .select(
                        *self.id_cols,
                        "re_site_id",
                        "re_type",
                        "area_per_mw",
                        "capacity_mw",
                        used_acres=pl.col("capacity_mw") * pl.col("area_per_mw") * 247,
                    )
                    .group_by(*self.id_cols, "re_type")
                    .agg(pl.col("used_acres", "capacity_mw").sum())
                    .with_columns(
                        mw_per_acre=pl.col("capacity_mw") / pl.col("used_acres"),
                        acre_per_mw=pl.col("used_acres") / pl.col("capacity_mw"),
                    )
                    .pivot(on="re_type", index=self.id_cols, values="used_acres")
                    .select(
                        *self.id_cols,
                        pl.col("solar").alias("solar_acres"),
                        pl.col("onshore_wind").alias("onshore_wind_acres"),
                    )
                )
            colo_ = colo_.join(re_select, on=self.id_cols, how="left", validate="1:1")
        except:  # noqa: E722
            pass

        utils = (
            colo_.select(
                "icx_id", "icx_gen", pl.col("icx_gen").str.split(",").alias("generator_id")
            )
            .explode("generator_id")
            .unique()
            .join(
                pl.from_pandas(self.ad.own).rename({"plant_id_eia": "icx_id"}),
                on=["icx_id", "generator_id"],
                how="left",
                validate="1:m",
            )
            .sort(
                "icx_id",
                "fraction_owned",
                "owner_utility_id_eia",
                descending=[False, True, True],
            )
            .with_columns(
                single_owner=pl.col("owner_utility_id_eia")
                .n_unique()
                .over("icx_id", "icx_gen")
                == 1,
                single_parent=pl.col("parent_name").n_unique().over("icx_id", "icx_gen") == 1,
            )
            .group_by("icx_id", "icx_gen")
            .agg(
                pl.col(
                    "owner_utility_id_eia",
                    "owner_utility_name_eia",
                    "fraction_owned",
                    "parent_name",
                    "utility_type_rmi",
                    "single_owner",
                    "single_parent",
                ).first()
            )
        )
        cols = ["latitude", "longitude", "plant_name_eia", "state", "county"]
        with_utils = (
            colo_.join(
                pl.from_pandas(self.ad.gens)
                .select(
                    pl.col("plant_id_eia").alias("icx_id"),
                    pl.col("operational_status").alias("status"),
                    pl.col("technology_description")
                    .replace_strict({v: k for k, v in TECH_CODES.items()}, default=None)
                    .alias("tech"),
                    *cols,
                )
                .group_by("icx_id", "tech", "status")
                .agg(pl.col(*cols).mode().first()),
                on=["icx_id", "tech", "status"],
                how="left",
                validate="m:1",
                suffix="_adgens",
            )
            .select(~cs.contains("_adgens"))
            .join(
                utils,
                on=["icx_id", "icx_gen"],
                validate="m:1",
                how="left",
                suffix="_utils",
            )
            .select(~cs.contains("_utils"))
        )
        rates = (
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
            .filter(
                (pl.col("report_date") == pl.col("report_date").max())
                & pl.col("customer_class").is_in(("commercial", "industrial"))
                & (pl.col("sales_mwh") > 0)
                & (pl.col("service_type") == "bundled")
            )
            .group_by("utility_name_eia", "utility_id_eia", "customer_class")
            .agg(pl.col("sales_revenue", "sales_mwh").sum())
            .with_columns(rate=pl.col("sales_revenue") / pl.col("sales_mwh"))
            .collect()
            .pivot("customer_class", index="utility_name_eia", values="rate")
            .select(
                pl.col("utility_name_eia")
                .replace({"TXU Energy Retail Co, LLC": "TXU Electric Co"})
                .alias("utility_name_eia_lse_"),
                pl.col("commercial").alias("com_rate"),
                pl.col("industrial").alias("ind_rate"),
            )
        )
        r_mw = [f"{t}_mw" for t in ("solar", "onshore_wind", "li", "fe", "new_fossil")]
        out = (
            with_utils.with_columns(
                cf_hist_ratio=pl.col("fossil_cf") / pl.col("historical_fossil_cf"),
                region=pl.col("ba_code").replace(REGION_MAP),
                pct_land_used=1 - pl.col("unused_sqkm_pct"),
                pct_chg_system_co2=pl.col("redispatch_sys_co2") / pl.col("baseline_sys_co2")
                - 1,
                pct_chg_system_cost=pl.col("redispatch_sys_cost_disc")
                / pl.col("baseline_sys_cost_disc")
                - 1,
                pct_overbuild=pl.sum_horizontal(cs.by_name(*r_mw, require_all=False))
                / pl.col("load_mw"),
                regimes=pl.col("regime").n_unique().over("icx_id", "icx_gen", "name"),
                own_short=pl.col("owner_utility_name_eia")
                .str.split(" ")
                .list.gather([0, 1], null_on_oob=True)
                .list.join(" "),
                plant_name_short=pl.col("plant_name_eia")
                .str.split(" ")
                .list.gather([0, 1], null_on_oob=True)
                .list.join(" "),
                utility_name_eia_lse_=pl.col("utility_name_eia_lse")
                .str.split("|")
                .list.first(),
                balancing_authority_code_eia=pl.col("balancing_authority_code_eia").replace(
                    {"CPLE": "DUK"}
                ),
            )
            .join(rates, on="utility_name_eia_lse_", how="left", validate="m:1")
            .select(pl.exclude("utility_name_eia_lse_"))
            .join(
                self._cc_cost(),
                on=["balancing_authority_code_eia", "state"],
                how="left",
                validate="m:1",
            )
        )
        try:
            out = (
                out.join(LAND, on="icx_id", how="left", validate="m:1")
                .with_columns(
                    required_acres=pl.sum_horizontal(
                        "solar_acres", "onshore_wind_acres", "load_mw"
                    )
                )
                .with_columns(
                    land_available=pl.col("buildable_acres").fill_null(0.0).fill_nan(0.0) * 1.5
                    >= pl.col("required_acres")
                )
            )
        except KeyError:
            pass
        out = out.join(
            self.best_at_site(out.filter(pl.col("land_available"))).select(
                *self.id_cols, best_at_site=pl.lit(True)
            ),
            on=self.id_cols,
            how="left",
            validate="m:1",
        ).with_columns(best_at_site=pl.col("best_at_site").fill_null(False))
        assert (
            out.group_by("icx_id", "icx_gen").agg(pl.sum("best_at_site"))["best_at_site"].max()
            == 1
        )
        return (
            out.join(
                self.good_screen(out).select(*self.id_cols, good=pl.lit(True)),
                on=self.id_cols,
                how="left",
            )
            .with_columns(good=pl.col("good").fill_null(False))
            .pipe(order_columns)
            .sort("state", "utility_name_eia_lse", "icx_id")
        )

    def _cc_cost(self) -> pl.DataFrame:
        atb = (
            clean_atb(
                case="Market",
                scenario="Moderate",
                report_year=2024,
                pudl_release=self.pudl_release,
            )
            .filter(
                (pl.col("technology_description") == "Natural Gas Fired Combined Cycle")
                & (pl.col("projection_year") == 2026)
            )
            .select("heat_rate_mmbtu_per_mwh", "fom_per_kw", "vom_per_mwh")
            .to_dict(as_series=False)
        )
        r_disc = ((1 + COSTS["discount"]) / (1 + COSTS["inflation"])) - 1
        return (
            pl_scan_pudl(
                "core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type",
                self.pudl_release,
            )
            .filter(
                (pl.col("report_year") == pl.col("report_year").max())
                & (pl.col("fuel_type_eiaaeo") == "natural_gas")
                & (pl.col("projection_year") >= 2026)
                & (pl.col("model_case_eiaaeo") == "reference")
            )
            .select(
                pl.col("electricity_market_module_region_eiaaeo").cast(pl.Utf8),
                "projection_year",
                fuel_cost_per_mwh=pl.col("fuel_cost_per_mmbtu")
                * atb["heat_rate_mmbtu_per_mwh"][0]
                / ((1 + COSTS["discount"]) ** (pl.col("projection_year") - 2026)),
                den=pl.lit(1) / ((1 + r_disc) ** (pl.col("projection_year") - 2026)),
            )
            .group_by("electricity_market_module_region_eiaaeo")
            .agg(pl.col("fuel_cost_per_mwh", "den").sum())
            .collect()
            .join(
                AEO_MAP,
                on="electricity_market_module_region_eiaaeo",
                how="inner",
                validate="1:m",
            )
            .select(
                "balancing_authority_code_eia",
                "state",
                cc_lcoe=(pl.col("fuel_cost_per_mwh") / pl.col("den"))
                + (4 / 3)
                * (2400 * f_pmt(r_disc, 20, -1) + atb["fom_per_kw"][0])
                / (8.760 * 0.75)
                + atb["vom_per_mwh"][0],
            )
        )

    def parse_log(self, run: str, *, local: bool = False) -> pl.DataFrame:
        run = self.norm_name(run)
        if local:
            return pl.read_ndjson(Path.home() / f"patio_data/{run}/log.jsonl")
        else:
            with self.fs.open(f"patio-results/colo_{run}/log.jsonl") as f:
                return pl.read_ndjson(f)

    def norm_name(self, run: str | int) -> str:
        if isinstance(run, int):
            return self.name_indices[run]
        return run.removeprefix("colo_")

    def summary_stats(self, cols=()):
        weights = pl.col("load_mw") / pl.col("load_mw").sum()
        return pl.concat(
            self.get_aligned(r).select(
                run=pl.lit(r),
                load_mw=pl.col("load_mw").sum(),
                weighted_ppa=(pl.col("ppa_ex_fossil_export_profit") * weights).sum(),
                weighted_pct_clean=(pl.col("pct_load_clean") * weights).sum(),
                weighted_pct_served=(pl.col("served_pct") * weights).sum(),
                **{f"weighted_{c}": (pl.col(c) * weights).sum() for c in cols},
            )
            for r in self.names
        )

    def for_dataroom(self, run=None, *, clip=False) -> pl.DataFrame:
        if run is None:
            run = self.last
        out = (
            self.summaries[self.norm_name(run)]
            .filter(pl.col("good"))
            .with_columns(pl.col("tech").replace(TECH_CODES), name=self.renamer)
            .select(*list(FANCY_COLS))
            .rename(FANCY_COLS)
        )
        if clip:
            out.write_clipboard()
        return out

    def for_xl(self, run=None, *, clip=False) -> pl.DataFrame:
        if run is None:
            run = self.last
        out = self.summaries[self.norm_name(run)].filter(pl.col("run_status") == "SUCCESS")
        if clip:
            out.write_clipboard()
        return out

    def case_sheets(self, run=None, *, land_screen=False) -> list[go.Figure]:
        if run is None:
            run = self.last
        run = self.norm_name(run)
        subplots = self.fig_case_subplots(
            run,
            test=False,
            land_screen=land_screen,
        )

        file = self.fig_path / f"{run} screened.pdf"
        if file.exists():
            file.unlink()
        with PdfWriter(file) as pdf:
            for plot in tqdm(subplots):
                # if "**" not in plot.layout.title.text:
                #     continue
                o = PdfReader(
                    io.BytesIO(
                        plot.to_image(
                            format="pdf", height=plot.layout.height, width=plot.layout.width
                        )
                    )
                )
                for page in o.pages:
                    if page.get_contents() is not None:
                        pdf.add_page(page)
                    else:
                        print(page)
        return subplots

    def fig_case_subplots(self, run=None, *, land_screen=False, test=False) -> list(go.Figure):
        if run is None:
            run = self.last

        colo = self.summaries[run].with_columns(name=self.renamer)
        if land_screen:
            colo = colo.filter(pl.col("land_available"))
        good = self.good_screen(colo)
        pl_good = (
            colo.filter(
                (pl.col("run_status") == "SUCCESS")
                & pl.col("good").max().over("icx_id", "icx_gen")
            )
            .sort("name", "regime")
            .with_columns(
                full_name=pl.concat_str("name", "regime", separator="<br>"),
                attr_rev_ptc=-(pl.col("attr_rev_full_ptc") - pl.col("attr_cost_curtailment")),
                attr_rev_export_clean=-pl.col("attr_rev_export_clean"),
            )
        )
        color_dict = dict(
            zip(
                pl_good.select("full_name").unique(maintain_order=True).to_series().to_list(),
                self.colors,
                strict=False,
            )
        ) | {
            "clean<br>limited": "#005d7f",
            "clean<br>reference": "#9bcce3",
            "dirty<br>limited": "#5f2803",
            "dirty<br>reference": "#a68462",
            "form<br>limited": "#3e3969",
            "form<br>reference": "#a59fce",
            "form_new_fossil<br>limited": "#ffcb05",
            "form_new_fossil<br>reference": "#ffe480",
            "moderate<br>limited": "#556940",
            "moderate<br>reference": "#bac671",
            "new_fossil<br>limited": "#c85c19",
            "new_fossil<br>reference": "#fbbb7d",
        }
        attr_color_dict = dict(
            zip(cs.expand_selector(pl_good, cs.contains("attr_")), self.colors, strict=False)
        )
        good = good.select(
            "icx_id",
            "icx_gen",
            "name",
            "regime",
            a=pl.when(pl.col("load_mw") >= 0).then(pl.lit("<b>")).otherwise(pl.lit("<i>")),
            b=pl.when(pl.col("load_mw") >= 0).then(pl.lit("</b>")).otherwise(pl.lit("</i>")),
        )
        pl_good = (
            pl_good.with_columns(
                color=pl.col("full_name").replace_strict(color_dict, default=None)
            )
            .join(good, on=["icx_id", "icx_gen", "name", "regime"], how="left")
            .with_columns(
                full_name=pl.concat_str(
                    pl.col("a").fill_null(pl.lit("")),
                    "full_name",
                    pl.col("b").fill_null(pl.lit("")),
                )
            )
        )
        flows = (
            self.flows[run]
            .with_columns(
                name=self.renamer,
                full_name=pl.concat_str(self.renamer, "regime", separator="<br>"),
            )
            .join(good, on=["icx_id", "icx_gen", "name", "regime"], how="left")
            .sort("name", "regime")
            .with_columns(
                full_name=pl.concat_str(
                    pl.col("a").fill_null(pl.lit("")),
                    "full_name",
                    pl.col("b").fill_null(pl.lit("")),
                )
            )
        )
        pl_good_parts = (
            pl_good.with_columns(pl.col("net_capex").fill_null(0.0))
            .sort("icx_id")
            .partition_by("icx_id", "icx_gen", as_dict=True)
        )
        rn = {
            "solar": "Solar",
            "onshore_wind": "Wind",
            "li": "Li Storage",
            "new_fossil": "New Fossil",
            "fe": "Fe Storage",
        }
        r_colors = {
            "New Fossil": "#c85c19",
            "Fe Storage": "#3e3969",
            "Solar": "#ffcb05",
            "Wind": "#005d7f",
            "Li Storage": "#7b76ad",
        }
        target_colors = {
            "Required Export": "#58595B",
            "Load": "#003B63",
            "Curtailment": "#eec7b7",
            "Surplus Export": "#808284",
            "Storage": "#7b76ad",
        }
        axes = {
            "xaxis": {"title": "% clean", "tickformat": ".0%", "range": [0.5, 1.05]},
            "yaxis": {"title": "ppa", "tickformat": "$.0f", "range": [0, 350]},
            "yaxis2": {"title": "ppa attribution", "tickprefix": "$"},
            "yaxis3": {"title": "% of load capacity", "tickformat": ".0%"},
            "yaxis4": {"title": "% of load energy", "tickformat": ".0%"},
            "yaxis5": {"title": "% of generation", "tickformat": ".0%", "range": [0, 1.19]},
            "yaxis6": {"title": "fossil capacity factor", "tickformat": ".0%"},
            "yaxis7": {"title": "capex/MW load", "tickprefix": "$"},
            "yaxis8": {"title": "acres", "title_standoff": 1},
        }
        axes = {k: v | {"title_standoff": 1} for k, v in axes.items()}
        subplots = []

        for j, (keys, pid) in enumerate(tqdm(pl_good_parts.items())):
            try:
                pid = pid.sort("name", "regime")
                names = pid.select("full_name").to_series()
                rows = max(8, len(pid))
                title = list(
                    pid.select(
                        pl.concat_str(
                            "balancing_authority_code_eia",
                            "plant_name_eia",
                            pl.col("tech").str.to_uppercase(),
                            "utility_name_eia",
                            separator=" ",
                            ignore_nulls=True,
                        )
                    )
                    .unique()
                    .to_series()
                )
                if len(title) != 1:
                    print(title)
                plt = (
                    make_subplots(
                        rows=rows,
                        cols=2,
                        specs=[[{"type": "scatter"}, {"type": "sankey"}]] * rows,
                        subplot_titles=[k for n in names for k in ["", n]],
                        vertical_spacing=0.025,
                        horizontal_spacing=0.02,
                    )
                    .update_layout(
                        title=("** " if keys[0] in good["icx_id"].to_numpy() else "")
                        + title[0],
                        template="ggplot2",
                        font_family=self.font,
                        height=50 + 380 * rows,
                        width=1400,
                        barmode="stack",
                        **axes,
                        legend={
                            "title": "Scenario name",
                            "xref": "paper",
                            "yref": "paper",
                            "x": 1.025,
                            "y": 0.97,
                        },
                        margin_r=200,
                    )
                    .add_annotation(
                        text=f"<b>Bolded</b> Scenario name means <br> Load Served ≥ {self.min_served:.2%}"
                        f"<br> & Hourly Matched Clean ≥ {self.min_pct_load_clean:.0%}<br> & PPA ≤ "
                        f"${self.max_ppa:,.0f}/MWh{'<br>Land confirmed' if land_screen else ''}",
                        xref="paper",
                        yref="paper",
                        align="left",
                        showarrow=False,
                        font_size=12,
                        # font=dict(size=12),
                        y=1,
                        x=1.025,
                        xanchor="left",
                    )
                )
                pid_flows = flows.filter(
                    (pl.col("icx_id") == keys[0])
                    & (pl.col("icx_gen") == keys[1])
                    & pl.col("full_name").is_in(names)
                )
                p_flows = pl.concat(
                    [
                        pl.DataFrame(
                            {
                                "full_name": ["existing fossil"],
                                "target_": ["Required Export"],
                                "value": [
                                    pid_flows.filter(
                                        (pl.col("target_") == "Required Export")
                                        & (pl.col("full_name") == names[0])
                                    )["value"].sum()
                                ],
                                "share": [1.0],
                            }
                        ),
                        pid_flows.group_by("full_name", "target_", maintain_order=True)
                        .agg(pl.sum("value"))
                        .select(
                            "full_name",
                            "target_",
                            value=pl.col("value").sum().over("full_name"),
                            share=pl.col("value") / pl.col("value").sum().over("full_name"),
                        ),
                    ]
                )
                source_flows = (
                    pid_flows.select(
                        "full_name",
                        pl.col("source_").replace({"Onshore Wind": "Wind"}),
                        "target_",
                        load=pl.col("value").sum().over("full_name", "target_"),
                        share=pl.col("value").sum().over("full_name", "source_")
                        / pl.col("value").sum().over("full_name", "target_"),
                    )
                ).filter(
                    pl.col("source_").is_in(("Solar", "Wind"))
                    & pl.col("target_").is_in(("Load",))
                )
                bkwgs = dict(showlegend=False, col=1)  # noqa: C408
                plt.add_bar(
                    x=["new combined<br>cycle"],
                    y=pid["cc_lcoe"].unique(),
                    name="CC",
                    base=[0],
                    marker_color="black",
                    row=2,
                    **bkwgs,
                )
                load = pid.select(
                    pl.concat_str(
                        pl.lit("<b>"),
                        pl.col("load_mw").cast(pl.Int32).cast(pl.Utf8),
                        pl.lit(" MW</b>"),
                    )
                ).to_series()
                plt.add_scatter(
                    x=pid["full_name"],
                    y=[
                        pid.select(
                            pl.sum_horizontal("attr_cost_clean", "attr_cost_load_fossil").max()
                        ).item()
                        * 1.05
                    ]
                    * len(pid),
                    text=load,
                    mode="text",
                    row=2,
                    textfont_color="black",
                    **bkwgs,
                )
                base = np.zeros(len(pid))
                costs = {
                    "attr_cost_clean": "Clean Cost",
                    "attr_cost_load_fossil": "Fossil Cost",
                }
                for re in costs:
                    plt.add_bar(
                        x=pid["full_name"],
                        y=pid[re],
                        name=costs[re],
                        text=[costs[re]] * len(pid),
                        base=base,
                        marker_color=attr_color_dict[re],
                        row=2,
                        **bkwgs,
                    )
                    base = base + pid[re].to_numpy()
                base = np.zeros(len(pid))
                revs = {"attr_rev_export_clean": "Export Revenue", "attr_rev_ptc": "PTC"}
                for re in revs:
                    plt.add_bar(
                        x=pid["full_name"],
                        y=pid[re],
                        name=revs[re],
                        text=[revs[re]] * len(pid),
                        base=base,
                        marker_color=attr_color_dict[re],
                        row=2,
                        **bkwgs,
                    )
                    base = base + pid[re].to_numpy()
                plt.add_scatter(
                    x=pid["full_name"],
                    y=pid["ppa_ex_fossil_export_profit"],
                    text=["<b>—PPA—</b>"] * len(pid),
                    textposition=["middle center"] * len(pid),
                    mode="text",
                    textfont_color="white",
                    row=2,
                    **bkwgs,
                )
                plt.add_bar(
                    x=["existing fossil"],
                    y=[0],  # pid[f"fossil_mw"] / pid["load_mw"],
                    name="Fossil",
                    marker_color="black",
                    row=3,
                    **bkwgs,
                )
                plt.add_scatter(
                    x=["existing fossil", *pid["full_name"]],
                    y=[
                        pid.select(
                            (
                                pl.sum_horizontal(
                                    "solar_mw",
                                    "li_mw",
                                    "onshore_wind_mw",
                                    "fe_mw",
                                    "new_fossil_mw",
                                )
                                / pl.col("load_mw")
                            ).max()
                        ).item()
                        * 1.05
                    ]
                    * (len(pid) + 1),
                    text=[f"<b>{pid['fossil_mw'].unique().item():.0f} MW</b>", *load],
                    mode="text",
                    row=3,
                    textfont_color="black",
                    **bkwgs,
                )
                for re in ("solar", "onshore_wind", "li", "new_fossil", "fe"):
                    plt.add_bar(
                        x=pid["full_name"],
                        y=pid[f"{re}_mw"] / pid["load_mw"],
                        name=rn[re],
                        text=rn[re].replace(" ", "<br>"),
                        marker_color=r_colors[rn[re]],
                        row=3,
                        **bkwgs,
                    )
                plt.add_bar(
                    x=["existing fossil"],
                    y=[0],
                    row=4,
                    **bkwgs,
                )
                for (target,), part in source_flows.partition_by(
                    "source_", as_dict=True
                ).items():
                    if target == "Solar":
                        plt.add_scatter(
                            x=part["full_name"],
                            y=[
                                source_flows.group_by("full_name")
                                .agg(pl.sum("share"))["share"]
                                .max()
                                * 1.05
                            ]
                            * len(part),
                            text=[f"<b>{f:,.0f} TWh</b>" for f in part["load"] * 1e-6],
                            row=4,
                            mode="text",
                            **bkwgs,
                        )
                    plt.add_bar(
                        x=part["full_name"],
                        y=part["share"],
                        name=part["source_"].unique().item(),
                        text=part["source_"],
                        marker_color=r_colors[target],
                        row=4,
                        **bkwgs,
                    )
                p_flows = p_flows.partition_by("target_", as_dict=True)
                for target in (
                    "Required Export",
                    "Load",
                    "Storage",
                    "Surplus Export",
                    "Curtailment",
                ):
                    part = p_flows.get((target,), None)
                    if part is None:
                        continue
                    if target == "Required Export":
                        plt.add_scatter(
                            x=part["full_name"],
                            y=[1.16] * len(part),
                            text=[f"<b>{f:,.0f} TWh</b>" for f in part["value"] * 1e-6],
                            row=5,
                            mode="text",
                            **bkwgs,
                        )
                    plt.add_bar(
                        x=part["full_name"],
                        y=part["share"],
                        name=part["target_"].unique().item(),
                        text=part["target_"].str.replace(" ", "<br>"),
                        marker_color=target_colors[target],
                        row=5,
                        **bkwgs,
                    )
                plt.add_bar(
                    x=["historical"],
                    y=pid["historical_fossil_cf"].unique(),
                    name="historical",
                    marker_color="black",
                    row=6,
                    **bkwgs,
                )
                plt.add_bar(x=[""], y=[0], name="", marker_color="black", row=7, **bkwgs)
                plt.add_bar(
                    x=["buildable"],
                    y=pid["buildable_acres"].fill_null(0.0).unique(),
                    marker_color="black",
                    row=8,
                    **bkwgs,
                )
                for i, item in enumerate(pid.iter_rows(named=True), start=1):
                    kwgs = bkwgs | dict(  # noqa: C408
                        x=[item["full_name"]],
                        name=item["full_name"],
                        marker_color=item["color"],
                    )
                    plt.add_scatter(
                        x=[item["pct_load_clean"]],
                        y=[item["ppa_ex_fossil_export_profit"]],
                        name=item["full_name"],
                        marker_size=np.log2(item["load_mw"]) * 2,
                        marker_color=item["color"],
                        # text=item["full_name"],
                        textposition=text_position(item, pid),
                        mode="markers+text",
                        showlegend=True,
                        marker_opacity=0.75,
                        row=1,
                        col=1,
                    )
                    rkwgs = bkwgs | dict(  # noqa: C408
                        x=[0.5, 0.55, 1.05],
                        mode="lines+text",
                        marker_color="black",
                        line_width=0.5,
                        row=1,
                    )
                    if (com_rate := pid["com_rate"][0]) is not None:
                        plt.add_scatter(
                            y=[com_rate] * 3, text=["", "commercial<br>rate", ""], **rkwgs
                        )
                    if (ind_rate := pid["ind_rate"][0]) is not None:
                        plt.add_scatter(
                            y=[ind_rate] * 3, text=["", "industrial<br>rate", ""], **rkwgs
                        )
                    plt.add_bar(y=[item["fossil_cf"]], row=6, **kwgs)
                    plt.add_bar(y=[item["net_capex"] / item["load_mw"]], row=7, **kwgs)
                    plt.add_scatter(
                        x=[item["full_name"]],
                        y=[(pid["net_capex"] / pid["load_mw"]).max() * 1.05],
                        text=[f"<b>${item['net_capex'] * 1e-6:,.0f} M</b>"],
                        row=7,
                        mode="text",
                        **bkwgs,
                    )
                    plt.add_bar(y=[item["required_acres"]], row=8, **kwgs)
                    san = sankey(
                        flows.filter(
                            pl_filter(
                                **{
                                    k: v
                                    for k, v in item.items()
                                    if k in ("icx_id", "icx_gen", "regime", "name")
                                }
                            )
                        )
                    )
                    plt.add_trace(san, row=i, col=2)
                subplots.append(plt)
                if test and j > 25:
                    break
            except Exception as exc:
                logging.error("%s", keys, exc_info=exc)
        return subplots

    def package_econ_data(self, run=None, addl_filter=None, *, local=False):
        if run is None:
            run = self.last
        run = self.norm_name(run)
        colo_dir = Path.home() / "patio_data" / f"colo_{run}"
        if not colo_dir.exists():
            colo_dir.mkdir()
        econ_dir = colo_dir / "econ"
        if not econ_dir.exists():
            econ_dir.mkdir()

        good = self.good_screen(self.summaries[run])
        if addl_filter:
            good = good.filter(addl_filter)
        selector = ~cs.by_dtype(pl.List(pl.Int64)) & ~cs.by_dtype(pl.Duration)
        all_itr = [
            (info, o_ids)
            for info in set(Info.from_df(good))
            for o_ids in (
                good.filter(info.filter()).select("regime", "ix").iter_rows(named=True)
            )
        ]
        for file in tqdm(
            ("summary", "annual", "cost_detail", "flows", "re_selected", "full"), position=0
        ):
            with (
                all_logging_disabled(),
                self.fs.open(f"patio-results/colo_{run}/colo_{file}.parquet") as f,
            ):
                data = pl.read_parquet(f)
            for info, o_ids in tqdm(all_itr, position=1):
                try:
                    ftsv_scr = info.filter(**o_ids)
                    pdir = econ_dir / info.file(**o_ids, suffix="")
                    if not pdir.exists():
                        pdir.mkdir()
                    data.filter(ftsv_scr).select(selector).write_csv(pdir / f"{file}.csv")
                except Exception as e:
                    print(f"{info} {o_ids} {e!r}")

    def compare(
        self,
        a,
        b,
        *,
        on=(
            "balancing_authority_code_eia",
            "plant_name_eia",
            "utility_name_eia_lse",
            "owner_utility_name_eia",
            "state",
            "icx_id",
            "tech",
            "status",
            "name",
            "regime",
        ),
        values=("load_mw", "served_pct", "pct_load_clean", "ppa_ex_fossil_export_profit"),
        clip=False,
    ) -> pl.DataFrame:
        c = (
            self.summaries[self.norm_name(a)]
            .filter(pl.col("run_status") == "SUCCESS")
            .select(*on, *values)
            .join(
                self.summaries[self.norm_name(b)]
                .filter(pl.col("run_status") == "SUCCESS")
                .select(*on, *values),
                on=on,
                suffix="_b",
                how="full",
                coalesce=True,
            )
        )
        c = c.select(list(dict.fromkeys(on) | dict.fromkeys(sorted(c.columns))))
        if clip:
            c.write_clipboard()
        return c

    def fig_scatter_geo(self, run=None, sixe_max=25, *, land_screen=False) -> go.Figure:
        if run is None:
            run = self.last
        run = self.norm_name(run)
        return (
            px.scatter_geo(
                self.get_aligned(run, land_screen=land_screen),
                "latitude",
                "longitude",
                hover_name="plant_name_eia",
                hover_data=[
                    "pct_load_clean",
                    "ba_code",
                    "tech",
                    "balancing_authority_code_eia",
                    "state",
                    "utility_name_eia",
                ],
                color="ppa_ex_fossil_export_profit",
                size="load_mw",
                height=800,
                width=900,
                locationmode="USA-states",
                size_max=sixe_max,
            )
            .update_geos(
                fitbounds="locations",
                # landcolor="rgb(230,232,234)",
                scope="usa",
            )
            .update_coloraxes(
                colorbar_title="PPA<br>Price",
                cmax=200,
                cmid=100,
                cmin=0,
                colorscale="portland",
                colorbar_tickformat="$.0f",
                colorbar_len=0.565,
            )
            .update_layout(
                legend_title="Percent<br>Clean Energy",
                showlegend=False,
                font_family=self.font,
                font_size=8,
                template="ggplot2",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
            )
            # .update_traces(textposition=improve_text_position(df["latitude"]))
            .update_traces(
                textposition="top center",
                # marker_opacity=1, marker_line_width=0
            )
        )

    def fig_selection_map(
        self,
        run=None,
        *,
        land_screen=False,
        px_size_max=8,
        fossil_size=0.15,
        selected_size=1,
        potential_size=0.8,
    ) -> go.Figure:
        if run is None:
            run = self.last
        run = self.norm_name(run)
        with (
            all_logging_disabled(),
            self.fs.open(f"patio-results/colo_{run}/all_re_sites.parquet") as f,
        ):
            all_re = (
                pl.scan_parquet(f)
                .cast({"re_site_id": pl.Int32})
                .filter(pl.col("re_type") != "offshore_wind")
                .join(
                    pl_scan_pudl("core_eia860m__changelog_generators", self.pudl_release)
                    .sort("valid_until_date", descending=True)
                    .group_by(pl.col("plant_id_eia").alias("icx_id"))
                    .agg(pl.col("latitude", "longitude").first()),
                    on="icx_id",
                    how="left",
                    validate="m:1",
                )
                .collect()
            )
        with (
            all_logging_disabled(),
            self.fs.open(f"patio-results/colo_{run}/colo_re_selected.parquet") as f,
        ):
            re_selected = pl.read_parquet(f).cast({"re_site_id": pl.Int32})

        potential = (
            all_re.filter(pl.col("distance") < 25)
            .select(
                plant_id="re_site_id",
                tech="re_type",
                icx_id="icx_id",
                color=pl.col("re_type").replace(
                    {"solar": "Solar", "onshore_wind": "Onshore Wind"}
                ),
                cat=pl.lit("potential"),
                size=potential_size
                * pl.col("capacity_mw_nrel_site").log()
                / pl.col("capacity_mw_nrel_site").log().max().over("icx_id"),
                latitude="latitude_nrel_site",
                longitude="longitude_nrel_site",
            )
            .unique()
            .sort("tech", "size", descending=[False, True])
        )
        selected = (
            self.get_aligned(run, land_screen=land_screen)
            .select("icx_id", "icx_gen", "regime", "name", "latitude", "longitude")
            .join(
                re_selected.with_columns(
                    re_site_id=pl.when(pl.col("re_type") == "solar")
                    .then(pl.lit(1))
                    .otherwise(pl.col("re_site_id"))
                )
                .group_by("icx_id", "icx_gen", "re_type", "re_site_id", "regime", "name")
                .agg(
                    pl.col("latitude_nrel_site", "longitude_nrel_site").first(),
                    pl.sum("capacity_mw"),
                ),
                on=["icx_id", "icx_gen", "regime", "name"],
                how="left",
            )
            .select(
                plant_id="re_site_id",
                tech="re_type",
                icx_id="icx_id",
                color=pl.col("re_type").replace(
                    {"solar": "Solar selected", "onshore_wind": "Onshore Wind selected"}
                ),
                cat=pl.lit("selected"),
                size=selected_size,
                # size=1.5*pl.col("capacity_mw").log(2)
                # / pl.col("capacity_mw").log(2).max().over("icx_id"),
                capacity_mw="capacity_mw",
                capacity_mw_lim=pl.lit(0.0),
                latitude=pl.when(pl.col("re_type") == "onshore_wind")
                .then(pl.col("latitude_nrel_site"))
                .otherwise(pl.col("latitude"))
                .cast(pl.Float64),
                longitude=pl.when(pl.col("re_type") == "onshore_wind")
                .then(pl.col("longitude_nrel_site"))
                .otherwise(pl.col("longitude"))
                .cast(pl.Float64),
            )
            .sort("tech", descending=False)
        )
        fossil = all_re.select(
            plant_id="icx_id",
            tech="icx_tech",
            icx_id="icx_id",
            color=pl.lit("Fossil"),
            cat=pl.lit("fossil"),
            size=pl.lit(fossil_size),
            latitude="latitude",
            longitude="longitude",
        ).unique()

        map_df = pl.concat([potential, selected, fossil], how="diagonal_relaxed").filter(
            pl.col("size") > 0.0
        )
        return (
            px.scatter_geo(
                map_df,
                lat="latitude",
                lon="longitude",
                size="size",
                color="color",
                template="ggplot2",
                hover_data="capacity_mw",
                size_max=px_size_max,
                color_discrete_map={k + " selected": v for k, v in COLOR_MAP.items()}
                | {
                    "Onshore Wind selected": "#529cba",
                    "aOnshore Wind selected": "#529cba",
                    "Fossil selected": "#000000",
                    "Load": "#000000",
                    "Fossil": "#000000",
                    # "Solar": "#fff8d5",
                    # "Onshore Wind": "#ebf5f9",
                    "Solar": "#ffeda9",
                    "Onshore Wind": "#c9e3f0",
                },
            )
            .update_geos(
                fitbounds="locations",
                scope="usa",  # landcolor="rgb(230,232,234)",
            )
            .update_traces(
                marker={"opacity": 1, "line": {"width": 0}}, selector={"mode": "markers"}
            )
        )

    def fig_supply_curve(self, run=None, *, land_screen=False) -> go.Figure:
        if run is None:
            run = self.last
        own_rn = {  # noqa: F841
            "Virginia Electric & Power Co": "Dominion",
            "Duke Energy Progress - (NC)": "Duke",
            "Duke Energy Carolinas, LLC": "Duke",
            "Southern Power Co": "Southern",
            "Old Dominion Electric Coop": "Old Dominion Coop",
            "North Carolina El Member Corp": "Member Corp",
            "SEPG Operating Services, LLC WCP": "SEPG",
            "Middle River Power II, LLC": "Middle River",
            "Doswell Ltd Partnership": "Doswell",
        }
        d_ = (
            self.get_aligned(run, land_screen=land_screen)
            .to_pandas()
            .sort_values(["ppa_ex_fossil_export_profit"])
            .assign(
                load_gw=lambda x: x.load_mw / 1000,
                x_=lambda x: ((x.load_gw.shift(1).fillna(0) / 2) + x.load_gw / 2).astype(
                    float
                ),
                x=lambda x: x.x_.cumsum(),
            )
        )

        return (
            go.Figure(
                go.Bar(
                    x=d_["x"],
                    width=d_["load_gw"],
                    y=d_["ppa_ex_fossil_export_profit"],
                    showlegend=False,
                    # text=d_["plant_name_short"]
                    # + "<br>"
                    # # + d_["state"]+ " "
                    #      + d_["balancing_authority_code_eia"].replace(
                    #     own_rn
                    # ),
                    customdata=d_[
                        [
                            "plant_name_eia",
                            "state",
                            "utility_name_eia",
                            "tech",
                            "solar_mw",
                            "onshore_wind_mw",
                            # "storage_mw",
                            "pct_load_clean",
                            "load_mw",
                        ]
                    ].fillna(0.0),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + "<b>%{text}</b><br><br>"
                    + "PPA: %{y:$,.0f}<br>"
                    + "Load MW: %{customdata[8]:.0f}<br>"
                    + "State: %{customdata[1]}<br>"
                    + "Utility: %{customdata[2]}<br>"
                    + "Tech: %{customdata[3]}<br>"
                    + "ES, Solar, Wind: %{customdata[6]:.0f}, %{customdata[4]:.0f}, %{customdata[5]:.0f}<br>"
                    + "Pct Clean: %{customdata[7]:.1%}<br>"
                    + "<extra></extra>",
                    marker=dict(  # noqa: C408
                        color=d_["pct_load_clean"],
                        cmin=0.5,
                        cmax=1,
                        colorbar=dict(  # noqa: C408
                            title="Percent<br>Clean<br>Energy",
                            tickformat=".0%",
                            outlinewidth=0,
                            tickcolor="rgb(237,237,237)",
                            ticklen=6,
                            ticks="inside",
                        ),
                        colorscale="portland_r",
                        line_width=0.01,
                    ),
                )
            )
            .update_yaxes(
                tickformat="$.0f",
                # range=[0, 100],
                title="PPA Price ($/MWh)",
                # zeroline=True,
                # zerolinecolor="black",
                # zerolinewidth=4,
                # title_standoff=10,
            )
            .update_xaxes(
                tickformat=",.0f",
                title="GW",
                # ticklen=10,
                # title_standoff=10,
            )
            .update_layout(
                template="ggplot2",
                font_family=self.font,
                height=int(500 * 1.25),
                width=int(700 * 1.25),
                margin_t=5,
                margin_b=5,
                margin_l=5,
                margin_r=0,
                # plot_bgcolor="rgb(255,255,255)",
                # paper_bgcolor="rgba(0,0,0,0)",
            )
            # .update_traces(textangle=-90)
            # .add_hline(y=100, row="all", col="all", line_color="white", line_width=1, opacity=1)
            # .write_html("varw.html", include_plotlyjs="cdn")
            # .write_image("varw_msft.pdf")
            # .write_image(f"{runs[r_num]}_var_width.pdf", height=580 * 0.8, width=1300 * 0.8)
            # .write_image(f"varw_meta.pdf")
        )
