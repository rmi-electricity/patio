from __future__ import annotations

import gc
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

import pandas as pd
import polars as pl
import polars.selectors as cs
from tqdm.asyncio import tqdm

from patio.constants import PATIO_PUDL_RELEASE, ROOT_PATH, TECH_CODES
from patio.helpers import (
    _git_commit_info,
)
from patio.model.colo_common import (
    FAIL_DISPATCH,
    FAIL_SELECT,
    FAIL_SERVICE,
    FAIL_SMALL,
    STATUS,
    SUCCESS,
    SUM_COL_ORDER,
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
                    if "ppa" in result and result.get("load_mw", 0) >= max(
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
        if "ppa" in result and result.get("load_mw", 0) >= max(
            50.0, mround(info.cap * 0.5, 25)
        ):
            LOGGER.info("skipping, good limited version exists", extra=xt)
            continue
        model_colo_config(conf, c_dir, d_dir, info)
    return None


def setup_plants_configs(config, plants, pudl_release=PATIO_PUDL_RELEASE, **kwargs):
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
        model.to_file()
    except Exception as exc:
        logger.error("%r", exc, extra=model.extra)
        logger.info("%r", exc, exc_info=exc, extra=model.extra)
        model.errors.append(repr(exc))
        model.to_file()
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
    model.add_df(hourly=hourly.collect(), full=econ_df, flows=flows)

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
            if not keep:
                for file in result_dir.glob("*.json"):
                    file.unlink()
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


if __name__ == "__main__":
    main()
