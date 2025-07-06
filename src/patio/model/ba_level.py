"""a model I first heard of while on a patio

This module contains the core logic of the model.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence  # noqa: TC003
from contextlib import nullcontext, suppress
from datetime import datetime, timedelta
from functools import cached_property, lru_cache
from io import BytesIO
from pathlib import Path
from traceback import TracebackException
from typing import Any, Literal

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
from dispatch import DispatchModel
from dispatch.constants import COLOR_MAP, PLOT_MAP
from dispatch.helpers import dispatch_key, zero_profiles_outside_operating_dates
from etoolbox.datazip import DataZip, IOMixin
from etoolbox.utils.cloud import (
    AZURE_CACHE_PATH,
    cached_path,
    rmi_cloud_fs,
)
from etoolbox.utils.cloud import (
    put as rmi_cloud_put,
)
from etoolbox.utils.pudl import pl_scan_pudl
from etoolbox.utils.pudl_helpers import simplify_columns
from pandas.util import hash_pandas_object
from platformdirs import user_documents_path
from scipy.optimize import Bounds, LinearConstraint, milp
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from patio.constants import (
    CLEAN_TD_MAP,
    ES_TECHS,
    FOSSIL_PRIME_MOVER_MAP,
    MTDF,
    OTHER_TD_MAP,
    PATIO_DOC_PATH,
    PATIO_PUDL_RELEASE,
    REGION_MAP,
    ROOT_PATH,
)
from patio.data.asset_data import AssetData

# from gencost.crosswalk import _prep_for_networkx, _subplant_ids_from_prepped_crosswalk
# from gencost.entity_ids import add_ba_code
from patio.data.entity_ids import add_ba_code
from patio.data.profile_data import ProfileData
from patio.exceptions import (
    NoMaxRE,
    PatioData,
    ScenarioError,
)
from patio.helpers import _git_commit_info, make_core_lhs_rhs, pl_distance, solver
from patio.model.ba_scenario import BAScenario
from patio.model.base import ScenarioConfig
from patio.model.colo_common import capture_stderr, capture_stdout
from patio.model.colo_core import BAD_COLO_BAS

LOGGER = logging.getLogger("patio")
# warnings.simplefilter("once")
logging.captureWarnings(True)
__all__ = ["BA", "BAs"]
RNG = np.random.default_rng(2021)
# trick the linter
_ = ES_TECHS

if not PATIO_DOC_PATH.exists():
    PATIO_DOC_PATH.mkdir()

MIN_UPTIME_BY_PRIME = pd.Series(
    {
        "ST": 4,  # steam turbine
        "GT": 0,  # gas turbine xCC
        "IC": 1,  # internal combustion
        "CA": 2,  # steam part of CC
        "CT": 2,  # gas turbine part of CC
        "CS": 2,  # single shaft CC
        "CC": 2,  # CC in planning
    }
)
"""CCS adders/factors.

NREL ATB 2022 v2, Coal_Retrofits vs Coal-CCS-90%, 2035 moderate case.
"""

INTERCON_MAP = {
    "130": "eastern",
    "177": "eastern",
    "186": "eastern",
    "195": "eastern",
    "2": "eastern",
    "210": "eastern",
    "22": "eastern",
    "531": "eastern",
    "552": "eastern",
    "556": "eastern",
    "569": "eastern",
    "57": "eastern",
    "58": "eastern",
    "656": "eastern",
    "658": "eastern",
    "AEC": "eastern",
    "AECI": "eastern",
    "APS": "western",
    "CAISO": "western",
    "DUKE": "eastern",
    "EPE": "western",
    "ERCO": "erco",
    "ETR": "eastern",
    "EVRG": "eastern",
    "FMPP": "eastern",
    "FPC": "eastern",
    "FPL": "eastern",
    "ISNE": "eastern",
    "JEA": "eastern",
    "LDWP": "western",
    "LGEE": "eastern",
    "MISO": "eastern",
    "NYIS": "eastern",
    "NEVP": "western",
    "PAC": "western",
    "PJM": "eastern",
    "PNM": "western",
    "PNW": "western",
    "PSCO": "western",
    "SC": "eastern",
    "SCEG": "eastern",
    "SEC": "eastern",
    "SOCO": "eastern",
    "SRP": "western",
    "SWPP": "eastern",
    "TEC": "eastern",
    "TEPC": "western",
    "TVA": "eastern",
    "WACM": "western",
    "WALC": "western",
    "WAUW": "western",
}


# def make_subplant_ids(
#     crosswalk: pd.DataFrame,
#     source_keys: list[str],
#     source_name: str,
#     target_keys: list[str],
#     target_name: str,
#     new_key_name: str,
# ) -> pd.DataFrame:
#     """Identify prime fuel sub-plants in the EPA/EIA crosswalk and EIA 860 graph.
#     Any row filtering should be done before this step.
#     Usage Example:
#     epacems = pudl.output.epacems.epacems(states=['ID']) # small subset for quick test
#     epacamd_eia = pudl_out.epacamd_eia()
#     filtered_crosswalk = filter_crosswalk(epacamd_eia, epacems)
#     crosswalk_with_subplant_ids = make_subplant_ids(filtered_crosswalk)
#     Note that sub-plant ids should be used in conjunction with `plant_id_eia` vs.
#     `plant_id_epa` because the former is more granular and integrated into CEMS during
#     the transform process.
#
#     Args:
#         crosswalk (pd.DataFrame): The epacamd_eia crosswalk
#         source_keys: columns that comprise the source composite key
#         source_name: name of the source composite key column
#         target_keys: columns that comprise the target composite key
#         target_name: name of the target composite key column
#         new_key_name: name of the new subplant column that makes up a
#             composite key with plant_id_eia
#     Returns:
#         pd.DataFrame: An edge list connecting EPA units to EIA generators, with
#             connected pieces issued a subplant_id
#     """
#
#     edge_list = _prep_for_networkx(
#         crosswalk,
#         source_keys=source_keys,
#         target_keys=target_keys,
#         source_name=source_name,
#         target_name=target_name,
#     )
#     edge_list = _subplant_ids_from_prepped_crosswalk(
#         edge_list,
#         source_name=source_name,
#         target_name=target_name,
#         new_key_name=new_key_name,
#     )
#     # edge_list = _convert_global_id_to_composite_id(edge_list, new_key_name=new_key_name)
#     return edge_list


class BA(IOMixin):
    _parquet_out = (
        "plant_data",
        "load",
        "profiles",
        "_re_profiles",
        "re_plant_specs",
        # "fuel_profile",
        # "_co2_profile",
        "cost_data",
        # "cems_change",
        # "mini_re_prof",
        # "old_re_specs",
        # "old_re_profs",
        # "old_re_prof",
        "adj_load",
        "missing",
        "proposed_clean_prof",
        "Ab",
        # "re_prof_ilr_adj",
        "net_load_prof",
        "fuel_curve",
        "augmented_load",
        "existing_re_prof",
        "re_plant_specs_pre_downselect",
    )
    re_spec_cols = [
        "capacity_mw",
        "re_type",
        "technology_description",
        "operating_date",
        "retirement_date",
        "ilr",
        "re_site_id",
        "energy_community",
        "category",
        "class_atb",
    ]

    def __init__(
        self,
        ba_code: str,
        profiles: pd.DataFrame,
        re_profiles: pd.DataFrame,
        re_plant_specs: pd.DataFrame,
        plant_data: pd.DataFrame,
        cost_data: pd.DataFrame,
        # cems_change: pd.DataFrame,
        baseload_replace: dict,
        missing: pd.DataFrame,
        fuel_curve: pd.DataFrame,
        queue,
        regime: str,
        ccs_convert: dict | None = None,
        scenario_configs: list[ScenarioConfig] | None = None,
        # old_re_prof: pd.DataFrame | None = None,
        # old_re_specs: pd.DataFrame | None = None,
        # old_re_profs: pd.DataFrame | None = None,
        existing_re_prof: pd.DataFrame | None = None,
        include_figs: bool = True,  # noqa: FBT001
        exclude_or_mothball: list | None = None,
        exclude: list | None = None,
        max_re: dict | None = None,
        year_mapper: dict | None = None,
        re_limits_dispatch: Literal["both", True, False] = "both",
        max_wind_distance: float = 100.0,
        max_solar_distance: float = 100.0,
        min_re_site_mw: float | None = None,
        cr_eligible_techs: list[str] | None = None,
        colo_techs: list[str] | None = None,
        colo_method: Literal["direct", "iter"] | None = None,
        old_clean_adj_net_load=True,
        colo_dir: str | None = None,
        **kwargs,
    ):
        for scen in scenario_configs:
            assert scen.nuclear_scen == 0, "nuclear scenarios are untested"
            assert scen.ccs_scen == 0, "ccs scenarios are untested"
        try:
            _ = int(ba_code)
            name = plant_data.respondent_name.iat[0]  # noqa: PD009
        except ValueError:
            name = ba_code
        self._metadata = {
            "ba_code": ba_code,
            "name": name,
            "eia_ba": plant_data.balancing_authority_code_eia.iat[0],  # noqa: PD009
            "year_mapper": year_mapper,
            "max_solar_distance": max_solar_distance,
            "max_wind_distance": max_wind_distance,
            "min_re_site_mw": min_re_site_mw,
            "cr_eligible_techs": cr_eligible_techs,
            "colo_techs": colo_techs,
            "regime": regime,
        }
        profiles, self._re_profiles = self.align_profiles(
            profiles.reindex(index=pd.to_datetime(profiles.index.to_series())),
            re_profiles,
        )
        if not scenario_configs:
            # for power we want all years to be the same length
            self._re_profiles = self._re_profiles.sort("datetime").lazy()
            profiles = profiles.sort_index()
            profiles_out = []
            re_profiles_out = []
            for yr in (
                self._re_profiles.select(pl.col("datetime").dt.year())
                .unique()
                .collect()
                .to_series()
            ):
                dfr = self._re_profiles.filter(pl.col("datetime").dt.year() == yr)
                dfp = profiles.query("datetime.dt.year == @yr")
                if len(dfp) > 8760:
                    re_profiles_out.append(
                        pl.concat(
                            [
                                dfr.select("datetime").filter(
                                    (
                                        (pl.col("datetime").dt.month() == 2)
                                        & (pl.col("datetime").dt.day() == 29)
                                    ).not_()
                                ),
                                dfr.select(pl.exclude("datetime")).head(8760),
                            ],
                            how="horizontal",
                        )
                    )
                    profiles_out.append(
                        dfp.head(8760).set_axis(
                            dfp.index.to_frame()
                            .query("~(datetime.dt.month == 2 & datetime.dt.day == 29)")
                            .index,
                            axis=0,
                        )
                    )
                else:
                    profiles_out.append(dfp)
                    re_profiles_out.append(dfr)
            profiles = pd.concat(profiles_out).sort_index()
            self._re_profiles = pl.concat(re_profiles_out).sort("datetime").collect()
            del profiles_out, re_profiles_out

        self.queue = queue
        self._metadata.update(
            colo_techs=colo_techs,
            # colo_config={"years": bad_re_yrs},
            colo_method=colo_method,
        )
        self.colo_dir: str = colo_dir
        max_re_distance = max(max_wind_distance, max_solar_distance)
        if max_re_distance is not None and re_plant_specs.distance.max() > max_re_distance:
            raise ValueError(
                f"max distance in re_plant_specs > {max_re_distance=}; filtering must occur before data is passed to {self.__class__.__qualname__}"
            )
        if (
            min_re_site_mw is not None
            and re_plant_specs.capacity_mw_nrel_site.min() < min_re_site_mw
        ):
            raise ValueError(
                f"min site capacity in re_plant_specs < {min_re_site_mw=}; filtering must occur before data is passed to {self.__class__.__qualname__}"
            )
        # self.old_clean_adj_net_load = old_clean_adj_net_load
        if re_limits_dispatch not in ("both", True, False):
            raise AssertionError("`re_limits_dispatch` must be one of ('both', True, False)")
        self.re_limits_dispatch = re_limits_dispatch
        self.plant_data = plant_data

        self.load = profiles.fillna(0.0)[self.fossil_list].sum(axis=1)
        # this needs to come after we calculate load to not have retirements affect
        # load profile
        self.profiles = zero_profiles_outside_operating_dates(
            profiles.fillna(0.0),
            self.plant_data.loc[profiles.columns, "operating_date"],
            self.plant_data.loc[profiles.columns, "retirement_date"],
        ).astype(np.float32)
        self._re_plant_specs = (
            re_plant_specs.assign(
                category="patio_clean",
                area_per_mw=lambda x: x.area_sq_km / x.capacity_mw_nrel_site,
                ones=1.0,
                icx_genid=lambda x: x.groupby(["icx_id", "icx_gen"]).icx_id.transform(
                    "ngroup"
                ),
                combi_id=lambda x: x[["icx_genid", "re_site_id", "re_type"]]
                .astype(str)
                .agg("_".join, axis=1),
                plant_id_eia=lambda x: hash_pandas_object(
                    x[["icx_genid", "re_site_id", "re_type"]]
                ).astype("int64"),
                cf_mult=lambda x: np.maximum(x.cf_atb / x.cf_ilr_prof_site, 1.0),
                capacity_mw_nrel_site_lim=lambda x: x.capacity_mw_nrel_site_lim.fillna(0.0),
                area_sq_km_lim=lambda x: x.area_sq_km_lim.fillna(0.0),
            )
            .sort_values("combi_id")
            .reset_index(drop=True)
            .merge(
                self.plant_data[["retirement_date", "operating_date", "ever_gas"]]
                .reset_index()
                .rename(columns={"plant_id_eia": "icx_id", "generator_id": "icx_gen"}),
                on=["icx_id", "icx_gen"],
                how="left",
                suffixes=(None, "_icx"),
            )
        )
        if regime == "reference":
            self.re_plant_specs = self._re_plant_specs.query("cr_eligible").copy()
        elif regime == "limited":
            self.re_plant_specs = (
                self._re_plant_specs.assign(
                    capacity_mw_nrel_site_ref=lambda x: x.capacity_mw_nrel_site,
                    area_sq_km_ref=lambda x: x.area_sq_km,
                    capacity_mw_nrel_site=lambda x: x.capacity_mw_nrel_site_lim,
                    area_sq_km=lambda x: x.area_sq_km_lim,
                )
                .query("cr_eligible & area_sq_km > @min_re_site_mw")
                .copy()
            )
        else:
            raise ValueError(f"regime must be one of ('reference', 'limited'), not {regime}")
        re_len = len(self.re_plant_specs)

        self.re_profiles, self.re_prof_ilr_adj = self.setup_re_profiles(self.re_plant_specs)

        self.ccs_convert = ccs_convert
        self.include_figs = include_figs
        self.no_limit_prime = ("CC",)

        self.cost_data = cost_data.loc[
            [i for i in cost_data.index if i[:2] in self.fossil_list], :
        ]
        self.fuel_curve = fuel_curve
        # self.cems_change = cems_change
        self.baseload_replace = baseload_replace
        self.max_re = max_re
        if exclude_or_mothball is None:
            self.exclude_or_mothball = []
        else:
            self.exclude_or_mothball = exclude_or_mothball
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude
        if scenario_configs is None:
            self.scenario_configs = [y for x in BAs.def_scen_configs for y in x.split_ccs()]
        else:
            self.scenario_configs = [y for x in scenario_configs for y in x.split_ccs()]

        self._scenarios: list[BAScenario] = []
        # self.n_years = self.profiles.index.year.nunique()
        # self.align_profiles()
        self.plant_data = self.plant_data.assign(
            # ramp_rate=lambda x: x.capacity_mw / x.ramp_hrs,
            # currently using CEMS ramp because 860 is too imprecise
            cems_ramp=np.maximum(self.profiles - self.profiles.shift(1), 0).max(),
            ramp_rate=lambda x: np.where(
                (x.technology_description == "GT")
                | (x.cems_ramp == 0.0)
                # TODO: assuming all proposed plants can ramp in 1hr is not great
                | (x.operational_status == "proposed"),
                x.capacity_mw,
                x.cems_ramp,
            ),
            # min_uptime to deal with deficits was bad, we're trying dynamic
            # storage reserves instead
            # min_uptime=lambda x: x.prime_mover_code.map(MIN_UPTIME_BY_PRIME),
        ).drop(columns=["cems_ramp"])

        # self.plant_data = self.plant_data.loc[self.profiles.columns, :]
        # self.profiles = self.profiles[self.plant_data.index]
        # assert np.all(
        #     self.plant_data.index == self.profiles.columns
        # ), "plant_data and req_profiles indexes do not match"
        # self.re_plant_specs = self.re_plant_specs
        # adj_load = self.load.to_numpy()

        # annual load should never drop below the leap-year adjusted level of the most
        # recent historical year
        self.augmented_load = self.load

        self.existing_re_prof = existing_re_prof.reindex(
            index=self.profiles.index, fill_value=0.0
        ).fillna(0.0)

        self.adj_load = np.maximum(
            0.0,
            self.augmented_load.to_numpy()
            - self.existing_re_prof.synthetic.to_numpy(dtype=float),
        )

        if self.clean_list:
            self.proposed_clean_prof = (
                zero_profiles_outside_operating_dates(
                    np.minimum(
                        self.profiles.loc[:, self.clean_list]
                        * self.plant_data.loc[self.clean_list, "ilr"],
                        1.0,
                    ),
                    self.plant_data.loc[self.clean_list, "operating_date"],
                    self.plant_data.loc[self.clean_list, "retirement_date"],
                    self.plant_data.loc[self.clean_list, "capacity_mw"],
                )
                .sum(axis=1)
                .to_numpy(float)
            )
        else:
            self.proposed_clean_prof = np.zeros_like(self.adj_load)

        self.net_load_prof = np.maximum(
            self.adj_load.astype(float) - self.proposed_clean_prof,
            0.0,
        )
        self.re_plant_specs = (
            self.re_plant_specs.merge(
                np.minimum(
                    self.re_prof_ilr_adj,
                    np.reshape(
                        self.net_load_prof / self.net_load_prof.max(),
                        (len(self.net_load_prof), 1),
                    ),
                )
                .sum()
                .to_frame("total_norm_match"),
                on=["re_site_id", "generator_id"],
                how="left",
                validate="m:1",
            )
            .merge(
                self.re_prof_ilr_adj.sum().to_frame("re_site_gen"),
                on=["re_site_id", "generator_id"],
                how="left",
                validate="m:1",
            )
            .assign(
                re_cost=lambda x: x.lcoe * x.re_site_gen,
                avoided_cost=lambda x: x.total_var_mwh * x.total_norm_match - x.re_cost,
                re_max_obj_coef=lambda x: x.groupby("re_type").lcoe.rank(
                    pct=True, ascending=False
                ),
            )
        )
        if not (test := self.re_plant_specs.query("total_var_mwh.isna()")).empty:
            LOGGER.warning(
                "%s unable to calculate avoided cost for %s re sites", self.ba_code, len(test)
            )
            self.re_plant_specs = self.re_plant_specs.fillna({"avoided_cost": 0.0})

        self.Ab = make_core_lhs_rhs(self.re_plant_specs)
        self.re_plant_specs_pre_downselect = self.re_plant_specs.copy()
        self.msg = []

        if self.scenario_configs:
            success, en = self.milp(
                c=-self.re_plant_specs.cf_atb,
                desc="energy",
                wind_lb={"reference": 80.0, "limited": 25.0}[self._metadata["regime"]],
            )
            if success is False:
                msg = "max energy MILP failed at BA level -> using LP"
                LOGGER.warning("%s %s", self.ba_code, msg)
                self.msg.append(msg)
                _, en = self.lp(c=-self.re_plant_specs.cf_atb)
            success, save = self.milp(
                c=-(
                    self.re_plant_specs.avoided_cost
                    - self.re_plant_specs.avoided_cost.min()
                    + 1
                ),
                desc="savings",
                wind_lb={"reference": 80.0, "limited": 25.0}[self._metadata["regime"]],
            )
            if success is False:
                msg = "max savings MILP failed at BA level -> using LP"
                LOGGER.warning("%s %s", self.ba_code, msg)
                self.msg.append(msg)
                _, save = self.lp(
                    c=-(
                        self.re_plant_specs.avoided_cost
                        - self.re_plant_specs.avoided_cost.min()
                        + 1
                    )
                )
            success, cap = self.milp(
                c=-self.re_plant_specs.re_max_obj_coef,
                desc="capacity",
                wind_lb={"reference": 80.0, "limited": 25.0}[self._metadata["regime"]],
            )
            if success is False:
                msg = "max capacity MILP failed at BA level -> using LP"
                LOGGER.warning("%s %s", self.ba_code, msg)
                self.msg.append(msg)
                _, cap = self.lp(c=-self.re_plant_specs.re_max_obj_coef)
            ops = pd.concat(
                [
                    en.to_frame("energy_max"),
                    cap.to_frame("cap_max"),
                    save.to_frame("savings_max"),
                ],
                axis=1,
            ).join(
                # want to always include RE where fossil generator is retiring
                self.re_plant_specs.set_index(["icx_genid", "re_site_id", "re_type"])
                .retirement_date_icx.notna()
                .astype(int),
                how="left",
            )
            if len(self.re_plant_specs) > 10000:
                ops = ops[np.any(ops > 0.0, axis=1)]
                msg = f"more than 10k re/fos site combos so downselecting at BA level ({len(self.re_plant_specs)} -> {len(ops)})"
                LOGGER.warning("%s %s", self.ba_code, msg)
                self.msg.append(msg)

            assert len(self.re_plant_specs_pre_downselect) == re_len, (
                "We lost RE sites pre-downselection"
            )

            self.re_plant_specs = self.re_plant_specs.merge(
                ops.reset_index(),
                on=["icx_genid", "re_site_id", "re_type"],
                how="inner",
                validate="1:1",
            )
            if len(self.re_plant_specs) < self.Ab.shape[1] - 1:
                self.Ab = make_core_lhs_rhs(self.re_plant_specs)
            self.max_energy = (
                self.re_plant_specs.energy_max
                @ self.re_plant_specs.cf_atb.to_numpy()
                * len(self.net_load_prof)
            )
        else:
            self.max_energy = 0.0

        self.counterfactual: BAScenario = self.make_cfl()
        self.missing = missing.assign(
            historical_mwh=lambda x: np.maximum(0.0, x.net_generation_mwh),
            historical_mmbtu=lambda x: x.fuel_consumed_for_electricity_mmbtu,
            historical_co2=lambda x: x.historical_mmbtu * x.co2_factor,
            historical_cost_fuel=lambda x: x.historical_mwh * x.fuel_per_mwh,
            historical_cost_vom=lambda x: x.historical_mwh * x.vom_per_mwh,
            historical_cost_fom=lambda x: x.capacity_mw * x.fom_per_kw * 1000,
            redispatch_mwh=lambda x: x.historical_mwh,
            redispatch_mmbtu=lambda x: x.fuel_consumed_for_electricity_mmbtu,
            redispatch_co2=lambda x: x.redispatch_mmbtu * x.co2_factor,
            redispatch_cost_fuel=lambda x: x.redispatch_mwh * x.fuel_per_mwh,
            redispatch_cost_vom=lambda x: x.redispatch_mwh * x.vom_per_mwh,
            redispatch_cost_fom=lambda x: x.capacity_mw * x.fom_per_kw * 1000,
        ).drop(
            columns=[
                "net_generation_mwh",
                "fuel_consumed_for_electricity_mmbtu",
                # "fuel_consumed_mmbtu",
                "vom_per_mwh",
                "fuel_per_mwh",
                "total_var_mwh",
                "fom_per_kw",
                "start_per_kw",
                "heat_rate",
                "co2_factor",
            ]
        )
        if "technology_description" not in self.missing:
            self.missing = self.missing.rename(
                columns={"technology_description_x": "technology_description"}
            )
        # if self.old_clean_adj_net_load:
        #     self.missing = pd.concat(
        #         [
        #             self.missing,
        #             self.old_re_specs.reset_index()
        #             .assign()
        #             .merge(
        #                 (self.old_re_profs * self.old_re_specs.capacity_mw)
        #                 .groupby(pd.Grouper(freq="YS"))
        #                 .sum()
        #                 .stack([0, 1])
        #                 .reset_index(name="redispatch_mwh"),
        #                 on=["plant_id_eia", "generator_id"],
        #                 validate="1:m",
        #             )
        #             .set_index(["plant_id_eia", "generator_id", "datetime"])
        #             .assign(
        #                 historical_mwh=lambda x: x.redispatch_mwh, final_ba_code=ba_code
        #             ),
        #         ]
        #     )

    def setup_re_profiles(self, re_specs):
        re_pro_help = re_specs.groupby(["re_site_id", "generator_id"])[
            ["cf_mult", "operating_date", "retirement_date", "ilr", "combined"]
        ].first()

        re_profiles_large = zero_profiles_outside_operating_dates(
            np.minimum(
                self._re_profiles.to_pandas()
                .set_index("datetime")[re_pro_help.combined]
                .set_axis(re_pro_help.index, axis=1)
                * re_pro_help.cf_mult,
                1.0,
            ),
            re_pro_help.operating_date,
            re_pro_help.retirement_date,
        )
        re_prof_ilr_adj = np.minimum(re_profiles_large * re_pro_help.ilr, 1.0)
        return re_profiles_large, re_prof_ilr_adj

    def make_mini_re_profiles(self):
        """Create aggregated RE profiles to use in ``equal_energy``."""
        if self.re_profiles.columns.nlevels == 1:
            return self.re_profiles
        mini_meta = (
            self.re_plant_specs.astype({"plant_id_eia": int})
            .groupby(["plant_id_eia", "generator_id"])
            .agg({"ba_weight": "sum", "ilr": "first"})
            .sort_index()
        )
        pd.testing.assert_index_equal(
            mini_meta.index,
            self.re_profiles.columns,
            check_names=False,
            check_order=True,
        )
        mini_re_prof = (
            (
                np.minimum(self.re_profiles * mini_meta.ilr.to_numpy(), 1.0)
                * mini_meta.ba_weight.to_numpy()
            )
            .groupby(level=1, axis=1)
            .sum()
        )
        re_t = ["onshore_wind", "solar", "offshore_wind"]
        fill = {x: 0.0 for x in re_t if x not in mini_re_prof}
        return mini_re_prof.assign(**fill)[re_t]

    def lp(
        self,
        Ab_eq: pd.DataFrame | None = None,  # noqa: N803
        Ab_ub: pd.DataFrame | None = None,  # noqa: N803
        c: pd.Series | None = None,
        lb: np.ndarray | None = None,
        time_limit=120,
    ) -> tuple[bool, pd.Series | str]:
        solver_ = solver()
        options = {
            cp.GUROBI: {
                "TimeLimit": time_limit,
                "Threads": 8,
                # "LogToConsole": 1,
                "reoptimize": True,
            },
            cp.HIGHS: {"time_limit": time_limit, "parallel": "on"},
            cp.COPT: {"TimeLimit": time_limit},
        }[solver_]

        Ab_ub = self.Ab if Ab_ub is None else pd.concat([self.Ab, Ab_ub])
        c = -1 * self.re_plant_specs.avoided_cost if c is None else c
        A_eq = None if Ab_eq is None else Ab_eq.iloc[:, :-1].to_numpy()
        b_eq = None if Ab_eq is None else Ab_eq.iloc[:, -1].to_numpy()
        lb = np.zeros_like(c) if lb is None else lb

        ub = (
            self.re_plant_specs.capacity_max_re_scen
            if "capacity_max_re_scen" in self.re_plant_specs
            else self.re_plant_specs.capacity_mw_nrel_site
        ).to_numpy()
        assert np.all(ub >= lb), "lb > ub"

        x = cp.Variable(self.re_plant_specs.shape[0])
        Ab_ub = Ab_ub.to_numpy()
        cons = [Ab_ub[:, :-1] @ x <= Ab_ub[:, -1], x >= lb, x <= ub]
        if Ab_eq is not None:
            cons.append(A_eq @ x == b_eq)

        p = cp.Problem(cp.Minimize(c.to_numpy() @ x), cons)
        p.solve(solver_, **options)
        if p.status == cp.OPTIMAL:
            return True, pd.Series(
                x.value,
                index=pd.MultiIndex.from_frame(
                    self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
                ),
            )
        return False, p.status

        # r = linprog(
        #     A_ub=Ab_ub.iloc[:, :-1].to_numpy(),
        #     b_ub=Ab_ub.iloc[:, -1].to_numpy(),
        #     A_eq=A_eq,
        #     b_eq=b_eq,
        #     c=c.to_numpy(),
        #     # bounds=(0, None),
        #     bounds=list(iter(np.vstack((lb, ub)).transpose())),
        #     method="highs",
        # )
        # if r.success is False:
        #     return False, r.message
        # return True, pd.Series(
        #     r.x,
        #     index=pd.MultiIndex.from_frame(
        #         self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
        #     ),
        # )

    def milp(
        self,
        c: pd.Series | None = None,
        Ab_eq: pd.DataFrame | None = None,  # noqa: N803
        Ab_ub: pd.DataFrame | None = None,  # noqa: N803
        lb: np.ndarray | None = None,
        wind_lb: float | None = 80.0,
        time_limit: float = 60,
        mip_rel_gap=1.5e-4,
        desc: str = "",
        **kwargs,
    ) -> tuple[bool, pd.Series | str]:
        c_ = -1 * self.re_plant_specs.avoided_cost if c is None else c
        Ab_ub_ = self.Ab if Ab_ub is None else pd.concat([self.Ab, Ab_ub])
        lb = np.zeros_like(c_) if lb is None else lb
        if isinstance(lb, pd.Series):
            lb = lb.to_numpy()
        solver_ = solver()
        options = {
            cp.GUROBI: {
                "TimeLimit": time_limit,
                "Threads": 8,
                "LogToConsole": 1,
                "reoptimize": True,
                "MIPGap": mip_rel_gap,
            },
            cp.HIGHS: {"time_limit": time_limit, "parallel": "on", "mip_rel_gap": mip_rel_gap},
            cp.COPT: {"TimeLimit": time_limit},
        }[solver_]

        ub = (
            self.re_plant_specs.capacity_max_re_scen
            if "capacity_max_re_scen" in self.re_plant_specs
            else self.re_plant_specs.capacity_mw_nrel_site
        ).to_numpy()
        lb_ = np.where(
            # we only use the wind_lb as the lower bound when there is not already
            # some wind at that site and the site is large enough to build as much
            # as the wind_lb
            self.re_plant_specs.re_type.str.contains("wind") & (lb == 0) & (ub > wind_lb),
            wind_lb,
            # sometimes rounding causes problems and the previous scenario's selection
            # is close and actually a little more than the upper bound which makes the
            # problem infeasible, so we fix that here
            np.where((lb > ub) & np.isclose(lb, ub), ub, lb),
        )
        if not np.all(ub >= lb_):
            df = (
                pd.concat(
                    {
                        "lb": pd.Series(lb_, index=self.re_plant_specs.combi_id),
                        "ub": pd.Series(ub, index=self.re_plant_specs.combi_id),
                    },
                    axis=1,
                )
                .query("lb > ub")
                .to_string()
                .replace("\n", "\n\t")
            )
            raise ScenarioError(f"{desc} lb > ub \n {df}")

        x = cp.Variable(self.re_plant_specs.shape[0], name="re_cap")
        if Ab_eq is not None:
            b_eq = Ab_eq.to_numpy()[:, -1]
            A_eq = Ab_eq.to_numpy()[:, :-1]
            eq = [A_eq @ x >= b_eq * 0.95, A_eq @ x <= b_eq * 1.05]
        else:
            eq = []

        semicon = (
            self.re_plant_specs.re_type.str.contains("wind") & (lb == 0.0) & (ub > wind_lb)
        )
        if semicon_idx := semicon[semicon].index.to_list():
            bin_var = cp.Variable(len(semicon_idx), name="bin", boolean=True)
            semicon_con = [
                x[semicon_idx] >= cp.multiply(lb_[semicon_idx], bin_var),
                x[semicon_idx] <= cp.multiply(ub[semicon_idx], bin_var),
            ]
        else:
            semicon_con = []
        con_idx = semicon[~semicon].index.to_list()
        too_little = (
            self.re_plant_specs.re_type.str.contains("wind") & (lb == 0.0) & (ub < wind_lb)
        )
        too_little_idx = too_little[too_little].index.to_list()
        Ab_ub__ = Ab_ub_.to_numpy()
        try:
            cp_con = [
                Ab_ub__[:, :-1] @ x <= Ab_ub__[:, -1],
                *eq,
                x[con_idx] >= lb_[con_idx],
                x[con_idx] <= ub[con_idx],
                *semicon_con,
                *([x[too_little_idx] == 0.0] if too_little_idx else []),
            ]
            p = cp.Problem(cp.Minimize(c_.to_numpy() @ x), cp_con)
            with capture_stdout() as c_out, capture_stderr() as err:
                p.solve(solver_, **options, verbose=True)
            try:
                if p.status != cp.OPTIMAL:
                    LOGGER.debug("\n %s", "\n".join((c_out.getvalue(), err.getvalue())))
            finally:
                c_out.close()
                err.close()
            if p.status == cp.OPTIMAL:  # noinspection PyUnreachableCode
                # noinspection PyUnreachableCode
                return True, pd.Series(  # noinspection PyUnreachableCode
                    x.value,
                    index=pd.MultiIndex.from_frame(
                        self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
                    ),
                )
            if (
                p.status == cp.USER_LIMIT
                and (mip_gap := p.solution.attr["solver_specific_stats"].MIPGAP) < 1e-3
            ):
                msg = f"{desc} using MILP results with {mip_gap=} after {time_limit=}"
                self.msg.append(msg)
                LOGGER.info("%s %s", self.ba_code, msg)
                return True, pd.Series(
                    x.value,
                    index=pd.MultiIndex.from_frame(
                        self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
                    ),
                )
            if p.status == cp.INFEASIBLE and wind_lb > 5:
                msg = f"{desc} MILP failed with {wind_lb=} -> trying again with {wind_lb / 2}"
                self.msg.append(msg)
                LOGGER.info("%s %s", self.ba_code, msg)
                return self.milp(
                    c=c,
                    Ab_eq=Ab_eq,
                    Ab_ub=Ab_ub,
                    lb=lb,
                    wind_lb=wind_lb / 2,
                    desc=desc,
                    time_limit=time_limit,
                    mip_rel_gap=mip_rel_gap,
                )
            return False, p.status
        except Exception as exc:
            LOGGER.error("%s %s failed: %r", self.ba_code, desc, exc)
            return False, cp.SOLVER_ERROR

        # q = (
        #     self.re_plant_specs.assign(mw=x.value, lb_=lb_, ub=ub)
        #     .join(pd.DataFrame(bin_var.value, index=semicon_idx, columns=["bin"]))[
        #         [
        #             co
        #             for co in (
        #                 "re_site_id",
        #                 "re_type",
        #                 "capacity_mw_nrel_site",
        #                 "capacity_max_re_scen",
        #             )
        #             if co in self.re_plant_specs
        #         ]
        #         + ["mw", "bin", "lb_", "ub"]
        #     ]
        #     .query("mw > 0 | bin > 0")
        # )

    def milp_scipy(
        self,
        c: pd.Series | None = None,
        Ab_eq: pd.DataFrame | None = None,  # noqa: N803
        Ab_ub: pd.DataFrame | None = None,  # noqa: N803
        lb: np.ndarray | None = None,
        wind_lb: float = 80.0,
        time_limit: float = 300,
        desc: str = "",
        mip_rel_gap=1.5e-4,
        disp=False,
        presolve=True,
    ) -> tuple[bool, pd.Series | str]:
        c_ = -1 * self.re_plant_specs.avoided_cost if c is None else c
        Ab_ub_ = self.Ab if Ab_ub is None else pd.concat([self.Ab, Ab_ub])
        if Ab_eq is None:
            eq_con = []
        else:
            b_eq = Ab_eq.iloc[:, -1].to_numpy()
            eq_con = [
                LinearConstraint(
                    A=Ab_eq.iloc[:, :-1].to_numpy(),
                    lb=b_eq * 0.95,
                    ub=b_eq * 1.05,
                )
            ]
        lb = np.zeros_like(c_) if lb is None else lb

        ub = (
            self.re_plant_specs.capacity_max_re_scen
            if "capacity_max_re_scen" in self.re_plant_specs
            else self.re_plant_specs.capacity_mw_nrel_site
        ).to_numpy()
        lb_ = np.where(
            # we only use the wind_lb as the lower bound when there is not already
            # some wind at that site and the site is large enough to build as much
            # as the wind_lb
            self.re_plant_specs.re_type.str.contains("wind") & (lb == 0) & (ub > wind_lb),
            wind_lb,
            # sometimes rounding causes problems and the previous scenario's selection
            # is close and actually a little more than the upper bound which makes the
            # problem infeasible, so we fix that here
            np.where((lb > ub) & np.isclose(lb, ub), ub, lb),
        )
        if not np.all(ub >= lb_):
            df = (
                pd.concat(
                    {
                        "lb": pd.Series(lb_, index=self.re_plant_specs.combi_id),
                        "ub": pd.Series(ub, index=self.re_plant_specs.combi_id),
                    },
                    axis=1,
                )
                .query("lb > ub")
                .to_string()
                .replace("\n", "\n\t")
            )
            raise ScenarioError(f"{desc} lb > ub \n {df}")

        r = milp(
            c=c_.to_numpy(),
            integrality=np.where(
                # these must follow the same logic as lb_ otherwise some wind sites
                # could be 0.0 even after an earlier scenario built there
                self.re_plant_specs.re_type.str.contains("wind")
                & (lb == 0.0)
                & (ub > wind_lb),
                2,
                0,
            ),
            constraints=[
                LinearConstraint(
                    A=Ab_ub_.iloc[:, :-1].to_numpy(),
                    ub=Ab_ub_.iloc[:, -1].to_numpy(),
                ),
                *eq_con,
            ],
            bounds=Bounds(lb=lb_, ub=ub),
            options={
                "time_limit": time_limit,
                "disp": disp,
                "mip_rel_gap": mip_rel_gap,
                "presolve": presolve,
            },
        )
        if r.success is True:
            return True, pd.Series(
                r.x,
                index=pd.MultiIndex.from_frame(
                    self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
                ),
            )
        if r.status == 1 and r.mip_gap < 1e-3:
            msg = f"{desc} using MILP results with {r.mip_gap=} after {time_limit=}"
            self.msg.append(msg)
            LOGGER.info("%s %s", self.ba_code, msg)
            return True, pd.Series(
                r.x,
                index=pd.MultiIndex.from_frame(
                    self.re_plant_specs[["icx_genid", "re_site_id", "re_type"]]
                ),
            )
        if r.status == 2 and presolve is True:
            msg = (
                f"{desc} MILP failed with {presolve=} -> trying again with False "
                f"(known bug in HIGHS https://github.com/scipy/scipy/issues/18907)"
            )
            self.msg.append(msg)
            LOGGER.info("%s %s", self.ba_code, msg)
            return self.milp_scipy(
                c=c,
                Ab_eq=Ab_eq,
                Ab_ub=Ab_ub,
                lb=lb,
                wind_lb=wind_lb,
                mip_rel_gap=mip_rel_gap,
                desc=desc,
                presolve=False,
            )
        if r.status == 2 and wind_lb > 5:
            msg = f"{desc} MILP failed with {wind_lb=} -> trying again with {wind_lb / 2}"
            self.msg.append(msg)
            LOGGER.info("%s %s", self.ba_code, msg)
            return self.milp_scipy(
                c=c,
                Ab_eq=Ab_eq,
                Ab_ub=Ab_ub,
                lb=lb,
                wind_lb=wind_lb / 2,
                mip_rel_gap=mip_rel_gap,
                desc=desc,
                presolve=presolve,
            )
        return False, r.message

    def make_cfl(self) -> BAScenario:
        cfl = BAScenario.__new__(BAScenario)
        cfl.ba = self
        cfl.cfl = True
        cfl.is_max_scen = False
        cfl.to_replace = []
        cfl.patio_chunks = MTDF.copy()
        cfl._data = MTDF.copy()
        # cfl.colo_dir = CACHE_PATH / datetime.now().strftime("%Y%m%d%H%M%S")
        cfl.config = ScenarioConfig(
            re_energy=0.0,
            nuclear_scen=0,
            storage_li_pct=0.0,
            storage_fe_pct=0.0,
            storage_h2_pct=0.0,
            ccs_scen=0,
        )
        cfl.re_cap = dict.fromkeys(("onshore_wind", "solar", "offshore_wind"), 0.0)
        cfl.re_plant_specs = (
            self.re_plant_specs.assign(capacity_mw=0.0)
            .set_index(self.plant_data.index.names)
            .query("re_type == 'y'")
        )

        # if self.old_clean_adj_net_load:
        re_specs = self.plant_data.loc[
            self.clean_list, [c for c in self.re_spec_cols if c in self.plant_data]
        ]
        re_profs = self.profiles[self.clean_list]
        cfl.es_specs = cfl.config.storage_specs(
            capacity=0.0,
            operating_date=self.re_profiles.index.min() - timedelta(30),
            category="patio_clean",
        )
        if re_specs.empty:
            re_specs = pd.DataFrame(
                {
                    "plant_id_eia": [-555],
                    "generator_id": ["solar"],
                    "capacity_mw": [0.0],
                    "technology_description": ["Solar Photovoltaic"],
                    "operating_date": [self.load.index.min()],
                    "retirement_date": [pd.NaT],
                    "ilr": [1.0],
                    "category": ["proposed_clean"],
                },
            ).set_index(self.plant_data.index.names)
            re_profs = pd.DataFrame(index=self.load.index, columns=re_specs.index).fillna(0.0)
        re_kwargs = dict(  # noqa: C408
            load_profile=pd.Series(self.adj_load, index=self.load.index),
            re_profiles=re_profs,
            re_plant_specs=re_specs,
        )
        # else:
        #     re_kwargs = dict(
        #         load_profile=self.load,
        #         re_profiles=pd.concat(
        #             [self.profiles[self.clean_list], self.old_re_profs], axis=1
        #         ),
        #         re_plant_specs=pd.concat(
        #             [
        #                 self.plant_data.loc[
        #                     self.clean_list,
        #                     [c for c in self.re_spec_cols if c in self.plant_data],
        #                 ],
        #                 self.old_re_specs,
        #             ]
        #         ),
        #     )
        re_pids = set(re_specs.index.get_level_values("plant_id_eia"))  # noqa: F841

        dm_kwargs = re_kwargs | dict(  # noqa: C408
            # load_profile=self.load,
            dispatchable_specs=self.plant_data.loc[self.fossil_list, :].assign(
                no_limit=lambda x: (
                    x.prime_mover.isin(self.no_limit_prime) & (x.fuel_group == "natural_gas")
                )
                | (x.operational_status == "proposed"),
            ),
            dispatchable_profiles=self.profiles[self.fossil_list],
            dispatchable_cost=self.cost_data,
            storage_specs=pd.concat(
                [
                    cfl.es_specs,
                    self.plant_data.loc[self.storage_list, :]
                    .query("plant_id_eia not in @re_pids")
                    .assign(reserve=0),
                    # for storage with the same pid as RE, t
                    self.plant_data.loc[self.storage_list, :]
                    .query("plant_id_eia in @re_pids")
                    .reset_index()
                    .groupby("plant_id_eia", as_index=False)
                    .agg(
                        {"capacity_mw": "sum", "generator_id": "first"}
                        | dict.fromkeys(self.plant_data.columns, "first")
                    )
                    .set_index(["plant_id_eia", "generator_id"])
                    .assign(reserve=0),
                ]
            ),
            config={"marginal_for_startup_rank": True},
            # re_profiles=pd.concat(
            #     [self.profiles[self.clean_list], self.old_re_profs], axis=1
            # ),
            # re_plant_specs=pd.concat(
            #     [self.plant_data.loc[self.clean_list, :], self.old_re_specs]
            # ),
        )
        dm = DispatchModel(**dm_kwargs)()
        max_defi = (dm.system_data.deficit / dm.load_profile).max()
        tot_deficit = dm.system_data.deficit.sum() / dm.load_profile.sum()
        if max_defi > 0.03 or tot_deficit > 0.02:
            self.no_limit_prime = ("CC", "GT")
            dm_kwargs["dispatchable_specs"] = dm_kwargs["dispatchable_specs"].assign(
                no_limit=lambda x: (
                    x.prime_mover.isin(self.no_limit_prime) & (x.fuel_group == "natural_gas")
                )
                | (x.operational_status == "proposed")
            )
            dm = DispatchModel(**dm_kwargs)()

        _, _, dmax, dcount = dm.system_level_summary(freq="YS").filter(like="deficit").max()
        dm.set_metadata("re_limits_dispatch", False)
        if (max_defi := (dm.system_data.deficit / self.augmented_load).max()) >= 0.06:
            if (
                self.ba_code
                in (
                    "EPE",
                    "SC",
                    "TEPC",
                    "WACM",
                    "LDWP",
                    "PSCO",
                    "APS",
                    "177",
                )
                or self._metadata["colo_techs"]
            ):
                LOGGER.warning(
                    "%s counterfactual max deficit is %.1f percent ",
                    self.ba_code,
                    max_defi * 100,
                )
            else:
                raise ScenarioError(
                    f"counterfactual max deficit is {max_defi:.2%}, it was expected to "
                    f"be less than 6%, this BA will not be in results"
                )

        cfl.dm = (dm,)
        cfl.no_limit_prime = self.no_limit_prime
        cfl._outputs = {}
        cfl.colo_hourly = pl.LazyFrame()
        cfl.colo_coef_ts = pl.LazyFrame()
        cfl.colo_coef_mw = pl.LazyFrame()
        cfl.colo_summary = pl.DataFrame()
        if colo_techs := self._metadata["colo_techs"]:
            cfl.setup_colo_data(colo_techs)

        cfl.msg = []
        cfl.good_scenario = dmax <= 0.15 and dcount <= 87
        for attr in ("to_replace", "mothball", "exclude", "ccs"):
            setattr(cfl, attr, [])
        return cfl

    @cached_property
    def fossil_list(self):
        return list(self.plant_data.query("category.str.contains('fossil')").index)

    @cached_property
    def clean_list(self):
        return list(
            self.plant_data.query(
                "category.str.contains('clean') & technology_description not in @ES_TECHS"
            ).index
        )

    @cached_property
    def storage_list(self):
        return list(self.plant_data.query("technology_description in @ES_TECHS").index)

    def _dzsetstate_(self, state):
        for to_fix in ("baseload_replace", "ccs_convert"):
            setattr(self, to_fix, {int(k): v for k, v in state.pop(to_fix).items()})
        self.scenario_configs = [ScenarioConfig(*x) for x in state.pop("scenario_configs")]
        has_scens = state["_metadata"].pop("has_scens")
        # counterfactual = state.pop("cfl")
        scenarios_data = state.pop("scenarios_data", [])

        for attr, val in state.items():
            if "output" not in attr:
                setattr(self, attr, val)

        if "augmented_load" not in state:
            self.augmented_load = self.load

        # The ones we use have lots of duplication that we want to avoid saving, so we
        # recreate the full version here
        self.re_profiles, self.re_prof_ilr_adj = self.setup_re_profiles(self.re_plant_specs)

        self.counterfactual: BAScenario = self.make_cfl()
        self._scenarios = []

        if has_scens:
            for data in tqdm(
                scenarios_data,
                desc=f"Rebuild scenarios {self.ba_code}",
            ):
                c = BAScenario.__new__(BAScenario)
                c._dzsetstate_({"ba": self} | data)
                self._scenarios.append(c)

    def _dzgetstate_(self) -> dict:
        state = {}
        for df_name in self._parquet_out:
            if (df_out := getattr(self, df_name)) is not None:
                if "prof" in df_name and isinstance(df_out, pd.DataFrame):
                    state[df_name] = df_out.astype(np.single)
                else:
                    state[df_name] = df_out

        state.update(
            {
                "_metadata": {
                    **self._metadata,
                    "has_scens": bool(self._scenarios),
                },
                "baseload_replace": self.baseload_replace,
                "ccs_convert": self.ccs_convert,
                "scenario_configs": self.scenario_configs,
                "include_figs": self.include_figs,
                # "cfl": self.counterfactual.__getstate__(),
                "exclude": self.exclude,
                "exclude_or_mothball": self.exclude_or_mothball,
                "max_re": self.max_re,
                "no_limit_prime": self.no_limit_prime,
                "re_limits_dispatch": self.re_limits_dispatch,
                "max_energy": self.max_energy,
                # "old_clean_adj_net_load": self.old_clean_adj_net_load,
            }
        )
        if len(self):
            state.update(
                {
                    f"{self.ba_code}_full_output": self.full_output(),
                    f"{self.ba_code}_allocated_output": self.allocated_output(),
                    f"{self.ba_code}_system_output": self.system_output(),
                    f"{self.ba_code}_messages": self.messages(),
                    f"{self.ba_code}_monthly_fuel": pd.concat(x.monthly_fuel() for x in self),
                }
            )
            for attr in (
                "colo_summary",
                "colo_hourly",
                "colo_coef_ts",
                # "colo_coef_ts_iter",
                "colo_coef_mw",
            ):
                try:
                    state[f"{self.ba_code}_{attr}"] = pl.concat(
                        (getattr(x, attr) for x in self), how="diagonal_relaxed"
                    )
                except Exception as exc:
                    LOGGER.error("%s output %s failed %r", self.ba_code, attr, exc)

            scens = []
            for scen in tqdm(
                self._scenarios,
                desc=f"Output scens {self.ba_code}",
                position=2,
                leave=None,
            ):
                try:
                    scens.append(scen._dzgetstate_())
                except Exception as exc:
                    LOGGER.error("serializing %s failed %r", str(scen), exc, exc_info=exc)
            state["scenarios_data"] = scens
        return state

    @property
    def scenarios(self) -> list[BAScenario]:
        if self._scenarios:
            return [scen for scen in self._scenarios if scen.good_scenario]
        return []

    @property
    def bad_scenarios(self) -> dict[ScenarioConfig, BAScenario]:
        if self._scenarios:
            return {scen.config: scen for scen in self._scenarios if not scen.good_scenario}
        return {}

    @property
    def ba_code(self):
        return self._metadata["ba_code"]

    @property
    def name(self):
        return self._metadata["name"]

    def messages(self):
        return pd.concat([x.messages() for x in self], axis=0).assign(
            ba_note=", ".join(self.msg)
        )
        # return pd.DataFrame(
        #     {
        #         "msg": [", ".join(x.msg) for x in self],
        #         "ba_code": [self.ba_code for _ in self],
        #         "scenario": [
        #             ("counterfactual" if x.cfl else x.config.for_idx) for x in self
        #         ],
        #     }
        # ).set_index(["ba_code", "scenario"])

    @property
    def ba_name(self):
        return self.ba_code if self.ba_code == self.name else f"{self.name} ({self.ba_code})"

    def align_profiles(self, profiles, re_profiles):
        """Sometimes CEMS and RE profiles have different begin/end dates or are missing hours
        this method selects the common interval for each profile and fills missing hours
        """
        begin = max(re_profiles["datetime"].min(), profiles.index.min())
        end = min(re_profiles["datetime"].max(), profiles.index.max())
        re_profiles = re_profiles.filter(
            (pl.col("datetime") >= begin) & (pl.col("datetime") <= end)
        )
        re_len = len(re_profiles)
        all_dates = re_profiles.select("datetime").upsample("datetime", every="1h")
        if len(all_dates) > re_len:
            re_cols = [c for c in re_profiles.columns if "__" in c]
            mo_hr = dict(  # noqa: C408
                month=pl.col("datetime").dt.month(), hour=pl.col("datetime").dt.hour()
            )
            missing = (
                pl.LazyFrame({"datetime": all_dates.to_series()})
                .filter(pl.col("datetime").is_in(re_profiles["datetime"]).not_())
                .with_columns(**mo_hr)
                .join(
                    re_profiles.lazy()
                    .with_columns(**mo_hr)
                    .group_by(["month", "hour"])
                    .agg(pl.col(*re_cols).mean().cast(pl.Float32)),
                    on=["month", "hour"],
                )
                .select("datetime", *re_cols)
                .collect()
            )
            re_profiles = pl.concat([re_profiles, missing]).sort("datetime")

        if len(re_profiles) > re_len:
            # this shouldn't really happen
            LOGGER.info(
                "%s RE data is not continuous, filling missing values with month/hour averages",
                self.ba_code,
            )
        profiles = profiles.loc[begin:end, :].fillna(0.0)
        if len(profiles) < len(re_profiles):
            # some of our processing can result in hours that have all nans in them being dropped
            # also, it's possible that issues with the underlying CEMS data means that hours are
            # missing, this is more likley to occur in BAs with few plants
            LOGGER.info(
                "%s CEMS is not continuous, filling missing values with 0", self.ba_code
            )
            profiles = profiles.reindex(index=re_profiles["datetime"].to_pandas()).fillna(0.0)
        return profiles, re_profiles

    @property
    def co2_profile(self) -> pd.DataFrame:
        lyear = str(self.profiles.index.year.max())
        intensity = self._co2_profile.loc[lyear:, :].sum() / self.profiles.loc[lyear:, :].sum()
        big_adj = self._co2_profile / self._co2_profile.groupby(
            pd.Grouper(freq="YS")
        ).transform("mean")
        return intensity * big_adj * self.profiles

    def run_all_scens(self):
        max_config = max(len(str(x)) for x in self.scenario_configs) + 24
        max_re_config = max(
            self.scenario_configs,
            key=lambda x: (
                x.re_energy,
                -x.nuclear_scen,
                -x.ccs_scen,
                x.storage_li_pct,
                x.storage_fe_pct,
                x.storage_h2_pct,
            ),
        )
        for_loop = tqdm(
            enumerate(
                dict.fromkeys([max_re_config]) | dict.fromkeys(sorted(self.scenario_configs))
            ),
            desc=f"Run scens {self.ba_code:12}".ljust(max_config, " "),
            total=len(self.scenario_configs),
            position=2,
            leave=None,
        )
        for _i, config in for_loop:
            for_loop.set_description(
                desc=f"Run scens {self.ba_code:12} {config}".ljust(max_config, " ")
            )
            try:
                if _i == 0:
                    scen = BAScenario(ba=self, config=config)
                    scen.is_max_scen = True
                    self.re_plant_specs = self.re_plant_specs.merge(
                        scen.re_plant_specs[["combi_id", "capacity_mw"]].rename(
                            columns={"capacity_mw": "capacity_max_re_scen"}
                        ),
                        on="combi_id",
                        how="right",
                    )
                    self.Ab = make_core_lhs_rhs(self.re_plant_specs)
                else:
                    scen = BAScenario(ba=self, config=config)
                if scen.is_duplicate_scen(self._scenarios):
                    raise ScenarioError("Scenario is a duplicate")

            except ScenarioError as exc:
                LOGGER.error("%s %s %r", self.ba_code, config.for_idx, exc)
            except Exception as exc:
                raise exc
            else:
                self._scenarios.append(scen)
                if _i == 0 and (colo_techs := self._metadata["colo_techs"]):
                    scen.setup_colo_data(colo_techs)

        # the max scenarios chunks were originally calculated before its parent was
        # created, so we need to re-calculate them here at the end
        no_rechunk = True
        for s in self._scenarios:
            if s.is_max_scen:
                s.rechunk_patio_clean()
                no_rechunk = False
        if no_rechunk:
            LOGGER.warning(
                "%s no successful max_scen to rechunk, scens=%s",
                self.ba_code,
                [s.config.for_idx for s in self._scenarios],
            )

        self._scenarios = sorted(self._scenarios, key=lambda x: tuple(x.config))
        assert True

    def parent_re_scenario(self, scen: BAScenario) -> BAScenario | None:
        if self._scenarios:  # noqa: SIM102
            if pars := [s for s in self._scenarios if s.config.is_re_child(scen.config)]:
                return max(pars, key=lambda x: x.config.re_energy)
        return None

    def parent_li_scenario(self, scen: BAScenario) -> BAScenario | None:
        if self._scenarios:  # noqa: SIM102
            if pars := [s for s in self._scenarios if s.config.is_li_child(scen.config)]:
                return max(pars, key=lambda x: (x.config.storage_li_pct, x.config.re_energy))
        return None

    def same_energy_scenario(self, scen: BAScenario) -> BAScenario | None:
        if self._scenarios:
            for p_scen in reversed(self._scenarios):
                if all(
                    getattr(scen.config, attr) == getattr(p_scen.config, attr)
                    for attr in ("re_energy", "nuclear_scen", "ccs_scen")
                ):
                    return p_scen
        return None

    def hourly_adjustment(self):
        """Hourly adjustment to ~normalize plant roles across years"""
        return (
            self.profile_check(mapper=self.plant_data.plant_role.to_dict())
            .assign(
                base=lambda x: x.base[-1] / x.base,
                mid=lambda x: x.mid[-1] / x.mid,
                peaker=lambda x: x.peaker[-1] / x.peaker,
            )
            .replace({np.inf: np.nan})
            .fillna(0.0)
            .reindex(index=self.profiles.index, method="ffill")[self.plant_data.plant_role]
            .to_numpy()
        )

    def profile_check(
        self,
        df: pd.DataFrame | None = None,
        mapper: str | list | dict | None = None,
        multiindex=False,
        norm=False,
    ):
        """Aggregate profiles using ``mapper_for_col_group`` and to annual frequency

        Args:
            df: profiles to check, if None, use self.profiles
            mapper: map (plant_id_eia, generator_id) -> "x" | ("x", "y", ...)
            multiindex: if you want to groupby multiple levels, set to True and make sure
                (plant_id_eia, generator_id) map to equal length tuples
            norm: normalize the result

        Returns:

        """  # noqa: D414
        if mapper is None:
            mapper = self.plant_data.plant_role.to_dict()
        elif isinstance(mapper, tuple | list):
            mapper = {
                k: tuple(v.values())
                for k, v in self.plant_data[list(mapper)].T.to_dict().items()
            }
        elif isinstance(mapper, str):
            mapper = self.plant_data[mapper].to_dict()
        # prep for and do column renaming
        if df is None:
            df = self.profiles.copy()
        df.columns = df.columns.values  # noqa: PD011
        df = df.rename(columns=mapper)
        for_level = [0]
        # turn columns back into multiindex
        if multiindex:
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            for_level = list(
                range(max(len(x) for x in mapper.values() if isinstance(x, tuple)))
            )
        out = df.groupby(level=for_level, axis=1).sum().groupby(pd.Grouper(freq="YS")).sum()
        if multiindex:
            out.columns = out.columns.map("_".join)
        out = out.assign(total=lambda x: x.sum(axis=1))
        if norm:
            return out / out.max()
        return out

    def full_output(self) -> pd.DataFrame:
        original = pd.concat(x.full_output() for x in self)
        # create historical data as a scenario with re_energy = -1.0
        not_hist_cats = (  # noqa: F841
            "old_clean",
            "proposed_clean",
            "proposed_fossil",
            "proposed_other",
        )
        hist = (
            self.counterfactual.full_output()
            .reset_index()
            .query("category not in @not_hist_cats")
            .assign(
                re_energy=-1.0,
                scenario="historical",
                redispatch_curt_adj_mwh=lambda x: x.redispatch_mwh,
            )
            .set_index(original.index.names)
            .drop(columns=[c for c in original if "redispatch" in c])
        )
        hist.columns = [c.replace("historical", "redispatch") for c in hist.columns]
        # x = original.reset_index().query(
        #     "category == 'patio_clean' "
        #     "& technology_description not in ('Batteries', 'H2 Storage') "
        #     "& datetime.dt.year == 2030"
        # ).sort_values(['plant_id_eia', 'scenario_add'])
        # y = x.pivot(
        #     index=["plant_id_eia", 'scenario_add', "generator_id"],
        #     columns="scenario",
        #     values="capacity_mw",
        # )
        icx_rn = {"icx_id": "associated_plant_id", "icx_gen": "associated_generator_id"}
        return (
            pd.concat([hist, original])[[c for c in original if "historical" not in c]]
            .rename(columns={"redispatch_co2": "redispatch_co2_tonne"})
            # .drop(columns=[c for c in original if "historical" in c])
            .assign(
                ccs_eligible=lambda x: x.index.to_frame()
                .droplevel(["ba_code", "scenario", "datetime"])
                .index.isin(self.ccs_convert[1]),
                historical_year=lambda x: x.index.get_level_values("datetime").year.map(
                    self._metadata["year_mapper"]
                ),
            )
            .reset_index()
            .merge(
                self.re_plant_specs[["plant_id_eia", "icx_id", "icx_gen"]],
                on="plant_id_eia",
                how="left",
                validate="m:1",
            )
            .rename(columns=icx_rn)
            .set_index(original.index.names)
        )

    def _is_ccs_ix(self, ix):
        """Is the index a row for a CCS plant in a CCS scenario"""
        if (ccs_scen := ix[5]) == 0:
            return False
        if ix[6:8] in self.ccs_convert.get(abs(ccs_scen), []):  # noqa: SIM103
            return True
        return False

    def system_output(self) -> pd.DataFrame:
        original = pd.concat([x.system_output() for x in self], axis=0)
        hist = (
            self.counterfactual.system_output(variant="historical")
            .reset_index()
            .assign(re_energy=-1.0, scenario="historical")
            .set_index(original.index.names)
        )
        r_base_cols = [
            f"{pre}_{res}"
            for pre in ("patio", "proposed", "old")
            for res in CLEAN_TD_MAP.values()
        ] + list(set(OTHER_TD_MAP.values()))
        full = [c + suff for suff in ("_mw", "_mwh") for c in r_base_cols if c + suff in hist]
        return pd.concat([hist[[*ScenarioConfig._fields, *full]], original], axis=0)[
            original.columns
        ]

    def allocated_output(self):
        return pd.concat([x.allocated_output() for x in self], axis=0)

    def get_figs(self):
        figdir = PATIO_DOC_PATH / "figs"
        if not figdir.exists():
            figdir.mkdir(parents=True)

        from pypdf import PdfWriter

        with (
            logging_redirect_tqdm(),
            PdfWriter(fileobj=figdir / f"{self.ba_code}_mwh_ms.pdf") as merger1,
            PdfWriter(fileobj=figdir / f"{self.ba_code}_daily.pdf") as merger2,
        ):
            for_tqdm = tqdm(
                self._scenarios,
                desc=f"{self.ba_code} Figs",
                total=len(self._scenarios),
            )
            for scen in for_tqdm:
                f1, f2, f3, f4 = scen.get_figs().values()
                print(f3.layout.title)
                f1.write_image(temp1 := BytesIO(), format="pdf")
                f2.write_image(temp2 := BytesIO(), format="pdf")
                f3.update_layout(
                    title=f3.layout.title.text + " re_limits_dispatch"
                ).write_image(temp3 := BytesIO(), format="pdf")
                f4.update_layout(
                    title=f4.layout.title.text + " re_limits_dispatch"
                ).write_image(temp4 := BytesIO(), format="pdf")
                merger1.append(temp1)
                merger2.append(temp2)
                merger1.append(temp3)
                merger2.append(temp4)

    def __len__(self):
        return len(self._scenarios) + 1 if hasattr(self, "counterfactual") else 0

    def __iter__(self):
        return iter([self.counterfactual, *self._scenarios])

    def __getitem__(self, item) -> BAScenario:
        return self._scenarios[item]

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"({', '.join(f'{k}={v}' for k, v in self._metadata.items())})"
        )

    def make_dm_mini(self, path, years=(2025, 2030)):
        f_year, l_year = years
        d_s = self.plant_data.loc[self.fossil_list, :].query(
            "(operating_date <= @l_year | operating_date.isna())"
            "& (retirement_date >= @f_year | retirement_date.isna())"
        )
        d_p = self.profiles.loc[str(f_year) : str(l_year), d_s.index]
        r_s = self.plant_data.loc[self.clean_list, :].query(
            "(operating_date <= @l_year | operating_date.isna())"
            "& (retirement_date >= @f_year | retirement_date.isna())"
        )
        l_period = f"{l_year}-12-31"  # noqa: F841
        with DataZip(path, "w") as z:
            z["load_profile"] = d_p.sum(axis=1)
            z["dispatchable_specs"] = d_s
            z["dispatchable_profiles"] = d_p
            z["dispatchable_cost"] = self.cost_data.loc[
                [i for i in self.cost_data.index if i[:2] in d_s.index], :
            ].query("datetime >= @f_year & datetime <= @l_period")
            z["storage_specs"] = (
                self.plant_data.loc[self.storage_list, :]
                .query(
                    "(operating_date <= @l_year | operating_date.isna())"
                    "& (retirement_date >= @f_year | retirement_date.isna())"
                )
                .assign(
                    roundtrip_eff=0.9,
                    duration_hrs=lambda x: (
                        x.energy_storage_capacity_mwh / x.capacity_mw
                    ).fillna(4),
                )
            )
            z["re_plant_specs"] = r_s
            z["re_profiles"] = self.profiles.loc[str(f_year) : str(l_year), r_s.index]


class BAs:
    """scalene patio/model/ba_level.py --profile-only 'ba_level,model' --profile-all"""

    def_scen_configs = sorted(
        set(
            # ScenarioConfig.from_sweeps(
            #     re=[0.6],
            #     # nuke=[0, 1],
            #     li=[0.25],
            # )
            ScenarioConfig.from_sweeps(
                re=[0.2, 0.4],
                # nuke=[0, 1],
                li=[0.0, 0.25],
            )
            + ScenarioConfig.from_sweeps(
                re=[0.6, 0.8],
                # nuke=[0, 1],
                li=[0.0, 0.25],
            )
            # + ScenarioConfig.from_sweeps(
            #     re=[0.25],
            #     nuke=[0, 1],
            #     li=[0.25],
            # )
            # + ScenarioConfig.from_sweeps(
            #     re=[0.25],
            #     li=[0.25],
            #     ccs=[0, 1],
            # ),
        )
    )
    bad_bas = ("Alaska", "HECO", "NBSO")

    def __init__(
        self,
        name: str | None = None,
        bas: list | None = None,
        profile_data: ProfileData | None = None,
        scenario_configs: list[ScenarioConfig] | None = None,
        regime: Literal["reference", "limited"] = "reference",
        solar_ilr: float = 1.34,
        data_kwargs: dict | None = None,
        by_plant: bool = False,  # noqa: FBT001
        data_source: str | None = None,
        pudl_release: str = PATIO_PUDL_RELEASE,
        queue=None,
        alt_path: str | None = None,
        econ_scen_dir: str = "",
        econ_suffix: str = "",
    ):
        if name is None:
            self.name = "BAs_" + datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.name = name
        if queue is None:
            self.queue = multiprocessing.Manager().Queue(-1)
        else:
            self.queue = queue
        self.pudl_release = pudl_release
        self.datetime = datetime.now().strftime("%Y%m%d%H%M")
        self.alt_path = alt_path
        self.econ_suffix = ("_" + econ_suffix.removeprefix("_")) if econ_suffix else ""
        self.econ_scen_dir = econ_scen_dir
        if profile_data is None:
            self._prod = ProfileData(
                AssetData(pudl_release=self.pudl_release),
                regime=regime,
                solar_ilr=solar_ilr,
            )
        else:
            self._prod = profile_data
        self.data_source = data_source
        if scenario_configs is None:
            self.scenario_configs = self.def_scen_configs
        else:
            self.scenario_configs = scenario_configs

        if bas is None:
            self.bas = [
                x
                for x in self.ad.all_modelable_generators().final_ba_code.unique()
                if x not in self.bad_bas
            ]
        else:
            self.bas = [
                x
                for x in bas
                if x in self.ad.all_modelable_generators().final_ba_code.unique()
            ]
            if not self.bas:
                raise RuntimeError(f"No modelable BAs in {bas}")
        self.errors: dict[str, defaultdict[str, list[TracebackException]]] = {
            "setup": defaultdict(list),
            "run": defaultdict(list),
            "close_re": defaultdict(list),
            "output": defaultdict(list),
        }
        self._dfs = {}
        if data_kwargs is None:
            self.data_kwargs = {}
        else:
            self.data_kwargs = data_kwargs
        self.by_plant = by_plant
        self.bad_scenarios = defaultdict(list)
        self.good_objs = []
        now = datetime.now().strftime("%Y%m%d%H%M")
        colo_dir = Path.home() / f"patio_data/colo_{now}"
        if self.data_kwargs.get("colo_techs"):
            (colo_dir / "data").mkdir(parents=True)
            # (result_dir := colo_dir / "results").mkdir()
            # (result_dir / "hourly").mkdir()
            # (result_dir / "coef_ts").mkdir()
            # (result_dir / "coef_mw").mkdir()
            with open(colo_dir / "colo.json", "w") as colo:
                json.dump(
                    {
                        "plants": [],
                        "regime": regime,
                        "pudl_release": pudl_release,
                        "created": now,
                        "data_commit": _git_commit_info(),
                    },
                    colo,
                    indent=4,
                )
        self.colo_dir: str = str(colo_dir.relative_to(Path.home()))

    @property
    def dir_path(self):
        dir_path = PATIO_DOC_PATH / self.datetime
        if not dir_path.exists():
            dir_path.mkdir()
        return dir_path

    @property
    def path(self):
        if self.alt_path is None:
            return self.dir_path / self.name
        return self.alt_path

    @classmethod
    def from_cloud(cls, name: str, data_source: str | None = None, econ_suffix: str = ""):
        fs = rmi_cloud_fs()
        name = name.removesuffix(".zip")
        prefix, dt = name.split("_")
        f = fs.open(f"az://patio-results/{dt}/{name}.zip")
        f.close()
        c_path = str(AZURE_CACHE_PATH / cached_path(f"patio-results/{dt}/{name}.zip"))
        with DataZip(c_path, "r") as z:
            kwargs = {}
            for k in (
                "solar_ilr",
                "data_kwargs",
                "bas",
                "by_plant",
                "data_source",
            ):
                with suppress(KeyError):
                    kwargs[k] = z[k]
            kwargs["regime"] = z.get("regime", "reference")
            kwargs["pudl_release"] = z.get("pudl_release", PATIO_PUDL_RELEASE)
            if data_source is not None:
                kwargs["data_source"] = data_source
            self = cls(
                name=z.get("name", f"BAs_{dt}"),
                profile_data=z.get("_prod", None),
                alt_path=c_path,
                econ_suffix=econ_suffix,
                **kwargs,
            )
            for x in ("data_kwargs", "bas", "bad_scenarios", "good_objs", "errors"):
                setattr(self, x, z[x])
            self.datetime = z.get("datetime", dt)
            self.queue = multiprocessing.Manager().Queue(-1)
            return self

    @classmethod
    def from_file(cls, name: str, data_source: str | None = None, econ_suffix: str = ""):
        prefix, dt = name.removesuffix(".zip").split("_")
        path = (PATIO_DOC_PATH / dt / name).with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with DataZip(path, "r") as z:
            kwargs = {}
            for k in (
                "solar_ilr",
                "data_kwargs",
                "bas",
                "by_plant",
                "data_source",
            ):
                with suppress(KeyError):
                    kwargs[k] = z[k]
            kwargs["regime"] = z.get("regime", "reference")
            kwargs["pudl_release"] = z.get("pudl_release", PATIO_PUDL_RELEASE)
            if data_source is not None:
                kwargs["data_source"] = data_source
            self = cls(
                name=z.get("name", f"BAs_{dt}"),
                profile_data=z.get("_prod", None),
                econ_suffix=econ_suffix,
                **kwargs,
            )
            for x in ("data_kwargs", "bas", "bad_scenarios", "good_objs", "errors"):
                setattr(self, x, z[x])
            self.datetime = z.get("datetime", dt)
            self.queue = multiprocessing.Manager().Queue(-1)
            return self

    @property
    def pd(self):
        return self._prod

    @property
    def ad(self):
        return self._prod.ad

    def _attrs_to_z(self, z: DataZip):
        z["solar_ilr"] = self.pd.solar_ilr
        z["regime"] = self.pd.regime
        # z["setup_errors"] = {
        #     k: "".join(v[0].format()) for k, v in self.errors["setup"].items()
        # }
        # z["run_errors"] = {
        #     k: "".join(v[0].format()) for k, v in self.errors["run"].items()
        # }
        for k in (
            "data_kwargs",
            "bas",
            "bad_scenarios",
            "good_objs",
            "errors",
            "by_plant",
            "datetime",
            "name",
            "data_source",
            "_prod",
            "pudl_release",
        ):
            try:
                z[k] = getattr(self, k)
            except Exception as exc:
                LOGGER.warning("unable to add %s to DataZip, %r", k, exc)

    def prep_all_data(self):
        with logging_redirect_tqdm():
            pad = (
                max(len(str(x)) for x in self.scenario_configs) if self.scenario_configs else 0
            ) + 24
            with DataZip(self.path.parent / (self.path.stem + "_data"), "w") as z:
                for_loop: Any[str] = tqdm(
                    self.bas, desc=f"Prep data {'':5}".ljust(pad, " "), position=0
                )
                for ba_code in for_loop:
                    for_loop.set_description(
                        f"Prep data {ba_code.rjust(5, ' ')}".ljust(pad, " ")
                    )
                    try:
                        z.reset_ids()
                        z[ba_code] = self.pd.get_ba_data(ba_code, **self.data_kwargs)
                    except PatioData as exc:
                        self.errors["setup"][ba_code].append(
                            TracebackException.from_exception(exc)
                        )
                    except Exception as exc:
                        self.errors["setup"][ba_code].append(
                            TracebackException.from_exception(exc)
                        )
                        LOGGER.error("%s %r", ba_code, exc, exc_info=True)  # noqa: G201
                self.log_error_counts("setup", "run")

    def run_all(self):
        pad = (
            max(len(str(x)) for x in self.scenario_configs) if self.scenario_configs else 0
        ) + 24
        with DataZip(self.path, "w") as z:
            for_loop: Any[str] = tqdm(
                self.bas, desc=f"Setup+Run {'':5}".ljust(pad, " "), position=0
            )
            for ba_code in for_loop:
                if ba_code is None:
                    continue
                with logging_redirect_tqdm():
                    for_loop.set_description(
                        f"Setup+Run {ba_code.rjust(5, ' ')}".ljust(pad, " ")
                    )
                    try:
                        if not self.scenario_configs and ba_code in BAD_COLO_BAS:
                            LOGGER.warning("ba_code=%s skipped for colo analysis", ba_code)
                            continue
                        if self.data_source is None:
                            data = self.pd.get_ba_data(
                                ba_code,
                                **self.data_kwargs,
                                colo_only=len(self.scenario_configs) == 0,
                            )
                        else:
                            with DataZip(Path.home() / self.data_source) as zs:
                                data = zs[ba_code]
                    except PatioData as exc:
                        self.errors["setup"][ba_code].append(
                            TracebackException.from_exception(exc)
                        )
                    except Exception as exc:
                        self.errors["setup"][ba_code].append(
                            TracebackException.from_exception(exc)
                        )
                        LOGGER.error("%s %r", ba_code, exc, exc_info=True)  # noqa: G201
                    else:
                        if self.by_plant:
                            self._plant_level_helper(ba_code, dz=z, data=data, pad=pad)
                        else:
                            self._setup_and_run_helper(ba_code, dz=z, data=data)
            self._attrs_to_z(z)

            self.log_error_counts("setup", "run")

    def _plant_level_helper(self, ba_code: str, dz, data, pad):
        plant_loop: Any[str] = tqdm(
            data["plant_data"]
            .query(
                "technology_description == 'Conventional Steam Coal' & category == 'existing_fossil'"
            )
            .reset_index()
            .plant_id_eia.unique(),
            desc=f"{ba_code:10}".ljust(pad, " "),
            position=1,
        )
        for pid in plant_loop:
            plant_loop.set_description(
                f"{ba_code.rjust(15, ' ')} plant_id={pid}".ljust(pad, " ")
            )
            try:
                data_ = self.pd.one_plant(pid, data)
            except NoMaxRE as exc:
                self.errors["setup"][ba_code].append(TracebackException.from_exception(exc))
            self._setup_and_run_helper(data_["ba_code"], dz=dz, data=data_)

    def get_fossils_specs(self):
        with logging_redirect_tqdm():
            pad = " " * max(len(str(x)) for x in self.scenario_configs)
            for_loop = tqdm(self.bas, desc=f"Get specs {'':5} {pad}", position=0)
            data = []
            for ba_code in for_loop:
                for_loop.set_description(f"Get specs {ba_code:5} {pad}")
                data.append(self.pd.get_ba_data(ba_code, **self.data_kwargs)["plant_data"])
        out = pd.concat(data, axis=0)
        out.to_parquet(user_documents_path() / "fossil_specs.parquet")

    def _setup_and_run_helper(self, ba_code, data, dz):
        if ba_code in dz:
            LOGGER.warning("Will be unable to save %s because it was already run", ba_code)
            return None
        try:
            ba_obj = BA(
                **(data | self.data_kwargs),
                scenario_configs=self.scenario_configs,
                queue=self.queue,
                colo_dir=self.colo_dir,
            )
            self.good_objs.append(ba_code)
            if self.scenario_configs:
                ba_obj.run_all_scens()
        except Exception as exc:
            LOGGER.error("%s %r", ba_code, exc, exc_info=exc)
            self.errors["run"][ba_code].append(TracebackException.from_exception(exc))
        else:
            if self.scenario_configs:
                dz.reset_ids()
                try:
                    dz[ba_code] = ba_obj
                    for bad_scen in ba_obj.bad_scenarios:
                        self.bad_scenarios[bad_scen].append(ba_code)
                except Exception as exc:
                    LOGGER.error("%s storing results failed %r", ba_code, exc, exc_info=exc)
                    self.errors["run"][ba_code].append(TracebackException.from_exception(exc))
                # ba_obj.cleanup()
        return None

    def log_error_counts(self, *types: str):
        for etype in types:
            _errors = self.errors.get(etype, None)
            if _errors is None:
                continue
            if _errors:
                error_count = Counter(v[0].exc_type.__qualname__ for v in _errors.values())
                count_str = f"""({
                    ", ".join(f"{k.replace('()', '')}={v}" for k, v in error_count.items())
                })"""
            else:
                count_str = ""
            LOGGER.error(
                "%s -> ~completed=%s, errors=%s %s",
                etype,
                len(self.good_objs),
                len(_errors),
                count_str,
            )

    def make_profile_figs(self, path, bas=None, extend_cems=True):
        if isinstance(path, str):
            path = ROOT_PATH / path

        from pypdf import PdfWriter

        with PdfWriter(fileobj=path.with_suffix(".pdf")) as merger:  # noqa: SIM117
            with logging_redirect_tqdm():
                if bas is None:
                    for_loop = tqdm(self.bas, desc="All profiles")
                else:
                    for_loop = tqdm(bas, desc="All profiles")
                for ba_code in for_loop:
                    for_loop.set_description("All profiles " + ba_code)
                    try:
                        l = self.pd.get_ba_data(
                            ba_code, re_by_plant=True, extend_cems=extend_cems
                        )
                        q = (
                            l["profiles"]  # noqa: PD013
                            .groupby([pd.Grouper(freq="D")])
                            .sum()
                            .stack([0, 1])
                            .reorder_levels([1, 2, 0])
                            .sort_index()
                            .reset_index()
                            .rename(columns={0: "mwh"})
                            .merge(
                                l["plant_data"].technology_description.reset_index(),
                                on=["plant_id_eia", "generator_id"],
                                validate="m:1",
                            )
                            .assign(
                                resource=lambda x: x.technology_description.replace(PLOT_MAP)
                            )
                            .groupby(["datetime", "resource"])
                            .mwh.sum()
                            .reset_index()
                            .assign(
                                day=lambda z: z.datetime.dt.day,
                                year=lambda z: z.datetime.dt.strftime("%Y"),
                                month=lambda z: z.datetime.dt.strftime("%B"),
                            )
                            .sort_values(["resource", "year", "month"], key=dispatch_key)
                        )
                        try:
                            _ = int(ba_code)
                            t = (
                                f"{l['plant_data'].final_ba_code.iat[0]} {l['plant_data'].respondent_name.iat[0]} "  # noqa: PD009
                                f"({', '.join(sorted(l['plant_data'].balancing_authority_code_eia.dropna().unique()))})"
                            )
                        except ValueError:
                            t = (
                                f"{ba_code} ({', '.join(sorted(l['plant_data'].respondent_name.dropna().unique()))})"
                                f"({', '.join(sorted(l['plant_data'].balancing_authority_code_eia.dropna().unique()))})"
                            )
                        f = (
                            px.bar(
                                q,
                                x="day",
                                y="mwh",
                                facet_col="month",
                                facet_row="year",
                                color="resource",
                                color_discrete_map=COLOR_MAP,
                                height=1750,
                                template="plotly",
                                width=1750,
                                title=t,
                                facet_row_spacing=0.002,
                                facet_col_spacing=0.006,
                            )
                            .for_each_annotation(
                                lambda a: a.update(text=a.text.split("=")[-1])
                            )
                            .update_traces(
                                marker_line_width=0.1,
                            )
                            .update_layout(bargap=0)
                        )
                        f.write_image(tempd := BytesIO(), format="pdf")
                        merger.append(tempd)
                    except Exception as exc:
                        LOGGER.error("%s %r", ba_code, exc)

    def __getitem__(self, item) -> BA:
        if isinstance(item, int):
            item = list(self.good_objs)[item]
        with DataZip(self.path, "r") as z0:
            return z0[item]

    def __repr__(self):
        return (
            self.__class__.__qualname__
            + f"(name={self.name}, working_BAs={list(self.good_objs)}, errors={list(self.errors)})"
        )

    def output(self):
        if "output" not in self._dfs:
            if not self.good_objs:
                raise RuntimeError("output only available after `run_all`")
            df_process = (
                ("output", "BA_full_output", self._add_missing_output_cols),
                ("allocated_output", "BA_allocated_output", self._add_missing_cols),
                (
                    "sys_output",
                    "BA_system_output",
                    lambda x: x.filter(regex="^(?!storage_).*"),
                ),
                ("msg_output", "BA_messages", lambda x: x),
                ("re_specs_out", "re_plant_specs_pre_downselect", lambda x: x),
                ("colo_summary", "BA_colo_summary", lambda x: x),
            )
            dfs_ = {df_: [] for df_, *_ in df_process}
            re_path = Path.home() / f"patio_data/re_data_{self.pd.regime}.zip"
            with (
                logging_redirect_tqdm(),
                DataZip(self.path, "r") as z0,
                DataZip(re_path) if re_path.exists() else nullcontext() as z1,
            ):
                for_tqdm = tqdm(self.good_objs, desc="FullOutput")
                for ba_ in for_tqdm:
                    for_tqdm.set_description(desc="FullOutput " + ba_)
                    for df_, key, _ in df_process:
                        try:
                            dfs_[df_].append(z0[ba_, key.replace("BA", ba_)])
                        except Exception as exc:
                            if df_ == "re_specs_out" and z1 is not None:
                                dfs_[df_].append(z1[f"{ba_}_meta"])
                                if len(dfs_["output"]) == 1:
                                    LOGGER.error("Loading RE from %r", re_path)
                            else:
                                LOGGER.error("%s %r %r", ba_, exc, df_)
                                self.errors["output"][ba_].append(
                                    TracebackException.from_exception(exc)
                                )
            for n, _, func in df_process:
                try:
                    if n == "colo_summary":
                        if dfs_[n]:
                            self._dfs[n] = pl.concat(dfs_[n], how="diagonal_relaxed")
                        continue
                    self._dfs[n] = pd.concat(dfs_[n], axis=0).sort_index().pipe(func)
                except Exception as exc:
                    LOGGER.error("unable to concatenate %s", n, exc_info=exc)
            # bad_bas = (
            #         self.summary((
            #     self._dfs["output"],
            #     self._dfs["allocated_output"],
            #     self._dfs["sys_output"],
            # ))
            #         .query(
            #             "scenario == 'counterfactual' & deficit_max_pct_net_load >= 0.06"
            #         )
            #         .ba_code
            # )
            # for n, *_ in df_process:
            #     if n not in self._dfs:
            #         continue
            #     if n == "colo_summary":
            #         self._dfs[n] = self._dfs[n].with_columns(ba_high_cfl_deficit=pl.col('ba_code').is_in(bad_bas))
            #
            #     elif "ba_code" in self._dfs[n].index.names:
            #         self._dfs[n] = self._dfs[n].assign(
            #             ba_high_cfl_deficit=lambda x: x.index.get_level_values(
            #                 "ba_code"
            #             ).isin(bad_bas)
            #         )
            #     elif "ba_code" in self._dfs[n].columns:
            #         self._dfs[n] = self._dfs[n].assign(
            #             ba_high_cfl_deficit=lambda x: x.ba_code.isin(bad_bas)
            #         )
        return (
            self._dfs["output"],
            self._dfs["allocated_output"],
            self._dfs["sys_output"],
        )

    def fuel_curve_comparison(self):
        if "fuel_output" not in self._dfs:
            fuel_out = []
            with logging_redirect_tqdm(), DataZip(self.path, "r") as z0:
                for_tqdm = tqdm(self.good_objs, desc="FuelOutput")
                for ba_code in for_tqdm:
                    for_tqdm.set_description(desc="FuelOutput " + ba_code)
                    try:
                        fuel_out.append(z0[ba_code, f"{ba_code}_monthly_fuel"])
                    except Exception as exc:
                        LOGGER.error("%s %r", ba_code, exc)
            self._dfs["fuel_output"] = pd.concat(fuel_out, axis=0).sort_index()
        fuel = self._dfs["fuel_output"].copy()
        series_map = {
            "curve_fuel_price": "linear_area",
            # "curve_fuel_price_exp": "exponential",
            # "curve_fuel_price_lin": "linear",
            "fuel_per_mmbtu": "original",
        }
        curve = self.ad.curve.filter(
            pl.col("ba_code").is_in(list(fuel.index.unique("ba_code")))
        ).to_pandas()
        expanded_curve = [curve.assign(scenario=x) for x in fuel.index.unique("scenario")]
        out = pd.concat(
            [
                fuel.reset_index()
                .rename(columns=series_map)
                .melt(
                    id_vars=[
                        "ba_code",
                        "plant_id_eia",
                        "generator_id",
                        "scenario",
                        "re_limits_dispatch",
                        "datetime",
                        "mthly_mmbtu",
                        "cumsum_mmbtu",
                        "pct_of_curve_mmbtu_max",
                        "heat_rate",
                    ],
                    value_vars=list(series_map.values()),
                    var_name="series",
                    value_name="final_fuel_cost_per_mmbtu",
                ),
                *expanded_curve,
            ]
        ).sort_values(
            ["ba_code", "scenario", "datetime", "series", "final_fuel_cost_per_mmbtu"]
        )
        return out

    def _add_missing_cols(self, df):
        missing = {
            "redispatch_mmbtu": 0.0,
            "redispatch_co2": 0.0,
            "redispatch_cost_fuel": 0.0,
            "redispatch_cost_vom": 0.0,
            "redispatch_cost_startup": 0.0,
        }
        return df.assign(**{k: v for k, v in missing.items() if k not in df})

    def _add_missing_output_cols(self, df):
        cols = ["plant_id_eia", "generator_id"]
        idx = df.index.names
        return (
            df.reset_index()
            .merge(
                self.ad.capex.sort_values([*cols, "datetime"])
                .groupby(["plant_id_eia", "generator_id"])[
                    ["capex_per_kw", "real_maint_capex_per_kw"]
                ]
                .last(),
                on=cols,
                how="left",
                validate="m:1",
            )
            .merge(
                self.ad.gens[["plant_id_eia", "generator_id", "technology_description"]],
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="m:1",
                suffixes=(None, "_r"),
            )
            .assign(
                technology_description=lambda x: x.technology_description.fillna(
                    x.technology_description_r
                )
            )
            .drop(columns=["technology_description_r"])
            .set_index(idx)
        )

    @staticmethod
    def update_trace(trc):
        if "selected" in trc.name:
            return trc.update(
                legendgroup="Selected sites",
                legendgrouptitle_text="Selected sites",
                name=trc.name.replace(" selected", ""),
            )
        elif "Fossil" in trc.name:
            return trc.update(
                legendgroup="Fossil",
                legendgrouptitle_text="Existing",
            )
        else:
            return trc.update(
                legendgroup="Potential sites",
                legendgrouptitle_text="Potential sites",
                # opacity=0.5
            )

    def make_all_maps(self, econ_select=True, re_energy=0.6, storage_li_pct=0.25, year=2035):
        from pypdf import PdfWriter

        with (
            logging_redirect_tqdm(),
            PdfWriter(fileobj=PATIO_DOC_PATH / f"figs/{self.name}_maps.pdf") as merger,
        ):
            for_tqdm = tqdm(self.bas, desc="Maps")
            for ba_code in for_tqdm:
                for_tqdm.set_description(desc="Maps " + ba_code)
                try:
                    fig = self.make_potential_selected_maps(
                        ba=ba_code,
                        econ_select=econ_select,
                        re_energy=re_energy,
                        storage_li_pct=storage_li_pct,
                        year=year,
                    )
                    merger.append(BytesIO(fig.to_image(format="pdf")))
                except Exception as exc:
                    LOGGER.error("%s %r", ba_code, exc)
                    self.errors["output"][ba_code].append(
                        TracebackException.from_exception(exc)
                    )

    def make_potential_selected_maps(
        self,
        ba: str | Sequence | None = None,
        owners: Sequence | None = None,
        year: int = 2031,
        selected: bool = True,  # noqa: FBT001
        potential: bool = True,  # noqa: FBT001
        fossil: bool = True,  # noqa: FBT001
        sensitivity: str | list | None = None,
        size_max: int = 6,
        subunitwidth: int = 1,
    ):
        """Args:
            ba: a patio BA code, a sequence of BA codes, or None to get all BAs
            owners: generation owners, if provided, ba is ignored
            year: year of econ selected to show
            selected: include selected sites
            potential: include all potential sites
            fossil: include fossil sites
            sensitivity: one or more sensitivities to display, by default all are displayed
            size_max: (default=6) max marker size, adjust for visual consistency
            subunitwidth: (default=1) control the width of state boundaries, can need
                to be increased to ~4 if vector versions will be resized

        Returns:

        """  # noqa: D414
        fos, re, sys_out, *_ = self.output()
        title = ba

        re = (
            self.econ_selected_allocated()
            .query("datetime.dt.year == @year & technology_description != 'Batteries'")
            .assign(
                re_type=lambda x: x.technology_description.replace(
                    {
                        "Solar Photovoltaic": "Solar",
                        "Offshore Wind Turbine": "Offshore Wind",
                        "Onshore Wind Turbine": "Onshore Wind",
                    }
                )
            )
        )
        if sensitivity is not None:
            sensitivity = [sensitivity] if isinstance(sensitivity, str) else sensitivity
            re = re.query("sensitivity in @sensitivity")

        if owners is not None:
            re = re.query(
                "utility_id_eia in @owners | parent_name in @owners | parent_ticker in @owners"
            )
            ba = list(
                self.ad.own.query(
                    "owner_utility_id_eia in @owners | parent_name in @owners | parent_ticker in @owners"
                )
                .final_ba_code.dropna()
                .unique()
            )
            title = ", ".join(map(str, owners))

        re_meta = self._dfs["re_specs_out"]

        if isinstance(ba, list | tuple):
            re = re.query("ba_code in @ba")
            re_meta = re_meta.query("ba_code in @ba")
            if owners is not None:
                re_meta = re_meta.merge(
                    self.ad.own.query(
                        "owner_utility_id_eia in @owners | parent_name in @owners | parent_ticker in @owners"
                    )
                    .drop_duplicates(subset=["plant_id_eia", "generator_id"])
                    .rename(columns={"plant_id_eia": "fos_id", "generator_id": "fos_gen"}),
                    on=["fos_id", "fos_gen"],
                    validate="m:1",
                    how="inner",
                )
            else:
                title = ", ".join(ba)
        elif ba is None:
            title = "All"
        else:
            re = re.query("ba_code == @ba")
            re_meta = re_meta.query("ba_code == @ba")

        re_meta = re_meta.merge(
            self.ad.gens[["plant_id_eia", "plant_name_eia"]].drop_duplicates(),
            left_on="fos_id",
            right_on="plant_id_eia",
            how="left",
            validate="m:1",
            suffixes=(None, "_r"),
        )
        join = lambda x: ", ".join(sorted(set(x)))  # noqa: E731
        re_potential = (
            re_meta.groupby(
                ["latitude_nrel_site", "longitude_nrel_site", "re_type"], as_index=False
            )
            .agg({"capacity_mw_nrel_site": "first", "plant_name_eia": join})
            .rename(columns={f"{x}_nrel_site": x for x in ("latitude", "longitude")})
            .assign(
                capacity_mw_nrel_site=lambda x: np.where(
                    x.re_type == "Fossil",
                    0.1,
                    x.capacity_mw_nrel_site
                    / x.groupby("re_type").capacity_mw_nrel_site.transform("max"),
                ),
                re_type=lambda x: x.re_type.str.title().str.replace("_", " "),
            )
            .sort_values("capacity_mw_nrel_site")
        )
        selected_ = (
            re.groupby(
                ["sensitivity", "latitude_nrel_site", "longitude_nrel_site", "re_type"],
                as_index=False,
            )
            .agg({"capacity_mw": "sum", "plant_name_eia": join})
            .rename(columns={f"{x}_nrel_site": x for x in ("latitude", "longitude")})
            .assign(
                re_type=lambda x: x.re_type.str.title().str.replace("_", " ") + " selected",
                capacity_mw_nrel_site=lambda x: 2 * x.capacity_mw / x.capacity_mw.max(),
            )
        )
        fossil_ = (
            re_meta.groupby(["fos_id"], as_index=False)[
                ["latitude", "longitude", "technology_description", "plant_name_eia"]
            ]
            .first()
            .assign(re_type="Fossil", capacity_mw_nrel_site=np.nan)
        )
        re = pd.concat(
            [
                *(
                    [re_potential.assign(sensitivity=s) for s in re.sensitivity.unique()]
                    if potential
                    else [MTDF.copy()]
                ),
                selected_ if selected else MTDF.copy(),
                *(
                    [
                        fossil_.assign(
                            sensitivity=s,
                            re_type=lambda x: np.where(
                                x.fos_id.isin(re[re.sensitivity == s].fos_id),  # noqa: B023
                                "Repowered Fossil",
                                "Fossil",
                            ),
                        ).sort_values("re_type")
                        for s in re.sensitivity.unique()
                    ]
                    if fossil
                    else [MTDF.copy()]
                ),
            ]
        )
        re = re.assign(
            capacity_mw_nrel_site=lambda x: x.capacity_mw_nrel_site.mask(
                x.re_type == "Repowered Fossil", x.capacity_mw_nrel_site.quantile(0.1)
            )
            .mask(x.re_type == "Fossil", x.capacity_mw_nrel_site.quantile(0.1))
            .fillna(1)
        )
        return (
            px.scatter_geo(
                re,
                lat="latitude",
                lon="longitude",
                color="re_type",
                hover_name="plant_name_eia",
                locationmode="USA-states",
                color_discrete_map={k + " selected": v for k, v in COLOR_MAP.items()}
                | {
                    "Onshore Wind selected": "#529cba",
                    "Offshore Wind selected": "#86984c",
                    "Repowered Fossil": "#000000",
                    "Fossil": "#e4e4e4",
                    "Solar": "#fff8d5",
                    "Offshore Wind": "#f2f3d7",
                    "Onshore Wind": "#ebf5f9",
                },
                size="capacity_mw_nrel_site",
                size_max=size_max,  # projection=
                opacity=1,
                height=200 + 300 * np.ceil(len(re.sensitivity.unique()) / 2),
                facet_row_spacing=0.02,
                facet_col="sensitivity",
                facet_col_wrap=2,
            )
            .update_geos(
                fitbounds="locations",
                landcolor="rgb(230,232,234)",
                scope="usa",
                subunitwidth=subunitwidth,
            )
            .update_traces(marker=dict(line_width=0))  # noqa: C408
            .for_each_trace(self.update_trace)
            .for_each_annotation(lambda a: a.update(text=a.text.split("sensitivity=")[-1]))
            .update_layout(
                title=title,
                legend_title=None,
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.01,
                legend_xanchor="right",
                legend_x=1,
            )
        )

    def compile_figs(self, path=None, startswith=""):
        if not self.good_objs:
            raise RuntimeError("output only available after `run_all`")

        if path is None:
            path = self.dir_path / f"{self.name}_{startswith}.pdf"

        if (ROOT_PATH / path).exists():
            raise FileExistsError(f"{(ROOT_PATH / path)} exists")
        from pypdf import PdfWriter

        with (
            logging_redirect_tqdm(),
            DataZip(self.path, "r") as z0,
            PdfWriter(fileobj=ROOT_PATH / path) as merger,
        ):
            for_tqdm = tqdm(self.good_objs, desc="Figs")
            for ba_code in for_tqdm:
                for_tqdm.set_description(desc="Figs " + ba_code)
                try:
                    for f in z0.namelist():
                        if "pdf" in f and f.startswith(f"{startswith}_{ba_code}"):
                            merger.append(BytesIO(z0.read(f)))
                except Exception as exc:
                    LOGGER.error("%s %r", ba_code, exc)
                    self.errors["output"][ba_code].append(
                        TracebackException.from_exception(exc)
                    )

    def upload_output(self):
        rmi_cloud_put(self.dir_path, "patio-results/")

    def write_output(self):
        renamer = {
            "datetime": "year",
            "operating_year": "operational_year",
            "operating_month": "operational_month",
        }
        fos, re, sys_out, *_ = self.output()
        with DataZip(self.dir_path / f"{self.name}_results", "w") as z:
            z["pudl_release"] = self.pudl_release
            # Adapt final outputs for easier compatibility with the patio economic model
            z["full"] = (
                fos.reset_index()
                .rename(columns=renamer)
                .assign(
                    year=lambda x: x.year.dt.year,
                    operating_date=lambda x: x.operating_date.mask(
                        x.category == "patio_clean", pd.NaT
                    ),
                    operational_year=lambda x: x.operating_date.dt.year,
                    operational_month=lambda x: x.operating_date.dt.month,
                    retirement_year=lambda x: x.retirement_date.dt.year,
                    retirement_month=lambda x: x.retirement_date.dt.month,
                )
            )
            if not re.empty:
                z.reset_ids()
                z["allocated"] = (
                    re.reset_index()
                    .rename(columns=renamer)
                    .assign(
                        year=lambda x: x.year.dt.year,
                        operating_date=lambda x: x.operating_date.mask(
                            x.category == "patio_clean",
                            pd.NaT,
                        ),
                        ccs_eligible=lambda x: np.where(
                            x.technology_description == "Conventional Steam Coal",
                            True,
                            False,
                        ),
                    )
                )
            if not sys_out.empty:
                z.reset_ids()
                z["system"] = (
                    sys_out.reset_index()
                    .rename(columns=renamer)
                    .assign(
                        year=lambda x: x.year.dt.year,
                        # dispatch_curtailment_pct=lambda x: x.curtailment_mwh
                        # / x.dispatch_load_mwh,
                        # dispatch_re_curtailment_pct=lambda x: x.re_curtailment_mwh
                        # / x.dispatch_re_mwh,
                        # dispatch_non_re_curtailment_pct_of_non_re_gen=lambda x: (
                        #     x.curtailment_mwh - x.re_curtailment_mwh
                        # )
                        # / (x.dispatch_load_mwh - x.dispatch_re_mwh),
                    )
                )
            if not self._dfs["msg_output"].empty:
                z.reset_ids()
                z["msg"] = self._dfs["msg_output"].reset_index()
            if not self._dfs["colo_summary"].is_empty():
                z.reset_ids()
                z["colo_summary"] = self._dfs["colo_summary"]
            if not (suma := self.summary()).empty:
                z.reset_ids()
                z["summary"] = suma
            log_count = 0
            for ext in (".log", ".jsonl"):
                try:
                    z.write(
                        PATIO_DOC_PATH / f"logs/{self.name.split('_')[1]}{ext}",
                        self.name + ext,
                    )
                    log_count += 1
                except Exception as exc:
                    LOGGER.info("unable to write log", exc_info=exc)
            if not log_count:
                LOGGER.error("unable to write log")

    # def gross_up(self):
    #     f, _, s, *_ = self.output()
    #     if "old_clean" in f.category:
    #         return None
    #
    #     all_old_re = []
    #
    #     for ba_code in tqdm(
    #         f.index.get_level_values("ba_code").unique(), desc="gross_up"
    #     ):
    #         if self.data_source is None:
    #             d = self.pd.get_ba_data(ba_code, **self.data_kwargs)
    #             old_re_specs = d["old_re_specs"]
    #             old_re_profs = d["old_re_profs"]
    #         else:
    #             with DataZip(Path.home() / self.data_source) as zs:
    #                 old_re_specs = zs[ba_code, "old_re_specs"]
    #                 old_re_profs = zs[ba_code, "old_re_profs"]
    #         old_re = (
    #             old_re_specs.reset_index()
    #             .assign()
    #             .merge(
    #                 (old_re_profs * old_re_specs.capacity_mw)
    #                 .groupby(pd.Grouper(freq="YS"))
    #                 .sum()
    #                 .stack([0, 1])
    #                 .reset_index(name="redispatch_mwh"),
    #                 on=["plant_id_eia", "generator_id"],
    #                 validate="1:m",
    #             )
    #         )
    #         all_old_re.append(
    #             pd.concat(
    #                 old_re.assign(scenario=scen)
    #                 for scen in f.index.get_level_values("scenario").unique()
    #             )
    #             .assign(ba_code=ba_code)
    #             .set_index(f.index.names)
    #         )
    #
    #     f = pd.concat([f, *all_old_re]).sort_index()
    #
    #     self._dfs["output"] = f
    #     self._dfs["sys_output"] = s
    #
    #     pass

    def deficits(self):
        _, _, sys_out, *_ = self.output()
        return (
            sys_out.reset_index()
            .groupby(
                [
                    "ba_code",
                    "re_energy",
                    "nuclear_scen",
                    "storage_li_pct",
                    "storage_fe_pct",
                    "storage_h2_pct",
                    "ccs_scen",
                    "excl_or_moth",
                    "no_limit_prime",
                ]
            )[
                [
                    "deficit_mwh",
                    "dirty_charge_mwh",
                    "curtailment_mwh",
                    "deficit_max_pct_net_load",
                    "deficit_gt_2pct_count",
                ]
            ]
            .max()
        )

    def pivot_column(self, column: str):
        _, _, s, *_ = self.output()
        return (
            s.reset_index()
            .assign(year=lambda x: x.datetime.dt.year)
            .pivot_table(index=s.index.names[:-1], columns=["year"], values=[column])
        )

    def summary(self, frs=None):
        if frs is None:
            f, r, s, *_ = self.output()
        else:
            f, r, s = frs
        f = f.reset_index()
        co2 = (
            f[f.category.isin(["existing_fossil", "proposed_fossil"])]
            .groupby(["ba_code", "scenario"], as_index=False)[
                ["redispatch_co2_tonne", "redispatch_mwh"]
            ]
            .sum()
        )
        co2 = (
            co2.merge(
                co2.query("scenario == 'historical'")[
                    ["ba_code", "redispatch_co2_tonne", "redispatch_mwh"]
                ],
                on="ba_code",
                validate="m:1",
                suffixes=("", "_hist"),
            )
            .merge(
                f.groupby(["ba_code", "scenario"], as_index=False)[["redispatch_co2_tonne"]]
                .sum()
                .query("scenario == 'historical'"),
                on="ba_code",
                validate="m:1",
                suffixes=("", "_full"),
            )
            .assign(
                avoided_co2_tonne=lambda x: x.redispatch_co2_tonne_hist
                - x.redispatch_co2_tonne,
                avoided_mwh=lambda x: x.redispatch_mwh_hist - x.redispatch_mwh,
                avoided_co2_pct=lambda x: x.avoided_co2_tonne / x.redispatch_co2_tonne_hist,
                avoided_co2_pct_incl_xpatio=lambda x: x.avoided_co2_tonne
                / x.redispatch_co2_tonne_full,
            )[
                [
                    "ba_code",
                    "scenario",
                    "avoided_mwh",
                    "avoided_co2_tonne",
                    "avoided_co2_pct",
                    "avoided_co2_pct_incl_xpatio",
                ]
            ]
        )
        r_base_cols = [
            f"{pre}_{res}"
            for pre in ("patio", "proposed", "old")
            for res in CLEAN_TD_MAP.values()
        ] + list(dict.fromkeys(OTHER_TD_MAP.values()))
        full = {c + "_mw": "max" for c in r_base_cols} | {
            c + "_mwh": "sum" for c in r_base_cols
        }

        aggs = {
            "re_curtailment_pct": "max",
            "curtailment_mwh": "sum",
            "re_curtailment_mwh": "sum",
            "deficit_pct": "max",
            "deficit_max_pct_net_load": "max",
            "deficit_gt_2pct_count": "max",
            "fossil_retirements_cum_mw": "max",
        }
        suma = (
            s.reset_index()
            .groupby(["ba_code", "scenario"])
            .agg(
                {k: v for k, v in aggs.items() if k in s}
                | {k: v for k, v in full.items() if k in s}
            )
            .merge(
                f.groupby(
                    ["ba_code", "scenario", "plant_id_eia", "generator_id"],
                    as_index=False,
                )[["implied_need_mw", "implied_need_mwh"]]
                .max()
                .groupby(["ba_code", "scenario"])[["implied_need_mw", "implied_need_mwh"]]
                .sum(),
                on=["ba_code", "scenario"],
                how="left",
                validate="1:1",
            )
            .merge(
                co2,
                on=["ba_code", "scenario"],
                how="left",
                validate="1:1",
            )
            .merge(
                f.query("scenario == 'historical'")
                .groupby(["ba_code", "datetime"], as_index=False)
                .capacity_mw.sum()
                .groupby("ba_code", as_index=False)
                .capacity_mw.max()
                .rename(columns={"capacity_mw": "system_mw"}),
                on="ba_code",
                how="left",
                validate="m:1",
            )
            .assign(
                ba_avg_reduction=lambda x: x.groupby(["ba_code"]).avoided_co2_tonne.transform(
                    "mean"
                )
            )
        )
        return suma

    def for_toy(self):
        f, r, s, *_ = self.output()
        re_first_cols = (
            "solar_mw",
            "onshore_wind_mw",
            # "solar_share",
            # "wind_share",
            "storage_li_mw",
            "storage_fe_mw",
        )
        re_max_cols = (
            "dirty_charge_mwh",
            "curtailment_mwh",
            "deficit_mwh",
            "deficit_max_pct_net_load",
            "deficit_gt_2pct_count",
            "storage_li_max_mw",
            "storage_fe_max_mw",
            "storage_li_mw_utilization",
            "storage_fe_mw_utilization",
        )
        scen_fields = [*ScenarioConfig._fields, "scenario", "re_limits_dispatch"]
        system = (
            s
            # .assign(
            #     solar_share=lambda x: x.solar_mw
            #     / x[["solar_mw", "onshore_wind_mw"]].sum(axis=1),
            #     wind_share=lambda x: 1 - x.solar_share,
            # )
            .reset_index()
            .groupby(scen_fields)
            .agg(dict.fromkeys(re_first_cols, "first") | dict.fromkeys(re_max_cols, "max"))
        )
        re = (
            r.reset_index()
            .groupby(["ba_code", *scen_fields, "re_type", "datetime"])[
                ["capacity_mw", "redispatch_mwh"]
            ]
            .sum()
            .assign(cf=lambda x: x.redispatch_mwh / (x.capacity_mw * hrs_per_year(x)))
            .pivot_table(
                index=["ba_code", *scen_fields],
                columns="re_type",
                values="cf",
                aggfunc="mean",
            )
            .rename(columns={"onshore_wind": "wind_cf", "solar": "solar_cf"})
            # .drop(columns=["Nuclear"])
        )
        # fos = (
        #     f[["historical_mwh", "redispatch_mwh", "capacity_mw"]]
        #     .reset_index()
        #     .groupby(
        #         ["ba_code", *ScenarioConfig._fields, "plant_id_eia", "generator_id"]
        #     )
        #     .agg(
        #         {
        #             "historical_mwh": "mean",
        #             "redispatch_mwh": "mean",
        #             "capacity_mw": "first",
        #         }
        #     )
        #     .query("plant_id_eia in (703, 2103)")
        #     .groupby(["ba_code", *ScenarioConfig._fields, "plant_id_eia"])
        #     .sum()
        #     .assign(
        #         historical_cf=lambda x: x.historical_mwh / (x.capacity_mw * 8760),
        #         redispatch_cf=lambda x: x.redispatch_mwh / (x.capacity_mw * 8760),
        #     )
        #     .reset_index()
        # )
        hist_f = (
            f.astype({"prime_mover_code": str})
            .replace({"prime_mover_code": FOSSIL_PRIME_MOVER_MAP})
            .groupby(["ba_code", *scen_fields, "plant_id_eia", "generator_id"])
            .agg(
                {
                    "redispatch_mwh": "mean",
                    "prime_mover_code": "first",
                }
            )
            .pivot_table(
                index=["ba_code", *scen_fields],
                columns="prime_mover_code",
                values=["redispatch_mwh"],
                aggfunc="sum",
            )
            .reorder_levels([1, 0], axis=1)
            .sort_index(axis=1)
        )
        hist_f.columns = map("_".join, hist_f.columns)
        re_mwh = (
            r.astype({"technology_description": str})
            .fillna({"redispatch_mwh": 0.0})
            .groupby(
                [
                    "ba_code",
                    *scen_fields,
                    "plant_id_eia",
                    "generator_id",
                    "re_plant_id",
                    "re_type",
                ]
            )
            .agg({"redispatch_mwh": "mean", "technology_description": "first"})
            .pivot_table(
                index=["ba_code", *scen_fields],
                columns="technology_description",
                values="redispatch_mwh",
                aggfunc="sum",
            )
        )
        re_mwh.columns = [c + "_mwh" for c in re_mwh.columns]
        out = (
            system.merge(
                re,
                left_index=True,
                right_index=True,
                validate="1:1",
                how="outer",
            )
            .reset_index()
            .merge(
                re_mwh.reset_index(),
                on=["ba_code", *scen_fields],
                validate="1:1",
                how="outer",
            )
            .merge(
                hist_f,
                on=["ba_code", *scen_fields],
                validate="1:1",
                how="outer",
            )
        )
        return out[
            dict.fromkeys(["ba_code"])
            | dict.fromkeys([x for x in out if x not in ScenarioConfig._fields])
        ].sort_values(["ba_code", "scenario", "re_limits_dispatch"])

    def profile_check(self, mapper=None, norm=False, all=False, cutoff=0.75):  # noqa: A002
        out = {}
        with DataZip(self.path, "r") as z0:
            for ba_code in self.good_objs:
                ba = z0[ba_code]
                out[ba_code] = ba.profile_check(mapper=mapper)
        out = pd.concat(out, axis=0, names=["ba_code"])
        out = out[["total"] + [x for x in out if "total" not in x]]
        norm_ = out / out.groupby(["ba_code"]).transform("max")
        if norm:
            return norm_
        if all:
            return out
        bad_bas = norm_[norm_.total < cutoff].index.get_level_values(0).unique()
        return pd.concat(
            [norm_.loc[bad_bas, "total"], out.loc[bad_bas, :].iloc[:, 1:]], axis=1
        ).dropna(how="all", axis=1)

    def top_parents_fig(
        self,
        n=20,
        by="parent_name",
        *,
        include_proposed=False,
        df_only=False,
        query=None,
    ):
        if not isinstance(by, str):
            raise TypeError(f"`by` must be a str, received {type(by)}")

        df = self.selected_re_for_fig(
            by=by, include_proposed=include_proposed, pivot=False, df_only=True
        )
        if query is not None:
            df = df.query(query)
        n = n * len(df.sensitivity.unique())
        utils = (
            df.assign(
                _cap=lambda x: x.capacity_gw.mask(x.category == "proposed_clean", 0),
                _sort=lambda x: x.groupby(["sensitivity", by])._cap.transform("sum"),
            )
            .sort_values(["_sort", "technology_description"], ascending=False)
            .assign(
                _grp=lambda x: x.groupby(
                    ["sensitivity", by], sort=False
                ).capacity_gw.transform("ngroup"),
            )
            .query("_grp < @n")
            .sort_values(["sensitivity", "_grp"], ascending=[True, False])
        )

        # util_order = utils[["_grp", by]].drop_duplicates()[ by].to_list()

        if df_only:
            return (
                utils.pivot(
                    index=["sensitivity", by],
                    columns="technology_description",
                    # columns=["category", "technology_description"],
                    values="capacity_gw",
                )
                .sort_index(axis=1, ascending=[False, True])
                .assign(_s=lambda x: x.sum(axis=1))
                .sort_values(["sensitivity", "_s"])
                .drop(columns=["_s"])
                # .loc[reversed(util_order), :]
            )

        return (
            px.bar(
                utils
                # .sort_values(["_grp"], ascending=[False])
                .assign(
                    r=lambda x: x.technology_description.map(PLOT_MAP | {"Wind": "Wind"}),
                    capacity_gw=lambda x: x.capacity_gw.mask(
                        x.category == "proposed_clean", -x.capacity_gw
                    ),
                ),
                y=by,
                x="capacity_gw",
                facet_col="sensitivity",
                facet_col_wrap=2,
                pattern_shape="category" if include_proposed else None,
                pattern_shape_map={"proposed_clean": "/", "patio_clean": ""},
                color="r",
                # category_orders={by: util_order},
                orientation="h",
                color_discrete_map=COLOR_MAP | {"Wind": COLOR_MAP["Onshore Wind"]},
                height=200 + 200 * np.ceil(len(df.sensitivity.unique()) / 2),
                facet_row_spacing=0.03,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("sensitivity=")[-1]))
            .update_layout(
                xaxis_title="GW",
                yaxis_title=None,
                legend_title=None,
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.01,
                legend_xanchor="right",
                legend_x=1,
            )
            .update_yaxes(showticklabels=True, matches=None)
        )

    def fig_summary(
        self,
        ba=None,
        region=None,
        parent=None,
        df_only=False,
        # sensitivity="max_earnings, refi_trigger=1.0, Debt_Repl_EIR=0.1, Equity_Repl_EIR=0.1",
    ):
        value_vars = [
            "MWh",
            "Earnings",
            "Emissions",
            "Costs",
            "Capex_Costs",
            "Opex",
            # "Costs_Disc",
            # "Capex_Costs_Disc",
            # "Opex_Disc",
            # "MWh_Disc",
            # "Earnings_Disc",
        ]
        id_vars = [
            "sensitivity",
            "ba_code",
            "portfolio",
            "technology_description",
            # "status",
            "datetime",
        ]
        df = self.econ_results().query(
            "is_irp_year "
            # "& sensitivity == @sensitivity "
            "& ~historical_actuals"
        )

        if region is not None:
            df = df[df.region == region]
        if ba is not None:
            df = df[df.ba_code == ba]
        if parent is not None:
            df = df[(df.parent_name == parent) | (df.parent_ticker == parent)]

        df = (
            df.assign(
                technology_description=lambda x: x.technology_description.map(PLOT_MAP).fillna(
                    "other"
                ),
                status=lambda x: x.category.map(
                    {
                        "patio_clean": "patio",
                        "old_clean": "old",
                        "proposed_clean": "proposed",
                        "system": "system",
                        "existing_fossil": "existing",
                        "existing_xpatio": "xpatio",
                        "proposed_fossil": "proposed",
                    }
                ).fillna("other"),
                datetime=lambda x: pd.to_datetime(x.operating_year.astype(str) + "-01-01"),
                portfolio=lambda x: np.where(x.selected, "selected", "counterfactual"),
            )
            .groupby(id_vars, as_index=False)[value_vars]
            .sum()
            .melt(id_vars=id_vars, value_vars=value_vars)
        )
        if df_only:
            return df

        def match_axis(axis):
            num = axis.anchor.lstrip("x")
            if num and (num := int(num)) % 2 == 0:
                axis.update(matches=f"y{num - 1}", showticklabels=False)
                return
            axis.update(matches=None, showticklabels=True)
            return

        return (
            px.bar(
                df,
                x="datetime",
                y="value",
                color="technology_description",
                # pattern_shape="status",
                color_discrete_map=COLOR_MAP,
                facet_col="portfolio",
                facet_row="variable",
                height=800,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .for_each_yaxis(match_axis)
        )

    def econ_results(self):
        if "econ_results" not in self._dfs:
            rules = pd.read_csv(
                ROOT_PATH / "patio/package_data/ba_rules.csv"
            ).drop_duplicates()

            fs = rmi_cloud_fs()

            LOGGER.warning("FILLING MISSING STATES WITH STATE IN BA WITH GREATEST CAPACITY")
            ix_cols = ("ba_code", "plant_id_eia", "generator_id", "category")
            scfl_map = {False: "selected", True: "cfl"}
            with fs.open(f"az://{self.econ_dir_path}/final_outputs.parquet") as f:
                df = (
                    pl.scan_parquet(f)
                    .filter(pl.col("historical_actuals").not_() & pl.col("is_irp_year"))
                    .group_by(
                        *ix_cols,
                        pl.col("counterfactual_baseline").replace_strict(
                            scfl_map, default=None
                        ),
                    )
                    .agg(pl.col("operating_year").max())
                    .collect()
                    .pivot(
                        values="operating_year", index=ix_cols, on="counterfactual_baseline"
                    )
                    .filter(pl.col("selected") != pl.col("cfl"))
                    .sort(*ix_cols)
                )
            if not df.is_empty():
                weird = df.to_pandas().to_string().replace("\n", "\n\t")
                LOGGER.error(
                    "The final year of some resources does not match between the selected and counterfactual scenario:\n    %s",
                    weird,
                )
            c_path = str(
                AZURE_CACHE_PATH / cached_path(f"{self.econ_dir_path}/final_outputs.parquet")
            )

            # good_bas = (
            #     self.output()[2]
            #     # .query("~ba_high_cfl_deficit")
            #     .reset_index()
            #     .ba_code.unique()
            # )
            na_subset = [
                "build_year",
                "operating_year",
                "MW",
                "Costs",
                "Capex_Costs",
                "Opex",
                "MWh",
                "Earnings",
                "Emissions",
                "Emissions_Reduced",
                "Disc",
                "Disc_E",
                "Costs_Disc",
                "Capex_Costs_Disc",
                "Opex_Disc",
                "MWh_Disc",
                "Earnings_Disc",
            ]
            eresults = pl.scan_parquet(c_path)
            if (
                len(
                    eresults.select("refi_trigger", "Debt_Repl_EIR", "Equity_Repl_EIR")
                    .unique()
                    .collect()
                )
                > 1
            ):
                raise RuntimeError("Need code here to deal with multiple EIR settings")

            self._dfs["econ_results"] = (
                eresults
                # .filter(
                #     pl.col("ba_code").is_in(list(good_bas))
                #     # & (
                #     #     pl.col("counterfactual_baseline")
                #     #     | (
                #     #         pl.col("counterfactual_baseline").is_not()
                #     #         & ~pl.col("historical_actuals")
                #     #     )
                #     # )
                #     # & pl.col("least_cost").is_not()
                #     # & (pl.col("refi_trigger") == 1.5)
                #     # & (pl.col("Debt_Repl_EIR") == 0.3)
                #     # & (pl.col("Equity_Repl_EIR") == 0.2)
                # )
                .rename({"Utility_ID_Econ": "owner_utility_id_eia"})
                .collect()
                .to_pandas()
                .dropna(
                    subset=na_subset,
                    axis=0,
                    how="all",
                )
                .astype(
                    {
                        "owner_utility_id_eia": int,
                        "plant_id_eia": int,
                        "build_year": int,
                        "operating_year": int,
                        "technology_description": "string",
                        "category": "string",
                    }
                )
                .replace(
                    {
                        "technology_description": {
                            "Utility-Scale Battery Storage": "Batteries",
                            "LandbasedWind": "Onshore Wind Turbine",
                            "UtilityPV": "Solar Photovoltaic",
                            "OffShoreWind": "Offshore Wind Turbine",
                            "Pumped Storage Hydropower": "Hydroelectric Pumped Storage",
                        }
                    }
                )
                .merge(
                    self.ad.own[
                        [
                            "owner_utility_id_eia",
                            "owner_utility_name_eia",
                            "parent_name",
                            "parent_ticker",
                            "balancing_authority_code_eia",
                        ]
                    ].drop_duplicates(subset=["owner_utility_id_eia"], keep="first"),
                    on="owner_utility_id_eia",
                    how="left",
                    validate="m:1",
                )
                .merge(rules, on="balancing_authority_code_eia", how="left", validate="m:1")
                .merge(
                    self.ad.gens[["plant_id_eia", "state"]].drop_duplicates(),
                    on=["plant_id_eia"],
                    validate="m:1",
                    how="left",
                )
                .merge(
                    self.ad.gens.rename(
                        columns={"technology_description": "incumbent_technology"}
                    )[
                        [
                            "plant_id_eia",
                            "generator_id",
                            "incumbent_technology",
                        ]
                    ],
                    on=["plant_id_eia", "generator_id"],
                    how="left",
                    validate="m:1",
                )
                .assign(
                    state=lambda x: x.state.fillna(
                        x.ba_code.map(
                            self.ad.gens.groupby(["final_ba_code", "state"])
                            .capacity_mw.sum()
                            .reset_index()
                            .sort_values(
                                ["final_ba_code", "capacity_mw"],
                                ascending=[True, False],
                            )
                            .groupby("final_ba_code")
                            .state.first()
                            .to_dict()
                        )
                    ),
                    region=lambda x: x.ba_code.map(REGION_MAP),
                    selected=lambda x: ~x.counterfactual_baseline & ~x.historical_actuals,
                    datetime=lambda x: pd.to_datetime(x.operating_year.astype(str) + "-01-01"),
                    incumbent_technology=lambda x: np.where(
                        x.category == "patio_clean", x.incumbent_technology, np.nan
                    ),
                )
                .pipe(self.add_sensitivity_col)
            )
        return self._dfs["econ_results"]

    def add_sensitivity_col(self, df):
        return df.assign(
            sensitivity=lambda y: y[
                [
                    "least_cost",
                    "refi_trigger",
                    "savings_tol",
                    "earnings_thresh",
                    "Debt_Repl_EIR",
                    "Equity_Repl_EIR",
                ]
            ]
            .replace({"least_cost": {True: "least_cost", False: "max_earnings"}})
            .assign(
                refi_trigger=lambda x: np.where(
                    x.refi_trigger != 0.0,
                    "refi_trigger=" + x.refi_trigger.astype(str),
                    "",
                ),
                savings_tol=lambda x: np.where(
                    x.savings_tol != 0.0,
                    "savings_tol=" + x.savings_tol.astype(str),
                    "",
                ),
                earnings_thresh=lambda x: np.where(
                    x.earnings_thresh != 0.0,
                    "earnings_thresh=" + x.earnings_thresh.astype(str),
                    "",
                ),
                Debt_Repl_EIR=lambda x: np.where(
                    x.Debt_Repl_EIR != 0.0,
                    "Debt_Repl_EIR=" + x.Debt_Repl_EIR.astype(str),
                    "",
                ),
                Equity_Repl_EIR=lambda x: np.where(
                    x.Equity_Repl_EIR != 0.0,
                    "Equity_Repl_EIR=" + x.Equity_Repl_EIR.astype(str),
                    "",
                ),
            )
            .astype(str)
            .agg(", ".join, axis=1)
            .str.replace(", ,", ",")
            .astype("category"),
        )

    def econ_comparison(self, by="region", column="Costs_Disc", query=None):
        df = self.econ_results().query("~historical_actuals")
        if query is not None:
            df = df.query(query)
        df = df.query("is_irp_year").assign(
            scenario=lambda x: np.where(x.selected, "selected", "counterfactual")
        )
        if isinstance(by, list | tuple):
            by = ["sensitivity", *by]
        elif isinstance(by, str):
            by = ["sensitivity", by]
        else:
            raise TypeError(f"`by` must be a list, tuple, or str; received {type(by)}")

        return df.pivot_table(index=by, columns="scenario", values=column, aggfunc="sum")

    # def selected_reduction(self, by):
    #     """Use final_econ to calculate emission and cost reductions
    #
    #     Args:
    #         by:
    #
    #     Returns:
    #
    #     """
    #     warnings.warn("WRONG DO NOT USE", DeprecationWarning)
    #
    #     econ_results_path = (
    #         ROOT_PATH
    #         / f"econ_results/{self.name}_resultsIRA_TRUE/final_econ-{self.name}_resultsIRA_TRUE.csv"
    #     )
    #     id_cols = ["utility_id_eia", "ba_code", "scenario"]
    #     cols = [
    #         "Costs_Disc_Sum",
    #         "Capex_Costs_Disc_Sum",
    #         "Opex_Disc_Sum",
    #         "MWh_Disc_Sum",
    #         "Earnings_Disc_Sum",
    #         "Emissions_Sum",
    #         "Emissions_Reduced_Sum",
    #         "Forward_Earnings_irp_year",
    #         "Costs_irp_year",
    #         "Capex_Costs_irp_year",
    #         "Opex_irp_year",
    #         "Emissions_fin_build_year",
    #         "Emissions_Reduced_fin_build_year",
    #         "MW",
    #     ]
    #
    #     econ_results = (
    #         pd.read_csv(econ_results_path)
    #         .rename(columns={"Utility_ID_Econ": "utility_id_eia"})
    #         .query("~counterfactual_baseline")[[*id_cols, *cols, "Cost_Reduction"]]
    #         .merge(
    #             self.ad.own[
    #                 [
    #                     "owner_utility_id_eia",
    #                     "parent_name",
    #                     "parent_ticker",
    #                     "owner_utility_name_eia",
    #                 ]
    #             ].drop_duplicates(),
    #             left_on="utility_id_eia",
    #             right_on="owner_utility_id_eia",
    #             how="left",
    #             validate="m:1",
    #         )
    #         .assign(
    #             cost_reduction_usd=lambda x: x.Cost_Reduction * x.MWh_Disc_Sum,
    #             scenario=lambda x: x.scenario.fillna(""),
    #         )
    #     )
    #     return (
    #         econ_results.groupby(by)
    #         .agg(
    #             {"scenario": lambda x: sorted(set(x))}
    #             | {k: "sum" for k in [*cols, "cost_reduction_usd"]}
    #         )
    #         .assign(
    #             Levelized_Costs=lambda x: x.Costs_Disc_Sum / x.MWh_Disc_Sum,
    #             Cost_Reduction=lambda x: x.cost_reduction_usd / x.MWh_Disc_Sum,
    #         )
    #     )

    # def emission_cost_reduction(self, by: str | list = "ba_code"):
    #     warnings.warn("MIGHT NOT WORK RIGHT (WIP)", UserWarning)
    #     if isinstance(by, tuple):
    #         by = list(by)
    #     econ_results_by_tech = (
    #         pd.read_parquet(
    #             ROOT_PATH
    #             / f"econ_results/{self.name}_resultsIRA_TRUE/final_econ_by_tech-{self.name}_resultsIRA_TRUE.parquet"
    #         )
    #         .astype({"scenario": str, "ba_code": str, "Utility_ID_Econ": int})
    #         .rename(columns={"Utility_ID_Econ": "utility_id_eia"})
    #         .assign(
    #             region=lambda x: x.ba_code.map(REGION_MAP),
    #             is_selected=lambda x: x.is_selected.replace({None: False}).astype(bool),
    #             counterfactual_baseline=lambda x: x.counterfactual_baseline.replace(
    #                 {None: False}
    #             ).astype(bool),
    #             year=lambda x: x.operating_year.astype(int),
    #             day=1,
    #             month=1,
    #             datetime=lambda x: pd.to_datetime(x[["year", "month", "day"]]),
    #             usd_per_mwh=lambda x: x.Costs_Disc / x.MWh_Disc,
    #         )
    #         .drop(columns=["year", "month", "day"])
    #         .query("(counterfactual_baseline & ~is_selected) | is_selected")
    #         .merge(
    #             self.ad.own[
    #                 ["owner_utility_id_eia", "parent_name", "parent_ticker"]
    #             ].drop_duplicates(),
    #             left_on="utility_id_eia",
    #             right_on="owner_utility_id_eia",
    #             how="left",
    #             validate="m:1",
    #         )
    #     )
    #
    #     # f, r, s = self.output()
    #     # s = generate_projection_from_historical(
    #     #     s.reset_index(),
    #     #     year_mapper={k: k for k in range(2025, 2040)}
    #     #     | dict(zip(range(2040, 2056), list(range(2036, 2040)) * 4)),
    #     # )
    #     #
    #     # econ_results_by_tech = econ_results_by_tech.merge(
    #     #     s[["ba_code", "scenario", "datetime", "re_curtailment_pct"]],
    #     #     on=["ba_code", "scenario", "datetime"],
    #     #     how="left",
    #     #     validate="m:1",
    #     # ).assign(
    #     #     mwh_cur_adj=lambda x: x.MWh_Disc.mask(
    #     #         x.Technology_FERC == "renewables",
    #     #         x.MWh_Disc * (1 - x.re_curtailment_pct),
    #     #     )
    #     # )
    #
    #     keep_region = {} if "region" in by else {"region": "first"}
    #     keep_parent = {} if "parent_name" in by else {"parent_name": "first"}
    #
    #     selected = (
    #         econ_results_by_tech.query("is_selected & ~counterfactual_baseline")
    #         .groupby(by, dropna=False)
    #         .agg(
    #             keep_parent
    #             | keep_region
    #             | {
    #                 "scenario": "last",
    #                 "parent_ticker": "first",
    #                 "Emissions": "sum",
    #                 "Emissions_Reduced": "sum",
    #                 "Costs_Disc": "sum",
    #                 "MWh_Disc": "sum",
    #                 "MWh": "sum",
    #             }
    #         )
    #         .rename(
    #             columns={
    #                 "Emissions": "Emissions_selected",
    #                 # "Emissions_Reduced": "Emissions_Reduced_reduced",
    #                 "Costs_Disc": "Costs_Disc_selected",
    #                 "MWh_Disc": "MWh_Disc_selected",
    #                 "MWh": "MWh_selected",
    #             }
    #         )
    #         .reset_index()
    #     )
    #     cfl = (
    #         econ_results_by_tech.query("counterfactual_baseline & ~is_selected")
    #         .groupby(by, dropna=False)[
    #             ["Emissions", "Emissions_Reduced", "Costs_Disc", "MWh_Disc", "MWh"]
    #         ]
    #         .sum()
    #         .rename(
    #             columns={
    #                 "Emissions": "Emissions_counterfactual",
    #                 "Emissions_Reduced": "Emissions_Reduced_counterfactual",
    #                 "Costs_Disc": "Costs_Disc_counterfactual",
    #                 "MWh_Disc": "MWh_Disc_counterfactual",
    #                 "MWh": "MWh_counterfactual",
    #             }
    #         )
    #         .reset_index()
    #     )
    #     full = selected.merge(cfl, on=by, validate="1:1").assign(
    #         mwh_diff=lambda x: x.MWh_Disc_selected / x.MWh_Disc_counterfactual - 1,
    #         unit_cost_selected=lambda x: (
    #             x.Costs_Disc_selected / x.MWh_Disc_selected
    #         ).replace({np.inf: np.nan}),
    #         unit_cost_counterfactual=lambda x: (
    #             x.Costs_Disc_counterfactual / x.MWh_Disc_counterfactual
    #         ).replace({np.inf: np.nan}),
    #         unit_cost_savings_pct=lambda x: -(
    #             x.unit_cost_selected / x.unit_cost_counterfactual - 1
    #         ).replace({np.inf: np.nan}),
    #         savingsish=lambda x: (x.unit_cost_counterfactual - x.unit_cost_selected)
    #         * x.MWh_Disc_counterfactual,
    #         savings_pct=lambda x: x.savingsish / x.Costs_Disc_counterfactual,
    #         unit_emissions_selected=lambda x: x.Emissions_selected
    #         / x.MWh_Disc_selected,
    #         unit_emissions_counterfactual=lambda x: x.Emissions_counterfactual
    #         / x.MWh_Disc_counterfactual,
    #         emissions_original=lambda x: x.Emissions_selected + x.Emissions_Reduced,
    #         emissions_savings_pct=lambda x: x.Emissions_Reduced / x.emissions_original,
    #         # costs_reduced=lambda x: x.costs_counterfactual - x.costs_selected,
    #         # savings_pct=lambda x: x.costs_reduced / x.costs_counterfactual,
    #         # em_r=lambda x: x.Emissions - x.Emissions_counterfactual,
    #         # emission_reduction_pct=lambda x: (x.Emissions - x.Emissions_counterfactual)
    #         # / x.Emissions_counterfactual,
    #     )
    #     return full

    def selected_re_for_fig(
        self,
        by="ba_code",
        *,
        include_proposed=False,
        pivot=True,
        combine_wind=True,
        df_only=False,
    ):
        if isinstance(by, str):
            by = [by]

        df = (
            self.econ_results()
            .query("~historical_actuals")
            .merge(
                self.ad.gens[["plant_id_eia", "generator_id", "retirement_date"]],
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="m:1",
            )
        )
        if combine_wind:
            df = df.replace(
                {
                    "technology_description": {
                        "OffShoreWind": "Wind",
                        "LandbasedWind": "Wind",
                        "Offshore Wind Turbine": "Wind",
                        "Onshore Wind Turbine": "Wind",
                    }
                }
            )
        df = (
            (
                df.query(
                    "category in ('patio_clean', 'proposed_clean') "
                    "& is_irp_year"
                    "& operating_year == 2032 & selected "
                    "& technology_description not in ('Pumped Storage Hydropower', 'Nuclear', 'Geothermal', 'Hydroelectric Pumped Storage')",
                    # "& (surplus | (replacement & retirement_date.notna()))",
                    engine="python",
                )
                .groupby(["sensitivity", *by, "category", "technology_description"])
                .MW.sum()
                / 1000
            )
            .reset_index()
            .rename(columns={"MW": "capacity_gw"})
        )
        if not include_proposed:
            df = df.query("category  == 'patio_clean'")
        if pivot:
            return df.pivot(
                index=["sensitivity", *by],
                columns=["category", "technology_description"],
                values="capacity_gw",
            ).sort_index(axis=1, ascending=False)
        if df_only:
            return df

        def t(trace):
            if "_p" in trace.name:
                trace.update(showlegend=False)

        # print(df[by])
        assert len(by) == 1, "when plotting, "
        return (
            px.bar(
                df.assign(
                    r=lambda x: x.technology_description.map(PLOT_MAP | {"Wind": "Wind"})
                    + np.where(x.category == "proposed_clean", "_p", ""),
                ),
                x=by[0],
                y="capacity_gw",
                facet_col="sensitivity",
                facet_col_wrap=2,
                color="r",
                # category_orders={
                #     by[0]: sorted(df[by].squeeze().unique()),
                #     "r": ["Storage_p", "Wind_p", "Solar_p", "Storage", "Wind", "Solar"],
                # },
                color_discrete_map=COLOR_MAP
                | {"Wind": COLOR_MAP["Onshore Wind"]}
                | {"Wind_p": "#529cba", "Storage_p": "#c7c4e2", "Solar_p": "#ffeda9"},
                height=200 + 200 * np.ceil(len(df.sensitivity.unique()) / 2),
                facet_row_spacing=0.1,
                orientation="v",
            )
            .update_layout(
                yaxis_title="GW",
                xaxis_title=None,
                legend_title=None,
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.01,
                legend_xanchor="right",
                legend_x=1,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("sensitivity=")[-1]))
            .for_each_trace(t)
        )

    def energy_comparison_fig(self, by="region", year=2035, df_only=False):
        warnings.warn("MIGHT NOT WORK RIGHT (WIP)", UserWarning)  # noqa: B028

        ORDERING = {
            "nuclear": "000",
            "coal": "001",
            "coal ccs": "0015",
            "gas cc": "002",
            "gas ct": "004",
            "gas rice": "005",
            "gas st": "003",
            "other fossil": "006",
            "other": "006",
            "hydro": "007",
            "biomass": "008",
            "solar": "011",
            "onshore wind": "009",
            "offshore wind": "010",
            "storage": "012",
            "curtailment": "013",
            "january": "101",
            "february": "102",
            "march": "103",
            "april": "104",
            "may": "105",
            "june": "106",
            "july": "107",
            "august": "108",
            "september": "109",
            "october": "110",
            "november": "111",
            "december": "112",
            "historical": "200",
            "redispatch": "201",
            "cr": "204",
            "clean repowering": "204",
            "clean<br>repowering": "204",
            "planned": "203",
            "current": "202",
        }

        def patio_key(item):
            if isinstance(item, pd.Series):
                return item.astype(str).str.casefold().replace(ORDERING)
            if isinstance(item, pd.Index):
                return pd.Index([ORDERING.get(x.casefold(), str(x)) for x in item])
            return ORDERING.get(item.casefold(), str(item))

        to_plot = self.for_en_comp(by=by, year=year)
        to_plot = (
            to_plot.query(
                "technology_description not in ('curtailment', 'deficit') "
                # "& scenario !='counterfactual'"
            )
            .assign(
                redispatch_mwh=lambda x: x.redispatch_mwh.mask(
                    x.technology_description == "curtailment", -x.redispatch_mwh
                )
                / 1e6
            )
            .astype({"technology_description": str})
            .replace(
                {
                    "technology_description": PLOT_MAP
                    | {
                        "Natural Gas Internal Combustion Engine": "Other",
                        "Natural Gas Steam Turbine": "Other",
                    }
                }
            )
            .replace(
                {
                    "technology_description": {
                        "Hydroelectric Pumped Storage": "Hydro",
                        "Other Fossil": "Other",
                        "Solar Thermal with Energy Storage": "Solar",
                        "Solar Thermal without Energy Storage": "Solar",
                        "Natural Gas with Compressed Air Storage": "Other",
                        "Other Natural Gas": "Other",
                        "Biomass": "Other",
                        "Geothermal": "Other",
                        "Flywheels": "Other",
                    },
                    "scenario": {
                        "2021": "Current",
                        "counterfactual": "Planned",
                        "selected": "Clean<br>Repowering",
                    },
                }
            )
            .groupby(["technology_description", by, "scenario"], observed=True)
            .redispatch_mwh.sum()
            .reset_index()
            .sort_values(["technology_description", by, "scenario"], key=patio_key)
        )
        if df_only:
            return to_plot
        return (
            px.bar(
                to_plot,
                x="scenario",
                y="redispatch_mwh",
                color="technology_description",
                facet_col=by,
                # facet_col_wrap=3,
                height=600,
                width=1000,
                # width=1100,
                facet_col_spacing=0.00001,
                color_discrete_map=COLOR_MAP | {"Other": COLOR_MAP["Other Fossil"]},
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .update_xaxes(
                title=None,
                tickangle=270,
            )
            .update_yaxes(
                gridcolor="#58595B",
            )
            .update_layout(
                font_family="Tw Cen MT Std",
                plot_bgcolor="#FFFFFF",
                # paper_bgcolor='#FFFFFF',
                font_color="#58595B",
                yaxis_title="TWh",
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.05,
                legend_xanchor="left",
                legend_x=-0.05,
                legend_title=None,
            )
        )

    # def for_en_comp(self, by="region", year=2035):
    #     warnings.warn("MIGHT NOT WORK RIGHT (WIP)", UserWarning)
    #
    #     if "for_en_comp" + by not in self._dfs:
    #         f, *_ = self.output()
    #         f = self.by_owner(f)
    #         selected = (
    #             self.econ_selected_full()
    #             .query("datetime.dt.year == @year")
    #             .groupby([by, "technology_description"])
    #             .redispatch_mwh.sum()
    #             .reset_index()
    #             .assign(scenario="selected")
    #         )
    #         cfl = (
    #             f.reset_index()
    #             .query("datetime.dt.year == @year")
    #             .assign(region=lambda x: x.ba_code.map(REGION_MAP))
    #             .query("scenario == 'counterfactual'")
    #             .groupby([by, "technology_description", "scenario"])
    #             .redispatch_mwh.sum()
    #             .reset_index()
    #             .assign(scenario="counterfactual")
    #         )
    #         base = (
    #             f.reset_index()
    #             .query("datetime.dt.year == 2021")
    #             .assign(region=lambda x: x.ba_code.map(REGION_MAP))
    #             .query("scenario == 'counterfactual'")
    #             .groupby([by, "technology_description", "scenario"])
    #             .redispatch_mwh.sum()
    #             .reset_index()
    #             .assign(scenario="2021")
    #         )
    #         self._dfs["for_en_comp" + by] = pd.concat([selected, cfl, base])
    #     return self._dfs["for_en_comp" + by]

    @lru_cache  # noqa: B019
    def _econ_selected_all_old(self):
        f, *_ = self.output()
        with rmi_cloud_fs().open(f"az://{self.econ_dir_path}/scenario_selected.parquet") as f:  # noqa: F811
            return (
                pd.read_parquet(f)  # noqa: PD013
                .astype({"scenario": str, "ba_code": str, "Utility_ID_Econ": int})
                .rename(columns={"Utility_ID_Econ": "utility_id_eia"})
                .assign(
                    region=lambda x: x.ba_code.map(REGION_MAP),
                    year=lambda x: x.build_year.astype(int),
                    day=1,
                    month=1,
                    datetime=lambda x: pd.to_datetime(x[["year", "month", "day"]]),
                )
                .query("is_selected")[["utility_id_eia", "ba_code", "datetime", "scenario"]]
                .pivot_table(
                    index="datetime",
                    columns=["utility_id_eia", "ba_code"],
                    values="scenario",
                    aggfunc="first",
                )
                .reindex(index=f.index.get_level_values("datetime").unique())
                .ffill()
                .bfill()
                .stack([0, 1])
                .reset_index(name="scenario")
                .sort_values(["utility_id_eia", "ba_code", "datetime"])
            )

    @property
    def econ_dir_path(self):
        prefix, dt = self.name.split("_")
        return f"patio-results/{dt}" + (f"/{self.econ_scen_dir}" if self.econ_scen_dir else "")

    @lru_cache  # noqa: B019
    def _econ_selected_all(self):
        f, *_ = self.output()
        with rmi_cloud_fs().open(f"az://{self.econ_dir_path}/scenario_selected.parquet") as f:  # noqa: F811
            df = (
                pd.read_parquet(f)
                .pipe(self.add_sensitivity_col)
                .astype({"scenario": str, "ba_code": str, "Utility_ID_Econ": int})
                .rename(columns={"Utility_ID_Econ": "utility_id_eia"})
                .assign(
                    region=lambda x: x.ba_code.map(REGION_MAP),
                    year=lambda x: x.build_year.astype(int),
                    day=1,
                    month=1,
                    datetime=lambda x: pd.to_datetime(x[["year", "month", "day"]]),
                )
                .query("is_selected")[
                    ["sensitivity", "utility_id_eia", "ba_code", "datetime", "scenario"]
                ]
            )
        sensitivities = []
        for sensi in df.sensitivity.unique():
            sensitivities.append(
                df.query("sensitivity == @sensi")  # noqa: PD013
                .pivot_table(
                    index="datetime",
                    columns=["utility_id_eia", "ba_code"],
                    values="scenario",
                    aggfunc="first",
                )
                .reindex(index=f.index.get_level_values("datetime").unique())
                .ffill()
                .bfill()
                .stack([0, 1])
                .reset_index(name="scenario")
                .sort_values(["utility_id_eia", "ba_code", "datetime"])
                .assign(sensitivity=sensi)
            )

        return pd.concat(sensitivities)

    def by_owner(self, df):
        c_to_allocate = [
            "capacity_mw",
            "redispatch_mwh",
            "redispatch_mmbtu",
            "redispatch_co2_tonne",
            "redispatch_cost_fuel",
            "redispatch_cost_vom",
            "redispatch_cost_startup",
            "redispatch_cost_fom",
            "implied_need_mw",
            "implied_need_mwh",
        ]
        patio_by_owner = (
            df.reset_index()
            .merge(
                self.ad.own[
                    [
                        "plant_id_eia",
                        "generator_id",
                        "owner_utility_id_eia",
                        "owner_utility_name_eia",
                        "fraction_owned",
                        "balancing_authority_code_eia",
                        "parent_name",
                        "parent_ticker",
                    ]
                ],
                on=["plant_id_eia", "generator_id"],
                how="left",
                suffixes=(None, "_r"),
            )
            .assign(utility_id_eia=lambda x: x.owner_utility_id_eia)
        )
        if "balancing_authority_code_eia_r" in patio_by_owner:
            patio_by_owner = patio_by_owner.assign(
                balancing_authority_code_eia=lambda x: x.balancing_authority_code_eia.fillna(
                    x.balancing_authority_code_eia_r
                )
            ).drop(columns=["balancing_authority_code_eia_r"])
        for c in c_to_allocate:
            if c in patio_by_owner:
                patio_by_owner[c] = patio_by_owner[c] * patio_by_owner.fraction_owned
        return patio_by_owner

    # def econ_selected_full(self):
    #     f, r, s, *_ = self.output()
    #
    #     curtailment = generate_projection_from_historical(
    #         s.reset_index(),
    #         year_mapper={k: k for k in range(2025, 2040)}
    #         | dict(zip(range(2040, 2056), list(range(2036, 2040)) * 4)),
    #     )[["ba_code", "scenario", "datetime", "re_curtailment_pct"]]
    #     f = self.by_owner(f)
    #     esa = self._econ_selected_all()
    #
    #     full = (
    #         f.query("category != 'patio_clean'")
    #         .merge(
    #             esa,
    #             on=["utility_id_eia", "ba_code", "scenario", "datetime"],
    #             how="inner",
    #             suffixes=(None, "_econ"),
    #         )
    #         .merge(
    #             curtailment,
    #             on=["ba_code", "scenario", "datetime"],
    #             how="inner",
    #             validate="m:1",
    #         )
    #         .assign(plant_id_prof_site=0)
    #     )
    #     LOGGER.warning("WE ARE LOSING IRP PLANTS THAT DO NOT HAVE UTILITY_IDS")
    #     missing_from_econ = (
    #         f.merge(
    #             esa.groupby(["ba_code", "datetime"])
    #             .agg({"scenario": lambda x: sorted(set(x))[0]})
    #             .reset_index(),
    #             on=["ba_code", "scenario", "datetime"],
    #             how="inner",
    #             validate="m:1",
    #         )
    #         .merge(
    #             full[["utility_id_eia", "ba_code", "datetime"]].drop_duplicates(),
    #             on=["utility_id_eia", "ba_code", "datetime"],
    #             how="outer",
    #             indicator=True,
    #             validate="m:1",
    #         )
    #         .query(
    #             "category not in ('patio_clean', 'system', 'old_clean') & _merge == 'left_only' & utility_id_eia.notna()"
    #         )
    #     )
    #
    #     full = pd.concat([full, missing_from_econ])
    #
    #     techs = list(RE_TECH) + ["Geothermal", "Nuclear"]
    #     full = pd.concat(
    #         [
    #             full,
    #             full.query(
    #                 "technology_description in @techs & category != 'existing_xpatio'"
    #             )
    #             .assign(
    #                 redispatch_mwh=lambda x: x.redispatch_mwh * x.re_curtailment_pct,
    #                 technology_description="curtailment",
    #                 category="system",
    #             )
    #             .groupby(
    #                 [
    #                     "utility_id_eia",
    #                     "ba_code",
    #                     "scenario",
    #                     "datetime",
    #                     "technology_description",
    #                 ]
    #             )
    #             .agg(
    #                 {
    #                     "redispatch_mwh": "sum",
    #                     "parent_name": "first",
    #                     "parent_ticker": "first",
    #                     "owner_utility_name_eia": "first",
    #                     "balancing_authority_code_eia": "first",
    #                 }
    #             )
    #             .reset_index(),
    #         ]
    #     )
    #
    #     full = full[
    #         list(
    #             dict.fromkeys(["utility_id_eia", "ba_code", "scenario", "datetime"])
    #             | dict.fromkeys(full)
    #         )
    #     ]
    #     allocated = (
    #         self.by_owner(r)
    #         .merge(
    #             esa,
    #             on=["utility_id_eia", "ba_code", "scenario", "datetime"],
    #             how="inner",
    #             suffixes=(None, "_econ"),
    #         )
    #         .assign(
    #             plant_id_eia=lambda x: x.re_plant_id,
    #             generator_id=lambda x: x.re_generator_id,
    #         )
    #         .groupby(
    #             [
    #                 "utility_id_eia",
    #                 "ba_code",
    #                 "plant_id_eia",
    #                 "generator_id",
    #                 "datetime",
    #             ]
    #         )
    #         .agg(
    #             {
    #                 "scenario": "first",
    #                 "re_energy": "first",
    #                 "storage_li_pct": "first",
    #                 "storage_fe_pct": "first",
    #                 "storage_h2_pct": "first",
    #                 "nuclear_scen": "first",
    #                 "ccs_scen": "first",
    #                 "excl_or_moth": "first",
    #                 "no_limit_prime": "first",
    #                 "re_limits_dispatch": "first",
    #                 "capacity_mw": "sum",
    #                 "redispatch_mwh": "sum",
    #                 "redispatch_cost_fom": "sum",
    #                 # 'implied_need_mw': 'first',
    #                 # 'implied_need_mwh': 'first',
    #                 "category": "first",
    #                 "technology_description": "first",
    #                 "operating_date": "first",
    #                 "ilr": "first",
    #                 "duration_hrs": "first",
    #                 "roundtrip_eff": "first",
    #                 "reserve": "first",
    #                 "energy_community": "first",
    #                 # 'fos_id': 'first',
    #                 # 'fos_gen': 'first',
    #                 "class_atb": "first",
    #                 "plant_id_prof_site": "first",
    #                 # 'distance': 'first',
    #                 "latitude_nrel_site": "first",
    #                 "longitude_nrel_site": "first",
    #                 "re_site_id": "first",
    #                 "operating_year": "first",
    #                 "retirement_year": "first",
    #                 "redispatch_mmbtu": "sum",
    #                 "redispatch_co2": "sum",
    #                 "redispatch_cost_fuel": "sum",
    #                 "redispatch_cost_vom": "sum",
    #                 "redispatch_cost_startup": "sum",
    #                 "owner_utility_name_eia": "first",
    #                 "balancing_authority_code_eia": "first",
    #                 "parent_name": "first",
    #                 "parent_ticker": "first",
    #             }
    #         )
    #         .rename(
    #             columns={
    #                 "latitude_nrel_site": "latitude",
    #                 "longitude_nrel_site": "longitude",
    #             }
    #         )
    #         .reset_index()
    #     )
    #
    #     return (
    #         pd.concat([full, allocated])
    #         .astype({"utility_id_eia": int})
    #         .assign(region=lambda x: x.ba_code.map(REGION_MAP))
    #         .sort_values(
    #             [
    #                 "ba_code",
    #                 "utility_id_eia",
    #                 "plant_id_eia",
    #                 "generator_id",
    #                 "datetime",
    #             ]
    #         )
    #     )

    def econ_selected_allocated(self):
        f, r, s, *_ = self.output()
        patio_by_owner = (
            self.by_owner(r)
            .merge(
                self.ad.gens[
                    [
                        "plant_id_eia",
                        "generator_id",
                        "retirement_date",
                        "plant_name_eia",
                        "state",
                    ]
                ],
                on=["plant_id_eia", "generator_id"],
                how="left",
            )
            .merge(
                pd.read_csv(ROOT_PATH / "patio/package_data/ba_rules.csv"),
                on="balancing_authority_code_eia",
                how="left",
            )
            .assign(
                rule=lambda x: np.where(x.retirement_date.notnull(), "replacement", "surplus"),  # noqa: PD004
            )
        )
        return patio_by_owner.merge(
            self._econ_selected_all(),
            on=["utility_id_eia", "ba_code", "scenario", "datetime"],
            how="inner",
            suffixes=(None, "_econ"),
        )

    def compare_hist_cfl(
        self,
        values="redispatch_mwh",
        index=(
            "plant_id_eia",
            "plant_name_eia",
            "technology_description",
            "datetime",
        ),
    ):
        if isinstance(index, tuple):
            index = list(index)
        f, *_ = self.output()
        return (
            f.reset_index()
            .query(
                "scenario in ('historical', 'counterfactual') "
                "& datetime.dt.year in (2021, 2022) "
                "& category == 'existing_fossil'"
            )
            .assign(
                redispatch_cost_fuel_original=lambda x: x.redispatch_cost_fuel_original.fillna(
                    x.redispatch_cost_fuel
                ),
                total_cost_original=lambda x: x[
                    [
                        "redispatch_cost_fuel_original",
                        "redispatch_cost_vom",
                        "redispatch_cost_startup",
                        "redispatch_cost_fom",
                    ]
                ].sum(axis=1),
                total_cost=lambda x: x[
                    [
                        "redispatch_cost_fuel",
                        "redispatch_cost_vom",
                        "redispatch_cost_startup",
                        "redispatch_cost_fom",
                    ]
                ].sum(axis=1),
            )
            .fillna({"plant_name_eia": "UNK"})
            .pivot_table(
                values=values,
                index=index,
                columns=["scenario"],
                aggfunc="sum",
            )
            # .query("historical > 0")
            # .assign(pct_diff=lambda x: x.counterfactual / x.historical - 1)
        )

    def mcoe_compare(self, suffix=""):
        ch1 = self.compare_hist_cfl(
            values=["redispatch_mwh", "total_cost_original", "total_cost"],
            index=["ba_code", "technology_description", "datetime"],
        )
        ch1.columns = map("_".join, ch1.columns)
        out = (
            ch1.reset_index()
            .assign(
                cfl_mcoe=lambda x: x.total_cost_counterfactual
                / x.redispatch_mwh_counterfactual,
                cfl_og_mcoe=lambda x: x.total_cost_original_counterfactual
                / x.redispatch_mwh_counterfactual,
                hist_mcoe=lambda x: x.total_cost_historical / x.redispatch_mwh_historical,
                cfl_total_mcoe=lambda x: x.groupby(
                    ["ba_code", "datetime"]
                ).total_cost_counterfactual.transform("sum")
                / x.groupby(["ba_code", "datetime"]).redispatch_mwh_counterfactual.transform(
                    "sum"
                ),
                cfl_og_total_mcoe=lambda x: x.groupby(
                    ["ba_code", "datetime"]
                ).total_cost_original_counterfactual.transform("sum")
                / x.groupby(["ba_code", "datetime"]).redispatch_mwh_counterfactual.transform(
                    "sum"
                ),
                hist_total_mcoe=lambda x: x.groupby(
                    ["ba_code", "datetime"]
                ).total_cost_historical.transform("sum")
                / x.groupby(["ba_code", "datetime"]).redispatch_mwh_historical.transform(
                    "sum"
                ),
            )
            .set_index(ch1.index.names)[
                [
                    "cfl_mcoe",
                    "cfl_og_mcoe",
                    "hist_mcoe",
                    "cfl_total_mcoe",
                    "cfl_og_total_mcoe",
                    "hist_total_mcoe",
                ]
            ]
        )
        out.columns = [c + suffix for c in out.columns]
        return out

    def sales861(self):
        ba_only = not any(x.isalnum() for x in self.bas)

        warnings.warn(
            "Use cast_2022_ba_relationships to get backcasted BAs",
            RuntimeWarning,
            stacklevel=2,
        )

        sales = (
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
            .filter(
                (pl.col("report_date") > "2007-01-01")
                & pl.col("state").is_in(("HI", "AK")).not_()
            )
            .group_by(
                "utility_id_eia",
                "state",
                "balancing_authority_code_eia",
                pl.col("report_date").dt.year().cast(pl.Int64).alias("historical_year"),
            )
            .agg(
                pl.first("utility_name_eia"),
                pl.sum("sales_mwh").alias("retail_sales_mwh"),
            )
            .collect()
        )

        # Attempt at re-allocating sales, but it doesn't work
        # ba_allocations = sales.filter(
        #     pl.col("historical_year") == pl.col("historical_year").max()
        # ).select(
        #     "utility_id_eia",
        #     "state",
        #     pl.col("balancing_authority_code_eia").alias("ba_from_allocation"),
        #     pct=(
        #         pl.col("retail_sales_mwh")
        #         / pl.col("retail_sales_mwh")
        #         .sum()
        #         .over(
        #             "utility_id_eia",
        #             "state",
        #         )
        #     ).fill_nan(1.0),
        # )
        #
        # allocated_sales = (
        #     sales.with_columns(
        #         util_state_sales=pl.col("retail_sales_mwh")
        #         .sum()
        #         .over("utility_id_eia", "state", "historical_year")
        #     )
        #     .join(ba_allocations, on=["utility_id_eia", "state"], how="outer")
        #     .select(
        #         "historical_year",
        #         "utility_id_eia",
        #         "utility_name_eia",
        #         "state",
        #         "ba_from_allocation",
        #         "balancing_authority_code_eia",
        #         "retail_sales_mwh",
        #         allocated_sales=pl.col("util_state_sales") * pl.col("pct"),
        #         n_bas=pl.col("ba_from_allocation")
        #         .n_unique()
        #         .over("utility_id_eia", "state", "historical_year"),
        #     )
        #     .sort(
        #         "historical_year",
        #         "utility_id_eia",
        #         "state",
        #         "ba_from_allocation",
        #         "balancing_authority_code_eia",
        #     )
        #     .group_by(
        #         "historical_year",
        #         "utility_id_eia",
        #         "state",
        #         "ba_from_allocation",
        #     )
        #     .agg(pl.col("utility_name_eia", "allocated_sales").first())
        # )

        return pl.from_pandas(
            sales.to_pandas().pipe(
                add_ba_code,
                new_ba_col="ba_code",
                apply_purchaser=False,
                drop_interim=True,
                ba_rollup_only=ba_only,
            )
        )

    def cast_2022_ba_relationships(self):
        # step 1) clean up and aggregate 861
        coop_map = pd.read_parquet(ROOT_PATH / "r_data/coop_ba_code_remap.parquet")
        transformed_sales = (
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
            .filter(
                (pl.col("report_date") > datetime(2007, 1, 1))
                & pl.col("state").is_in(("HI", "AK")).not_()
            )
            .collect()
            .to_pandas()
            .merge(
                coop_map.rename(
                    columns={
                        "dist_utility_id_eia": "utility_id_eia",
                        "ba_code": "ba_code_coop_map",
                    }
                )[["utility_id_eia", "ba_code_coop_map"]],
                on=["utility_id_eia"],
                how="left",
            )
            .assign(
                balancing_authority_code_eia=lambda x: np.where(
                    x["ba_code_coop_map"].notna(),
                    x["ba_code_coop_map"],
                    x["balancing_authority_code_eia"],
                )
            )
            .query('report_date > "2007-01-01" & state != "HI" & state != "AK"')
            .assign(
                report_date=lambda x: pd.to_datetime(x["report_date"], format="%Y-%m-%d"),
                historical_year=lambda x: x["report_date"].dt.year.astype(int),
            )
            .groupby(
                [
                    "utility_id_eia",
                    "state",
                    "balancing_authority_code_eia",
                    "historical_year",
                ]
            )
            .agg({"utility_name_eia": "first", "sales_mwh": "sum"})
            .rename(
                columns={
                    "utility_name_eia": "utility_name",
                    "sales_mwh": "retail_sales_mwh",
                }
            )
            .reset_index()
        )

        # step 2) get unique reported bas at utility state level
        # for the last year in which that utility reported to 861
        # max n bas is 8

        # step A: split up all unique bas for each utility/state/year combination

        split_bas = (
            transformed_sales.groupby(["utility_id_eia", "state", "historical_year"])[
                "balancing_authority_code_eia"
            ]
            .unique()
            .reset_index()
            .assign(
                clean_ba=lambda x: x["balancing_authority_code_eia"]
                .astype(str)
                .str.replace("[", "")
                .str.replace("]", "")
                .str.replace("'", "")
            )
            .drop_duplicates(subset=["utility_id_eia", "state"], keep="last")["clean_ba"]
            .astype(str)
            .str.split(" ", n=8, expand=True)
        )

        # step B: get utility, state, year index for concat
        unique_bas = (
            transformed_sales.groupby(["utility_id_eia", "state", "historical_year"])[
                "balancing_authority_code_eia"
            ]
            .unique()
            .reset_index()
            .drop_duplicates(subset=["utility_id_eia", "state"], keep="last")
        )

        # bas_last_report_year = (
        #     pd.concat([unique_bas, split_bas], axis=1)
        #     .rename(
        #         columns={
        #             0: "ba_code_1",
        #             1: "ba_code_2",
        #             2: "ba_code_3",
        #             3: "ba_code_4",
        #             4: "ba_code_5",
        #             5: "ba_code_6",
        #             6: "ba_code_7",
        #             7: "ba_code_8",
        #         }
        #     )
        #     .melt(
        #         id_vars=["utility_id_eia", "state", "historical_year"],
        #         value_vars=[
        #             "ba_code_1",
        #             "ba_code_2",
        #             "ba_code_3",
        #             "ba_code_4",
        #             "ba_code_5",
        #             "ba_code_6",
        #             "ba_code_7",
        #             "ba_code_8",
        #         ],
        #         value_name="balancing_authority_code_eia",
        #         var_name="n_ba_code",
        #     )
        #     .query("balancing_authority_code_eia.notnull()")
        #     .drop(columns=["n_ba_code"])
        # )

        # step C: concat to get pivoted out like columns, then melt for merge
        bas_last_report_year = (
            pd.concat([unique_bas, split_bas], axis=1)
            .rename(
                columns={
                    0: "ba_code_1",
                    1: "ba_code_2",
                    2: "ba_code_3",
                    3: "ba_code_4",
                    4: "ba_code_5",
                    5: "ba_code_6",
                    6: "ba_code_7",
                    7: "ba_code_8",
                }
            )
            .drop(columns=["balancing_authority_code_eia"])
            .melt(
                id_vars=["utility_id_eia", "state", "historical_year"],
                value_vars=[
                    "ba_code_1",
                    "ba_code_2",
                    "ba_code_3",
                    "ba_code_4",
                    "ba_code_5",
                    "ba_code_6",
                    "ba_code_7",
                    "ba_code_8",
                ],
                value_name="balancing_authority_code_eia",
                var_name="n_ba_code",
            )
            .query("balancing_authority_code_eia.notnull()")
            .drop(columns=["n_ba_code"])
        )
        # calculate ba share of each utility/state combo's total sales
        ba_sales_share_2022 = transformed_sales.assign(
            balancing_authority_code_eia=lambda x: np.where(
                x["balancing_authority_code_eia"] == "UNK",
                pd.NA,
                x["balancing_authority_code_eia"],
            ),
            ba_share_of_latest_state_sales=lambda x: x.groupby(
                [
                    "utility_id_eia",
                    "state",
                    "balancing_authority_code_eia",
                    "historical_year",
                ]
            )["retail_sales_mwh"].transform("sum")
            / x.groupby(["utility_id_eia", "state", "historical_year"])[
                "retail_sales_mwh"
            ].transform("sum"),
        ).merge(
            bas_last_report_year,
            on=[
                "utility_id_eia",
                "state",
                "historical_year",
                "balancing_authority_code_eia",
            ],
            how="right",
            indicator=True,
        )[
            # leave sales_mwh column out since we only care about latest ba's and share
            [
                "utility_id_eia",
                "state",
                "balancing_authority_code_eia",
                "ba_share_of_latest_state_sales",
            ]
        ]
        ba_only = not any(x.isalnum() for x in self.bas)

        return (
            transformed_sales[
                [
                    "utility_id_eia",
                    "state",
                    "historical_year",
                    "utility_name",
                    "retail_sales_mwh",
                ]
            ]
            .merge(ba_sales_share_2022, on=["utility_id_eia", "state"], how="left")
            .assign(
                retail_sales_mwh=lambda x: x["retail_sales_mwh"]
                * x["ba_share_of_latest_state_sales"]
            )
            .groupby(
                [
                    "utility_id_eia",
                    "state",
                    "balancing_authority_code_eia",
                    "historical_year",
                ]
            )["retail_sales_mwh"]
            .sum()
            .reset_index()
            .pipe(
                add_ba_code,
                new_ba_col="ba_code",
                apply_purchaser=False,
                drop_interim=True,
                ba_rollup_only=ba_only,
            )
            .sort_values(by=["utility_id_eia", "state", "historical_year"], ascending=True)
        )

    def compare_rmodel_to_sales(self):
        full, *_ = self.output()
        # making way to convert sales in historical years to sales in future operating years
        y = pl.from_pandas(
            full.reset_index()[["datetime", "historical_year"]].drop_duplicates()
        ).select(
            pl.col("datetime").dt.year().cast(pl.Int64).alias("operating_year"),
            "historical_year",
        )

        sales = self.sales861()
        sales_suma = (
            sales.group_by("ba_code", "historical_year")
            .agg(pl.sum("retail_sales_mwh"))
            .join(y, on="historical_year")
        )
        res_out = (
            pl.from_pandas(full.reset_index())
            .filter(pl.col("scenario") == "counterfactual")
            .with_columns(operating_year=pl.col("datetime").dt.year().cast(pl.Int64))
            .group_by("ba_code", "operating_year")
            .agg(pl.sum("redispatch_curt_adj_mwh"))
        )
        return (
            res_out.join(
                sales_suma, on=["ba_code", "operating_year"], how="full", coalesce=True
            )
            .sort("ba_code")
            .select(
                "ba_code",
                pl.col("ba_code")
                .replace_strict(INTERCON_MAP, default=None)
                .alias("interconnect"),
                "operating_year",
                "redispatch_curt_adj_mwh",
                "retail_sales_mwh",
                (pl.col("redispatch_curt_adj_mwh") / pl.col("retail_sales_mwh") - 1).alias(
                    "pct_diff"
                ),
            )
        )

    def backup_utility_name_entity(self):
        return (
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
            .filter(pl.col("data_maturity") > "final")
            .select("utility_id_eia", "utility_name_eia", "report_date", "entity_type")
            .unique()
            .sort("utility_id_eia", "report_date")
            .group_by("utility_id_eia")
            .agg(pl.col("utility_name_eia", "entity_type").last())
            .with_columns(
                pl.col("entity_type").replace_strict(
                    {
                        "Municipal": "M",
                        "Behind the Meter": "Q",
                        "Retail Power Marketer": "Q",
                        "Wholesale Power Marketer": "Q",
                        "Unregulated": "Unregulated",
                        "Facility": "Facility",
                        "Political Subdivision": "P",
                        "Federal": "F",
                        "Community Choice Aggregator": "M",
                        "Power Marketer": "Q",
                        "Investor Owned": "I",
                        "State": "S",
                        "Cooperative": "C",
                    },
                    default=None,
                )
            )
            # .collect()
        )

    def get_util_ids(self, name_pattern: str) -> tuple[int]:
        """Get a list of utility_id_eia based on name_pattern.

        Args:
            name_pattern: pattern to match against utility_name_eia
                in core_eia861__yearly_sales

        Returns: tuple of utility_id_eia

        """
        all = (  # noqa: A001
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
            .select("utility_id_eia", "utility_name_eia")
            .lazy()
            .filter(pl.col("utility_name_eia").str.contains(name_pattern))
            .unique(subset="utility_id_eia")
            .sort("utility_name_eia")
            .collect()
        )
        if all.is_empty():
            raise ValueError(f"Could not find any utilities matching {name_pattern}.")
        print(all)
        return tuple(all["utility_id_eia"])

    def by_lse(self, *, least_cost: bool = True) -> pl.DataFrame:
        """Get results by LSE.

        Get results by LSE by calculating transactions based on EIA 861

        Args:
            least_cost: if True get results for least cost scenario,
                otherwise max earnings

        Returns:

        """  # noqa: D414
        disc_col = "Disc"
        full, *_ = self.output()

        # add patio BA codes to sales, this might be the first step down a
        # bad path...
        # sales = self.sales861()
        warnings.warn(
            "Now include purchase/sales but still work needed to fully get results by LSE.",
            RuntimeWarning,
            stacklevel=2,
        )
        sales = pl.from_pandas(self.cast_2022_ba_relationships()).filter(
            pl.col("ba_code") != "NBSO"
        )
        if "NYIS" not in full.index.get_level_values("ba_code"):
            sales = sales.filter(pl.col("ba_code") != "NYIS")

        econ_results = (
            pl.from_pandas(self.econ_results())
            .filter(
                pl.col("is_irp_year")
                & (pl.col("least_cost") if least_cost else pl.col("least_cost").not_())
                & pl.col("historical_actuals").not_()
            )
            .with_columns(
                ba_code=pl.col("ba_code").cast(pl.Utf8),
                portfolio=pl.when(pl.col("selected"))
                .then(pl.lit("selected"))
                .otherwise(pl.lit("counterfactual")),
            )
            .rename({"owner_utility_id_eia": "utility_id_eia"})
        )

        # Test that MWh match between econ and resource results in counterfactual
        econ_suma = (
            econ_results.filter(pl.col("portfolio") == "counterfactual")
            .group_by("ba_code", "sensitivity", pl.col("operating_year").alias("year"))
            .agg(pl.sum("MWh").alias("econ_mwh"))
            .to_pandas()
        )
        res_suma = (
            full.reset_index()
            .query("scenario == 'counterfactual'")
            .assign(year=lambda x: x.datetime.dt.year)
            .groupby(["ba_code", "year"], as_index=False)[["redispatch_curt_adj_mwh"]]
            .sum()
        )
        comp = econ_suma.merge(
            res_suma, on=["ba_code", "year"], how="inner", validate="m:1"
        ).assign(
            compare=lambda x: np.isclose(x.econ_mwh, x.redispatch_curt_adj_mwh, rtol=1e-4),
            compare2=lambda x: x.econ_mwh / x.redispatch_curt_adj_mwh - 1,
        )
        bad = comp[~np.isclose(comp.econ_mwh, comp.redispatch_curt_adj_mwh, rtol=1e-4)]
        assert bad.empty, (
            "BA MWh do not match for counterfactual scenarios between econ and resource results"
        )

        # extend historical / operating year map beyond 2039
        y = pl.from_pandas(
            full.reset_index()[["datetime", "historical_year"]].drop_duplicates()
        ).select(
            pl.col("datetime").dt.year().cast(pl.Int64).alias("operating_year"),
            "historical_year",
        )
        addl_years = zip(
            range(2040, econ_results["operating_year"].max() + 1),
            9 * y.filter(pl.col("operating_year") >= 2036)["historical_year"].to_list(),
            strict=False,
        )
        hist_year_map = pl.concat(
            [y, pl.DataFrame(addl_years, orient="row", schema=y.columns)]
        ).lazy()

        # checking BA changes

        # sales_ = sales.group_by("utility_id_eia", "ba_code")

        sales_ = (
            sales.lazy()
            # roll up sales to BA/util level
            .group_by("ba_code", "utility_id_eia", "historical_year", maintain_order=True)
            .agg(pl.sum("retail_sales_mwh"))
            .sort("ba_code", "utility_id_eia", "historical_year")
            # map on future operating year based on assigned historical year, inner
            # join downselects for us
            .join(hist_year_map, on="historical_year", how="inner")
            .filter(pl.col("operating_year").is_in(econ_results["operating_year"]))
            # regroup by operating year and add utility_type
            .group_by("ba_code", "operating_year", "utility_id_eia", maintain_order=True)
            .agg(pl.first("retail_sales_mwh"))
            .with_columns(utility_type=pl.lit("LSE"))
        )
        util_info = self.utility_type()

        merged = (
            # aggregate econ results to uitlity / year level and calculate LCOE
            econ_results.lazy()
            .group_by("portfolio", "ba_code", "utility_id_eia", "operating_year")
            .agg(
                pl.col("MWh", "MWh_Disc", "Costs_Disc", "Emissions").sum(),
                pl.first(disc_col),
            )
            .with_columns(
                lcoe=pl.when(pl.col("MWh_Disc").fill_null(0.0) > 0)
                .then(pl.col("Costs_Disc") / pl.col("MWh_Disc"))
                .otherwise(pl.lit(0.0)),
                emission_intensity=pl.when(pl.col("MWh_Disc").fill_null(0.0) > 0)
                .then(pl.col("Emissions") / pl.col("MWh_Disc"))
                .otherwise(pl.lit(0.0)),
            )
            # # add on historical year
            # .join(
            #     hist_year_map.select("operating_year", "historical_year"),
            #     on="operating_year",
            #     how="left",
            # )
            # join on 861 retail sales
            .join(
                sales_,
                on=["operating_year", "utility_id_eia", "ba_code"],
                how="full",
                coalesce=True,
            )
            .with_columns(
                pl.col("retail_sales_mwh").fill_null(0.0),
                pl.col("MWh").fill_null(0.0),
                pl.col("utility_type").fill_null("GEN"),
            )
            # bring util types and names
            .join(util_info, how="left", on="utility_id_eia", validate="m:1")
            .select(
                "portfolio",
                "ba_code",
                "utility_id_eia",
                "utility_name_eia",
                "utility_type",
                "operating_year",
                "MWh",
                "Emissions",
                "retail_sales_mwh",
                pl.col("lcoe").fill_null(0.0),
                pl.col("emission_intensity").fill_null(0.0).fill_nan(0.0),
                pl.when(pl.col("retail_sales_mwh") > pl.col("MWh"))
                .then(pl.col("retail_sales_mwh") - pl.col("MWh"))
                .otherwise(pl.lit(0.0))
                .alias("purchase_mwh"),
                pl.when(pl.col("retail_sales_mwh") <= pl.col("MWh"))
                .then(pl.col("MWh") - pl.col("retail_sales_mwh"))
                .otherwise(pl.lit(0.0))
                .alias("sfr_mwh"),
                pl.col("Costs_Disc").fill_null(0.0),
                # "sales_revenue",
                # "historical_year",
                pl.col(disc_col)
                .fill_null(
                    pl.col(disc_col).mean().over("ba_code", "operating_year", "entity_type")
                )
                .fill_null(pl.col(disc_col).mean().over("ba_code", "operating_year")),
            )
        )
        # for LSE only utilities need to create identical selected and counterfactual versions
        merged = (
            pl.concat(
                [
                    merged.filter(pl.col("portfolio").is_not_null()),
                    merged.filter(pl.col("portfolio").is_null()).with_columns(
                        portfolio=pl.lit("counterfactual")
                    ),
                    merged.filter(pl.col("portfolio").is_null()).with_columns(
                        portfolio=pl.lit("selected")
                    ),
                ]
            )
            .sort("portfolio", "ba_code", "operating_year", "utility_id_eia")
            .with_columns(
                ba_sfr_total_mwh=pl.col("sfr_mwh")
                .sum()
                .over("portfolio", "ba_code", "operating_year"),
                ba_purchase_total_mwh=pl.col("purchase_mwh")
                .sum()
                .over("portfolio", "ba_code", "operating_year"),
            )
            .with_columns(
                ba_weighted_price=(
                    pl.col("lcoe") * pl.col("sfr_mwh") / pl.col("ba_sfr_total_mwh")
                )
                .sum()
                .over("portfolio", "ba_code", "operating_year")
                .fill_nan(0.0),
                ba_weighted_intensity=(
                    pl.col("emission_intensity")
                    * pl.col("sfr_mwh")
                    / pl.col("ba_sfr_total_mwh")
                )
                .sum()
                .over("portfolio", "ba_code", "operating_year")
                .fill_nan(0.0),
            )
            .filter(
                (pl.col("MWh").sum().over("ba_code") > 0.0)
                | (pl.col("retail_sales_mwh").sum().over("ba_code") > 0.0)
            )
        )

        ba_level = (
            merged.group_by("portfolio", "ba_code", "operating_year")
            .agg(
                pl.col("MWh", "retail_sales_mwh").sum(),
                pl.first("ba_weighted_price"),
                pl.first("ba_weighted_intensity"),
            )
            .with_columns(
                pl.col("ba_code")
                .replace_strict(INTERCON_MAP | {"ERCO": "western"}, default=None)
                .alias("interconnect"),
                pl.when(pl.col("retail_sales_mwh") > pl.col("MWh"))
                .then(pl.col("retail_sales_mwh") - pl.col("MWh"))
                .otherwise(pl.lit(0.0))
                .alias("purchase_mwh"),
                pl.when(pl.col("retail_sales_mwh") <= pl.col("MWh"))
                .then(pl.col("MWh") - pl.col("retail_sales_mwh"))
                .otherwise(pl.lit(0.0))
                .alias("sfr_mwh"),
            )
            .with_columns(
                inter_sfr_total_mwh=pl.col("sfr_mwh")
                .sum()
                .over("portfolio", "interconnect", "operating_year"),
                # inter_purchase_total_mwh=pl.col("purchase_mwh")
                # .sum()
                # .over("portfolio", "interconnect", "operating_year"),
            )
            .with_columns(
                inter_weighted_price=(
                    pl.col("ba_weighted_price")
                    * pl.col("sfr_mwh")
                    / pl.col("inter_sfr_total_mwh")
                )
                .sum()
                .over("portfolio", "interconnect", "operating_year")
                .fill_nan(0.0),
                inter_weighted_intensity=(
                    pl.col("ba_weighted_intensity")
                    * pl.col("sfr_mwh")
                    / pl.col("inter_sfr_total_mwh")
                )
                .sum()
                .over("portfolio", "interconnect", "operating_year")
                .fill_nan(0.0),
            )
            .sort("portfolio", "interconnect", "ba_code", "operating_year")
        )

        issues = (
            ba_level.group_by("portfolio", "interconnect", "operating_year")
            .agg(pl.col("MWh", "retail_sales_mwh").sum())
            .with_columns(pct_diff=pl.col("MWh") / pl.col("retail_sales_mwh") - 1)
            .filter(pl.col("pct_diff").abs() > 0.05)
            .sort("portfolio", "interconnect", "operating_year")
            .collect()
            .pivot(
                values="pct_diff",
                index=["portfolio", "operating_year"],
                on="interconnect",
            )
            .to_pandas()
            .to_string()
            .replace("\n", "\n\t")
        )
        LOGGER.warning("Interconnect mismatches \n %s", issues)

        out = (
            merged.join(
                ba_level.select(
                    "portfolio",
                    "ba_code",
                    "operating_year",
                    "inter_weighted_price",
                    "inter_weighted_intensity",
                ).unique(),
                on=["portfolio", "ba_code", "operating_year"],
                how="left",
            )
            .with_columns(
                pct_purchased_from_ba=pl.when(
                    pl.col("ba_sfr_total_mwh") >= pl.col("ba_purchase_total_mwh")
                )
                .then(pl.lit(1.0))
                .otherwise(pl.col("ba_sfr_total_mwh") / pl.col("ba_purchase_total_mwh"))
            )
            .select(
                "portfolio",
                "ba_code",
                "utility_id_eia",
                "utility_name_eia",
                pl.col("utility_id_eia").alias("Utility_ID"),
                pl.when(pl.col("portfolio") == "selected")
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("selected"),
                pl.when(pl.col("portfolio") == "counterfactual")
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("counterfactual_baseline"),
                pl.lit(False).alias("historical_actuals"),
                pl.lit(True).alias("is_irp_year"),
                pl.lit(econ_results["sensitivity"].unique()[0]).alias("sensitivity"),
                pl.col("portfolio").alias("scenario"),
                "operating_year",
                pl.when(pl.col("purchase_mwh") > 0)
                .then(
                    pl.col("purchase_mwh")
                    * (
                        pl.col("pct_purchased_from_ba") * pl.col("ba_weighted_price")
                        + (1 - pl.col("pct_purchased_from_ba"))
                        * pl.col("inter_weighted_price")
                    )
                )
                .when(pl.col("sfr_mwh") > 0)
                .then(-pl.col("sfr_mwh") * pl.col("lcoe"))
                .otherwise(pl.lit(0.0))
                .alias("Costs"),
                pl.when(pl.col("purchase_mwh") > 0)
                .then(
                    pl.col("purchase_mwh")
                    * (
                        pl.col("pct_purchased_from_ba") * pl.col("ba_weighted_intensity")
                        + (1 - pl.col("pct_purchased_from_ba"))
                        * pl.col("inter_weighted_intensity")
                    )
                )
                .when(pl.col("sfr_mwh") > 0)
                .then(-pl.col("sfr_mwh") * pl.col("emission_intensity"))
                .otherwise(pl.lit(0.0))
                .alias("Emissions"),
                (pl.col("purchase_mwh") - pl.col("sfr_mwh")).alias("MWh"),
                pl.when(pl.col("purchase_mwh") > 0)
                .then(pl.lit("Purchases"))
                .when(pl.col("sfr_mwh") > 0)
                .then(pl.lit("Sales for Resale"))
                .otherwise(pl.lit("drop"))
                .alias("technology_description"),
                disc_col,
            )
            .with_columns(
                Costs_Disc=pl.col("Costs") * pl.col(disc_col),
                MWh_Disc=pl.col("MWh") * pl.col(disc_col),
                category=pl.lit("transaction"),
            )
            .filter(pl.col("technology_description") != "drop")
            .join(
                econ_results.lazy()
                .select(
                    pl.col("sensitivity").cast(pl.Utf8),
                    "refi_trigger",
                    "savings_tol",
                    "earnings_thresh",
                    "least_cost",
                    "Debt_Repl_EIR",
                    "Equity_Repl_EIR",
                )
                .unique(),
                on="sensitivity",
                how="left",
                validate="m:1",
            )
            .join(
                econ_results.filter(pl.col("utility_id_eia") == pl.col("Utility_ID"))
                .lazy()
                .select(
                    "utility_id_eia",
                    "parent_name",
                    "parent_ticker",
                    "owner_utility_name_eia",
                    "entity_type",
                )
                .unique(),
                on="utility_id_eia",
                how="left",
                validate="m:1",
            )
            .join(
                econ_results.lazy().select("ba_code", "region").unique(),
                on="ba_code",
                how="left",
                validate="m:1",
            )
            # load-only LSEs don't have entity_types from econ_results
            .join(
                util_info.select("utility_id_eia", "entity_type"),
                on="utility_id_eia",
                validate="m:1",
                how="left",
            )
            .with_columns(pl.col("entity_type").fill_null(pl.col("entity_type_right")))
            .drop("entity_type_right")
        )

        # sellers = (
        #     merged.filter(pl.col("net_purchases") < 0.0)
        #     .with_columns(
        #         ba_sales=pl.col("net_purchases")
        #         .sum()
        #         .over("portfolio", "ba_code", "operating_year"),
        #     )
        #     .with_columns(
        #         weighted_cost_contribution=pl.col("lcoe")
        #         * pl.col("net_purchases")
        #         / pl.col("ba_sales")
        #     )
        #     .with_columns(
        #         weighted_cost=pl.col("weighted_cost_contribution")
        #         .sum()
        #         .over("portfolio", "ba_code", "operating_year")
        #     )
        # )
        #
        # buyers = merged.filter(pl.col("net_purchases") >= 0.0)

        warnings.warn(
            "not all columns fully implemented in purchase/sales block",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            pl.concat([econ_results.lazy(), out], how="diagonal_relaxed")
            .with_columns(
                pl.when(pl.col("utility_id_eia").is_in(sales["utility_id_eia"]))
                .then(pl.lit("LSE"))
                .otherwise(pl.lit("GEN"))
                .alias("utility_type"),
            )
            .sort(
                "portfolio",
                "ba_code",
                "utility_id_eia",
                "operating_year",
                "technology_description",
            )
            .collect()
        )  # , sellers

    def utility_type(self):
        util_info = (
            pl_scan_pudl("out_eia__yearly_utilities", self.pudl_release)
            .filter(pl.col("data_maturity") == "final")
            .select("utility_id_eia", "utility_name_eia", "report_date", "entity_type")
            .sort("utility_id_eia", "report_date")
            .group_by("utility_id_eia")
            .agg(pl.last("utility_name_eia"), pl.last("entity_type"))
            .join(
                self.backup_utility_name_entity(),
                on="utility_id_eia",
                validate="1:1",
                how="full",
                coalesce=True,
            )
            .filter(pl.col("utility_id_eia").is_not_null())
            .select(
                "utility_id_eia",
                pl.col("utility_name_eia_right")
                .fill_null(pl.col("utility_name_eia"))
                .alias("utility_name_eia"),
                pl.col("entity_type_right")
                .fill_null(pl.col("entity_type"))
                .alias("entity_type"),
            )
        )
        return util_info

    def by_lse_fig(self, util_ids: str | int | tuple[int] | None = None):
        """Bar chart showing MW, MWh, and discounted cost for selected and BAU
        portfolios for selected utilities

        Args:
            util_ids: can be a single utility_id_eia, a sequence of utility_id_eia,
                a pattern matched on utility_name_eia from core_eia861__yearly_sales,
                the default is None and results in a figure that aggregates all
                utilities together, including gen-only ones


        """
        if util_ids is None:
            blse = self.by_lse()
        elif isinstance(util_ids, int):
            blse = self.by_lse().filter(pl.col("utility_id_eia") == util_ids)
        elif isinstance(util_ids, str):
            blse = self.by_lse().filter(
                pl.col("utility_id_eia").is_in(self.get_util_ids(util_ids))
            )
        else:
            blse = self.by_lse().filter(pl.col("utility_id_eia").is_in(util_ids))
        df = (
            blse.with_columns(
                category=pl.col("category").replace({"old_clean": "existing_xpatio"}),
                technology_description=pl.col("technology_description").replace(
                    PLOT_MAP
                    | {
                        "NaturalGas_FE": "Other Fossil",
                        "Hydroelectric Pumped Storage": "Hydro",
                        "Solar Thermal with Energy Storage": "Solar",
                        "Solar Thermal without Energy Storage": "Solar",
                        "Other Natural Gas": "Other Fossil",
                        "Flywheels": "Storage",
                    }
                ),
                portfolio=pl.col("portfolio").replace({"counterfactual": "bau"}),
            )
            .sort("operating_year", descending=False)
            .group_by(
                "portfolio",
                "category",
                "technology_description",
                "plant_id_eia",
                "generator_id",
            )
            .agg(pl.last("MW"), pl.col("MWh", "Costs_Disc").sum())
            .group_by(
                "portfolio",
                "category",
                "technology_description",
                # "generator_id",
            )
            .agg(pl.sum("MW"), pl.col("MWh", "Costs_Disc").sum())
            .melt(
                id_vars=[
                    "portfolio",
                    "category",
                    "technology_description",
                    # "operating_year"
                ],
                value_vars=["MW", "MWh", "Costs_Disc"],
            )
            .sort(
                "category",
                "technology_description",
            )
        )
        return px.bar(
            df.filter(
                (
                    (pl.col("technology_description") == "Transmission")
                    & (pl.col("variable") == "MW")
                ).not_()
            ).sort("portfolio"),
            x="portfolio",
            y="value",
            color="technology_description",
            facet_col="variable",
            facet_col_spacing=0.075,
            color_discrete_map=COLOR_MAP
            | {
                "Purchases": "#a7a9ac",
                "Sales for Resale": "#a7a9ac",
                "Transmission": "#58585b",
            },
            template="plotly",
        ).update_yaxes(
            matches=None,
            showticklabels=True,
        )

    def cr_site_list(self, least_cost=True):
        er = (
            self.econ_results().query("least_cost")
            if least_cost
            else self.econ_results().query("~least_cost")
        )
        by = [
            "ba_code",
            "owner_utility_id_eia",
            "re_plant_id",
            "re_generator_id",
            "plant_id_eia",
            "generator_id",
        ]
        er = (
            er.query("category == 'patio_clean' & is_irp_year")
            .astype({"re_plant_id": int})
            .sort_values([*by, "operating_year"], ascending=True)
            .groupby(by, observed=True, as_index=False)
            .agg(
                dict.fromkeys(
                    (
                        "MW",
                        "technology_description",
                        "owner_utility_name_eia",
                        "parent_name",
                        "parent_ticker",
                        "balancing_authority_code_eia",
                        "incumbent_technology",
                        "region",
                        "scenario",
                        "state",
                        "sensitivity",
                    ),
                    "last",
                )
                | dict.fromkeys(
                    (
                        "Capex",
                        "Costs",
                        "Capex_Costs",
                        "No_Policy_Capex_Costs",
                        "Fuel_Costs",
                        "FOM",
                        "VOM",
                        "Startup_Costs",
                        "Opex",
                        "MWh",
                        "MWh_no_curt",
                        "Earnings",
                        "Emissions",
                        "Emissions_Reduced",
                        "Disc",
                        "Disc_E",
                        "Costs_Disc",
                        "Capex_Costs_Disc",
                        "No_Policy_Capex_Costs_Disc",
                        "Opex_Disc",
                        "MWh_Disc",
                        "Earnings_Disc",
                    ),
                    "sum",
                )
            )
        )

        specs = []
        with DataZip(self.path, "r") as z0:
            for ba in self.good_objs:
                specs.append(z0[ba, "re_plant_specs"])
        re_specs = pd.concat(specs)[
            [
                "plant_id_eia",
                "fos_id",
                "fos_gen",
                "latitude",
                "longitude",
                "latitude_nrel_site",
                "longitude_nrel_site",
                "distance",
                "fos_cap",
                "total_var_mwh",
                "sc_gid",
                "capacity_factor",
                "global_horizontal_irradiance",
                "capacity_mw_nrel_site",
                "area_sq_km",
                "distance_to_transmission_km",
                "class_atb",
                "capex_atb",
                "cf_atb",
                "fom_atb",
                "plant_id_prof_site",
                "latitude_prof_site",
                "longitude_prof_site",
                "cf_prof_site",
                "cf_ilr_prof_site",
                "distance_prof_site",
                "wind_speed_120meters",
                "wind_speed_150meters",
                "landfall_distance_to_transmission_km",
                "bathymetry_meters",
                "distance_to_coast_km",
                "re_site_id",
                "energy_community",
                "lcoe",
                "ilr",
                "area_per_mw",
                "total_norm_match",
                "re_site_gen",
                "re_cost",
                "avoided_cost",
                "re_max_obj_coef",
                "energy_max",
                "cap_max",
                "savings_max",
                "capacity_max_re_scen",
            ]
        ].rename(
            columns={
                "plant_id_eia": "re_plant_id",
                "fos_id": "plant_id_eia",
                "fos_gen": "generator_id",
            }
        )
        return er.merge(
            re_specs,
            on=["re_plant_id", "plant_id_eia", "generator_id"],
            validate="1:1",
            how="left",
        )

    def gen_list(self, cr_only=True, sort_cols=False, include_tx=False):  # noqa: D417
        """Create a list of plant-techs and the CR opportunity at them.

        Args:
            cr_only: defualt is True, if False include all plants, not just the ones with CR
            sort_cols: clean up the column names and order, only tested with cr_only=False

        """

        def fix_ids(df):
            to_add = []
            for a, b in [("6452", "7801"), ("10005", "22500"), ("10171", "11249")]:
                date = df[df.owner_utility_id_eia == f"{a} and {b}"]["next_irp_date"].item()
                to_add.append(
                    pd.DataFrame(
                        {"owner_utility_id_eia": [a, b], "next_irp_date": [date, date]}
                    )
                )
                df = df[df.owner_utility_id_eia != f"{a} and {b}"]

            return pd.concat([df, *to_add]).astype({"owner_utility_id_eia": int})

        coals = ("ANT", "BIT", "LIG", "SGC", "SUB", "WC", "RC", "SC")
        c2 = (
            pl_scan_pudl("core_eia860__scd_generators", self.pudl_release)
            .filter(
                (pl.col("operational_status") != "proposed")
                & (
                    pl.col("technology_description").str.contains("Coal")
                    | pl.col("energy_source_code_1").is_in(coals)
                    | pl.col("energy_source_code_2").is_in(coals)
                    | pl.col("energy_source_code_3").is_in(coals)
                    | pl.col("energy_source_code_4").is_in(coals)
                    | pl.col("energy_source_code_5").is_in(coals)
                    | pl.col("energy_source_code_6").is_in(coals)
                )
            )
            .select("plant_id_eia")
            .unique()
            .collect()
            .to_series()
            .to_pandas()
        )

        suf = [
            ", Inc.",
            " Inc.",
            ", Inc",
            " Inc",
            ", LLC.",
            " LLC.",
            ", L.L.C.",
            " L.L.C.",
            ", LLC",
            " LLC",
            ", Corp.",
            " Corp.",
            ", Corp",
            " Corp",
            ", LP.",
            " LP.",
            ", L.P.",
            " L.P.",
            ", LP",
            " LP",
            "Membershipration",
        ]
        er_sensitivity = self.econ_results().query("is_irp_year")
        base = (
            er_sensitivity.query("selected")
            .assign(
                objective=lambda x: x.least_cost.replace(
                    {True: "least_cost", False: "max_earnings"}
                )
            )
            .astype({"technology_description": str})
            .copy()
        )
        # repowered = base.query("category == 'patio_clean'").plant_id_eia.unique()
        all_techs = self.ad.gens[
            ["plant_id_eia", "generator_id", "technology_description"]
        ].astype({"technology_description": str})
        capacity = (
            self.ad.gens.assign(
                clean_repowering_type=lambda x: np.where(
                    x.replacement
                    & x.cr_eligible
                    & (x.retirement_date.dt.year.fillna(2100) <= 2035),
                    "replacement",
                    np.where(
                        x.surplus
                        & x.cr_eligible
                        & (x.retirement_date.dt.year.fillna(2100) > 2035),
                        "surplus",
                        "no_cr",
                    ),
                )
            )
            .astype({"technology_description": str})
            .groupby(
                ["plant_id_eia", "generator_id", "technology_description"],
                as_index=False,
            )
            .agg(
                {
                    "state": "first",
                    "capacity_mw": "sum",
                    "clean_repowering_type": lambda x: ", ".join(
                        sorted(set(a for a in x if a != "no_cr"))  # noqa: C401
                    ),
                }
            )
        )
        parent = (
            base.query("category == 'existing_fossil'")
            .groupby(
                [
                    "region",
                    "plant_id_eia",
                    "generator_id",
                    "technology_description",
                    "parent_name",
                    "owner_utility_name_eia",
                    "owner_utility_id_eia",
                ],
                observed=True,
                as_index=False,
            )
            .MW.sum()
            .sort_values(
                ["region", "plant_id_eia", "technology_description", "MW"],
                ascending=[True, True, True, False],
            )
            .groupby(
                ["region", "plant_id_eia", "generator_id", "technology_description"],
                as_index=False,
                observed=True,
            )[["parent_name", "owner_utility_name_eia", "owner_utility_id_eia"]]
            .first()
            .assign(
                parent_name=lambda x: x.parent_name.str.replace("|".join(suf), "").str.replace(
                    "Generation & Transmission Association", "G&T"
                ),
                owner_utility_name_eia=lambda x: x.owner_utility_name_eia.str.replace(
                    "|".join(suf), ""
                ).str.replace("Generation & Transmission Association", "G&T"),
            )
            .astype({"owner_utility_id_eia": int})
        )

        cr_agg = {
            "Costs_Disc": "sum",
            "Emissions": "sum",
            "MWh": "sum",
        }
        re_rename = {
            "Onshore Wind Turbine": "Wind",
            "Offshore Wind Turbine": "Wind",
            "Solar Photovoltaic": "Solar",
        }

        base_with_tech = base.query(
            "category in ('patio_clean', 'existing_fossil', 'refinancing', 'transmission_lines')"
        ).merge(
            all_techs,
            on=["plant_id_eia", "generator_id"],
            how="left",
            validate="m:1",
            suffixes=("_o", None),
        )

        clean_rep = (
            base_with_tech.groupby(
                [
                    "objective",
                    "region",
                    "plant_id_eia",
                    "generator_id",
                    "technology_description",
                    "operating_year",
                ],
                as_index=False,
                observed=True,
            )
            .agg(cr_agg | {"Disc": "mean"})
            .groupby(["objective", "region", "plant_id_eia", "technology_description"])
            .agg(cr_agg | {"Disc": "sum"})
            .join(  # join on costs by category (fossil incumbent vs CR)
                base_with_tech.replace(
                    {
                        "category": {
                            "existing_fossil": "Costs_Disc_Fossil",
                            "refinancing": "Costs_Disc_Fossil",
                            "patio_clean": "Costs_Disc_CR",
                            "transmission_lines": "Costs_Disc_CR",
                        }
                    }
                ).pivot_table(
                    index=[
                        "objective",
                        "region",
                        "plant_id_eia",
                        "generator_id",
                        "technology_description",
                    ],
                    columns="category",
                    values="Costs_Disc",
                    aggfunc="sum",
                    observed=True,
                )
            )
            .join(  # bring in CR capactiy
                base.query("category in ('patio_clean',)")
                .assign(clean_tech=lambda x: x.technology_description.replace(re_rename))
                .merge(
                    all_techs,
                    on=["plant_id_eia", "generator_id"],
                    how="left",
                    validate="m:1",
                    suffixes=("_o", None),
                )
                .groupby(
                    [
                        "objective",
                        "region",
                        "plant_id_eia",
                        "generator_id",
                        "technology_description",
                        "clean_tech",
                        "operating_year",
                    ],
                    as_index=False,
                    observed=True,
                )
                .MW.sum()
                .pivot_table(
                    index=[
                        "objective",
                        "region",
                        "plant_id_eia",
                        "generator_id",
                        "technology_description",
                    ],
                    columns="clean_tech",
                    values="MW",
                    aggfunc="last",
                    observed=True,
                ),
                how="right" if cr_only else "left",
            )
            .assign(
                total_clean=lambda x: x[
                    [c for c in ("Solar", "Wind", "Batteries") if c in x]
                ].sum(axis=1)
            )
            .reset_index()
        )

        clean_rep = clean_rep.pivot(
            index=[
                "region",
                "plant_id_eia",
                "generator_id",
                "technology_description",
            ],
            columns="objective",
            values=[
                c
                for c in (
                    "Costs_Disc",
                    "Costs_Disc_CR",
                    "Costs_Disc_Fossil",
                    "Emissions",
                    "MWh",
                    "Disc",
                    "Batteries",
                    "Solar",
                    "Wind",
                    "total_clean",
                )
                if c in clean_rep
            ],
        )
        # clean_rep.columns=list(map("_".join, clean_rep.columns))
        # clean_rep = clean_rep.set_axis(list(map("_".join, clean_rep.columns)), axis=1).reset_index(level="Disc")
        with rmi_cloud_fs().open(
            "az://patio-data/20241031/Copy of IRP Service 37_EQ Research_RMI_8.31.23.xlsx"
        ) as f:
            irp_data = (
                pd.read_excel(
                    f,
                    sheet_name="IRP Data",
                    header=1,
                    na_values=["np.nan"],
                    usecols=["EIA ID", "Anticipated Date for Next IRP"],
                )
                .rename(
                    columns={
                        "EIA ID": "owner_utility_id_eia",
                        "Anticipated Date for Next IRP": "next_irp_date",
                    }
                )
                .dropna(subset="owner_utility_id_eia")
                .pipe(fix_ids)
            )

        cr = (
            clean_rep.set_axis(list(map("_".join, clean_rep.columns)), axis=1)
            .assign(Disc=lambda x: x.Disc_max_earnings.fillna(x.Disc_least_cost))
            .drop(columns=["Disc_max_earnings", "Disc_least_cost"])
            .sort_index()
            .join(
                er_sensitivity.query(
                    "least_cost & counterfactual_baseline & ~selected & category in ('existing_fossil',)"
                )
                .rename(
                    columns={
                        "Costs_Disc": "Costs_Disc_cfl",
                        "Emissions": "Emissions_cfl",
                        "MWh": "MWh_cfl",
                    }
                )
                .groupby(
                    [
                        "region",
                        "plant_id_eia",
                        "generator_id",
                        "technology_description",
                    ],
                    observed=True,
                )
                .agg(
                    {
                        "Costs_Disc_cfl": "sum",
                        "Emissions_cfl": "sum",
                        "MWh_cfl": "sum",
                    }
                ),
                how="left",
                validate="1:1",
            )
            .sort_values(["region", "total_clean_max_earnings"], ascending=[True, False])
            .assign(
                rank_max_earnings=lambda x: x.groupby(
                    ["region"], sort=False
                ).total_clean_max_earnings.transform("rank", method="dense", ascending=False)
            )
            .sort_values(["region", "total_clean_least_cost"], ascending=[True, False])
            .assign(
                rank_least_cost=lambda x: x.groupby(
                    ["region"], sort=False
                ).total_clean_least_cost.transform("rank", method="dense", ascending=False)
            )
            .reset_index()
            .merge(
                parent,
                on=["region", "plant_id_eia", "generator_id", "technology_description"],
                how="left",
                validate="1:1",
            )
            .merge(
                capacity,
                on=["plant_id_eia", "generator_id", "technology_description"],
                how="left",
                validate="m:1",
            )
            .merge(
                self.ad.gens[["plant_id_eia", "plant_name_eia"]].drop_duplicates(),
                on="plant_id_eia",
                how="left",
                validate="m:1",
            )
            .replace({"technology_description": PLOT_MAP})
            .assign(
                clean_repowering_type=lambda x: x.clean_repowering_type.mask(
                    x.total_clean_max_earnings + x.total_clean_least_cost == 0, np.nan
                ),
                annualized_npv_least_cost=lambda x: x.Costs_Disc_least_cost / x.Disc,
                annualized_fossil_npv_least_cost=lambda x: x.Costs_Disc_Fossil_least_cost
                / x.Disc,
                annualized_cr_npv_least_cost=lambda x: x.Costs_Disc_CR_least_cost / x.Disc,
                annualized_npv_max_earnings=lambda x: x.Costs_Disc_max_earnings / x.Disc,
                annualized_fossil_npv_max_earnings=lambda x: x.Costs_Disc_Fossil_max_earnings
                / x.Disc,
                annualized_cr_npv_max_earnings=lambda x: x.Costs_Disc_CR_max_earnings / x.Disc,
                annualized_npv_cfl=lambda x: x.Costs_Disc_cfl / x.Disc,
                utilization_least_cost=lambda x: x.MWh_least_cost
                / (x.capacity_mw * 8766 * 30),
                utilization_max_earnings=lambda x: x.MWh_max_earnings
                / (x.capacity_mw * 8766 * 30),
                utilization_cfl=lambda x: x.MWh_cfl / (x.capacity_mw * 8766 * 30),
                annual_emissions_least_cost=lambda x: x.Emissions_least_cost / 30,
                annual_emissions_max_earnings=lambda x: x.Emissions_max_earnings / 30,
                annual_emissions_cfl=lambda x: x.Emissions_cfl / 30,
            )
            .merge(
                irp_data,
                on="owner_utility_id_eia",
                how="left",
            )
            .sort_values("total_clean_max_earnings", ascending=False)
            .assign(
                coal_site=lambda x: x.plant_id_eia.isin(c2),
            )
        )

        assert np.all(
            np.isclose(
                cr[
                    [
                        "annualized_fossil_npv_max_earnings",
                        "annualized_cr_npv_max_earnings",
                    ]
                ].sum(axis=1),
                cr.annualized_npv_max_earnings,
            )
            | cr.annualized_npv_max_earnings.isna()
        ), "Max Earnings NPVs don't add up"
        assert np.all(
            np.isclose(
                cr[["annualized_fossil_npv_least_cost", "annualized_cr_npv_least_cost"]].sum(
                    axis=1
                ),
                cr.annualized_npv_least_cost,
            )
            | cr.annualized_npv_least_cost.isna()
        ), "Least Cost NPVs don't add up"

        if not sort_cols:
            return cr
        prm = [
            "plant_name_eia",
            "generator_id",
            "technology_description",
            "capacity_mw",
            "owner_utility_name_eia",
            "parent_name",
            "state",
            "region",
            "clean_repowering_type",
            "plant_id_eia",
            "owner_utility_id_eia",
            "total_clean_least_cost",
            "total_clean_max_earnings",
            "Batteries_least_cost",
            "Batteries_max_earnings",
            "Solar_least_cost",
            "Solar_max_earnings",
            "Wind_least_cost",
            "Wind_max_earnings",
            "MWh_cfl",
            "MWh_least_cost",
            "MWh_max_earnings",
            "Costs_Disc_cfl",
            "Costs_Disc_CR_least_cost",
            "Costs_Disc_CR_max_earnings",
            "Costs_Disc_Fossil_least_cost",
            "Costs_Disc_Fossil_max_earnings",
            "Emissions_cfl",
            "Emissions_least_cost",
            "Emissions_max_earnings",
            "annualized_fossil_npv_least_cost",
            "annualized_cr_npv_least_cost",
            "annualized_fossil_npv_max_earnings",
            "annualized_cr_npv_max_earnings",
            "annualized_npv_cfl",
            "utilization_least_cost",
            "utilization_max_earnings",
            "utilization_cfl",
            "annual_emissions_least_cost",
            "annual_emissions_max_earnings",
            "annual_emissions_cfl",
            "next_irp_date",
            "coal_site",
        ]
        return cr[prm].rename(columns={c: c.replace("_cfl", "_bau") for c in prm})

    def plant_list(self, cr_only=True, sort_cols=False, include_tx=False):  # noqa: D417
        """Create a list of plant-techs and the CR opportunity at them.

        Args:
            cr_only: defualt is True, if False include all plants, not just the ones with CR
            sort_cols: clean up the column names and order, only tested with cr_only=False

        """

        def fix_ids(df):
            to_add = []
            for a, b in [("6452", "7801"), ("10005", "22500"), ("10171", "11249")]:
                date = df[df.owner_utility_id_eia == f"{a} and {b}"]["next_irp_date"].item()
                to_add.append(
                    pd.DataFrame(
                        {"owner_utility_id_eia": [a, b], "next_irp_date": [date, date]}
                    )
                )
                df = df[df.owner_utility_id_eia != f"{a} and {b}"]

            return pd.concat([df, *to_add]).astype({"owner_utility_id_eia": int})

        coals = ("ANT", "BIT", "LIG", "SGC", "SUB", "WC", "RC", "SC")
        c2 = (
            pl_scan_pudl("core_eia860__scd_generators", self.pudl_release)
            .filter(
                (pl.col("operational_status") != "proposed")
                & (
                    pl.col("technology_description").str.contains("Coal")
                    | pl.col("energy_source_code_1").is_in(coals)
                    | pl.col("energy_source_code_2").is_in(coals)
                    | pl.col("energy_source_code_3").is_in(coals)
                    | pl.col("energy_source_code_4").is_in(coals)
                    | pl.col("energy_source_code_5").is_in(coals)
                    | pl.col("energy_source_code_6").is_in(coals)
                )
            )
            .select("plant_id_eia")
            .unique()
            .collect()
            .to_series()
            .to_pandas()
        )

        suf = [
            ", Inc.",
            " Inc.",
            ", Inc",
            " Inc",
            ", LLC.",
            " LLC.",
            ", L.L.C.",
            " L.L.C.",
            ", LLC",
            " LLC",
            ", Corp.",
            " Corp.",
            ", Corp",
            " Corp",
            ", LP.",
            " LP.",
            ", L.P.",
            " L.P.",
            ", LP",
            " LP",
            "Membershipration",
        ]
        er_sensitivity = self.econ_results().query("is_irp_year")
        base = (
            er_sensitivity.query("selected")
            .assign(
                objective=lambda x: x.least_cost.replace(
                    {True: "least_cost", False: "max_earnings"}
                )
            )
            .copy()
        )
        # repowered = base.query("category == 'patio_clean'").plant_id_eia.unique()
        all_techs = self.ad.gens[["plant_id_eia", "generator_id", "technology_description"]]
        capacity = (
            self.ad.gens.assign(
                clean_repowering_type=lambda x: np.where(
                    x.replacement
                    & x.cr_eligible
                    & (x.retirement_date.dt.year.fillna(2100) <= 2035),
                    "replacement",
                    np.where(
                        x.surplus
                        & x.cr_eligible
                        & (x.retirement_date.dt.year.fillna(2100) > 2035),
                        "surplus",
                        "no_cr",
                    ),
                )
            )
            .groupby(["plant_id_eia", "technology_description"], as_index=False)
            .agg(
                {
                    "state": "first",
                    "capacity_mw": "sum",
                    "clean_repowering_type": lambda x: ", ".join(
                        sorted(set(a for a in x if a != "no_cr"))  # noqa: C401
                    ),
                }
            )
        )
        parent = (
            base.query("category == 'existing_fossil'")
            .groupby(
                [
                    "region",
                    "plant_id_eia",
                    "technology_description",
                    "parent_name",
                    "owner_utility_name_eia",
                    "owner_utility_id_eia",
                ],
                as_index=False,
            )
            .MW.sum()
            .sort_values(
                ["region", "plant_id_eia", "technology_description", "MW"],
                ascending=[True, True, True, False],
            )
            .groupby(["region", "plant_id_eia", "technology_description"], as_index=False)[
                ["parent_name", "owner_utility_name_eia", "owner_utility_id_eia"]
            ]
            .first()
            .assign(
                parent_name=lambda x: x.parent_name.str.replace("|".join(suf), "").str.replace(
                    "Generation & Transmission Association", "G&T"
                ),
                owner_utility_name_eia=lambda x: x.owner_utility_name_eia.str.replace(
                    "|".join(suf), ""
                ).str.replace("Generation & Transmission Association", "G&T"),
            )
            .astype({"owner_utility_id_eia": int})
        )

        cr_agg = {
            "Costs_Disc": "sum",
            "Emissions": "sum",
            "MWh": "sum",
        }
        re_rename = {
            "Onshore Wind Turbine": "Wind",
            "Offshore Wind Turbine": "Wind",
            "Solar Photovoltaic": "Solar",
        }
        if include_tx:
            base_ = base.query(
                "category in ('patio_clean', 'existing_fossil', 'refinancing', 'transmission_lines')"
            )
        else:
            base_ = base.query("category in ('patio_clean', 'existing_fossil', 'refinancing')")

        base_with_tech = base_.merge(
            all_techs,
            on=["plant_id_eia", "generator_id"],
            how="left",
            validate="m:1",
            suffixes=("_o", None),
        )

        clean_rep = (
            base_with_tech.groupby(
                [
                    "objective",
                    "region",
                    "plant_id_eia",
                    "technology_description",
                    "operating_year",
                ],
                as_index=False,
            )
            .agg(cr_agg | {"Disc": "mean"})
            .groupby(["objective", "region", "plant_id_eia", "technology_description"])
            .agg(cr_agg | {"Disc": "sum"})
            .join(  # join on costs by category (fossil incumbent vs CR)
                base_with_tech.replace(
                    {
                        "category": {
                            "existing_fossil": "Costs_Disc_Fossil",
                            "refinancing": "Costs_Disc_Fossil",
                            "patio_clean": "Costs_Disc_CR",
                            "transmission_lines": "Costs_Disc_CR",
                        }
                    }
                ).pivot_table(
                    index=[
                        "objective",
                        "region",
                        "plant_id_eia",
                        "technology_description",
                    ],
                    columns="category",
                    values="Costs_Disc",
                    aggfunc="sum",
                )
            )
            .join(  # join on mwh by category (fossil incumbent vs CR)
                base_with_tech.replace(
                    {
                        "category": {
                            "existing_fossil": "MWh_Fossil",
                            "refinancing": "MWh_Fossil",
                            "patio_clean": "MWh_CR",
                            "transmission_lines": "MWh_CR",
                        }
                    }
                ).pivot_table(
                    index=[
                        "objective",
                        "region",
                        "plant_id_eia",
                        "technology_description",
                    ],
                    columns="category",
                    values="MWh",
                    aggfunc="sum",
                )
            )
            .join(  # bring in CR capactiy
                base.query("category in ('patio_clean',)")
                .assign(clean_tech=lambda x: x.technology_description.replace(re_rename))
                .merge(
                    all_techs,
                    on=["plant_id_eia", "generator_id"],
                    how="left",
                    validate="m:1",
                    suffixes=("_o", None),
                )
                .groupby(
                    [
                        "objective",
                        "region",
                        "plant_id_eia",
                        "technology_description",
                        "clean_tech",
                        "operating_year",
                    ],
                    as_index=False,
                )
                .MW.sum()
                .pivot_table(
                    index=[
                        "objective",
                        "region",
                        "plant_id_eia",
                        "technology_description",
                    ],
                    columns="clean_tech",
                    values="MW",
                    aggfunc="last",
                ),
                how="right" if cr_only else "left",
            )
            .assign(total_clean=lambda x: x[["Solar", "Wind", "Batteries"]].sum(axis=1))
            .reset_index()
            .pivot(
                index=["region", "plant_id_eia", "technology_description"],
                columns="objective",
                values=[
                    "Costs_Disc",
                    "Costs_Disc_CR",
                    "Costs_Disc_Fossil",
                    "Emissions",
                    "MWh",
                    "MWh_Fossil",
                    "MWh_CR",
                    "Disc",
                    "Batteries",
                    "Solar",
                    "Wind",
                    "total_clean",
                ],
            )
        )
        # clean_rep.columns=list(map("_".join, clean_rep.columns))
        # clean_rep = clean_rep.set_axis(list(map("_".join, clean_rep.columns)), axis=1).reset_index(level="Disc")

        with rmi_cloud_fs().open(
            "az://patio-data/20241031/Copy of IRP Service 37_EQ Research_RMI_8.31.23.xlsx"
        ) as f:
            irp_data = (
                pd.read_excel(
                    f,
                    sheet_name="IRP Data",
                    header=1,
                    na_values=["np.nan"],
                    usecols=["EIA ID", "Anticipated Date for Next IRP"],
                )
                .rename(
                    columns={
                        "EIA ID": "owner_utility_id_eia",
                        "Anticipated Date for Next IRP": "next_irp_date",
                    }
                )
                .dropna(subset="owner_utility_id_eia")
                .pipe(fix_ids)
            )

        cr = (
            clean_rep.set_axis(list(map("_".join, clean_rep.columns)), axis=1)
            .assign(Disc=lambda x: x.Disc_max_earnings.fillna(x.Disc_least_cost))
            .drop(columns=["Disc_max_earnings", "Disc_least_cost"])
            .sort_index()
            .join(
                er_sensitivity.query(
                    "least_cost & counterfactual_baseline & ~selected & category in ('existing_fossil',)"
                )
                .rename(
                    columns={
                        "Costs_Disc": "Costs_Disc_cfl",
                        "Emissions": "Emissions_cfl",
                        "MWh": "MWh_cfl",
                    }
                )
                .groupby(["region", "plant_id_eia", "technology_description"])
                .agg(
                    {
                        "Costs_Disc_cfl": "sum",
                        "Emissions_cfl": "sum",
                        "MWh_cfl": "sum",
                    }
                ),
                how="left",
                validate="1:1",
            )
            .sort_values(["region", "total_clean_max_earnings"], ascending=[True, False])
            .assign(
                rank_max_earnings=lambda x: x.groupby(
                    ["region"], sort=False
                ).total_clean_max_earnings.transform("rank", method="dense", ascending=False)
            )
            .sort_values(["region", "total_clean_least_cost"], ascending=[True, False])
            .assign(
                rank_least_cost=lambda x: x.groupby(
                    ["region"], sort=False
                ).total_clean_least_cost.transform("rank", method="dense", ascending=False)
            )
            .reset_index()
            .merge(
                parent,
                on=["region", "plant_id_eia", "technology_description"],
                how="left",
                validate="1:1",
            )
            .merge(
                capacity,
                on=["plant_id_eia", "technology_description"],
                how="left",
                validate="m:1",
            )
            .merge(
                self.ad.gens.sort_values("plant_name_eia", key=lambda x: x.str.len())
                .groupby("plant_id_eia", as_index=False)
                .plant_name_eia.first(),
                on="plant_id_eia",
                how="left",
                validate="m:1",
            )
            .replace({"technology_description": PLOT_MAP})
            .assign(
                clean_repowering_type=lambda x: x.clean_repowering_type.mask(
                    x.total_clean_max_earnings + x.total_clean_least_cost == 0, np.nan
                ),
                annualized_npv_least_cost=lambda x: x.Costs_Disc_least_cost / x.Disc,
                annualized_fossil_npv_least_cost=lambda x: x.Costs_Disc_Fossil_least_cost
                / x.Disc,
                annualized_cr_npv_least_cost=lambda x: x.Costs_Disc_CR_least_cost / x.Disc,
                annualized_npv_max_earnings=lambda x: x.Costs_Disc_max_earnings / x.Disc,
                annualized_fossil_npv_max_earnings=lambda x: x.Costs_Disc_Fossil_max_earnings
                / x.Disc,
                annualized_cr_npv_max_earnings=lambda x: x.Costs_Disc_CR_max_earnings / x.Disc,
                annualized_npv_cfl=lambda x: x.Costs_Disc_cfl / x.Disc,
                utilization_least_cost=lambda x: x.MWh_least_cost
                / (x.capacity_mw * 8766 * 30),
                utilization_max_earnings=lambda x: x.MWh_max_earnings
                / (x.capacity_mw * 8766 * 30),
                utilization_cfl=lambda x: x.MWh_cfl / (x.capacity_mw * 8766 * 30),
                annual_emissions_least_cost=lambda x: x.Emissions_least_cost / 30,
                annual_emissions_max_earnings=lambda x: x.Emissions_max_earnings / 30,
                annual_emissions_cfl=lambda x: x.Emissions_cfl / 30,
            )
            .merge(
                irp_data,
                on="owner_utility_id_eia",
                how="left",
            )
            .sort_values("total_clean_max_earnings", ascending=False)
            .assign(
                coal_site=lambda x: x.plant_id_eia.isin(c2),
            )
        )

        assert np.all(
            np.isclose(
                cr[
                    [
                        "annualized_fossil_npv_max_earnings",
                        "annualized_cr_npv_max_earnings",
                    ]
                ].sum(axis=1),
                cr.annualized_npv_max_earnings,
            )
            | cr.annualized_npv_max_earnings.isna()
        ), "Max Earnings NPVs don't add up"
        assert np.all(
            np.isclose(
                cr[["annualized_fossil_npv_least_cost", "annualized_cr_npv_least_cost"]].sum(
                    axis=1
                ),
                cr.annualized_npv_least_cost,
            )
            | cr.annualized_npv_least_cost.isna()
        ), "Least Cost NPVs don't add up"

        if not sort_cols:
            return cr
        prm = [
            "plant_name_eia",
            "technology_description",
            "capacity_mw",
            "owner_utility_name_eia",
            "parent_name",
            "state",
            "region",
            "clean_repowering_type",
            "plant_id_eia",
            "owner_utility_id_eia",
            "total_clean_least_cost",
            "total_clean_max_earnings",
            "Batteries_least_cost",
            "Batteries_max_earnings",
            "Solar_least_cost",
            "Solar_max_earnings",
            "Wind_least_cost",
            "Wind_max_earnings",
            "MWh_cfl",
            "MWh_least_cost",
            "MWh_max_earnings",
            "MWh_CR_least_cost",
            "MWh_CR_max_earnings",
            "MWh_Fossil_least_cost",
            "MWh_Fossil_max_earnings",
            "Costs_Disc_cfl",
            "Costs_Disc_CR_least_cost",
            "Costs_Disc_CR_max_earnings",
            "Costs_Disc_Fossil_least_cost",
            "Costs_Disc_Fossil_max_earnings",
            "Emissions_cfl",
            "Emissions_least_cost",
            "Emissions_max_earnings",
            "annualized_fossil_npv_least_cost",
            "annualized_cr_npv_least_cost",
            "annualized_fossil_npv_max_earnings",
            "annualized_cr_npv_max_earnings",
            "annualized_npv_cfl",
            "utilization_least_cost",
            "utilization_max_earnings",
            "utilization_cfl",
            "annual_emissions_least_cost",
            "annual_emissions_max_earnings",
            "annual_emissions_cfl",
            "next_irp_date",
            "coal_site",
        ]
        return cr[prm].rename(columns={c: c.replace("_cfl", "_bau") for c in prm})

    def cost_waterfall(
        self, source_data: pd.DataFrame | pl.DataFrame, least_cost, by="region"
    ):
        """Args:
            source_data: econ_results or by_lse
            least_cost: only required for econ_results
            by: column to group by e.g. region, utility_id_eia

        Returns:

        """  # noqa: D414
        cost_cat_map = pd.DataFrame(
            [
                ("existing_clean", "Capex_Costs", "Other Capex"),
                ("existing_fossil", "Capex_Costs", "Other Capex"),
                ("existing_other", "Capex_Costs", "Other Capex"),
                ("existing_xpatio", "Capex_Costs", "Other Capex"),
                ("old_clean", "Capex_Costs", "Other Capex"),
                ("proposed_clean", "Capex_Costs", "Other Capex"),
                ("proposed_fossil", "Capex_Costs", "Other Capex"),
                ("proposed_transmission_upgrades", "Capex_Costs", "Other Capex"),
                ("patio_clean", "Capex_Costs", "Clean repowering Capex"),
                ("refinancing", "Capex_Costs", "EIR"),
                ("transmission_lines", "Capex_Costs", "Other Capex"),
                ("existing_fossil", "Opex", "Fossil Opex"),
                ("proposed_clean", "Opex", "Other Opex"),
                ("proposed_fossil", "Opex", "Fossil Opex"),
                ("proposed_transmission_upgrades", "Opex", "Other Opex"),
                ("patio_clean", "Opex", "Clean repowering Opex"),
                ("transmission_lines", "Opex", "Other Opex"),
                ("existing_other", "Opex", "Other Opex"),
            ],
            columns=["category", "cost_cat", "new_cat"],
        )
        if isinstance(source_data, pd.DataFrame):
            source_data = source_data
        elif isinstance(source_data, pl.DataFrame):
            source_data = source_data.to_pandas()
        source_data = (
            source_data.query("least_cost") if least_cost else source_data.query("~least_cost")
        )
        for_wtfl = (
            (
                source_data.query(
                    "is_irp_year & category in ('patio_clean', 'refinancing', 'proposed_fossil', 'existing_fossil')"
                )
                .assign(
                    selected_=lambda x: np.where(x.selected, "selected", "cfl"),
                    #                               _cat=lambda: x.category.map(
                    #    {'patio_clean': 'patio_clean',
                    # 'transmission_lines': 'transmission_lines',
                    # 'proposed_fossil': 'proposed_fossil',
                    # 'proposed_clean': 'proposed_clean',
                    # 'proposed_transmission_upgrades': 'proposed_transmission_upgrades',
                    # 'existing_fossil': 'existing_fossil',
                    # 'existing_xpatio': 'existing_xpatio',
                    # 'old_clean': 'old_clean',
                    # 'existing_other': 'existing_other',
                    # 'existing_clean': 'existing_clean',
                    # 'refinancing': 'EIR'}
                )
                .groupby([by, "selected_", "category"], as_index=False)[
                    ["Capex_Costs_Disc", "Opex_Disc"]
                ]
                .sum()
            )
            .melt(
                id_vars=[
                    by,
                    "selected_",
                    "category",
                ],
                value_vars=["Capex_Costs_Disc", "Opex_Disc"],
                var_name="cost_cat",
            )
            .replace({"cost_cat": {"Capex_Costs_Disc": "Capex_Costs", "Opex_Disc": "Opex"}})
            .merge(cost_cat_map, on=["category", "cost_cat"], how="left")
            .groupby([by, "selected_", "new_cat"])
            .agg(
                {
                    "cost_cat": "first",
                    "value": "sum",
                }
            )
        )
        return for_wtfl

    def transactions_drive_cr_cost_increase(self):
        """Cases where the change in transaction revenue > clean repowering cost increase."""
        blse = self.by_lse()
        return (
            blse.filter(
                pl.col("utility_id_eia").is_in(
                    blse.pivot(
                        on="portfolio",
                        index="utility_id_eia",
                        values="Costs_Disc",
                        aggregate_function="sum",
                    ).filter(
                        (pl.col("counterfactual") < pl.col("selected"))
                        & (pl.col("counterfactual") / pl.col("selected"))
                        .is_between(1 - 1e-6, 1 + 1e-6)
                        .not_()
                    )["utility_id_eia"]
                )
                & (pl.col("utility_type") == "LSE")
            )
            .with_columns(
                col=pl.concat_str(
                    pl.col("portfolio"),
                    pl.when(pl.col("category") == "transaction")
                    .then(pl.lit("transaction"))
                    .otherwise(pl.lit("owned")),
                    separator="_",
                ),
            )
            .pivot(
                on=["col"],
                index=["utility_id_eia", "utility_type"],
                values=["Costs_Disc"],
                aggregate_function="sum",
            )
            .with_columns(
                counterfactual=pl.col("counterfactual_owned").fill_null(0.0)
                + pl.col("counterfactual_transaction"),
                selected=pl.col("selected_owned").fill_null(0.0)
                + pl.col("selected_transaction"),
            )
            .join(
                blse.filter(pl.col("utility_name_eia").is_not_null())
                .select("utility_id_eia", "utility_name_eia")
                .unique(),
                on="utility_id_eia",
                how="left",
                validate="m:1",
            )
            .filter(
                (pl.col("selected") - pl.col("counterfactual"))
                < (pl.col("selected_transaction") - pl.col("counterfactual_transaction"))
            )
            .select(
                "utility_id_eia",
                "utility_name_eia",
                "counterfactual_transaction",
                "selected_transaction",
                "counterfactual",
                "selected",
                "counterfactual_owned",
                "selected_owned",
            )
            .sort("counterfactual", descending=True)
            .join(
                pl_scan_pudl("core_eia861__yearly_operational_data_misc", self.pudl_release)
                .filter(
                    (pl.col("data_maturity") == "final")
                    & (pl.col("report_date") == "2022-01-01")
                )
                .select(
                    "utility_id_eia",
                    (pl.col("sales_for_resale_mwh") / pl.col("total_disposition_mwh"))
                    .fill_null(0.0)
                    .alias("hist_resale_share"),
                )
                .collect(),
                on="utility_id_eia",
                how="left",
            )
        )

    def brownfields(self, radius=10) -> pl.LazyFrame:
        # import geopandas as gpd

        # acres = (
        #     gpd.read_file(path, layer="ACRES")[
        #         [
        #             "REGISTRY_ID",
        #             "PRIMARY_NAME",
        #             "LOCATION_ADDRESS",
        #             "CITY_NAME",
        #             "COUNTY_NAME",
        #             "FIPS_CODE",
        #             "STATE_CODE",
        #             "POSTAL_CODE",
        #             "LATITUDE83",
        #             "LONGITUDE83",
        #             "HUC8_CODE",
        #             "ACCURACY_VALUE",
        #             "COLLECT_MTH_DESC",
        #             "REF_POINT_DESC",
        #             "CREATE_DATE",
        #             "UPDATE_DATE",
        #             "LAST_REPORTED_DATE",
        #             "FAC_URL",
        #             "PGM_SYS_ID",
        #             "PGM_SYS_ACRNM",
        #             "INTEREST_TYPE",
        #             "PROGRAM_URL",
        #             "PGM_REPORT_URL",
        #             "PUBLIC_IND",
        #             "ACTIVE_STATUS",
        #             "FEDERAL_AGENCY_NAME",
        #             "HUC_12",
        #             "FEDERAL_LAND_IND",
        #             "FED_FACILITY_CODE",
        #             "EPA_REGION_CODE",
        #             "KEY_FIELD",
        #         ]
        #     ]
        #     .rename(columns={"LATITUDE83": "latitude", "LONGITUDE83": "longitude"})
        #     .pipe(simplify_columns)
        # )
        # https://www.epa.gov/cleanups/cimc-how-download-data
        repowered = (  # noqa: F841
            pl.from_pandas(self.econ_results().reset_index())
            .filter(
                pl.col("selected")
                & pl.col("sensitivity").cast(pl.Utf8).str.contains("least_cost")
                & (pl.col("category") == "patio_clean")
                & (pl.col("datetime").dt.year() == 2035)
                & (pl.col("technology_description") != "Batteries")
            )
            .select(
                pl.col("plant_id_eia").cast(pl.Int64),
            )
            .unique()
            .to_series()
            .to_list()
        )
        with rmi_cloud_fs().open(
            "az://patio-data/20241031/CIMC_Brownfields_Download.csv"
        ) as f:
            bfields = pd.read_csv(f, low_memory=False).pipe(simplify_columns)

        return (
            pl.from_pandas(bfields)
            .lazy()
            .with_columns(
                needs_cleanup=(
                    pl.col("indicate_whether_cleanup_is_necessary")
                    .fill_null("Y")
                    .replace("U", "Y")
                    == "Y"
                )
                & (
                    pl.col(
                        "indicate_whether_cleanup_treatment_technology_ies_were_implemented"
                    ).fill_null("N")
                    == "N"
                ),
                available=(
                    pl.sum_horizontal(
                        "redevelopment_land_use_residential",
                        "redevelopment_land_use_greenspace",
                        "redevelopment_land_use_industrial",
                        "redevelopment_land_use_commercial",
                    ).fill_null(0.0)
                    == 0.0
                )
                & pl.col("redevelopment_start_date").is_null(),
            )
            .group_by(pl.col("property_id").cast(pl.Int64))
            .agg(
                pl.col(
                    "property_name",
                    "property_owner",
                    "street_address",
                    "zip_code",
                    "city",
                    "state",
                ).first(),
                pl.first("latitude").alias("latitude_bf"),
                pl.first("longitude").alias("longitude_bf"),
                pl.max("size_in_acres"),
                pl.col("needs_cleanup").all(),
                pl.col("available").all(),
            )
            .filter(pl.col("size_in_acres").is_not_null() & pl.col("available"))
            .sort("property_id")
            .join(
                pl.from_pandas(self.ad.gens.query("plant_id_eia in @repowered"))
                .select(
                    "plant_id_eia",
                    pl.col("final_ba_code").alias("ba_code"),
                    "latitude",
                    "longitude",
                )
                .lazy()
                .unique(),
                how="cross",
            )
            .pipe(
                pl_distance,
                lat1="latitude",
                lat2="latitude_bf",
                lon1="longitude",
                lon2="longitude_bf",
            )
            .filter(pl.col("distance") <= radius)
        )

    def unused_clean_repowering(self, detail=True):
        """Identify additional capacity / area at selected CR sites.

        Calculates how much additional RE could be installed at each fossil generator
        selected for clean repowering. Provides both installable capacity and area
         assuming that all selected clean repowering is actually built.

        """
        f, *_ = self.output()
        core = (
            pl.from_pandas(self.econ_results().reset_index().astype({"re_plant_id": "Int64"}))
            .filter(
                pl.col("selected")
                & pl.col("sensitivity").cast(pl.Utf8).str.contains("least_cost")
                & (pl.col("category") == "patio_clean")
                & (pl.col("datetime").dt.year() == 2035)
                & (pl.col("technology_description") != "Batteries")
            )
            .select(
                pl.col("re_plant_id").cast(pl.Int64).alias("plant_id_eia"),
                pl.col("plant_id_eia").cast(pl.Int64).alias("fos_id"),
                pl.col("generator_id").alias("fos_gen"),
                "technology_description",
                pl.col("MW").alias("selected_mw"),
                "incumbent_technology",
                "state",
                "owner_utility_id_eia",
                "owner_utility_name_eia",
                "parent_name",
            )
            .sort("fos_id", "fos_gen", "technology_description", "selected_mw")
        )
        selected = (
            core.group_by("plant_id_eia", "technology_description")
            .agg(
                pl.sum("selected_mw"),
                pl.col("fos_id").unique().alias("fos_ids"),
            )
            .join(
                pl.from_pandas(self._dfs["re_specs_out"])
                .select("plant_id_eia", "technology_description", "area_per_mw")
                .unique(),
                on=["plant_id_eia", "technology_description"],
            )
            .select(
                "plant_id_eia",
                "technology_description",
                "selected_mw",
                "fos_ids",
                selected_sqkm=(pl.col("selected_mw") * pl.col("area_per_mw"))
                .sum()
                .over("plant_id_eia"),
            )
        )
        free = selected.join(
            pl.from_pandas(self._dfs["re_specs_out"]).join(
                core.group_by("fos_id", "fos_gen", "incumbent_technology", "state").agg(
                    pl.col(
                        "owner_utility_id_eia", "owner_utility_name_eia", "parent_name"
                    ).unique(),
                ),
                on=["fos_id", "fos_gen"],
                how="inner",
            ),
            on=["plant_id_eia", "technology_description"],
            how="full",
            coalesce=True,
        ).with_columns(
            # sum over to account for multiple RE types at a given NREL site
            available_sqkm=pl.col("area_sq_km")
            - pl.col("selected_sqkm").fill_null(0.0).sum().over("re_site_id"),
            available_mw=(
                pl.col("area_sq_km")
                - pl.col("selected_sqkm").fill_null(0.0).sum().over("re_site_id")
            )
            / pl.col("area_per_mw"),
        )
        if detail:
            return free
        return (
            free.group_by("ba_code", "fos_id", "fos_gen", "technology_description")
            .agg(
                pl.col("fos_cap", "incumbent_technology", "state").first(),
                pl.col("selected_mw", "selected_sqkm", "available_mw", "available_sqkm").sum(),
                pl.col(
                    "owner_utility_id_eia", "owner_utility_name_eia", "parent_name"
                ).first(),
                pl.col("plant_id_eia").unique().alias("re_plant_ids"),
            )
            .sort("ba_code", "fos_id", "fos_gen", "technology_description")
        )

    def clean_repowering_by_generator(
        self, utility: int | None = None, ba: str | list[str] | None = None
    ) -> pl.DataFrame:
        if ba is None and isinstance(utility, int):
            blse = self.by_lse().filter(pl.col("utility_id_eia") == utility)
        elif ba is None and isinstance(utility, str):
            blse = self.by_lse().filter(pl.col("utility_id_eia").str.contains(utility))
        elif utility is None and isinstance(ba, str):
            blse = self.by_lse().filter(pl.col("ba_code") == ba)
        elif utility is None and isinstance(ba, list | tuple):
            blse = self.by_lse().filter(pl.col("ba_code").is_in(ba))
        else:
            raise TypeError("must provide either utility or ba")
        blse = blse.with_columns(pl.col("generator_id").cast(pl.Utf8))

        return (
            blse.filter(pl.col("category") == "existing_fossil")
            .pivot(
                values="MWh",
                index=["plant_id_eia", "generator_id", "technology_description"],
                on="portfolio",
                aggregate_function="sum",
            )
            .join(
                pl.from_pandas(self.ad.gens).select(
                    "plant_id_eia", "plant_name_eia", "generator_id"
                ),
                on=["plant_id_eia", "generator_id"],
                how="left",
            )
            .join(
                blse.filter(
                    (pl.col("category") == "patio_clean") & (pl.col("operating_year") == 2032)
                )
                .groupby("plant_id_eia", "generator_id")
                .agg(pl.sum("MW").alias("clean_repowering_mw")),
                on=["plant_id_eia", "generator_id"],
                how="left",
            )
            .select(
                "plant_id_eia",
                "generator_id",
                "plant_name_eia",
                "technology_description",
                pl.col("counterfactual").alias("counterfactual_mwh"),
                pl.col("selected").alias("selected_mwh"),
                (pl.col("counterfactual") - pl.col("selected")).alias("reduction_mwh"),
                "clean_repowering_mw",
            )
            .sort("reduction_mwh", descending=True)
        )


# def patio_key(item):
#     if isinstance(item, pd.Series):
#         return item.astype(str).str.casefold().replace(ORDERING)
#     if isinstance(item, pd.Index):
#         return pd.Index([ORDERING.get(x.casefold(), str(x)) for x in item])
#     return ORDERING.get(item.casefold(), str(item))

ORDERING = {
    "nuclear": "000",
    "coal": "001",
    "coal ccs": "0015",
    "gas cc": "002",
    "gas ct": "004",
    "gas rice": "005",
    "gas st": "003",
    "other fossil": "006",
    "other": "006",
    "hydro": "007",
    "biomass": "008",
    "solar": "011",
    "onshore wind": "009",
    "offshore wind": "010",
    "storage": "012",
    "curtailment": "013",
    "january": "101",
    "february": "102",
    "march": "103",
    "april": "104",
    "may": "105",
    "june": "106",
    "july": "107",
    "august": "108",
    "september": "109",
    "october": "110",
    "november": "111",
    "december": "112",
    "historical": "200",
    "redispatch": "201",
    "selected": "202",
    "CR": "202",
    "counterfactual": "203",
    "Planned": "203",
    "2021": "204",
    "Current": "204",
}


def hrs_per_year(df):
    if "datetime" in df:
        return np.where(
            df.datetime.dt.is_leap_year,
            8784,
            8760,
        )
    return np.where(
        df.index.get_level_values("datetime").is_leap_year,
        8784,
        8760,
    )


if __name__ == "__main__":
    """
    scalene patio/model/ba_level.py --profile-only 'model' --profile-all
    """
    self = BAs(
        # name="Coops_" + datetime.now().strftime("%Y%m%d%H%M"),
        solar_ilr=1.3,
        data_kwargs={"re_by_plant": True, "include_figs": False},
        bas=[
            "PJM",
            # "552",
            # "554",
            # "556",
            # "560",
            # "562",
            # "569",
            # "58",
            # "AEC",
            # "AECI",
            # "CAISO",  # Arizona Electric Pwr Coop Inc?
            # "DUKE",
            # "ERCO",
            # "MISO",
            # "PAC",
            # "PJM",
            # "SEC",
            # "SOCO",
            # "SWPP",
            # "WALC",  # Arizona Electric Pwr Coop Inc?
        ],
        scenario_configs=[
            ScenarioConfig(0.7, 0, 0.0, 4, 0),
            ScenarioConfig(0.8, 0, 0.0, 4, 0),
            # ScenarioConfig(0.9, 0, 0.0, 4, 0),
            # ScenarioConfig(1.0, 0, 0.0, 4, 0),
        ],
    )
    self.run_all()
    # self.select_utils(
    #     util_ids=(
    #         39347,
    #         16624,
    #         12658,
    #         4716,
    #         1692,
    #         17632,
    #         1307,
    #         18315,
    #         40230,
    #         30151,
    #         7570,
    #         9267,
    #         5580,
    #         924,
    #         21554,
    #         7353,
    #         20447,
    #         796,
    #         20910,
    #         189,
    #         25422,
    #         17583,
    #         13683,
    #         7349,
    #         7004,
    #         17568,
    #         2172,
    #         13994,
    #         40229,
    #         3522,
    #         13670,
    #         807,
    #         40211,
    #         3258,
    #         4363,
    #         19558,
    #         40307,
    #         11824,
    #         19389,
    #     ),
    #     export="xl",
    # )
