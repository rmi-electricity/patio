import gc
import json
import logging
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
from dispatch import DispatchModel, zero_profiles_outside_operating_dates
from dispatch.constants import COLOR_MAP, PLOT_MAP
from etoolbox.datazip import DataZip
from pandas.core.util.hashing import hash_pandas_object
from scipy.optimize import linprog
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from patio.constants import (
    BAD_COLO_BAS,
    CCS_FACTORS,
    CLEAN_TD_MAP,
    ES_TECHS,
    MTDF,
    OTHER_TD_MAP,
    RE_TECH,
    ROOT_PATH,
    SIMPLE_TD_MAP,
    TD_MAP,
    TECH_CODES,
)
from patio.data.profile_data import get_714profile
from patio.exceptions import ScenarioError
from patio.helpers import df_query, generate_projection_from_historical_pl, make_core_lhs_rhs
from patio.model.base import (
    ScenarioConfig,
    calc_redispatch_cost,
    fuel_auc,
    optimize_equal_energy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger("patio")
MIN_HIST_FOSSIL = 0.05


class BAScenario:
    def __init__(self, ba=None, config=ScenarioConfig(1, 0, 0.25, 0, 0)):  # noqa: B008
        self.cfl = False
        self.ba = ba
        self.config = config
        self.is_max_scen = False
        self.no_limit_prime = self.ba.no_limit_prime
        self.to_replace = self.ba.baseload_replace.get(self.config.nuclear_scen, [])
        self.msg = []
        if self.config.excl_or_moth == "exclude":
            self.exclude = self.ba.exclude + self.ba.exclude_or_mothball
            self.mothball = []
        else:
            self.exclude = self.ba.exclude
            self.mothball = self.ba.exclude_or_mothball
        temp = self.ba.ccs_convert.get(abs(self.config.ccs_scen), [])
        self.ccs = [
            x
            for x in self.ba.plant_data.index.intersection(temp)
            if x[0] not in self.to_replace
        ]
        if temp and len(self.ccs) != len(temp):
            LOGGER.debug(
                "%s %s, dropping generators from CCS because the plant will be replaced "
                "by nuclear or the generator was excluded because of missing data: %s",
                self.ba.ba_code,
                self.config,
                set(temp) - set(self.ccs),
            )

        # max_re = (
        #     np.inf
        #     if ba.max_re is None
        #     else np.array(
        #         [ba.max_re.get(x, 0) for x in self.ba.mini_re_prof.columns], dtype=float
        #     )
        # )

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        # if ba.max_re is not None:
        #     re_cap = optimize(
        #         req_prof=self.ba.adj_load.astype(float),
        #         re_profs=self.ba.mini_re_prof.to_numpy(),
        #         pct_to_replace=self.config.re_share_of_total,
        #         max_re_cap=max_re,
        #     )
        # else:
        if (p_scen := self.ba.same_energy_scenario(self)) is not None:
            LOGGER.info(
                "%s %s using RE capacities from %s",
                self.ba.ba_code,
                self.config.for_idx,
                p_scen.config.for_idx,
            )
            self.msg.append("using previous scenario capacities")
            self.re_plant_specs = p_scen.re_plant_specs.copy()
            self.re_cap = p_scen.re_cap.copy()
            self._data = p_scen._data.copy()
            if p_scen.is_max_scen:
                self.patio_chunks = self.chunk_re(
                    self.re_plant_specs.reset_index(), self.ba.parent_re_scenario(self)
                )
            else:
                self.patio_chunks = p_scen.patio_chunks.copy()
        else:
            (
                re_plant_specs,
                self._data,
                self.re_cap,
                self.patio_chunks,
            ) = self.select_re_site_capacity()
            self.re_plant_specs = self.adj_re_specs(re_plant_specs).set_index(
                ["plant_id_eia", "generator_id"]
            )
        self.es_specs = self.allocate_es()
        self.dm: tuple[DispatchModel, ...] = self.make_dm()
        _, _, dmax, dcount = (
            self.dm[0].system_level_summary(freq="YS").filter(like="deficit").max()
        )
        self.good_scenario = dmax <= 0.15 and dcount <= 87
        self._outputs = {}
        self.colo_hourly = pl.LazyFrame()
        self.colo_coef_ts = pl.LazyFrame()
        # self.colo_coef_ts_iter = pl.LazyFrame()
        self.colo_coef_mw = pl.LazyFrame()
        self.colo_summary = pl.DataFrame()

    def select_re_site_capacity(self):
        en_max = self.ba.re_plant_specs.set_index(
            ["icx_genid", "re_site_id", "re_type"]
        ).energy_max
        target_en = self.ba.net_load_prof.sum() * self.config.re_share_of_total
        if target_en > self.ba.max_energy:
            target_en = self.ba.max_energy * 0.90
            LOGGER.warning(
                "%s %s Insufficient renewables to replace target energy will only replace %.1f pct",
                self.ba.ba_code,
                self.config.for_idx,
                100 * target_en / self.ba.net_load_prof.sum(),
            )

            # new_target = np.round(self.ba.max_energy / self.ba.net_load_prof.sum(), 3)
            # LOGGER.warning(
            #     "%s %s target energy is %s x what can be generated by allowed "
            #     "renewables -> this scenario will only replace %s of fossil energy",
            #     self.ba.ba_code,
            #     self.config.for_idx,
            #     np.round(target_en / self.ba.max_energy, 3),
            #     new_target,
            # )
            # self.msg.append(
            #     f"Insufficient renewables to replace {self.config.re_energy:.1%}, will "
            #     f"only replace {new_target:.1%}"
            # )
            # return (
            #     self.ba.re_plant_specs.merge(
            #         en_max.reset_index(name="capacity_mw"),
            #         on=["icx_genid", "re_site_id", "re_type"],
            #         validate="1:1",
            #     ).query("capacity_mw > 0"),
            #     pd.concat({"energy max": en_max.groupby(level=2).sum()}, axis=1),
            #     en_max.groupby("re_type").sum().to_dict(),
            # )
        if (par := self.ba.parent_re_scenario(self)) is not None:
            lb = (
                self.ba.re_plant_specs[["combi_id"]]
                .merge(
                    par.re_plant_specs[["combi_id", "capacity_mw"]],
                    on="combi_id",
                    how="left",
                )
                .fillna(0.0)
                .capacity_mw.to_numpy()
            )
        else:
            lb = np.zeros_like(self.ba.re_plant_specs.capacity_mw_nrel_site)

        energy_Ab = (
            (
                self.ba.re_plant_specs.groupby("combi_id").cf_atb.first()
                * len(self.ba.net_load_prof)
            )
            .to_frame("energy")
            .T.fillna(0.0)
            .join(pd.DataFrame({"rhs": target_en}, index=["energy"]))
        )
        success, re_size0 = self.ba.milp(
            Ab_eq=energy_Ab,
            lb=lb,
            desc=f"{self.config.for_idx} initial econ",
            wind_lb={"reference": 80.0, "limited": 25.0}[self.ba._metadata["regime"]],
        )
        if success is False:
            msg = (
                "initial economic selection with min wind size failed, running "
                "without min wind size constraint"
            )
            self.msg.append(f"{msg}: {re_size0}")
            LOGGER.warning("%s %s %s", self.ba.ba_code, self.config.for_idx, msg)
            success, re_size0 = self.ba.lp(Ab_eq=energy_Ab)
        if success is False:
            self.msg.append(f"Initial economic selection failed: {re_size0}")
            raise ScenarioError(
                f"{self.ba.ba_code} {self.config.for_idx} initial economic selection "
                f"failed. {re_size0}"
            )
        if len(self.ba.re_plant_specs.re_type.unique()) == 1:
            # if there is only one type of RE, then we can skip the profile matching
            # and cost re-optimization
            self.msg.append(
                "only one type of renewable so skipping profile matching and cost re-optimization"
            )
            re_out = self.ba.re_plant_specs.merge(
                re_size0.reset_index(name="capacity_mw"),
                on=["icx_genid", "re_site_id", "re_type"],
                validate="1:1",
            ).query("capacity_mw > 0")
            return (
                re_out,
                pd.concat(
                    {
                        "energy max": en_max.groupby(level=2).sum(),
                        "pre match": re_size0.groupby(level=2).sum(),
                    },
                    axis=1,
                ),
                re_size0.groupby("re_type").sum().to_dict(),
                self.chunk_re(re_out, par),
            )

        cap_max, lo, re_cap, re_size1, up = self.prof_match_and_final_econ(
            energy_Ab=energy_Ab, re_size0=re_size0, lb=lb
        )
        re_out = self.ba.re_plant_specs.merge(
            re_size1.reset_index(name="capacity_mw"),
            on=["icx_genid", "re_site_id", "re_type"],
            validate="1:1",
        ).query("capacity_mw > 0")

        re_chunks = self.chunk_re(re_out, par)
        assert np.isclose(re_out.capacity_mw.sum(), re_chunks.capacity_mw.sum()), (
            "RE chunking failed"
        )

        _data = pd.concat(
            {
                "energy max": en_max.groupby(level=2).sum(),
                "capacity max": cap_max.groupby(level=2).sum(),
                "pre match": re_size0.groupby(level=2).sum(),
                "match target": re_cap,
                "final": re_size1.groupby(level=2).sum(),
            },
            axis=1,
        )
        LOGGER.info(
            "%s %s post profile match used (%s, %s) bounds\n%s",
            *(self.ba.ba_code, self.config.for_idx, up, lo),
            _data.round(0).to_string().replace("\n", "\n\t"),
        )
        return (
            re_out,
            _data,
            re_size1.groupby("re_type").sum().to_dict(),
            re_chunks,
        )

    def chunk_re(self, re_out, par):
        chunk_cols = [
            "technology_description",
            "re_type",
            "plant_id_eia",
            "capacity_mw",
            "category",
        ]
        re_chunks = (
            re_out[chunk_cols]
            .assign(
                scenario_add=self.config.for_idx,
                gen_share=1.0,
                re_generator_id=lambda x: hash_pandas_object(x[["re_type", "scenario_add"]])
                .astype("int64")
                .astype(str),
            )
            .copy()
        )
        if par is not None:
            old_chunks = par.patio_chunks.query("technology_description != 'Batteries'")
            re_chunks = pd.concat(
                [
                    old_chunks,
                    re_chunks.merge(
                        old_chunks.groupby("plant_id_eia", as_index=False).capacity_mw.sum(),
                        on="plant_id_eia",
                        suffixes=(None, "_old"),
                        how="left",
                    )
                    .assign(
                        capacity_mw=lambda x: x.capacity_mw - x.capacity_mw_old.fillna(0.0)
                    )
                    .query("capacity_mw > 0")
                    .drop(columns=["capacity_mw_old"]),
                ]
            ).assign(
                gen_share=lambda x: x.capacity_mw
                / x.groupby("plant_id_eia").capacity_mw.transform("sum")
            )
        return re_chunks

    def prof_match_and_final_econ(self, energy_Ab, re_size0, lb):  # noqa: N803
        # for the purposes of selecting the amount of solar vs wind, if the LP
        # step didn't select any of one, we assume a simple regional average
        # to create profiles for the equal energy step
        re_size_agg = re_size0.groupby(["re_site_id", "re_type"]).sum()
        _test = re_size_agg.groupby(level=1).sum().to_frame("rhs")
        # see https://github.com/rmi-electricity/patio-model/issues/148#issuecomment-1696563612
        if len(z := _test.query("rhs == 0 & re_type != 'offshore_wind'").index) > 0:
            for rtype in z:
                re_size_agg.loc[(slice(None), rtype)] = 1.0
        # determine max amount of each re type consistent with economic selection
        cap_kwargs = {
            "c": -self.ba.re_plant_specs.re_max_obj_coef,
            "Ab_ub": -self.ba.re_plant_specs.pivot_table(
                index="re_type", columns="combi_id", values="ones", aggfunc="first"
            )
            .fillna(0.0)
            .join(_test),
            "lb": lb,
        }
        success, cap_max = self.ba.milp(
            **cap_kwargs,
            desc=f"{self.config.for_idx} capacity",
            wind_lb={"reference": 80.0, "limited": 25.0}[self.ba._metadata["regime"]],
        )
        if success is False:
            msg = "max cap with min wind size failed, running without min wind size constraint"
            self.msg.append(f"{msg}: {cap_max}")
            LOGGER.warning("%s %s %s", self.ba.ba_code, self.config.for_idx, msg)
            success, cap_max = self.ba.lp(**cap_kwargs)
            if success is False:
                self.msg.append(f"max cap failed w/o min wind size: {cap_max}")
                raise ScenarioError(f"max cap failed w/o min wind size: {cap_max}")
        re_profs = (self.ba.re_prof_ilr_adj * re_size_agg).groupby(level=1, axis=1).sum()
        re_profs = (re_profs / re_profs.max()).dropna(axis=1, how="all")
        success, re_cap = optimize_equal_energy(
            target=self.ba.net_load_prof,
            re_profs=re_profs,
            Ab_ub=pd.DataFrame(
                np.eye(re_profs.shape[1]),
                index=re_profs.columns,
                columns=re_profs.columns,
            ).join(cap_max.groupby(level=2).sum().to_frame("rhs")),
            Ab_eq=re_profs.sum().to_frame("energy").T.join(energy_Ab.rhs),
        )
        if success is False:
            msg = (
                f"Selection for profile match failed, using first economic "
                f"selection, optimizer returned: {re_cap}"
            )
            self.msg.append(msg)
            LOGGER.error("%s %s %s", self.ba.ba_code, self.config.for_idx, msg)
            re_size1, up, lo = re_size0, None, None
            re_cap = re_size1.groupby(level=2).sum()
        else:
            re_A = (
                self.ba.re_plant_specs.pivot_table(
                    columns="combi_id",
                    index="re_type",
                    values="ones",
                    aggfunc="first",
                )
                .fillna(0.0)
                .join(re_cap.to_frame("rhs").query("rhs > 0.5"), how="inner")
            )
            for up, lo in [
                (1.15, 0.85),
                (1.25, 0.75),
                (1.5, 0.5),
                (2, 0.5),
                (3, 0.15),
                (4, 0),
            ]:
                success, re_size1 = self.ba.milp(
                    Ab_eq=energy_Ab,
                    Ab_ub=pd.concat([re_A, -re_A]).assign(
                        rhs=lambda x: np.where(x.rhs > 0, x.rhs * up, x.rhs * lo)  # noqa: B023
                    ),
                    lb=lb,
                    wind_lb={"reference": 80.0, "limited": 25.0}[self.ba._metadata["regime"]],
                    desc=f"{self.config.for_idx} final econ ({up}, {lo})",
                )
                if success is True:
                    break
            if success is False:
                er_df = (
                    pd.concat(
                        {
                            "pre match": re_size0.groupby(level=2).sum(),
                            "match target": re_cap,
                        },
                        axis=1,
                    )
                    .round(0)
                    .to_string()
                    .replace("\n", "\n\t")
                )
                self.msg.append(f"Unable to re-optimize cost after profile match {re_size1}")
                raise ScenarioError(f"post profile match failed {re_size1}\n{er_df}")
        return cap_max, lo, re_cap, re_size1, up

    def _dzsetstate_(self, state):
        _ = state.pop("figs", None)
        self.config = ScenarioConfig(**state.pop("config"))
        for attr, val in state.items():
            setattr(self, attr, val)
        self.dm = self.make_dm()
        self._outputs = {}
        self.colo_hourly = pl.LazyFrame()
        self.colo_coef_ts = pl.LazyFrame()
        # self.colo_coef_ts_iter = pl.LazyFrame()
        self.colo_coef_mw = pl.LazyFrame()
        self.colo_summary = pl.DataFrame()

    def _dzgetstate_(self) -> dict:
        return {
            "config": self.config._asdict(),
            "no_limit_prime": self.no_limit_prime,
            "cfl": self.cfl,
            "is_max_scen": self.is_max_scen,
            "re_cap": self.re_cap,
            "good_scenario": self.good_scenario,
            "re_plant_specs": self.re_plant_specs,
            "es_specs": self.es_specs,
            "patio_chunks": self.patio_chunks,
            "to_replace": self.to_replace,
            "mothball": self.mothball,
            "exclude": self.exclude,
            "ccs": self.ccs,
            "figs": self.get_figs() if self.ba.include_figs else {},
            "intermed": self._outputs,
        }

    def monthly_fuel(self):
        return self.outputs_wrapper(self._outputs["monthly_fuel"])

    def get_figs(self):
        stem = self.config.for_idx
        return {
            f"mwh_ms_{stem}_no_limit_fig": self.dm[0]
            .plot_output("redispatch_mwh", freq="MS")
            .update_layout(title=f"{self.ba.ba_name} {stem}", height=1500),
            f"daily_{stem}_no_limit_fig": self.dm[0]
            .plot_all_years()
            .update_layout(title=f"{self.ba.ba_name} {stem}"),
            f"mwh_ms_{stem}_fig": self.dm[1]
            .plot_output("redispatch_mwh", freq="MS")
            .update_layout(title=f"{self.ba.ba_name} {stem}", height=1500),
            f"daily_{stem}_fig": self.dm[1]
            .plot_all_years()
            .update_layout(title=f"{self.ba.ba_name} {stem}"),
        }

    def is_duplicate_scen(self, others: Sequence[Self]) -> bool:
        return any(self.config[1:] == x.config[1:] and self.re_cap == x.re_cap for x in others)

    @property
    def ccs_pid_gid(self):
        return [f"{i}_{g}" for i, g in self.ccs] if self.ccs else []

    @staticmethod
    def add_pid_gid(df, fos=True):
        idc = ["icx_id", "icx_gen"] if fos else ["plant_id_eia", "generator_id"]
        idx = False
        if any(x in df.index.names for x in idc):
            idx = df.index.names
            df = df.reset_index()
        df = df.assign(pid_gid=lambda x: x[idc].astype(str).agg("_".join, axis=1))
        return df.set_index(idx) if idx else df

    # @property
    # def allocated_storage_specs(self):
    #     f_col_map = {
    #         "plant_id_eia": "fos_id",
    #         "generator_id": "fos_gen",
    #         "capacity_mw": "fos_capacity_mw",
    #     }
    #     return (
    #         self.ba.plant_data.query("category == 'existing_fossil'")
    #         .pipe(self.add_pid_gid, fos=False)
    #         .query(
    #             "plant_id_eia not in @self.to_replace & pid_gid not in @self.ccs_pid_gid"
    #         )
    #         .reset_index()[list(f_col_map)]
    #         .assign(m=1, ba_weight=lambda x: x.capacity_mw / x.capacity_mw.sum())
    #         .rename(columns=f_col_map)
    #         .merge(
    #             self.dm[0]
    #             .storage_specs.query("category == 'patio_clean'")
    #             .dropna(axis=1, how="all")
    #             .reset_index()
    #             .assign(m=1, distance=0.0),
    #             on="m",
    #         )
    #         .assign(capacity_mw=lambda x: x.capacity_mw * x.ba_weight)
    #         .drop(columns=["m"])
    #     )

    def adj_re_specs(self, re_plant_specs):
        df = (
            re_plant_specs.copy()
            # .assign(
            #     capacity_mw=lambda x: x.ba_weight * x.generator_id.replace(self.re_cap),
            #     # generator_id=lambda x: x.r_type,
            # )
        )
        if not any((self.to_replace, self.ccs)):
            return df
        if self.to_replace:
            nuke = (
                self.ba.plant_data.query("plant_id_eia in @self.to_replace")
                .reset_index()
                .assign(
                    icx_id=lambda x: x.plant_id_eia,
                    icx_gen=lambda x: x.generator_id,
                    icx_capacity=lambda x: x.capacity_mw,
                    ba_weight=lambda x: x.capacity_mw / x.capacity_mw.sum(),
                    technology_description="Nuclear",  # calculating storage capacity depends on this name
                    distance=0,
                    category="patio_clean",
                    # ba_code=lambda x: x.final_ba_code,
                    ilr=1.0,
                    operating_date=self.ba.re_profiles.index.min() - timedelta(30),
                    retirement_date=pd.NaT,
                    ba_distance=0.0,
                )
            )
            LOGGER.debug(
                "%s %s nuclear replaces %s of %s",
                self.ba.ba_code,
                self.config,
                df.query("icx_id in @self.to_replace")
                .groupby(["technology_description"])
                .capacity_mw.sum()
                .to_dict(),
                df.groupby(["technology_description"]).capacity_mw.sum().to_dict(),
            )
            assert not set(nuke.icx_id.unique()) - set(self.to_replace), (
                "unexpected plants in nuke replacement relative to baseload replacement config setting"
            )
            assert not set(
                self.ba.plant_data.index.get_level_values("plant_id_eia")
            ).intersection(set(self.to_replace) - set(nuke.icx_id.unique())), (
                "failed to replace some baseload plants that should be replaced by nukes"
            )
            df = pd.concat(
                [
                    # doing it here means we gave the fossil that will be replaced
                    # with nuclear its RE and then we excluded it, that means
                    # the total amount of RE is less than determined by equal_energy
                    df.query("icx_id not in @self.to_replace"),
                    nuke[[c for c in df if c in nuke]],
                ],
                axis=0,
            ).reset_index(drop=True)
        if self.ccs:
            # remove RE associated with fossil generators that get CCS doing it
            # here means we gave the fossil that will be mitigated with CCS its RE
            # and then we excluded it, that means the total amount of RE is less
            # than determined by equal_energy
            return (
                df.pipe(self.add_pid_gid)
                .query("pid_gid not in @self.ccs_pid_gid")
                .drop(columns=["pid_gid"])
            )
        return df

    def make_dm(self) -> tuple[DispatchModel, DispatchModel]:
        re_plant_specs = pd.concat(
            [
                self.re_plant_specs[self.ba.re_spec_cols],
                self.ba.plant_data.loc[
                    self.ba.clean_list,
                    [c for c in self.ba.re_spec_cols if c in self.ba.plant_data],
                ],
            ],
            axis=0,
        ).sort_index(axis=0)
        old_ids = list(
            self.re_plant_specs[["re_site_id", "re_type"]].itertuples(index=False, name=None)
        )
        re_profs = pd.concat(
            [
                self.ba.re_profiles.loc[:, old_ids].set_axis(
                    self.re_plant_specs.index, axis=1
                ),
                self.ba.profiles[self.ba.clean_list],
            ],
            axis=1,
        ).sort_index(axis=1)
        if self.to_replace:
            re_profs = pd.concat(
                [
                    re_profs,
                    pd.DataFrame(
                        1.0,
                        index=self.ba.re_profiles.index,
                        columns=list(
                            self.re_plant_specs.query("technology_description == 'Nuclear'")[
                                ["plant_id_eia", "generator_id"]
                            ].itertuples(index=False, name=None)
                        ),
                    ),
                ],
                axis=1,
            ).sort_index(axis=1)

        if diff := (set(re_profs) - set(re_plant_specs.index)):
            lost = set(
                self.ba.re_plant_specs.pipe(self.add_pid_gid, fos=True)
                .query("icx_id in @self.to_replace | pid_gid in @self.ccs_pid_gid")[
                    ["plant_id_eia", "generator_id"]
                ]
                .itertuples(index=False, name=None)
            )
            assert not diff - lost, (
                f"{self.config.for_idx} RE plants in re_profs but removed from specs "
                f"not associated with Nuclear or CCS replacement {sorted(diff - lost)}"
            )
            re_profs = re_profs[re_plant_specs.index]

        pd.testing.assert_index_equal(
            re_plant_specs.index, re_profs.columns, exact=False, check_exact=False
        )
        cost_data = self.ba.cost_data.copy()  # .drop(columns=["heat_rate"])
        if self.mothball:
            cost_data = cost_data.assign(
                total_var_mwh=lambda x: x.total_var_mwh.mask(
                    x.index.get_level_values("plant_id_eia").isin(self.mothball),
                    x.groupby(["datetime"]).total_var_mwh.transform("max") + 1.0,
                ),
                start_per_kw=lambda x: x.start_per_kw.mask(
                    x.index.get_level_values("plant_id_eia").isin(self.mothball),
                    x.groupby(["datetime"]).start_per_kw.transform("max") + 1.0,
                ),
            )
        disp_profiles = self.ba.profiles[self.ba.fossil_list].copy()
        disp_specs = self.ba.plant_data.loc[self.ba.fossil_list, :].assign(
            exclude=lambda x: x.index.get_level_values("plant_id_eia").isin(
                # mark retired and replaced plants for exclusion
                self.to_replace + self.exclude
            ),
            # no_limit=lambda x: np.where(
            #     x.exclude,
            #     False,
            #     x.prime_mover.isin(self.config.no_limit_prime.split(","))
            #     | (x.operational_status == "proposed"),
            # ),
        )
        if self.ccs:
            locs = [ind for ind in cost_data.index if ind[:2] in self.ccs]
            cost_data.loc[locs, ["vom_per_mwh", "fom_per_kw"]] = CCS_FACTORS[
                "vom_per_mwh", "fom_per_kw"
            ]
            cost_data.loc[locs, ["fuel_per_mwh", "heat_rate", "co2_factor"]] = (
                cost_data.loc[locs, ["fuel_per_mwh", "heat_rate", "co2_factor"]]
                * CCS_FACTORS["fuel_per_mwh", "heat_rate", "co2_factor"]
            )
            if self.config.ccs_scen < 0:
                # in the negative version of the scenario, set total var cost to 0 to make it run
                # a lot, an approximation of the effect of rich PTC payments in early years
                cost_data.loc[locs, "total_var_mwh"] = 0.0
            else:
                cost_data.loc[locs, "total_var_mwh"] = cost_data.loc[
                    locs, ["vom_per_mwh", "fuel_per_mwh"]
                ].sum(axis=1)
            # chop the top off fossil profiles that get CCS to represent increased parasitic load
            disp_profiles.loc[:, self.ccs] = np.minimum(
                disp_profiles.max().loc[self.ccs].to_numpy() / CCS_FACTORS.at["heat_rate"],
                disp_profiles.loc[:, self.ccs],
            )

        # if we only have one re site, we can DC-couple the Li storage to it
        # if len(rid := self.re_plant_specs.plant_id_eia.unique()) == 1:
        #     es_kwargs["ids"] = (rid.item(), -2, -3)
        # if self.ba.old_clean_adj_net_load:
        re_kwargs = dict(  # noqa: C408
            load_profile=pd.Series(self.ba.adj_load, index=self.ba.load.index),
            re_profiles=re_profs,
            re_plant_specs=re_plant_specs,
        )
        # else:
        #     re_kwargs = dict(
        #         load_profile=self.ba.load,
        #         re_profiles=pd.concat([re_profs, self.ba.old_re_profs], axis=1),
        #         re_plant_specs=pd.concat([re_plant_specs, self.ba.old_re_specs]),
        #         # .assign(
        #         #     category=lambda x: x.category.fillna("patio_clean")
        #         # )
        #     )
        es_specs = self.ba.plant_data.loc[self.ba.storage_list, :].assign(reserve=0)
        if not (
            dups := es_specs.query(
                "plant_id_eia in @re_plant_specs.index.get_level_values('plant_id_eia') & plant_id_eia.duplicated(keep=False)"
            )
        ).empty:
            o_idx = [x for x in es_specs.index if x not in dups.index]
            es_specs = pd.concat(
                [
                    es_specs.loc[o_idx, :],
                    (
                        dups.reset_index()
                        .groupby("plant_id_eia", as_index=False)
                        .agg(
                            dict.fromkeys(dups.columns, "first")
                            | {"capacity_mw": "sum", "generator_id": lambda x: str(tuple(x))}
                        )
                        .set_index(["plant_id_eia", "generator_id"])
                    ),
                ]
            )
        dm_kwargs = re_kwargs | dict(  # noqa: C408
            dispatchable_specs=disp_specs,
            dispatchable_cost=cost_data,
            storage_specs=pd.concat(
                [
                    # self.config.storage_specs(**es_kwargs),
                    self.es_specs,
                    es_specs,
                ],
            ),
            config={"marginal_for_startup_rank": True},
        )

        dms = self._adjust_profs_make_dm(dm_kwargs, disp_profiles.copy(), re_profs)
        dms[-1]()

        if len(self.no_limit_prime) == 1:
            max_deficit = (dms[-1].system_data.deficit / dms[-1].load_profile).max()
            tot_deficit = dms[-1].system_data.deficit.sum() / dms[-1].load_profile.sum()
            if max_deficit > 0.03 or tot_deficit > 0.02:
                # If the deficits are too high, re-run with limits on CCs
                self.no_limit_prime = ("CC", "GT")
                dms = self._adjust_profs_make_dm(dm_kwargs, disp_profiles, re_profs)
                dms[-1]()
        if len(dms) > 1:
            dms[0]()

            assert np.all(
                np.all(dms[0].dispatchable_profiles >= dms[1].dispatchable_profiles)
            ), "no_limit_re profiles not always greater than limited equivalent"
        return dms

    def allocate_es(self):
        tot_storage = self.config.storage_li_pct * self.re_plant_specs.capacity_mw.sum()
        storage_at_solar = min(
            self.re_plant_specs.query("re_type == 'solar'").capacity_mw.sum(),
            tot_storage,
        )

        allocate_by_type: Callable[[pd.DataFrame], np.ndarray[Any, np.dtype[float]]] = (  # noqa: E731
            lambda df: np.where(
                df.re_type == "solar",
                storage_at_solar
                * df.capacity_mw
                / df.query("re_type == 'solar'").capacity_mw.sum(),
                (tot_storage - storage_at_solar)
                * df.capacity_mw
                / df.query("re_type != 'solar'").capacity_mw.sum(),
            )
        )

        par = self.ba.parent_li_scenario(self)
        if par is None:
            for_es = (
                self.re_plant_specs.reset_index()
                .copy()
                .assign(
                    new_cap=lambda x: allocate_by_type(x),
                )
            )
        else:
            for_es = (
                self.re_plant_specs.reset_index()
                .copy()
                .merge(
                    par.dm[0]
                    .storage_specs.query("category == 'patio_clean'")
                    .reset_index()
                    .rename(columns={"capacity_mw": "previous_mw"})[
                        ["plant_id_eia", "previous_mw"]
                    ],
                    on="plant_id_eia",
                    validate="1:1",
                    how="left",
                )
                .assign(
                    previous_mw=lambda x: x.previous_mw.fillna(0.0),
                    target=lambda x: np.maximum(
                        x.previous_mw,
                        allocate_by_type(x),
                    ),
                )
            )
            res = linprog(
                c=-for_es.re_type.map({"solar": 2}).fillna(1).to_numpy(),
                A_ub=np.vstack([np.eye(len(for_es)), -np.eye(len(for_es))]),
                b_ub=np.hstack([for_es.target.to_numpy(), -for_es.previous_mw.to_numpy()]).T,
                A_eq=np.ones((1, len(for_es))),
                b_eq=np.ones(1) * tot_storage,
            )
            for_es = for_es.assign(new_cap=res.x)
        assert np.isclose(tot_storage, for_es.new_cap.sum()), (
            "Allocated storage does not match target"
        )
        es = for_es.assign(
            capacity_mw=lambda x: x.new_cap,
            technology_description="Batteries",
            generator_id="es",
            ilr=1.0,
            re_type="es",
            duration_hrs=4,
            roundtrip_eff=0.9,
            reserve=0.0,
        ).set_index(["plant_id_eia", "generator_id"])[
            [
                "technology_description",
                "capacity_mw",
                "ilr",
                "re_type",
                "duration_hrs",
                "roundtrip_eff",
                "reserve",
                "operating_date",
                "retirement_date",
                "category",
                "energy_community",
            ]
        ]
        if self.config.storage_li_pct == 0.0:
            es = es.iloc[[0], :]

        """
        ES can increase when either re_energy goes up OR storage_li_pct increases
        this means that there isn't a single storage scenario parent and that
        it is not possible to reliably chunk storage until the very end
        """
        # chunk_cols = ["technology_description", "re_type", "capacity_mw"]
        # es_chunks = (
        #     es[chunk_cols]
        #     .reset_index()
        #     .assign(
        #         scenario_add=self.config.for_idx,
        #         gen_share=1.0,
        #         re_generator_id=lambda x: hash_pandas_object(
        #             x[["re_type", "scenario_add"]]
        #         )
        #         .astype("int64")
        #         .astype(str),
        #     )
        #     .copy()
        # )
        # if par is not None:
        #     old_chunks = par.patio_chunks.query("technology_description == 'Batteries'")
        #     es_chunks = pd.concat(
        #         [
        #             old_chunks,
        #             es_chunks.merge(
        #                 old_chunks.groupby(
        #                     "plant_id_eia", as_index=False
        #                 ).capacity_mw.sum(),
        #                 on="plant_id_eia",
        #                 suffixes=(None, "_old"),
        #                 how="left",
        #             )
        #             .assign(
        #                 capacity_mw=lambda x: x.capacity_mw
        #                 - x.capacity_mw_old.fillna(0.0)
        #             )
        #             .query("capacity_mw > 0")
        #             .drop(columns=["capacity_mw_old"]),
        #         ]
        #     ).assign(
        #         gen_share=lambda x: x.capacity_mw
        #         / x.groupby("plant_id_eia").capacity_mw.transform("sum")
        #     )
        # assert np.isclose(es.capacity_mw.sum(), es_chunks.capacity_mw.sum())
        return es  # , es_chunks

    def _adjust_profs_make_dm(self, dm_kwargs, disp_profiles, re_profs):
        # because we need to be able to make more subtle changes to `disp_profiles`,
        # including to represent dispatch constrained by RE output at the same site
        # we don't use the `no_limit` flag in `disp_specs`, we have to handle that
        # change here directly, this is ok because we don't do anything with the
        # historical results, that is taken care of by the counterfactual scenario
        disp_specs = dm_kwargs["dispatchable_specs"]
        no_limit = np.where(
            disp_specs.exclude,
            False,
            (
                disp_specs.prime_mover.isin(self.no_limit_prime)
                & (disp_specs.fuel_group == "natural_gas")
            )
            | (disp_specs.operational_status == "proposed"),
        )
        if np.any(no_limit):
            disp_profiles.loc[:, no_limit] = zero_profiles_outside_operating_dates(
                np.maximum(
                    disp_profiles.loc[:, no_limit],
                    disp_specs.loc[no_limit, "capacity_mw"].to_numpy(),
                ),
                disp_specs.loc[no_limit, "operating_date"],
                disp_specs.loc[no_limit, "retirement_date"],
            )
        if self.ba.re_limits_dispatch == "both" or self.ba.re_limits_dispatch is False:
            dm_no_limit = DispatchModel(**dm_kwargs, dispatchable_profiles=disp_profiles)
            dm_no_limit.set_metadata("re_limits_dispatch", False)
            if self.ba.re_limits_dispatch is False:
                return (dm_no_limit,)
        if self.ba.re_limits_dispatch == "both" or self.ba.re_limits_dispatch is True:
            re = self.re_plant_specs.query("technology_description != 'Nuclear'")
            # res = re.set_index(["plant_id_eia", "generator_id"]).index
            res = re.index
            # assume that fossil and associated RE share an interconnection, the max
            # of which is the fossil generator's capacity
            disp_profiles_ = np.minimum(
                disp_profiles,
                np.maximum(
                    disp_specs.capacity_mw.to_numpy()
                    - (
                        (
                            np.minimum(re_profs[res] * re.ilr.to_numpy(), 1.0)
                            * re.capacity_mw.to_numpy()
                        )
                        .set_axis(re.set_index(["icx_id", "icx_gen"]).index, axis=1)
                        .groupby(level=[0, 1], axis=1)
                        .sum()
                        .reindex_like(disp_profiles)
                        .fillna(0.0)
                    ),
                    0.0,
                ),
            ).copy()
            assert (disp_profiles >= disp_profiles_).all().all()
            dm_limit = DispatchModel(**dm_kwargs, dispatchable_profiles=disp_profiles_)
            dm_limit.set_metadata("re_limits_dispatch", True)
            if self.ba.re_limits_dispatch is True:
                return (dm_limit,)
        return dm_no_limit, dm_limit

    @property
    def re_capacity_mw(self):
        """Capacity of RE adjusted for nuclear."""
        un_adjusted = sum(self.re_cap.values())
        if un_adjusted == 0.0:
            return 0.0
        out = self.re_plant_specs.query(
            "technology_description in ('Solar Photovoltaic', 'Onshore Wind Turbine', 'Offshore Wind Turbine')"
        ).capacity_mw.sum()
        assert out / un_adjusted > 0.65, "adjusted RE capacity is less than 65%"
        return out

    def hrs_to_check(self, cutoff=0.01):
        """Hours with positive deficits are ones where not all of net load was served
        we want to be able to easily check the two hours immediately before these
        positive deficit hours
        """
        return pd.concat((self.dm[i].hrs_to_check(cutoff=cutoff) for i in (0, 1)), axis=1)

    def checks(self, cutoff):
        dm = self.dm[0]
        self.ba.profiles.loc[dm.hrs_to_check(cutoff), :]
        dm.redispatch.loc[dm.hrs_to_check(cutoff), :]
        dm.storage_dispatch.loc[dm.hrs_to_check(cutoff), :]

    def storage_durations(self):
        """Number of hours during which state of charge was in various duration bins."""
        return pd.concat((self.dm[i].storage_durations() for i in (0, 1)), axis=1)

    def storage_capacity(self):
        """Number of hours where storage charge or discharge was in various bins."""
        return pd.concat((self.dm[i].storage_capacity() for i in (0, 1)), axis=1)

    def re_allocate_deficit(self, df):
        """Spread 2x the annual deficit across operating fossil."""
        if (
            deficit := df.query("technology_description == 'deficit'").redispatch_mwh
        ).sum() < 1 or self.cfl:
            return df
        assert df.index.is_unique, "df index not unique, cannot allocate"
        f = (
            df.reset_index()
            .merge(deficit.reset_index(), on=["datetime"], suffixes=(None, "_def"))
            .assign(
                redispatch_mwh_orig=lambda x: x.redispatch_mwh,
                _hrs=lambda x: np.where(x.datetime.dt.is_leap_year, 8784, 8760),
                _cf=lambda x: x.redispatch_mwh / (x.capacity_mw * x._hrs),
                redispatch_mwh=lambda x: x.redispatch_mwh_orig
                + 2
                * x.redispatch_mwh_def
                * np.where(
                    (x.category == "existing_fossil") & (x._cf < 0.75),
                    x.redispatch_mwh_orig
                    / x.query("category == 'existing_fossil' & _cf < 0.75")
                    .groupby("datetime")
                    .redispatch_mwh_orig.transform("sum"),
                    0.0,
                ),
                cf=lambda x: x.redispatch_mwh / (x.capacity_mw * x._hrs),
                redispatch_mmbtu=lambda x: np.where(
                    (x.redispatch_mwh_orig != x.redispatch_mwh),
                    x.redispatch_mmbtu * x.redispatch_mwh / x.redispatch_mwh_orig,
                    x.redispatch_mmbtu,
                ),
                redispatch_co2=lambda x: np.where(
                    (x.redispatch_mwh_orig != x.redispatch_mwh),
                    x.redispatch_co2 * x.redispatch_mwh / x.redispatch_mwh_orig,
                    x.redispatch_co2,
                ),
                redispatch_cost_fuel=lambda x: np.where(
                    (x.redispatch_mwh_orig != x.redispatch_mwh),
                    x.redispatch_cost_fuel * x.redispatch_mwh / x.redispatch_mwh_orig,
                    x.redispatch_cost_fuel,
                ),
                redispatch_cost_vom=lambda x: np.where(
                    (x.redispatch_mwh_orig != x.redispatch_mwh),
                    x.redispatch_cost_vom * x.redispatch_mwh / x.redispatch_mwh_orig,
                    x.redispatch_cost_vom,
                ),
            )
            .set_index(df.index.names)
        )
        if not ((f.cf <= 1.00000001) | ~np.isfinite(f.cf)).all():
            raise ScenarioError("allocation of deficit produces CFs > 1")
        return f.loc[df.index, list(df) + ["redispatch_mwh_orig"]]

    def dm_full_output(self):
        if "full_output" not in self._outputs:
            cols = [
                "re_limits_dispatch",
                "capacity_mw",
                "historical_mwh",
                "historical_mmbtu",
                "historical_co2",
                "historical_cost_fuel",
                "historical_cost_vom",
                "historical_cost_startup",
                "historical_cost_fom",
                "redispatch_mwh",
                "redispatch_curt_adj_mwh",
                "redispatch_mmbtu",
                "redispatch_co2",
                "redispatch_cost_fuel",
                "redispatch_cost_fuel_original",
                "redispatch_cost_vom",
                "redispatch_cost_startup",
                "redispatch_cost_fom",
                "implied_need_mw",
                "implied_need_mwh",
                "category",
                "plant_name_eia",
                "utility_id_eia",
                "utility_name_eia",
                "respondent_name",
                "final_ba_code",
                "balancing_authority_code_eia",
                "state",
                "latitude",
                "longitude",
                "technology_description",
                "prime_mover_code",
                "prime_mover",
                "fuel_group",
                "operational_status",
                "operational_status_code",
                "operating_date",
                "operating_month",
                "operating_year",
                "retirement_date",
                "retirement_month",
                "retirement_year",
                "ilr",
                "plant_role",
                "ramp_rate",
                "exclude",
                "no_limit",
                "min_uptime",
                "duration_hrs",
                "roundtrip_eff",
                "reserve",
                "energy_community",
                "curve_fuel_price",
                "curtailment_mwh",
                # "curve_fuel_price_exp",
                # "curve_fuel_price_lin",
                "pct_of_curve_mmbtu_max",
                "class_atb",
                "scenario_add",
            ]

            no_lim = (
                self.dm[0]
                .full_output(freq="YS")
                .assign(
                    # we have to null retirement_date for by-plant analysis to work
                    retirement_date=lambda x: x.retirement_date.fillna(
                        x.planned_generator_retirement_date
                    )
                )
                .pipe(self.re_allocate_deficit)
                .pipe(self.add_curve_price, dm=self.dm[0])
                .pipe(self.chunk_output)
            )

            if len(self.dm) == 1:
                out = (
                    pd.concat([no_lim, self.ba.missing])
                    .pipe(self.allocate_curtailment, dm=self.dm[0])
                    .assign(
                        re_limits_dispatch=self.dm[0].re_limits_dispatch,
                        implied_need_mw=0.0,
                        implied_need_mwh=0.0,
                    )
                    .fillna({"category": "system"})
                )
                self._outputs["full_output"] = out[[x for x in cols if x in out]]

                # if not self.cfl:
                #     cfl = self.ba.counterfactual.dm_full_output().query(
                #         "category == 'proposed_clean'"
                #     )
                #     assert (
                #         self._outputs["full_output"]
                #         .query("category == 'proposed_clean'")[cfl.columns]
                #         .compare(cfl)
                #         .empty
                #     ), f"{self.config.for_idx} proposed clean does not match counterfactual"

                return self._outputs["full_output"]
            assert not self.dm[0].re_limits_dispatch and self.dm[1].re_limits_dispatch, (  # noqa: PT018
                "re_limit_dispatch DispatchModels are in the wrong order"
            )
            in_ = np.maximum(self.dm[0].redispatch - self.dm[1].redispatch, 0.0)
            in_cols = [f"{p}_{g}" for p, g in in_.columns]
            pids = {f"{p}_{g}": p for p, g in in_.columns}
            gids = {f"{p}_{g}": g for p, g in in_.columns}
            in_pl = pl.from_pandas(in_.set_axis(in_cols, axis=1).reset_index()).lazy()

            kwargs = {"id_vars": "datetime", "variable_name": "pid_gid"}
            implied_need = (
                in_pl.with_columns(
                    # cumulative sum that resets when there is a zero
                    pl.col(in_cols).cumsum()
                    - pl.when(pl.col(in_cols) == 0.0)
                    .then(pl.col(in_cols).cumsum())
                    .otherwise(pl.lit(None))
                    .forward_fill()
                )
                .melt(**kwargs, value_name="implied_need_mwh")
                .join(
                    in_pl.melt(**kwargs, value_name="implied_need_mw"),
                    on=["datetime", "pid_gid"],
                )
                .sort(["datetime", "pid_gid"])
                .group_by_dynamic("datetime", every="1y", period="1y", by=["pid_gid"])
                .agg(pl.col(["implied_need_mw", "implied_need_mwh"]).max())
                .with_columns(
                    pl.col("pid_gid").replace_strict(pids, default=None).alias("plant_id_eia"),
                    pl.col("pid_gid").replace_strict(gids, default=None).alias("generator_id"),
                )
                .drop("pid_gid")
            )

            out = pd.concat(
                [
                    pd.concat(
                        [
                            no_lim.join(
                                implied_need.collect()
                                .to_pandas()
                                .set_index(["plant_id_eia", "generator_id", "datetime"]),
                                how="left",
                            ),
                            self.ba.missing,
                        ]
                    ).assign(re_limits_dispatch=self.dm[0].re_limits_dispatch),
                    pd.concat(
                        [
                            self.dm[1]
                            .full_output(freq="YS")
                            .pipe(self.re_allocate_deficit)
                            .pipe(self.add_curve_price, dm=self.dm[1])
                            .pipe(self.chunk_output),
                            self.ba.missing,
                        ]
                    )
                    .pipe(self.allocate_curtailment, dm=self.dm[1])
                    .assign(
                        re_limits_dispatch=self.dm[1].re_limits_dispatch,
                        # we have to null retirement_date for by-plant analysis to work
                        retirement_date=lambda x: x.retirement_date.fillna(
                            x.planned_retirement_date
                        ),
                    ),
                ]
            ).fillna({"category": "system", "implied_need_mw": 0.0, "implied_need_mwh": 0.0})
            self._outputs["full_output"] = out[[x for x in cols if x in out]]
        return self._outputs["full_output"]

    def allocate_curtailment(self, df: pd.DataFrame, dm: DispatchModel):
        dm_re_gen = dm.re_profiles_ac.sum(axis=1)
        dm_disp_gen = dm.redispatch.sum(axis=1)
        dm_curtailment = dm.system_data.curtailment
        existing_re = self.ba.existing_re_prof.sum(axis=1)
        assert len(existing_re) == len(dm_curtailment)
        curtailment = (
            dm_curtailment.to_frame()
            .assign(
                old_clean=lambda x: np.maximum(0.0, np.minimum(x.curtailment, existing_re)),
                patio_clean=lambda x: np.maximum(
                    0.0, np.minimum(x.curtailment - x.old_clean, dm_re_gen)
                ),
                existing_fossil=lambda x: np.maximum(
                    0.0,
                    np.minimum(
                        x.curtailment - x[["old_clean", "patio_clean"]].sum(axis=1),
                        dm_disp_gen,
                    ),
                ),
            )
            .drop(columns=["curtailment"])
            .groupby(pd.Grouper(freq="YS"))
            .sum()
        )
        assert np.isclose(dm_curtailment.sum(), curtailment.sum().sum())

        total = (
            existing_re.to_frame("old_clean")
            .assign(
                patio_clean=dm_re_gen,
                # because we already have adjusted fossil generation to account for DM
                # deficits, we have to use that data instead of dispatch from DM
                existing_fossil=df.query("category in ('existing_fossil', 'proposed_fossil')")
                .groupby("datetime")
                .redispatch_mwh.sum(),
            )
            .groupby(pd.Grouper(freq="YS"))
            .sum()
        )

        curt_pct_by_year = (
            (
                (curtailment / total)
                .fillna(0.0)
                .assign(
                    proposed_clean=lambda x: x.patio_clean,
                    proposed_fossil=lambda x: x.existing_fossil,
                )
            )
            .reset_index()
            .melt(
                id_vars="datetime",
                value_vars=[*total.columns, "proposed_clean", "proposed_fossil"],
                var_name="category_",
                value_name="curtailment_pct",
            )
        )

        out = (
            df.assign(
                category_=lambda x: np.where(
                    x.technology_description.isin(ES_TECHS),
                    "storage",
                    np.where(
                        (x.category == "existing_xpatio")
                        & x.technology_description.isin(RE_TECH),
                        "old_clean",
                        x.category,
                    ),
                )
            )
            .reset_index()
            .merge(
                curt_pct_by_year,
                on=["datetime", "category_"],
                how="left",
                validate="m:1",
            )
            .fillna({"curtailment_pct": 0.0})
            .assign(
                curtailment_mwh=lambda x: x.redispatch_mwh * x.curtailment_pct,
                redispatch_curt_adj_mwh=lambda x: np.where(
                    x.technology_description == "curtailment",
                    0.0,
                    x.redispatch_mwh - x.curtailment_mwh,
                ),
            )
            .set_index(df.index.names)
            # .drop(columns=["category_"])
        )

        test = (
            out.query("technology_description != 'curtailment'")
            .groupby(level="datetime")[["redispatch_mwh", "redispatch_curt_adj_mwh"]]
            .sum()
            .assign(
                curt=df.query("technology_description == 'curtailment'")
                .droplevel(level=[0, 1])
                .redispatch_mwh
            )
        )
        if not np.isclose(
            test.redispatch_mwh - test.redispatch_curt_adj_mwh, test.curt, rtol=1e-4
        ).all():
            b = (
                out.replace(
                    {
                        "category_": {
                            "proposed_fossil": "existing_fossil",
                            "proposed_clean": "patio_clean",
                        }
                    }
                )
                .groupby(["category_", "datetime"])[["curtailment_mwh"]]
                .sum()
                .query("curtailment_mwh > 0")
                .unstack(0)
                .droplevel(0, axis=1)
                .fillna(0.0)
            )
            c = np.ceil(curtailment.sort_index(axis=1)).compare(
                np.ceil(b.sort_index(axis=1)),
                result_names=("category_level", "asset_level"),
            )

            # THE PROBLEM IS PROBABLY TO DO WITH DEFICIT REALLOCATION
            c_str = c.to_string().replace("\n", "\n\t")

            raise ScenarioError(
                f"{self.config.for_idx} curtailment allocation failed. Mismatches \n {c_str}"
            )

        return out

    def chunk_output(self, df):
        if self.cfl:
            return df

        out = (
            df.reset_index()
            .merge(
                self.patio_chunks[
                    [
                        "plant_id_eia",
                        "re_generator_id",
                        "scenario_add",
                        "gen_share",
                        "re_type",
                        "category",
                    ]
                ],
                on=["plant_id_eia", "re_type", "category"],
                how="left",
            )
            .fillna({"gen_share": 1.0})
            .assign(
                generator_id=lambda x: x.re_generator_id.fillna(x.generator_id),
                capacity_mw=lambda x: x.capacity_mw * x.gen_share,
                redispatch_mwh=lambda x: x.redispatch_mwh * x.gen_share,
                redispatch_cost_fom=lambda x: x.redispatch_cost_fom * x.gen_share,
            )
            .set_index(df.index.names)
        )
        assert np.isclose(out.capacity_mw.sum(), df.capacity_mw.sum()), (
            "Chunking output did not preserve capacity"
        )
        return out

    def rechunk_patio_clean(self):
        if (par := self.ba.same_energy_scenario(self)) is not None:
            self.patio_chunks = par.patio_chunks.copy()
        else:
            self.patio_chunks = self.chunk_re(
                self.re_plant_specs.reset_index(), self.ba.parent_re_scenario(self)
            )
        assert np.isclose(
            self.re_plant_specs.capacity_mw.sum(), self.patio_chunks.capacity_mw.sum()
        ), "Rechunking max scenario did not preserve patio clean capacity"

    def add_curve_price(self, df: pd.DataFrame, dm: DispatchModel):
        hr = self.ba.cost_data.heat_rate.unstack([0, 1]).reindex(
            index=self.ba.load.index, method="ffill"
        )
        id_cols = list(self.ba.plant_data.index.names)
        gas_ids = self.ba.plant_data.query("fuel_group == 'natural_gas'").reset_index()[
            id_cols
        ]
        # monthly gas consumption by generator from DispatchModel
        gas_monthly = (
            dm.grouper(dm.redispatch * hr, by=None, freq="MS", col_name="mthly_mmbtu")
            .reset_index()
            # this is how we select the gas generators
            .merge(gas_ids, on=id_cols, how="inner", validate="m:1")
        )
        # create MS datetime to month integer mapping
        dt_idx = (
            pd.Series(
                pd.date_range(hr.index.min(), hr.index.max(), freq="MS"),
                name="datetime",
            )
            .reset_index()
            .rename(columns={"index": "dt_idx"})
        )
        # add on historical price and other metrics, calculate cumulative monthly
        # fuel consumption for each generator in each month add columns that define
        # the edges of its fuel consumption, ie the x-min and x-max of its region under
        # the fuel curve, include curve max and dt index
        edges = (
            gas_monthly.merge(
                self.ba.cost_data[
                    ["fuel_per_mmbtu", "heat_rate", "slope", "intercept"]
                ].reset_index(),
                on=self.ba.cost_data.index.names,
                how="left",
                validate="1:1",
            )
            .sort_values(["datetime", "fuel_per_mmbtu"])
            .assign(
                cumsum_mmbtu=lambda x: x.groupby("datetime").mthly_mmbtu.transform("cumsum"),
                end=lambda x: x.cumsum_mmbtu,
                start=lambda x: np.where(
                    np.roll(x.datetime, shift=1) == x.datetime,
                    np.roll(x.end, shift=1) + 1e-6,
                    0.0,
                ),
            )
            .merge(dt_idx, on="datetime", how="left", validate="m:1")
            .merge(
                self.ba.fuel_curve.groupby(["datetime"])
                .fuel_mmbtu_cumsum.max()
                .to_frame("curve_max"),
                on="datetime",
            )
        )
        # add additional (x,y) values to the fuel price curve so that all x-min and
        # x-max generator edges have values in the curve, any new point takes the
        # y-value (price) of the point immediate left of the new one
        curve = (
            pd.concat(
                [
                    self.ba.fuel_curve,
                    edges.melt(
                        id_vars=["datetime", "slope", "intercept", "curve_max"],
                        value_vars=["start", "end"],
                        value_name="fuel_mmbtu_cumsum",
                    ).assign(
                        cost_per_mmbtu=lambda x: np.where(
                            x.fuel_mmbtu_cumsum > x.curve_max,
                            x.intercept + x.slope * (x.fuel_mmbtu_cumsum - x.curve_max),
                            np.nan,
                        ),
                    )[["datetime", "fuel_mmbtu_cumsum", "cost_per_mmbtu"]],
                ]
            )
            .sort_values(["datetime", "fuel_mmbtu_cumsum"])
            .merge(dt_idx, on="datetime", how="left", validate="m:1")
            .assign(
                cost_per_mmbtu=lambda x: x.cost_per_mmbtu.ffill(),
            )[["dt_idx", "fuel_mmbtu_cumsum", "cost_per_mmbtu"]]
            .drop_duplicates()
        )

        gas = (
            gas_monthly.merge(
                self.ba.cost_data[
                    ["fuel_per_mmbtu", "heat_rate", "fuel_mmbtu_max"]
                ].reset_index(),
                on=self.ba.cost_data.index.names,
                how="left",
                validate="1:1",
            )
            .merge(
                edges.assign(
                    auc_price=lambda x: fuel_auc(
                        edges=x[["dt_idx", "start", "end"]].to_numpy(dtype=np.float64),
                        curve=curve.to_numpy(dtype=np.float64),
                    )
                    / x.mthly_mmbtu,
                )[["plant_id_eia", "generator_id", "datetime", "auc_price"]],
                on=["plant_id_eia", "generator_id", "datetime"],
                how="left",
                validate="1:1",
            )
            .sort_values(["datetime", "fuel_per_mmbtu"])
            .assign(
                cumsum_mmbtu=lambda x: x.groupby("datetime").mthly_mmbtu.transform("cumsum"),
                pct_of_curve_mmbtu_max=lambda x: x.groupby("datetime").cumsum_mmbtu.transform(
                    "max"
                )
                / x.groupby("datetime").fuel_mmbtu_max.transform("max"),
                redispatch_cost_fuel_curve=lambda x: (x.auc_price * x.mthly_mmbtu).fillna(0.0),
            )
        )
        if not gas.query("mthly_mmbtu > 1e-8 & redispatch_cost_fuel_curve == 0").empty:
            gas = gas.assign(
                auc_price=lambda x: x.groupby("datetime").auc_price.ffill(),
                redispatch_cost_fuel_curve=lambda x: (x.auc_price * x.mthly_mmbtu).fillna(0.0),
            )
            LOGGER.warning(
                "%s %s forward filling auc_price",
                self.ba.ba_code,
                self.config.for_idx,
            )
        outputs_cols = [
            "fuel_per_mmbtu",
            "heat_rate",
            "mthly_mmbtu",
            "cumsum_mmbtu",
            "pct_of_curve_mmbtu_max",
            "redispatch_cost_fuel_curve",
            "auc_price",
        ]
        for_outputs = gas.set_index(["plant_id_eia", "generator_id", "datetime"])[
            outputs_cols
        ].assign(
            redispatch_cost_fuel_og=lambda x: x.mthly_mmbtu * x.fuel_per_mmbtu,
            curve_fuel_price=lambda x: x.redispatch_cost_fuel_curve / x.mthly_mmbtu,
            re_limits_dispatch=dm.re_limits_dispatch,
        )
        if "monthly_fuel" in self._outputs:
            self._outputs["monthly_fuel"] = pd.concat(
                [self._outputs["monthly_fuel"], for_outputs]
            )
        else:
            self._outputs["monthly_fuel"] = for_outputs

        bad_gas_cost = gas[(gas.mthly_mmbtu > 1e-8) & (gas.redispatch_cost_fuel_curve == 0)]
        if not bad_gas_cost.empty:
            bad_years = (  # noqa: F841
                bad_gas_cost.reset_index()
                .assign(year=lambda x: x.datetime.dt.year)
                .groupby(id_cols)
                .agg({"year": set})
            )
            bad_gas_cost.to_csv(
                ROOT_PATH / f"{self.ba.ba_code}_{self.config.for_idx}_bad_gas_cost.csv"
            )
            # raise AssertionError(
            #     f"{self.ba.ba_code}/{self.config.for_idx} missing reqd data for {bad_years}"
            # )
        g = (
            gas.groupby([*id_cols, pd.Grouper(key="datetime", freq="YS")])
            .agg(
                {
                    "mthly_mmbtu": "sum",
                    "redispatch_cost_fuel_curve": "sum",
                    "pct_of_curve_mmbtu_max": "max",
                }
            )
            .assign(
                curve_fuel_price=lambda x: np.where(
                    x.mthly_mmbtu == 0.0,
                    0.0,
                    x.redispatch_cost_fuel_curve / x.mthly_mmbtu,
                ),
            )
            .reset_index()
        )
        return (
            df.reset_index()
            .merge(g, on=[*id_cols, "datetime"], how="left", validate="1:1")
            .set_index(df.index.names)
            .assign(
                redispatch_cost_fuel_original=lambda x: x.redispatch_cost_fuel,
                # because we scaled mmbtus in re_allocate_deficit, the mmbtus from
                # DispatchModel are not quite right, so we just apply the annual prices
                # we calculated to the adjusted mmbtus
                redispatch_cost_fuel=lambda x: np.where(
                    x.curve_fuel_price.isna(),
                    x.redispatch_cost_fuel_original,
                    x.curve_fuel_price * x.redispatch_mmbtu,
                ),
            )
        )

    def full_output(self):
        try:
            return self.outputs_wrapper(self.dm_full_output())
        except ScenarioError as exc:
            LOGGER.error(
                "%s %s unable to create full output %r",
                self.ba.ba_code,
                self.config.for_idx,
                exc,
            )
            return MTDF.copy()

    def system_output(self, variant="redispatch"):
        es = self.ba.plant_data.query("technology_description == 'Batteries'").reset_index()
        storage_rollup = {
            stat: list(es[es.operational_status == stat].plant_id_eia.unique())
            for stat in es.operational_status.unique()
        }
        out = (
            self.dm[0]
            .system_level_summary(freq="YS", storage_rollup=storage_rollup)
            .assign(re_limits_dispatch=self.dm[0].re_limits_dispatch)
        )
        if len(self.dm) > 1:
            out = pd.concat(
                [
                    out,
                    self.dm[1]
                    .system_level_summary(freq="YS", storage_rollup=storage_rollup)
                    .assign(re_limits_dispatch=self.dm[1].re_limits_dispatch),
                ]
            )
        out = out[list({"re_limits_dispatch": None} | dict.fromkeys(out))]

        if pid := self.ba.ba_code.partition("_")[2]:
            out = out.rename(columns={k: k.replace(f"_-{pid}_", "_-1_") for k in out})

        # adjust output for missing and old re
        cats = ["existing_xpatio", "old_clean"]  # + (  # noqa: F841
        #     ["old_clean"] if self.ba.old_clean_adj_net_load else []
        # )
        dmfo_ = self.dm_full_output()
        dmfo = dmfo_.query("category in @cats").reset_index()
        for_out = (
            dmfo.query("technology_description in @RE_TECH")
            .groupby(["datetime", "re_limits_dispatch"])
            .redispatch_mwh.sum()
            .reset_index(name="missing_re_mwh")
            .merge(
                dmfo.groupby(["datetime", "re_limits_dispatch"])
                .redispatch_mwh.sum()
                .reset_index(name="missing_load_mwh"),
                on=["datetime", "re_limits_dispatch"],
            )
            .merge(
                dmfo_.query("category != 'system'")
                .groupby(["datetime", "re_limits_dispatch"])
                .redispatch_mwh.sum()
                .reset_index(name="all_gen_mwh"),
                on=["datetime", "re_limits_dispatch"],
            )
        )
        out = (
            out.filter(regex="^(?!storage_).*")
            .reset_index()
            .rename(
                columns={
                    k: "dispatch_" + k
                    for k in (
                        "load_mwh",
                        "re_mwh",
                        "curtailment_pct",
                        "re_curtailment_pct",
                    )
                }
            )
            .merge(
                self.ba.augmented_load.groupby(pd.Grouper(freq="YS"))
                .sum()
                .reset_index(name="load_mwh"),
                on="datetime",
            )
            .merge(for_out, on=["datetime", "re_limits_dispatch"])
            .assign(
                load_mwh=lambda x: x[["load_mwh", "missing_load_mwh"]].sum(axis=1),
                re_mwh=lambda x: x[["dispatch_re_mwh", "missing_re_mwh"]].sum(axis=1),
                curtailment_pct=lambda x: x.curtailment_mwh / x.all_gen_mwh,
                re_curtailment_pct=lambda x: x.re_curtailment_mwh / x.re_mwh,
                non_re_curtailment_pct_of_non_re_gen=lambda x: (
                    x.curtailment_mwh - x.re_curtailment_mwh
                )
                / (x.all_gen_mwh - x.re_mwh),
                dispatch_non_re_curtailment_pct_of_non_re_gen=lambda x: (
                    x.curtailment_mwh - x.re_curtailment_mwh
                )
                / (x.dispatch_load_mwh - x.re_mwh),
            )
            .set_index(out.index.names)
        )

        es_rename = dict(enumerate(("storage_li_mw", "storage_fe_mw", "storage_h2_mw")))

        resource_deets = (
            self.dm_full_output()
            .replace(
                {
                    "category": {
                        "old_clean": "old_",
                        "patio_clean": "patio_",
                        "system": "del",
                        "proposed_clean": "proposed_",
                        "existing_fossil": "",
                        "proposed_fossil": "",
                        "existing_xpatio": "del",
                    }
                }
            )
            .rename(columns={"capacity_mw": "mw", f"{variant}_mwh": "mwh"})
            .replace({"technology_description": TD_MAP})
            .query("category != 'del'& technology_description not in @ES_TECHS")
            .assign(
                header=lambda x: x[["category", "technology_description"]]
                .astype(str)
                .agg("".join, axis=1),
            )
            .pivot_table(
                index=[
                    "datetime",
                    "re_limits_dispatch",
                ],
                columns="header",
                values=["mw", "mwh"],
                aggfunc=np.sum,
            )
            .reorder_levels([1, 0], axis=1)
        )
        resource_deets.columns = map("_".join, resource_deets.columns)
        retirements = (
            self.dm_full_output()[["re_limits_dispatch", "capacity_mw", "retirement_date"]]
            .reset_index()
            .query(
                "datetime == datetime.min() & retirement_date.notna() & ~re_limits_dispatch"
            )
            .groupby(pd.Grouper(key="retirement_date", freq="YS"))[["capacity_mw"]]
            .sum()
            .sort_index()
            .assign(fossil_retirements_cum_mw=lambda x: x.capacity_mw.cumsum())
            .reindex(index=out.index.get_level_values("datetime").unique(), method="ffill")
            .fillna(0.0)[["fossil_retirements_cum_mw"]]
        )

        return self.outputs_wrapper(
            out.rename(
                columns=lambda x: x.replace("_-1_", "_li_")
                .replace("_-2_", "_fe_")
                .replace("_-3_", "_h2_")
            )
            .assign(
                **self.config.storage_specs(self.re_capacity_mw)
                .capacity_mw.reset_index(drop=True)
                .rename(index=es_rename)
                .to_dict(),
            )
            .join(retirements, how="left", validate="m:1")
            .reset_index()
            .merge(resource_deets.reset_index(), on=["datetime", "re_limits_dispatch"])
            .set_index("datetime")
        )

    def messages(self):
        return self.outputs_wrapper(
            # pd.concat([pd.DataFrame(self.msg, columns=["notes"]), self._data], axis=1)
            self._data.assign(notes=", ".join(self.msg))
        )

    def allocated_output(self):
        """RE summary by fossil plant."""
        # remove RE that is only there to turn net load from past historical years
        # into net load for weather years
        if self.cfl:
            return MTDF.copy()
        dm_out = (
            self.dm_full_output()
            .pipe(self.add_pid_gid, fos=False)
            .query("category == 'patio_clean' | pid_gid in @self.ccs_pid_gid")
            .drop(columns=["pid_gid"])
            .dropna(axis=1, how="all")
        )

        re_out = (
            dm_out.reset_index()
            .merge(
                self.re_plant_specs.reset_index()[
                    [
                        "icx_id",
                        "icx_gen",
                        # "class_atb",
                        "plant_id_prof_site",
                        "plant_id_eia",
                        "distance",
                        "latitude_nrel_site",
                        "longitude_nrel_site",
                        "re_site_id",
                    ]
                ],
                on=["plant_id_eia"],
                validate="m:1",
                suffixes=(None, "_spec"),
            )
            .merge(
                self.ba.plant_data.energy_community.reset_index(),
                left_on=["icx_id", "icx_gen"],
                right_on=["plant_id_eia", "generator_id"],
                how="left",
                validate="m:1",
                suffixes=(None, "_icx"),
            )
            .assign(
                energy_community=lambda x: x.energy_community.mask(
                    x.generator_id == "es",
                    x.energy_community | x.energy_community_icx.fillna(False),
                )
            )
            .drop(columns=["plant_id_eia_icx", "generator_id_icx", "energy_community_icx"])
            .assign(
                re_plant_id=lambda x: x.plant_id_eia,
                re_generator_id=lambda x: x.generator_id,
                generator_id=lambda x: x.icx_gen,
                plant_id_eia=lambda x: x.icx_id.astype(int),
                operating_year=pd.NA,
                retirement_year=pd.NA,
                category="patio_clean",
            )
            .set_index(
                [
                    "plant_id_eia",
                    "generator_id",
                    "re_plant_id",
                    "re_generator_id",
                    "datetime",
                ]
            )
            .sort_index()
        )

        # re_out = (
        #     pd.concat(
        #         [
        #             dm_out.reset_index().merge(
        #                 self.re_plant_specs.reset_index()[
        #                     [
        #                         "fos_id",
        #                         "fos_gen",
        #                         "class_atb",
        #                         "plant_id_prof_site",
        #                         "plant_id_eia",
        #                         "distance",
        #                         "latitude_nrel_site",
        #                         "longitude_nrel_site",
        #                         "re_site_id",
        #                     ]
        #                 ],
        #                 on=["plant_id_eia"],
        #                 validate="m:1",
        #             )
        #         ]
        #         + [
        #             dm.storage_summary(by=None, freq="YS")
        #             .query("category == 'patio_clean'")
        #             .dropna(axis=1, how="all")
        #             .reset_index()
        #             .merge(
        #                 self.re_plant_specs.reset_index()[
        #                     [
        #                         "plant_id_eia",
        #                         "fos_id",
        #                         "fos_gen",
        #                         "plant_id_prof_site",
        #                         "energy_community",
        #                     ]
        #                 ],
        #                 on="plant_id_eia",
        #                 how="left",
        #                 validate="m:1",
        #             )
        #             .merge(
        #                 self.ba.plant_data.energy_community.reset_index(),
        #                 left_on=["fos_id", "fos_gen"],
        #                 right_on=["plant_id_eia", "generator_id"],
        #                 how="left",
        #                 validate="m:1",
        #                 suffixes=(None, "_fos"),
        #             )
        #             .assign(
        #                 re_limits_dispatch=dm.re_limits_dispatch,
        #                 energy_community=lambda x: x.energy_community
        #                 | x.energy_community_fos.fillna(False),
        #             )
        #             .drop(
        #                 columns=[
        #                     "plant_id_eia_fos",
        #                     "generator_id_fos",
        #                     "energy_community_fos",
        #                 ]
        #             )
        #             for dm in self.dm
        #         ]
        #     )
        #     .assign(
        #         re_plant_id=lambda x: x.plant_id_eia,
        #         re_type=lambda x: x.generator_id.where(
        #             x.generator_id.isin(RE_TECH_R), x.technology_description
        #         ),
        #         generator_id=lambda x: x.fos_gen.fillna(x.generator_id),
        #         plant_id_eia=lambda x: x.fos_id.fillna(x.plant_id_eia).astype(int),
        #         operating_year=pd.NA,
        #         retirement_year=pd.NA,
        #         category="patio_clean",
        #     )
        #     .set_index(
        #         ["plant_id_eia", "generator_id", "re_plant_id", "re_type", "datetime"]
        #     )
        #     .sort_index()
        # )

        return self.outputs_wrapper(re_out)

    def outputs_wrapper_pl(
        self, df: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        str_scen = "counterfactual" if self.cfl else self.config.for_idx
        df_cols = df.columns
        new_cols = (
            {
                k: pl.lit(v).cast(pl.Categorical)
                for k, v in zip(self.ix_names, self.ix, strict=False)
                if k not in df_cols
            }
            | {"scenario": pl.lit(str_scen).cast(pl.Categorical)}
            | {k: pl.lit(v) for k, v in self.config._asdict().items() if k not in df_cols}
            | {"no_limit_prime": pl.lit(",".join(self.no_limit_prime)).cast(pl.Categorical)}
        )
        return df.with_columns(**new_cols).select(*list(new_cols), *df_cols)

    def outputs_wrapper(self, df):
        str_scen = "counterfactual" if self.cfl else self.config.for_idx
        df_idx = df.index.names
        df = pd.concat(
            [df],
            axis=0,
            keys=[(*self.ix, str_scen, *self.config, ",".join(self.no_limit_prime))],
            names=[
                *self.ix_names,
                "scenario",
                *self.config._fields,
                "no_limit_prime",
            ],
        )
        if "re_limits_dispatch" not in df:
            return df
        mapper = {True: ", re_limits_dispatch", False: ""}
        return (
            df.reset_index()
            .assign(scenario=lambda x: x.scenario.str.cat(x.re_limits_dispatch.map(mapper)))
            .set_index([*self.ix_names, *df_idx, "scenario"])
        )

    def setup_colo_data(
        self,
        selection: dict,
    ):
        if self.to_replace:
            # additional re_spec and profile things have to happen with nuclear that
            # are skipped when redispatching the system after colo load added
            raise RuntimeError("Colo load not compatible with nuclear replacement.")
        if self.ba.ba_code in BAD_COLO_BAS:
            LOGGER.warning("ba_code=%s skipped for colo analysis", self.ba.ba_code)
            return None
        min_ret_yr = selection["min_retirement_year"]  # noqa: F841
        min_op_yr = selection["min_operating_year"]  # noqa: F841
        reg_rank = selection["reg_rank"]  # noqa: F841
        ids = (
            self.ba._re_plant_specs.query(
                "(retirement_date_icx.isna() | retirement_date_icx.dt.year > @min_ret_yr) & "
                "((operating_date_icx.dt.year >= @min_op_yr & icx_tech in @OTHER_TD_MAP) | icx_tech not in @OTHER_TD_MAP)"
                "& reg_rank in @reg_rank"
            )
            .groupby(["icx_id", "icx_gen"], as_index=False)[
                ["icx_tech", "icx_capacity", "icx_status", "ever_gas"]
            ]
            .first()
            .groupby(["icx_id", "icx_tech", "icx_status"], as_index=False)
            .agg({"icx_capacity": "sum", "icx_gen": tuple, "ever_gas": "max"})
            .astype({"ever_gas": bool})
            .pipe(df_query, selection)[["icx_id", "icx_gen", "icx_capacity"]]
        )
        filters = (
            self.ba.plant_data.merge(
                self.ba._re_plant_specs[["icx_id", "icx_gen"]]
                .drop_duplicates()
                .rename(columns={"icx_id": "plant_id_eia", "icx_gen": "generator_id"}),
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="1:1",
                indicator=True,
            )
            .pipe(df_query, selection)
            .assign(
                retirement=lambda x: x.retirement_date.isna()
                | (x.retirement_date.dt.year > selection["min_retirement_year"]),
                operating=lambda x: x.operating_date.dt.year
                >= selection["min_operating_year"],
                size=lambda x: x.groupby(
                    ["plant_id_eia", "operating", "retirement"]
                ).capacity_mw.transform("sum")
                >= selection.get("capacity_mw", {"item": 0.0})["item"],
                re_data=lambda x: x["_merge"] == "both",
            )
            .groupby(
                [
                    "plant_id_eia",
                    "operational_status",
                    "re_data",
                    "operating",
                    "retirement",
                    "size",
                ],
                as_index=False,
            )
            .capacity_mw.sum()
            .groupby(["operational_status", "re_data", "retirement", "operating", "size"])
            .capacity_mw.agg(["count", "sum"])
        )
        LOGGER.warning(
            "%s filtering for colo analysis:\n%s",
            self.ba.ba_code,
            filters.to_string().replace("\n", "\n\t"),
        )
        if not ids.empty:
            json_path = Path.home() / self.ba.colo_dir / "colo.json"
            self.setup_ba_colo_data(first_year=2024)

            with tqdm_logging_redirect():
                with open(json_path) as cjson:
                    json_plants = json.load(cjson)
                plants_data = []
                for pid, gens, cap in tqdm(
                    ids.itertuples(index=False),
                    total=len(ids),
                    position=4,
                    leave=False,
                    desc=f"{self.ba.ba_code} colo data setup",
                ):
                    try:
                        p = self.setup_re_colo_data(pid, gens, cap, first_year=2024)
                    except Exception as exc:
                        LOGGER.error("(pid=%s, gens=%s, cap=%s) %r", pid, gens, cap, exc)
                    else:
                        if p:
                            plants_data.append(p)
                json_plants["plants"].extend(plants_data)
                with open(json_path, "w") as cjson:
                    json.dump(json_plants, cjson, indent=4)

    def setup_ba_colo_data(self, first_year):
        self.dm[0].dispatchable_specs = self.dm[0].dispatchable_specs.assign(
            simple_td=lambda x: x.technology_description.map(SIMPLE_TD_MAP)
        )
        self.dm[0].re_plant_specs = self.dm[0].re_plant_specs.assign(
            simple_td=lambda x: x.technology_description.map(SIMPLE_TD_MAP)
        )
        ferc714 = generate_projection_from_historical_pl(
            get_714profile(
                tuple(self.ba.plant_data.respondent_id_ferc714.dropna().astype(int).unique()),
                self.ba.pudl_release,
            )
            .pivot(on="respondent_id_ferc714", index="datetime", values="mwh")
            .sort("datetime"),
            year_mapper=self.ba._metadata["year_mapper"],
        ).cast({"datetime": pl.Datetime("ns")})
        baseline = pd.concat(
            [
                self.dm[0]
                .system_summary_core(freq="h")
                .loc[
                    :,
                    ["load_mwh", "deficit_mwh", "curtailment_mwh", "curtailment_pct"],
                ]
                .assign(
                    curtailment_pct=lambda x: np.maximum(
                        0.0, np.minimum(1.0, x.curtailment_pct.fillna(0.0))
                    )
                )
                .rename(
                    columns={
                        "load_mwh": "baseline_sys_load",
                        "deficit_mwh": "baseline_sys_deficit",
                        "curtailment_mwh": "baseline_sys_curtailment",
                        "curtailment_pct": "baseline_sys_curtailment_pct",
                    }
                ),
                self.dm[0].redispatch_lambda().rename("baseline_sys_lambda"),
                self.dm[0].net_load_profile.rename("baseline_sys_net_load"),
                (
                    self.dm[0].redispatch
                    * (
                        self.dm[0].dispatchable_cost.heat_rate
                        * self.dm[0].dispatchable_cost.co2_factor
                    )
                    .reset_index()
                    .pivot(
                        index="datetime",
                        columns=["plant_id_eia", "generator_id"],
                        values=0,
                    )
                    .reindex_like(self.dm[0].redispatch, method="ffill")
                )
                .sum(axis=1)
                .rename("baseline_sys_co2"),
                calc_redispatch_cost(self.dm[0]).rename("baseline_sys_cost"),
                self.dm[0].redispatch.sum(axis=1).rename("baseline_sys_dispatchable"),
                self.dm[0].re_profiles_ac.sum(axis=1).rename("baseline_sys_renewable"),
                self.dm[0]
                .storage_dispatch.T.groupby(level=0)
                .sum()
                .T.assign(baseline_sys_storage=lambda x: x.discharge - x.gridcharge)[
                    "baseline_sys_storage"
                ],
            ],
            axis=1,
        )
        re_specs = self.dm[0].re_plant_specs.copy()
        re_profs = self.ba.profiles.loc[f"{first_year}-01-01" :, self.ba.clean_list]
        if set(re_profs.columns) != set(re_specs.index):
            re_profs = re_profs.reindex(columns=re_specs.index).fillna(0.0)

        out = {
            "baseline": pl.from_pandas(baseline, include_index=True)
            .filter(pl.col("datetime").dt.year() >= first_year)
            .join(ferc714, on="datetime", how="left"),
            "ba_dm_data": {
                "dispatchable_profiles": self.dm[0].dispatchable_profiles.query(
                    "datetime.dt.year >= @first_year"
                ),
                "dispatchable_specs": self.dm[0].dispatchable_specs,
                "dispatchable_cost": pl.from_pandas(
                    self.dm[0].dispatchable_cost.reset_index()
                ).filter(pl.col("datetime").dt.year() >= first_year),
                "storage_specs": self.dm[0].storage_specs,
                "re_profiles": re_profs,
                "re_plant_specs": re_specs,
            },
            "plant_data": self.ba.plant_data.reset_index(),
            "fuel_curve": self.ba.fuel_curve.query("datetime.dt.year >= @first_year"),
        }
        with DataZip(
            Path.home() / f"{self.ba.colo_dir}/{self.ba.ba_code}.zip",
            "w",
        ) as dz:
            for k, v in out.items():
                dz[k] = v
        gc.collect()
        return None

    def setup_re_colo_data(self, pid, gens, poi, first_year):
        re_first = (
            "generator_id",
            "icx_genid",
            "icx_tech",
            "icx_status",
            "combi_id",
            "combined",
            "area_per_mw",
            "ones",
            "plant_id_prof_site",
            "retirement_date",
            "operating_date",
            "cf_mult",
            "ilr",
            "latitude_nrel_site",
            "longitude_nrel_site",
            "distance_prof_site",
            "distance_to_transmission_km",
            "mean_cf_ac",
            "mean_lcoe",
            "lcot",
            "total_lcoe",
            "lcoe",
            "trans_cap_cost_per_mw_ac",
            "class_atb",
            "cf_atb",
            "reinforcement_cost_per_mw_ac",
            "reinforcement_dist_km",
            "reg_mult",
            "energy_community",
            "state",
            "utility_id_eia_lse",
            "utility_name_eia_lse",
            "balancing_authority_code_eia",
            "respondent_id_ferc714",
        )
        re = (
            self.ba._re_plant_specs.query("icx_id == @pid & icx_gen in @gens")
            .merge(
                self.re_plant_specs.assign(cr_mw=lambda x: x.capacity_mw).cr_mw.reset_index(),
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="1:1",
            )
            .fillna({"cr_mw": 0.0})
            .merge(
                self.es_specs.assign(es_mw=lambda x: x.capacity_mw).reset_index()[
                    ["plant_id_eia", "es_mw"]
                ],
                on="plant_id_eia",
                how="left",
                validate="1:1",
            )
            .groupby(["re_site_id", "re_type"], as_index=False)
            .agg(
                {"plant_id_eia": tuple, "icx_capacity": "sum"}
                | dict.fromkeys(re_first, "first")
                | {
                    "capacity_mw_nrel_site": "max",
                    "area_sq_km": "max",
                    "capacity_mw_nrel_site_lim": "max",
                    "area_sq_km_lim": "max",
                    "distance": "max",
                    "cr_mw": "sum",
                    "es_mw": "sum",
                }
            )
        )
        other_gens_selected = (
            self.re_plant_specs.assign(used_site_type_mw=lambda x: x.capacity_mw)
            .query("(icx_id != @pid | icx_gen not in @gens)")
            .groupby(["re_site_id", "re_type"], as_index=False)
            .agg({"used_site_type_mw": "sum", "area_sq_km": "first", "area_per_mw": "first"})
            .assign(
                used_site_type_area=lambda x: x.used_site_type_mw * x.area_per_mw,
                used_site_area=lambda x: x.groupby("re_site_id").used_site_type_area.transform(
                    "sum"
                ),
            )
        )
        re = (
            re.merge(
                other_gens_selected,
                on=["re_site_id", "re_type"],
                how="left",
                validate="1:1",
                suffixes=(None, "_other_icx"),
            )
            .assign(
                area_sq_km=lambda x: x.area_sq_km - x.used_site_area.fillna(0.0),
                capacity_mw_nrel_site=lambda x: x.capacity_mw_nrel_site
                - x.used_site_type_mw.fillna(0.0),
                area_sq_km_lim=lambda x: x.area_sq_km_lim - x.used_site_area.fillna(0.0),
                capacity_mw_nrel_site_lim=lambda x: x.capacity_mw_nrel_site_lim
                - x.used_site_type_mw.fillna(0.0),
            )
            .sort_values(["re_type", "distance"], ascending=[False, True])
            .assign(
                cum_cap_mw=lambda x: x.groupby("re_type").capacity_mw_nrel_site.transform(
                    "cumsum"
                )
                - x.groupby("re_type").capacity_mw_nrel_site.transform("first"),
                cum_cap_mw_lim=lambda x: x.groupby(
                    "re_type"
                ).capacity_mw_nrel_site_lim.transform("cumsum")
                - x.groupby("re_type").capacity_mw_nrel_site_lim.transform("first"),
            )
            .query("cum_cap_mw_lim < @poi * 50")
        )
        icx_tech = re.icx_tech.unique()[0]
        status = re.icx_status.unique()[0]
        _fuels = self.ba.plant_data.query(
            "plant_id_eia == @pid & generator_id in @gens"
        ).fuel_group
        if len(_fuels.unique()) == 1:
            fuel = _fuels.unique().item()
        else:
            fuel = _fuels.mode().item()
            LOGGER.warning(
                "%s %s %s %s has multiple fuels, selecting %s from %s",
                *(self.ba.ba_code, pid, gens, icx_tech, fuel, list(_fuels.unique())),
            )
        assert make_core_lhs_rhs(re, fossil=False).query("rhs.isna()").empty, (
            "rhs adjustment failed: idx mismatch"
        )
        re_pro_ = self.ba.setup_re_profiles(re)[1]
        assert re_pro_.isna().sum().sum() == 0, "re_profiles contains NaNs"

        if icx_tech in OTHER_TD_MAP:
            icx_pro_all = (
                self.dm[0].redispatch.loc[:, [(pid, gid) for gid in gens]].sum(axis=1)
            )
            icx_histpro_all = self.ba.profiles.loc[:, [(pid, gid) for gid in gens]].sum(axis=1)
            var_all = (
                self.ba.cost_data.query("plant_id_eia == @pid & generator_id in @gens")
                .groupby(pd.Grouper(freq="MS", level="datetime"))["total_var_mwh"]
                .mean()
            )
        elif icx_tech == "Nuclear":
            icx_pro_all = pd.Series(poi, index=re_pro_.index)
            icx_histpro_all = pd.Series(poi, index=re_pro_.index)
            var_all = (
                pd.Series(0.5, index=re_pro_.index, name="total_var_mwh")
                .groupby(pd.Grouper(freq="MS", level="datetime"))
                .mean()
            )
        elif icx_tech in CLEAN_TD_MAP:
            icx_pro_all = pd.Series(poi, index=re_pro_.index)
            icx_histpro_all = pd.Series(poi, index=re_pro_.index)
            var_all = (
                pd.Series(0.0, index=re_pro_.index, name="total_var_mwh")
                .groupby(pd.Grouper(freq="MS", level="datetime"))
                .mean()
            )
        icx_tech = {v: k for k, v in TECH_CODES.items()}[icx_tech]
        pro_cols = re[["combined", "combi_id"]].itertuples(name=None, index=False)
        with DataZip(
            Path.home()
            / f"{self.ba.colo_dir}/{self.ba.ba_code}_{pid}_{icx_tech}_{status}.zip",
            "w",
        ) as dz:
            dz["re"] = pl.from_pandas(re)
            dz["re_pro"] = self.ba._re_profiles.filter(
                pl.col("datetime").dt.year() >= first_year
            ).select("datetime", *[pl.col(a).alias(b) for a, b in pro_cols])
            dz["icx_pro"] = pl.from_pandas(
                pd.concat({"icx": icx_pro_all, "icx_hist": icx_histpro_all}, axis=1),
                include_index=True,
            ).filter(pl.col("datetime").dt.year() >= first_year)
            dz["var_all"] = pl.from_pandas(var_all, include_index=True)
        gc.collect()
        return {
            "ba": self.ba.ba_code,
            "pid": pid,
            "gens": gens,
            "tech": icx_tech,
            "status": status,
            "cap": poi,
            "fuel": fuel,
            "ferc_id": int(re.respondent_id_ferc714.unique().item()),
        }

    @property
    def ix(self) -> tuple:
        return (self.ba.ba_code,)

    @property
    def ix_names(self) -> tuple:
        return ("ba_code",)

    def __repr__(self):
        return (
            self.__class__.__qualname__ + f"(ba_code={self.ba.ba_code}, config={self.config})"
        )


def plot_year_hourly(hourly, year):
    _re_types = list(
        {"solar", "onshore_wind", "offshore_wind"}.intersection(
            set(hourly.dropna(axis=1, how="all").columns)
        )
    )
    q = (
        hourly.assign(
            year=lambda x: x.index.year,
            month=lambda x: x.index.month,
            hour=lambda x: x.groupby(pd.Grouper(freq="M")).load.transform("cumcount"),
            curtailment=lambda x: -x.curtailment,
            useful=lambda x: x.load + x.export,
        )
        .melt(
            id_vars=["year", "month", "hour"],
            value_vars=[
                *_re_types,
                "es",
                "fossil",
                # "load_fossil",
                # "export_fossil",
                "curtailment",
                "useful",
            ],
            value_name="MW",
        )
        .replace(
            {
                "variable": PLOT_MAP
                | {
                    "es": "Storage",
                    "load_fossil": "Local Fossil",
                    "fossil": "Fossil",
                    "export_fossil": "Export Fossil",
                }
            }
        )
    ).query("year == @year")
    le_kw = dict(  # noqa: C408
        mode="lines",
        name="Load+Export",
        line_dash="dot",
        marker={"color": "black"},
    )
    z = (
        px.bar(
            q.query("variable != 'useful'"),
            y="MW",
            x="hour",
            color="variable",
            facet_col="month",
            facet_col_wrap=3,
            facet_row_spacing=0.02,
            color_discrete_map=COLOR_MAP
            | {
                "Local Fossil": COLOR_MAP["Gas CC"],
                "Fossil": COLOR_MAP["Gas CC"],
                "Export Fossil": COLOR_MAP["Gas CT"],
            },
        )
        .update_traces(
            marker_line_width=0.0,
        )
        .add_scatter(
            x=q.query("variable == 'useful' & month == 12").hour,
            y=q.query("variable == 'useful' & month == 12").MW,
            row=1,
            col=3,
            **le_kw,
        )
        .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        .update_layout(
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1.05,
            legend_xanchor="left",
            legend_x=-0.05,
            legend_title=None,
            template="ggplot2",
        )
    )
    for m, r, c in ((m, int(5 - np.ceil(m / 3)), 1 + (m - 1) % 3) for m in range(1, 12)):  # noqa: B007
        q_ = q.query("variable == 'useful' & month == @m")
        z = z.add_scatter(x=q_.hour, y=q_.MW, row=r, col=c, showlegend=False, **le_kw)
    return z
