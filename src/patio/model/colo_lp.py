import logging
import operator
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property, reduce, singledispatchmethod
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NamedTuple, Self
from warnings import catch_warnings

import cvxpy as cp
import gurobipy  # noqa: TC002
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from dispatch import DispatchModel
from etoolbox.datazip import DataZip, IOMixin
from etoolbox.utils.cloud import rmi_cloud_fs
from polars import col as c
from scipy.signal import argrelextrema

import patio.model.colo_resources as colo_res_mod
from patio.constants import (
    OTHER_TD_MAP,
    PATIO_PUDL_RELEASE,
    ROOT_PATH,
    SIMPLE_TD_MAP,
    TECH_CODES,
)
from patio.helpers import (
    generate_projection_from_historical_pl as gen_proj,
)
from patio.helpers import (
    make_core_lhs_rhs,
    solver,
)
from patio.model.base import adjust_profiles, calc_redispatch_cost
from patio.model.colo_common import (
    AEO_MAP,
    COSTS,
    FAIL_DISPATCH,
    FAIL_SELECT,
    FAIL_SERVICE,
    FAIL_SMALL,
    INFEASIBLE,
    STATUS,
    SUCCESS,
    add_missing_col,
    aeo,
    capture_stderr,
    capture_stdout,
    hstack,
    keep_nonzero_cols,
    nt,
    prof,
    safediv,
    timer,
    to_dict,
)
from patio.model.colo_resources import (
    DecisionVariable,
    ExportIncumbentFossil,
    FeStorage,
    LoadIncumbentFossil,
    LoadNewFossil,
    LoadNewFossilWithBackup,
    Storage,
    is_resource_selection,
)

OPT_THREADS = 2


def encode_param(self, _, item):
    k = item.__dict__  # noqa: F841
    shape = item.shape[0] if len(item.shape) == 1 else item.shape
    return {
        "__type__": "cvxpy.Parameter",
        "items": {
            "name": item.name(),
            "shape": shape,
            "value": item.value.tolist() if shape else item.value,
        },
    }


def _encode_pl_df(self, name: str, df: pl.DataFrame, **kwargs) -> dict:
    """Write a polars df in the ZIP as parquet."""
    try:
        df.write_parquet(temp := BytesIO())
    except Exception as exc:
        raise RuntimeError(f"Could not write {name!r} to parquet {exc!r}") from exc
    return {
        "__type__": "plDataFrame",
        "__loc__": self._encode_loc_helper(f"{name}.parquet", df, temp.getvalue()),
    }


def _encode_ndarray(self, name: str, data: np.ndarray, **kwargs) -> dict:
    np.save(temp := BytesIO(), data, allow_pickle=False)
    return {
        "__type__": "ndarray",
        "__loc__": self._encode_loc_helper(f"{name}.npy", data, temp.getvalue()),
    }


DataZip.ENCODERS.update(
    {cp.Parameter: encode_param, pl.DataFrame: _encode_pl_df, np.ndarray: _encode_ndarray}
)
DataZip.DECODERS.update({"cvxpy.Parameter": lambda self, obj: cp.Parameter(**obj["items"])})


def nan_to_none(in_dict: dict) -> dict:
    for k, v in in_dict.items():
        try:
            if isinstance(v, float) and np.isnan(v):
                in_dict[k] = None
        except Exception as exc:
            raise TypeError(f"Failed to convert {k}: {v} to None: {exc!r}") from exc
    return in_dict


RE_NAME = {
    "ba": "ba_code",
    "pid": "icx_id",
    "gens": "icx_gen",
    "tech": "tech",
    "fuel": "fuel",
    "status": "status",
    "cap": "fossil_mw",
}
CONVERTERS = {
    pd.Series: {
        "ba": lambda x: x,
        "pid": lambda x: x,
        "gens": lambda x: tuple(x.split(",")),
        "tech": lambda x: x,
        "fuel": lambda x: x,
        "status": lambda x: x,
        "cap": lambda x: x,
    },
    pl.DataFrame: {
        "ba": lambda x: x[0],
        "pid": lambda x: x[0],
        "gens": lambda x: tuple(x[0].split(",")),
        "tech": lambda x: x[0],
        "fuel": lambda x: x[0],
        "status": lambda x: x[0],
        "cap": lambda x: x[0],
    },
}


class Info(NamedTuple):
    ba: str
    pid: int
    gens: Sequence[str]
    tech: str
    status: str
    cap: float
    fuel: str
    years: tuple[int, int] = ()
    max_re: float = 0.0

    def file(self, regime=None, ix=None, suffix=".json", **kwargs):
        if regime is None and ix is None:
            return f"{self.ba}_{self.pid}_{self.tech}_{self.status}{suffix}"
        return f"{self.ba}_{self.pid}_{self.tech}_{self.status}_{regime}_{ix}{suffix}"

    def extra(self, regime="", name=None, **kwargs):
        return {
            "ba_code": self.ba,
            "plant_id": self.pid,
            "tech": self.tech[:4],
            "status": self.status[:4],
            "regime": regime[:3],
            "config": "" if name is None else name,
        }

    def pixs(self, regime, name, ix, param, **kwargs):
        return {
            "ba_code": self.ba,
            "icx_id": self.pid,
            "icx_gen": ",".join(self.gens),
            "regime": regime,
            "name": name,
            "ix": ix,
        } | param

    def filter(self, **kwargs):
        return reduce(
            operator.and_,
            [pl.col("icx_id") == self.pid, pl.col("icx_gen") == ",".join(self.gens)]
            + [pl.col(k) == v for k, v in kwargs.items()],
        )
        # return (
        #     (pl.col("icx_id") == self.pid)
        #     & (pl.col("icx_gen") == ",".join(self.gens))
        #     & (pl.col("regime") == regime)
        #     & (pl.col("name") == name)
        # )

    @classmethod
    def fix(cls, i: Self) -> Self:
        return cls(*[tuple(k["items"]) if isinstance(k, dict) else k for k in i])

    @singledispatchmethod
    @classmethod
    def from_df(cls, df) -> Self:
        raise NotImplementedError

    @from_df.register
    @classmethod
    def _(cls, df: pd.Series) -> Self:
        converter = CONVERTERS[pd.Series]
        return cls(**{n: converter[n](df[o]) for n, o in RE_NAME.items()})

    @from_df.register
    @classmethod
    def _(cls, df: pd.DataFrame) -> Self:
        converter = CONVERTERS[pd.Series]
        return [
            cls(**{n: converter[n](df_[o]) for n, o in RE_NAME.items()})
            for _, df_ in df.iterrows()
        ]

    @from_df.register
    @classmethod
    def _(cls, df: pl.DataFrame) -> Self | list[Self]:
        converter = CONVERTERS[pd.Series]
        out = [
            cls(**{n: converter[n](df_[o]) for n, o in RE_NAME.items()})
            for df_ in df.iter_rows(named=True)
        ]
        if len(out) == 1:
            return out[0]
        return out


@dataclass(repr=False, frozen=True, slots=True)
class Data:
    re_specs: pl.LazyFrame
    re_pro: pl.LazyFrame
    ba_pro: pl.LazyFrame
    mcoe: pl.DataFrame
    opt_years: list
    re_ids: tuple
    i: Info
    src_path: Path
    ba_data: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert all(
            a == b
            for a, b in zip(
                self.re_pro.collect_schema().names(),
                ["datetime"]
                + self.re_specs.select("combi_id").collect().to_series().to_list(),
                strict=False,
            )
        )
        assert not self.re_pro.null_count().collect().sum_horizontal().item(), (
            "re_pro has null values"
        )

    @classmethod
    def from_dz(cls, src_path, info, re_filter: pl.Expr | None = None):
        with DataZip(src_path / f"{info.ba}_{info.pid}_{info.tech}_{info.status}.zip") as z:
            p_data = {k: v for k, v in z.items()}  # noqa: C416
        re_specs = p_data.get("re", p_data.get("re_specs")).with_columns(
            technology_description=pl.col("re_type").replace(TECH_CODES)
        )
        if re_filter is not None:
            re_specs = re_specs.filter(re_filter)
        re_ids = tuple(re_specs["combi_id"])
        re_pro = p_data.get("re_pro", p_data.get("re_pro"))
        with DataZip(src_path / f"{info.ba}.zip") as z:
            baseline = z["baseline"]
            plant_data = z["plant_data"]
            disp_cost = (
                z["ba_dm_data"]["dispatchable_cost"].cast({"datetime": pl.Datetime()}).lazy()
            )
        re_filter = (
            re_filter.meta.serialize(format="json")
            if isinstance(re_filter, pl.Expr)
            else "no_filter"
        )
        return cls(
            re_specs=re_specs.lazy(),
            re_pro=re_pro.select(pl.col("datetime").cast(pl.Datetime()), *re_ids).lazy(),
            ba_pro=p_data.get("icx_pro", p_data.get("ba_pro"))
            .join(baseline, on="datetime", how="left")
            .cast({"datetime": pl.Datetime()})
            .lazy(),
            mcoe=p_data.get("var_all").cast({"datetime": pl.Datetime()}),
            opt_years=p_data.get("opt_years"),
            re_ids=re_ids,
            i=info,
            src_path=src_path,
            ba_data={"dispatchable_cost": disp_cost, "plant_data": plant_data},
            meta={"re_filter": re_filter},
        )

    def load_ba_data(self, path=None):
        if "ba_dm_data" not in self.ba_data:
            if path is None:
                path = self.src_path / f"{self.i.ba}.zip"
            with DataZip(path) as z:
                b_data = {k: v for k, v in z.items() if k not in self.ba_data}
            b_data.pop("baseline")
            b_data["ba_dm_data"].pop("dispatchable_cost")
            self.ba_data.update(b_data)

    def del_ba_data(self):
        for k in ("ba_dm_data", "fuel_curve"):
            del self.ba_data[k]

    def __getitem__(self, item):
        if item in self.ba_data:
            return self.ba_data[item]
        if item in self.ba_data.get("ba_dm_data", []):
            return self.ba_data["ba_dm_data"][item]
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)  # noqa: B904


class Model(IOMixin):
    cost_cols = [
        "re_site_id",
        "re_type",
        "capex_raw",
        "life_adj",
        "dur",
        "itc_adj",
        "reg_mult",
        "tx_capex_raw",
        "distance",
        "opex_raw",
        "ptc",
        "ptc_gen",
    ]

    def __init__(
        self,
        ix: int,
        name: str,
        i: Info,
        d: Data,
        *,
        life: int,
        build_year: int,
        atb_scenario: str,
        aeo_scenario: str,
        aeo_report_year: int,
        mkt_rev_mult: float = 0.5,
        num_crit_hrs: int = 0,
        max_pct_fos: float = 0.2,
        max_pct_hist_fos: float = 1.0,
        fos_load_cost_mult: float = 1.0,
        gas_window_max_hrs: int = 24,
        gas_window_max_mult: float = 1.0,
        gas_window_seasonal: bool = False,
        stored_fuel_hrs: int = 48,
        backup_max_hrs_per_year: int = 250,
        solar_degrade_per_year: float = 0.0,
        load_icx_max_mult: float = 0.75,
        con_fossil_load_hrly: bool = True,
        regime: Literal["limited", "reference"] = "limited",
        logger: logging.Logger | None = None,
        ptc: float = COSTS["ptc"],
        itc: float = COSTS["itc"],
        errors: list | None = None,
        dvs: dict[str, dict] | None = None,
        result_dir: Path | None = None,
        pudl_release: str = PATIO_PUDL_RELEASE,
    ):
        self.ix = ix
        self.name = name
        self.i: Info = i
        self.d: Data = d
        self.life = life
        self.build_year = build_year
        self.atb_scenario = atb_scenario
        self.aeo_report_year = aeo_report_year
        self.aeo_scenario = aeo_scenario
        self.pudl_release = pudl_release
        self._params = {
            "mkt_rev_mult": mkt_rev_mult,
            "max_pct_fos": cp.Parameter(name="max_pct_fos", value=max_pct_fos),
            "max_pct_hist_fos": cp.Parameter(name="max_pct_hist_fos", value=max_pct_hist_fos),
            "fos_load_cost_mult": fos_load_cost_mult,
            "regime": regime,
            "num_crit_hrs": None,
            "gas_window_max_hrs": gas_window_max_hrs,
            "stored_fuel_hrs": stored_fuel_hrs,
            "backup_max_hrs_per_year": backup_max_hrs_per_year,
            "gas_window_max_mult": gas_window_max_mult,
            "solar_degrade_per_year": solar_degrade_per_year,
            "load_icx_max_mult": load_icx_max_mult,
            "gas_window_seasonal": gas_window_seasonal,
            "con_fossil_load_hrly": con_fossil_load_hrly,
        }
        self.result_dir = result_dir
        self.ptc = ptc
        self.itc = itc
        self.ptc_years = self.d.opt_years[:10]
        self.r_disc = ((1 + COSTS["discount"]) / (1 + COSTS["inflation"])) - 1
        self.fcr = self.r_disc / (1 - (1 + self.r_disc) ** -self.life)
        max_opt, len_opt = max(self.d.opt_years) + 1, len(self.d.opt_years)
        self.yr_mapper = {
            y: self.d.opt_years[-(4 - i % 4)]
            for i, y in enumerate(range(max_opt, max_opt + self.life - len_opt))
        }
        cost_years = self.d.opt_years + list(self.yr_mapper.values())
        assert len(cost_years) == self.life, "remap of years to full life of project failed"

        # this export requirement incorporates both historical and baseline dispatch
        # it makes no sense (and screws up the LP) for there to be an export requirement
        # and curtailment at the same time
        self.export_profs = self.d.ba_pro.with_columns(
            export_requirement=pl.min_horizontal(
                self.i.cap, pl.max_horizontal("icx", "icx_hist")
            ),
        ).select(
            "datetime",
            "export_requirement",
            curtailment_pct=pl.when(pl.col("export_requirement") == 0)
            .then(pl.col("baseline_sys_curtailment_pct"))
            .otherwise(0.0)
            .fill_null(0.0),
        )
        self.re_land = make_core_lhs_rhs(
            self.d.re_specs.collect().to_pandas(), limited=True, fossil=False
        )
        self.b_land = cp.Parameter(
            len(self.re_land),
            name="b_land",
            value=self.re_land.loc[
                :, {"limited": "rhs_lim", "reference": "rhs"}[self.regime]
            ].to_numpy(nt),
        )
        self.errors: list[str] = [] if errors is None else errors
        if logger is None:
            logger = logging.getLogger("patio")
        self.logger = logger

        self.dvs: dict[str, DecisionVariable] = {}
        self.selected: None | cp.Problem = None
        self.dispatchs: dict[int, cp.Problem] = {}
        self.results: dict[tuple, dict] = {}
        self.solver_logs: dict[tuple, str] = {}
        self.update_years()
        # update max_re
        x = cp.Variable(len(self.d.re_ids), name="re")
        p = cp.Problem(
            cp.Maximize(cp.sum(x)),
            [self.re_land.to_numpy()[:, :-2] @ x <= self.b_land.value, x >= 0.0],
        )
        self.i = self.i._replace(max_re=p.solve().item())
        self._crit_hrs = {self.i.years: cp.Parameter(8760 * 2, name="crit_hrs_select")} | {
            (yr,): cp.Parameter(8760, name=f"crit_hrs_{yr}") for yr in self.d.opt_years
        }
        self.num_crit_hrs = num_crit_hrs
        self._state = {}  #  {"dvs": dvs}
        if dvs is not None:
            self.d.load_ba_data()
            self.add_dvs_from_config_dict(dvs)
        self._dfs: dict[str, pl.DataFrame] = {}
        self._out_result_dict: dict[str, float | int | str | list] = {}
        self.status = INFEASIBLE

    def add_dvs_from_config_dict(self, dv_dict: dict[str, dict]):
        for dv_n, params in dv_dict.items():
            getattr(colo_res_mod, dv_n)(self, **params)
        return None

    def add_dv(self, dv: DecisionVariable):
        if dv.cat in self.dvs:
            self.dvs[dv.cat + "1"] = dv
        else:
            self.dvs[dv.cat] = dv

    @property
    def max_pct_fos(self):
        return self._params["max_pct_fos"]

    @max_pct_fos.setter
    def max_pct_fos(self, value):
        self._params["max_pct_fos"].value = value

    @property
    def max_pct_hist_fos(self):
        return self._params["max_pct_hist_fos"]

    @max_pct_hist_fos.setter
    def max_pct_hist_fos(self, value):
        self._params["max_pct_hist_fos"].value = value

    @property
    def fos_load_cost_mult(self):
        return self._params["fos_load_cost_mult"]

    @property
    def mkt_rev_mult(self):
        return self._params["mkt_rev_mult"]

    @property
    def regime(self):
        return self._params["regime"]

    @regime.setter
    def regime(self, value: Literal["reference", "limited"]):
        if value != self.regime:
            self.b_land.value = self.re_land.loc[
                :, {"limited": "rhs_lim", "reference": "rhs"}[value]
            ].to_numpy(nt)
            self._params["regime"] = value
            self.update_years()
            self.propagate_new_selection_years()

    @property
    def num_crit_hrs(self):
        return self._params["num_crit_hrs"]

    @num_crit_hrs.setter
    def num_crit_hrs(self, value: int):
        if value != self.num_crit_hrs:
            if self.num_crit_hrs is not None:
                self.logger.warning(
                    "num_crit_hrs not a parameter, changing it deletes previous solutions same as full reset",
                    extra=self.extra,
                )
            for yr in self._crit_hrs:
                self._crit_hrs[yr].value = self.mk_crit_hrs_mask(value, yr)
            self._params["num_crit_hrs"] = value
            self.selected = None
            self.dispatchs = {}

    @property
    def gas_window_max_hrs(self):
        return self._params["gas_window_max_hrs"]

    @property
    def gas_window_max_mult(self):
        return self._params["gas_window_max_mult"]

    @property
    def stored_fuel_hrs(self):
        return self._params["stored_fuel_hrs"]

    @property
    def backup_max_hrs_per_year(self):
        return self._params["backup_max_hrs_per_year"]

    @property
    def solar_degrade_per_year(self):
        return self._params["solar_degrade_per_year"]

    @property
    def load_icx_max_mult(self):
        return self._params["load_icx_max_mult"]

    @property
    def gas_window_seasonal(self):
        return self._params["gas_window_seasonal"]

    @property
    def con_fossil_load_hrly(self):
        return self._params["con_fossil_load_hrly"]

    @cached_property
    def fuels(self):
        return {f for dv in self for f in dv.fuels}

    def mk_crit_hrs_mask(self, nh, yr):
        crit = lambda co: pl.col(co).rank(descending=True).over(pl.col("datetime").dt.year())  # noqa: E731
        return (
            prof(self.d.ba_pro, yr, cs.all)
            .select((crit("baseline_sys_net_load") <= nh) | (crit("baseline_sys_load") <= nh))
            .cast(pl.Int32)
            .collect()
            .to_series()
            .to_numpy()
        )

    def update_params(self, params, regime=None, ix=None, name=None, **kwargs):
        for k, v in params.items():
            if isinstance(getattr(self, k), cp.Parameter) or k == "regime":
                setattr(self, k, v)
        if regime is not None:
            self.regime = regime
        if ix is not None:
            self.ix = ix
        if name is not None:
            self.name = name
        self._dfs = {}
        self._out_result_dict = {}
        self.status = INFEASIBLE

    def propagate_new_selection_years(self):
        old_yrs = next(filter(lambda x: len(x) == 2, self._crit_hrs))
        if self.i.years != old_yrs:
            if (param := self._crit_hrs.pop(old_yrs)) is not None:
                param.value = self.mk_crit_hrs_mask(self.num_crit_hrs, self.i.years)
                self._crit_hrs = {self.i.years: param, **self._crit_hrs}
            for dv in self.dvs:
                for attr_n in ("cost_cap", "cost", "x"):
                    attr = getattr(self[dv], attr_n)
                    if (og := attr.pop(old_yrs, None)) is not None:
                        setattr(self[dv], attr_n, {self.i.years: og, **attr})
                if hasattr(dv, "yr_fact_map"):
                    self[dv].yr_fact_map = dict(
                        zip(self.i.years, self[dv].yr_fact_map.values(), strict=False)
                    )

    @property
    def load_mw(self) -> float | None:
        if self["load"].x_cap.value is None:
            return None
        if self.status == FAIL_SMALL:
            return 50
        return self["load"].x_cap.value.item()

    def update_years(self):
        renew = self.d.re_pro.filter(pl.col("datetime").dt.year() >= 2028).collect()
        r_var = cp.Variable(renew.shape[1] - 1, name="r_var")
        cons = [
            self.re_land.to_numpy(nt)[:, :-2] @ r_var <= self.b_land,
            r_var >= 0.0,
            r_var <= 2e4,
        ]
        result = cp.Problem(
            cp.Maximize(cp.sum(renew.select(cs.numeric()).sum().to_numpy() @ r_var)), cons
        )
        result.solve(cp.HIGHS)
        new_years: tuple[int, ...] = tuple(
            pl.LazyFrame(
                [renew["datetime"], pl.Series("re", renew.select(cs.numeric()) @ r_var.value)]
            )
            .rolling("datetime", period="7d")
            .agg(pl.sum("re"))
            .with_row_index()
            .filter(c.index > 7 * 24 - 1)
            .group_by_dynamic("datetime", every="1y")
            .agg(pl.min("re"))
            .sort("re")
            .collect()["datetime"]
            .dt.year()[:2]
        )
        self.logger.info(
            "replacing %s selection years with %s", self.i.years, new_years, extra=self.extra
        )
        self.i = self.i._replace(years=new_years)

    def pre_check(self, load):
        renew = self["renewables"]
        yr, nh = (0, 1), 0
        cons = [
            renew.annual_gen()
            >= self.d.re_pro.filter(pl.col("datetime").dt.year() >= 2024)
            .group_by_dynamic("datetime", every="1y")
            .agg(cs.last().count())
            .collect()
            .to_numpy()[:, 1]
            * load
            * (1 - self.max_pct_fos),
            renew.land(yr, nh)
            <= self.b_land
            - hstack([load * self["load"].sqkm_per_mw], np.zeros(len(self.re_land) - 1)),
            *renew.bounds(yr, nh),
        ]
        result = cp.Problem(cp.Minimize(cp.sum(renew.x_cap)), cons)
        result.solve(cp.HIGHS)
        if result.status != cp.OPTIMAL:
            self.errors.append(f"Insufficient local renewables load=50 {result.status}")
            self.logger.info(self.errors[-1], extra=self.extra)
            self.status = FAIL_SMALL
        return self

    def select_resources(self, time_limit=1000):
        if self.selected is None:
            self.selected = cp.Problem(
                self.objective(self.i.years), self.constraints(self.i.years)
            )
        solver_ = solver()
        options = {
            cp.GUROBI: {
                "TimeLimit": time_limit,
                "Threads": OPT_THREADS,
                "LogToConsole": 1,
                "reoptimize": True,
            },
            cp.HIGHS: {"time_limit": time_limit, "parallel": "on"},
            cp.COPT: {"TimeLimit": time_limit, "LpMethod": 4},
        }[solver_]
        with (
            capture_stdout() as c_out,
            capture_stderr() as err,
            catch_warnings(record=True) as w_list,
        ):
            try:
                self.selected.solve(solver_, **options, verbose=True)
                if self.selected.status != cp.OPTIMAL and solver_ == cp.GUROBI:
                    self.selected.solve(solver_, **options, verbose=True, BarHomogeneous=1)

            except (cp.SolverError, ValueError) as exc:
                self.errors.append(
                    repr(exc)
                    + " "
                    + "\n".join(
                        f"{''.join(w.filename.partition('cvxpy')[1:])}:{w.lineno} {w.message!r}"
                        for w in w_list
                    )
                )
            if to_warn := [
                f"{''.join(w.filename.partition('cvxpy')[1:])}:{w.lineno} {w.message!r}"
                for w in w_list
                if "cvxpy" in w.filename
                and not w.message.args[0].startswith("Your problem has too many parameters")
            ]:
                self.logger.warning("%s", "\n".join(to_warn), extra=self.extra)

        try:
            self.solver_logs[self.i.years] = "\n".join((c_out.getvalue(), err.getvalue()))
        finally:
            c_out.close()
            err.close()
        self.logger.debug("\n %s", self.solver_logs[self.i.years], extra=self.extra)
        if self.selected.status != cp.OPTIMAL:
            self.errors.append(f"Sizing failed {self.selected.status}")
            self.logger.info(self.errors[-1], extra=self.extra)
            self.status = FAIL_SELECT
            return self
        if self.load_mw < 50:
            self.errors.append(f"Sizing failed to serve ≥50MW {self.load_mw=:.2f}")
            self.logger.info(self.errors[-1], extra=self.extra)
            self.status = FAIL_SMALL
            return self
        self.status = SUCCESS
        return self

    def round(self):
        for dv in self:
            dv.round()

    def dispatch(self, yr, time_limit=1000):
        yrt = (yr,)
        if yr not in self.dispatchs:
            self.dispatchs[yr] = cp.Problem(self.objective(yrt), self.constraints(yrt))
        solver_ = solver()
        options = {
            cp.GUROBI: {
                "TimeLimit": time_limit,
                "Threads": OPT_THREADS,
                "LogToConsole": 1,
                "reoptimize": True,
            },
            cp.HIGHS: {"time_limit": time_limit, "parallel": "on"},
            cp.COPT: {"TimeLimit": time_limit},
        }[solver_]

        with capture_stdout() as c_out, capture_stderr() as c_err:
            try:
                self.dispatchs[yr].solve(solver_, **options, verbose=True)
                if self.dispatchs[yr].status != cp.OPTIMAL and solver_ == cp.GUROBI:
                    self.dispatchs[yr].solve(
                        solver_, **options, verbose=True, BarHomogeneous=1
                    )
            except cp.SolverError as exc:
                self.errors.append(repr(exc))
        try:
            self.solver_logs[yr] = "\n".join((c_out.getvalue(), c_err.getvalue()))

        finally:
            c_out.close()
            c_err.close()
        if self.dispatchs[yr].status != cp.OPTIMAL:
            self.errors.append(
                f"{yr} dispatch failed {self.load_mw=:.2f} {self.dispatchs[yr].status}"
            )
            self.logger.info(self.errors[-1], extra=self.extra)
            self.status = FAIL_DISPATCH
            self.logger.debug(
                "DISPATCH FAILED: %s\n %s", yr, self.solver_logs[yr], extra=self.extra
            )
        return self

    def dispatch_all(self, time_limit):
        if not any(dv.x_cap.value for dv in self.dvs.values() if dv.x_cap is not None):
            raise RuntimeError("must run select_resources first or use Model.from_run")
        for yr in self.d.opt_years:
            self.dispatch(yr, time_limit)
            if self.status == FAIL_DISPATCH:
                break
        return self

    def objective(self, yr: tuple) -> cp.Minimize:
        if is_resource_selection(yr):
            in_sum = (1 + COSTS["inflation"]) / (1 + COSTS["discount"])
            # the first data year represents the first 10 years with PTC, the remaining
            # data years represent the next life - 10 years without PTC with those remaining
            # years split as evenly as possible across the data years not used for PTC
            # years, i.e. all but the first data year
            yr_fact_map = {yr[0]: np.sum(in_sum ** np.arange(10))}
            no_ptc_incr = int((self.life - 10) / (len(yr) - 1))
            start = 10
            for y in yr[1:]:
                stop = start + no_ptc_incr if y != yr[-1] else self.life
                yr_fact_map[y] = np.sum(in_sum ** np.arange(start, stop))
                start = stop
        else:
            yr_fact_map = {}
        return cp.Minimize(cp.sum([dv.objective(yr, yr_fact_map) for dv in self]))

    def constraints(self, yr: tuple) -> list[cp.Constraint]:
        cons = [
            cp.sum([dv.load(yr) for dv in self]) == 0.0,
            cp.sum([dv.clean_export(yr) for dv in self]) >= 0.0,
            cp.sum([dv.fossil_load(yr) for dv in self]) <= 0.0,
            cp.sum([dv.icx_ops(yr) for dv in self]) <= self.i.cap,
            cp.sum([dv.incumbent_ops(yr) for dv in self]) <= self.i.cap,
            cp.sum([dv.land(yr) for dv in self]) <= self.b_land,
            cp.sum([dv.export_req(yr) for dv in self]) >= self.b_exp_req(yr),
        ]
        if "natural_gas" in self.fuels and self.i.fuel == "natural_gas":
            cons.append(
                cp.sum([dv.gas_window_max(yr) for dv in self]) <= self.b_rolling_gas(yr)
            )
        if self.con_fossil_load_hrly:
            cons.append(cp.sum([dv.fossil_load_hrly(yr) for dv in self]) <= 0.0)
        if not is_resource_selection(yr) and "distillate_fuel_oil" in self.fuels:
            cons.extend(
                [
                    cp.sum([dv.stored_fuel(yr) for dv in self])
                    <= self.stored_fuel_hrs * self.load_mw,
                    cp.sum([dv.backup_annual(yr) for dv in self])
                    <= self.backup_max_hrs_per_year * self.load_mw,
                ]
            )
        if LoadIncumbentFossil in self and self.num_crit_hrs > 0:
            cons.append(
                cp.multiply(self._crit_hrs[yr].value, cp.sum([dv.critical(yr) for dv in self]))
                >= 0.0
            )
        if ExportIncumbentFossil in self:
            cons.append(cp.sum([dv.fossil_hist(yr) for dv in self]) <= self.b_fos_hist(yr))
        soc_ixs = [0] + prof(self.d.ba_pro, yr, cs.datetime).with_row_index().filter(
            (c.datetime.dt.month() == 12)
            & (c.datetime.dt.day() == 31)
            & (c.datetime.dt.hour() == 23)
        ).collect()["index"].to_list()
        for dv in self:
            cons.extend(dv.single_dv(yr, soc_ixs))
            cons.extend(dv.bounds(yr))
        return cons

    def b_exp_req(self, yr):
        return (
            prof(self.export_profs, yr, cs.by_name("export_requirement"))
            .collect()
            .to_series()
            .to_numpy()
        )

    def b_fos_hist(self, yr):
        return self.max_pct_hist_fos * max(
            self.d.ba_pro.group_by_dynamic("datetime", every="1y")
            .agg(pl.sum("icx_hist"))
            .select("icx_hist")
            .median()
            .collect()
            .item(),
            prof(self.d.ba_pro, yr).select("icx_hist").sum().collect().item(),
        )

    def b_rolling_gas(self, yr):
        out = (
            self.export_profs.with_columns(
                rmax=pl.sum("export_requirement").rolling(
                    "datetime", period=f"{self.gas_window_max_hrs}h"
                ),
            )
            .select(
                pl.max("rmax").over(pl.col("datetime").dt.month()) * self.gas_window_max_mult
            )
            .head(8760 * len(yr))
            .collect()
            .to_series()
            .to_numpy()
        )
        if self.gas_window_seasonal:
            return out
        return max(out)

    def check_service(self):
        hrly_check = (
            self.hourly()
            .with_columns(
                src=pl.sum_horizontal(
                    cs.by_name(
                        *self.re_types,
                        "es",
                        "export_fossil",
                        "load_fossil",
                        "backup",
                        require_all=False,
                    )
                ),
                disp=pl.sum_horizontal("export_requirement", "load"),
            )
            .with_columns(miss=pl.col("src") - pl.col("disp"))
            .filter(pl.col("miss") < -1)
            .collect()
        )

        if not hrly_check.is_empty():
            pct_hrs = len(hrly_check) / (len(self.d.opt_years) * 8760)
            miss_pct = -(
                hrly_check.select("miss").sum()
                / self.hourly()
                .select(pl.sum_horizontal("load", "export_requirement").sum())
                .collect()
            ).item()
            self.add_to_result_dict(
                served_pct=1 - miss_pct,
                unserved_mwh=abs(hrly_check.select("miss").sum().item()),
                unserved_pct_hrs=pct_hrs,
                max_unserved_mw=abs(hrly_check.select("miss").min().item()),
            )
            self.errors.append(
                f"{miss_pct:.6%} of load + export requirement not served across {pct_hrs:.6%} of hours, max miss is {-hrly_check.select('miss').min().item():.0f}"
            )
            self.status = FAIL_SERVICE

    @cached_property
    def re_types(self):
        return list(
            {"solar", "onshore_wind", "offshore_wind"}.intersection(
                set(self.d.re_specs.select("re_type").unique().collect().to_series())
            )
        )

    @cached_property
    def es_types(self):
        return [x._type for x in self if isinstance(x, Storage)]

    @cached_property
    def re_selected(self):
        return (
            self.d.re_specs.with_columns(capacity_mw=self["renewables"].x_cap.value)
            .filter(pl.col("capacity_mw") > 0)
            .collect()
        )

    def hourly(self, *, selection=False) -> pl.LazyFrame:
        result = (
            pl.concat(
                (dv.hourly(selection=selection) for dv in self),
                how="align",
            )
            .join(
                self.export_profs.rename({"curtailment_pct": "assumed_curtailment_pct"}),
                on="datetime",
                how="left",
                maintain_order="left",
            )
            .join(
                self.d.ba_pro.select(
                    "datetime",
                    pl.col("icx_hist").alias("historical_fossil"),
                    pl.col("icx").alias("baseline_export_fossil"),
                    "baseline_sys_load",
                    # "baseline_sys_deficit",
                    # 'baseline_sys_curtailment',
                    "baseline_sys_curtailment_pct",
                    "baseline_sys_lambda",
                    # 'baseline_sys_net_load',
                    "baseline_sys_co2",
                    "baseline_sys_cost",
                    # "baseline_sys_dispatchable",
                    # "baseline_sys_renewable",
                    # "baseline_sys_storage",
                ),
                on="datetime",
                how="left",
                validate="1:1",
                maintain_order="left",
            )
            .sort("datetime")
        )
        cols = result.collect_schema().names()
        if "load_fossil" not in cols:
            result = result.with_columns(
                load_fossil=pl.lit(0.0), c_load_fossil=pl.lit(0.0), load_fossil_co2=pl.lit(0.0)
            )
        if "load_stored_fuel" not in cols:
            result = result.with_columns(
                load_stored_fuel=pl.lit(0.0), c_load_stored_fuel=pl.lit(0.0)
            )
        result = result.with_columns(
            load_fossil_co2=pl.sum_horizontal("load_fossil_co2"),
            es=pl.sum_horizontal(cs.contains("discharge"))
            - pl.sum_horizontal(cs.contains("_charge")),
            fossil=pl.sum_horizontal("export_fossil", "load_fossil"),
            export=pl.sum_horizontal("export_clean", "export_fossil"),
            stored_fuel=pl.sum_horizontal("load_stored_fuel"),
        ).with_columns(
            gas_fossil=pl.col("fossil") - pl.col("stored_fuel"),
            clean_for_load=pl.min_horizontal("load", pl.sum_horizontal(*self.re_types, "es")),
        )
        if selection:
            return result.with_columns(critical_hour=self._crit_hrs[self.i.years].value)
        return (
            result.with_columns(
                critical_hour=hstack(
                    v.value for k, v in self._crit_hrs.items() if k != self.i.years
                ),
                _export_clean=pl.max_horizontal(
                    pl.min_horizontal(
                        pl.sum_horizontal(*self.re_types, "es") - pl.col("load"), self.i.cap
                    ),
                    0.0,
                ),
                _export=pl.min_horizontal(
                    pl.sum_horizontal(*self.re_types, "es", "export_fossil", "load_fossil")
                    - pl.col("load"),
                    self.i.cap,
                ),
            )
            .with_columns(
                _export_fossil=pl.col("_export") - pl.col("_export_clean"),
                _curtailment=pl.sum_horizontal(*self.re_types, "es", "fossil")
                - pl.col("load")
                - pl.col("_export"),
            )
            .with_columns(
                _load_fossil=pl.col("fossil") - pl.col("_export_fossil"),
                baseline_sys_load_net_of_clean_export=pl.col("baseline_sys_load")
                - pl.col("export_clean"),
            )
            .with_columns(
                _hr_mkt_rev=c.baseline_sys_lambda
                * self.mkt_rev_mult
                * (1 - c.assumed_curtailment_pct),
            )
            .with_columns(
                _ptc=pl.when(c.datetime.dt.year().is_in(self.ptc_years))
                .then(self.ptc)
                .otherwise(0.0),
                rev_clean=c.export_clean * -c.c_export_clean,
                rev_fossil=c.export_fossil * c._hr_mkt_rev,
            )
            .with_columns(
                baseline_rev_fossil=c.baseline_export_fossil * c._hr_mkt_rev,
                cost_fossil=pl.sum_horizontal(
                    pl.col("gas_fossil") * pl.col("c_export_fossil"),
                    pl.col("stored_fuel") * pl.col("c_load_stored_fuel"),
                ),
                cost_load_fossil=pl.sum_horizontal(
                    (pl.col("load_fossil") - pl.col("stored_fuel")) * pl.col("c_load_fossil"),
                    pl.col("stored_fuel") * pl.col("c_load_stored_fuel"),
                ),
                cost_export_fossil=c.export_fossil * c.c_export_fossil,
                cost_curt=c.curtailment * c.c_curtailment
                + c.export_clean * c.assumed_curtailment_pct * c._ptc,
                starts=((c.fossil != 0) & (c.fossil.shift(1) == 0))
                .cast(pl.Int64)
                .fill_null(0),
            )
        )

    def finance_df(self, hourly: pl.LazyFrame | pl.DataFrame | None = None) -> pl.LazyFrame:
        core_cols = (
            "datetime",
            "type",
            "capacity_mw",
            "capex_gross",
            "itc",
            "tx_capex",
            "fom",
            "ptc",
        )
        cost_summary = (
            pl.concat([dv.cost_summary(every="1y") for dv in self], how="diagonal_relaxed")
            .with_columns(
                type=pl.col("type").replace({"fossil": "load_fossil", "rice": "load_fossil"}),
                hly=pl.sum_horizontal(~cs.by_name(*core_cols, require_all=False)),
            )
            .with_columns(
                cost=pl.max_horizontal(0.0, pl.col("hly")),
                rev=-pl.min_horizontal(0.0, pl.col("hly")),
            )
            .select(cs.by_name(*core_cols, "cost", "rev", require_all=False))
            .select(~cs.starts_with("capacity_mw"))
            .sort("type", "datetime")
            .pivot(on=["type"], index="datetime")
            .pipe(add_missing_col, "cost_load_fossil")
            .pipe(self.add_mapped_yrs)
            .rename({"rev_export": "rev_clean", "cost_curtailment": "cost_curt"})
            .with_columns(
                cost_fossil=pl.sum_horizontal("cost_export_fossil", "cost_load_fossil")
            )
            .with_columns(cs.by_dtype(pl.Float64).fill_null(0.0))
            .pipe(
                keep_nonzero_cols,
                keep=[
                    "cost_curt",
                    "cost_load_fossil",
                    "cost_export_fossil",
                    "cost_fossil",
                    *[f"ptc_{t}" for t in self.re_types],
                ],
            )
            .lazy()
        )
        if hourly is None:
            hourly = self.hourly().pipe(self.add_mapped_yrs)
        hourly = hourly.lazy()
        if any("redispatch" in co for co in hourly.collect_schema().names()):
            col0 = {
                "redispatch_ptc": pl.sum_horizontal(cs.starts_with("ptc_"))
                - c.redispatch_cost_curt
            }
            col1 = {
                "redispatch_cost": pl.sum_horizontal("cost_clean", "redispatch_cost_fossil")
                - c.redispatch_ptc
            }
        else:
            col0, col1 = {}, {}
        return (
            cost_summary.join(
                hourly.group_by_dynamic("datetime", every="1y").agg(
                    cs.by_name(
                        "rev_fossil",
                        "baseline_rev_fossil",
                        "baseline_sys_cost",
                        "redispatch_rev_clean",
                        "redispatch_rev_fossil",
                        "redispatch_cost_fossil",
                        "redispatch_cost_curt",
                        "redispatch_sys_cost",
                        "redispatch_cost_export_fossil",
                        require_all=False,
                    ).sum()
                ),
                on="datetime",
                how="left",
            )
            .sort("datetime")
            .with_columns(
                year=pl.col("datetime").dt.year(),
                fin_disc=np.reshape(
                    ((1 + COSTS["inflation"]) / (1 + COSTS["discount"]))
                    ** np.arange(self.life),
                    (self.life,),
                ),
            )
            .with_columns(
                cost_clean=pl.sum_horizontal(cs.contains("capex") | cs.contains("itc"))
                + pl.when(pl.col("datetime") == pl.col("datetime").min())
                .then((pl.sum_horizontal(cs.starts_with("fom_")) * pl.col("fin_disc")).sum())
                .otherwise(0.0),
                full_ptc=pl.sum_horizontal(cs.starts_with("ptc_")),
                ptc=pl.sum_horizontal(cs.starts_with("ptc_")) - c.cost_curt,
                **col0,
            )
            .with_columns(cost=pl.sum_horizontal("cost_clean", "cost_fossil") - c.ptc, **col1)
            .with_columns(func=c.cost - c.rev_clean)
            .pipe(
                lambda df: df.select(
                    list(dict.fromkeys(["year"] + sorted(df.collect_schema().names())))
                )
            )
        )

    def energy_df(self, hourly: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if hourly is None:
            hourly = self.hourly().pipe(self.add_mapped_yrs)
        hourly = hourly.lazy()
        if any("redispatch" in co for co in hourly.collect_schema().names()):
            red_cols = [
                "redispatch_export_fossil",
                "redispatch_sys_deficit",
                "redispatch_sys_co2",
            ]
        else:
            red_cols = []
        return (
            hourly.lazy()
            .group_by(pl.col("datetime").dt.year().alias("year"))
            .agg(
                cs.by_name(
                    "load",
                    "fossil",
                    "export",
                    "curtailment",
                    "export_clean",
                    "export_fossil",
                    "load_fossil",
                    "load_stored_fuel",
                    "historical_fossil",
                    "baseline_export_fossil",
                    "baseline_sys_deficit",
                    "baseline_sys_co2",
                    "baseline_sys_load",
                    "baseline_sys_load_net_of_clean_export",
                    *red_cols,
                    require_all=False,
                ).sum()
            )
            .sort("year")
            .with_columns(
                phys_disc=np.reshape(
                    (1 / (1 + COSTS["discount"])) ** np.arange(self.life), (self.life,)
                )
            )
            .pipe(
                lambda df: df.select(
                    list(dict.fromkeys(["year"] + sorted(df.collect_schema().names())))
                )
            )
        )

    def gas_violation_df[T: pl.LazyFrame | pl.DataFrame](self, hourly: T) -> T:
        return (
            hourly.with_columns(
                rolling_max=pl.sum("export_requirement").rolling(
                    "datetime", period=f"{self.gas_window_max_hrs}h"
                )
            )
            .with_columns(
                rolling_max=pl.max("rolling_max").over(pl.col("datetime").dt.month()),
                gas_use=pl.sum_horizontal("export_fossil", "load_fossil")
                - pl.sum_horizontal(
                    cs.by_name("load_stored_fuel", "backup", require_all=False)
                ),
            )
            .with_columns(
                capacity_vio=pl.max_horizontal(pl.col("gas_use") - self.i.cap, 0.0),
                rolling_vio=pl.max_horizontal(
                    pl.col("gas_use")
                    .sum()
                    .rolling("datetime", period=f"{self.gas_window_max_hrs}h")
                    - pl.col("rolling_max"),
                    0.0,
                ),
            )
            .with_columns(
                rolling_rel=pl.col("rolling_vio") / pl.col("rolling_max"),
                capacity_rel=pl.col("capacity_vio") / self.i.cap,
                run_num=((pl.col("rolling_vio").shift(1) == 0) & (pl.col("rolling_vio") != 0))
                .fill_null(True)
                .cum_sum(),
            )
            .with_columns(
                not_largest_violation=pl.sum("rolling_vio").over("run_num"),
                longest_violation=pl.col("rolling_vio")
                .replace({0: None})
                .count()
                .over("run_num"),
            )
            .select(
                pl.max("not_largest_violation"),
                pl.max("longest_violation"),
                total_capacity_violation=pl.sum("capacity_vio"),
                total_rolling_violation=pl.sum("rolling_vio"),
                max_capacity_violation_pct=pl.max("capacity_rel"),
                max_rolling_violation_pct=pl.max("rolling_rel"),
            )
        )

    @property
    def selected_cost(self):
        return pl.concat(
            [
                v.cost_cap[self.i.years].with_columns(capacity_mw=v.x_cap.value)
                for v in self
                if v.cost_cap
            ],
            how="diagonal_relaxed",
        )

    def cost_detail(self):
        return pl.concat(
            [
                self.selected_cost.insert_column(
                    0, pl.lit(" ".join(map(str, self.i.years))).alias("year")
                )
            ]
            + [
                pl.concat(
                    [
                        v.cost_cap[(yr,)].with_columns(capacity_mw=v.x_cap.value)
                        for v in self
                        if v.cost_cap
                    ],
                    how="vertical_relaxed",
                ).insert_column(0, pl.lit(str(yr)).alias("year"))
                for yr in self.d.opt_years
            ],
            how="diagonal_relaxed",
        )

    def create_df_for_econ_model(
        self, flows_hourly: pl.LazyFrame, hourly: pl.LazyFrame
    ) -> pl.LazyFrame:
        re = (
            self["renewables"]
            .hourly(by_type=False)
            .pipe(self.add_mapped_yrs)
            .unpivot(
                cs.numeric(),
                index="datetime",
                variable_name="combi_id",
                value_name="redispatch_mwh",
            )
            .join(self.re_selected.lazy(), on="combi_id", how="inner", validate="m:1")
            .select(
                pl.col("re_site_id").alias("plant_id_eia"),
                pl.col("generator_id").alias("source"),
                "datetime",
                "redispatch_mwh",
            )
        )
        costs = (
            self.d.ba_data["dispatchable_cost"]
            .filter(
                (pl.col("datetime").dt.year() >= self.d.opt_years[0])
                & (c.plant_id_eia == self.i.pid)
                & c.generator_id.is_in(self.i.gens)
            )
            .group_by("datetime", "plant_id_eia")
            .agg(cs.numeric().mean())
            .with_columns(
                datetime=pl.col("datetime").cast(pl.Datetime()), source=pl.lit("fossil")
            )
            .collect()
            .join(
                flows_hourly.group_by_dynamic("datetime", every="1mo")
                .agg(pl.sum("starts"))
                .collect(),
                on="datetime",
                how="left",
                validate="1:1",
            )
            .pipe(self.add_mapped_yrs)
            .sort("datetime")
            .lazy()
        )
        if LoadNewFossil in self:
            new_fos_tech = self["fossil"].tech
            mcoe = (
                self["fossil"]
                .hourly()
                .with_columns(
                    cost=pl.col("load_fossil")
                    * pl.col("c_load_fossil")
                    / self.fos_load_cost_mult
                )
                .group_by(pl.col("datetime").dt.year())
                .agg(pl.col("cost").sum())
                .collect()
            )
            new_fos_specs = [
                self["fossil"]
                .cost_cap[(y,)]
                .with_columns(
                    datetime=pl.lit(f"{y}-01-01").str.to_datetime(),
                    capacity_mw=pl.lit(self["fossil"].x_cap.value.item()),
                    mcoe=pl.lit(mcoe.filter(pl.col("datetime") == y)["cost"].item()),
                )
                .lazy()
                for y in self.d.opt_years
            ]
        else:
            new_fos_tech = None
            new_fos_specs = []

        # costs = []
        # for dv in self:
        #     a = dv.cost_summary(
        #         every="1y",
        #         cap_grp=[
        #             pl.col("re_site_id").alias("plant_id_eia"),
        #             pl.col("re_type").alias("type"),
        #             "generator_id",
        #         ],
        #         h_grp=[
        #             pl.col("re_site_id").alias("plant_id_eia"),
        #             pl.col("re_type").alias("type"),
        #         ],
        #     )
        #     if a.is_empty():
        #         a = dv.cost_summary(every="1y").with_columns(
        #             plant_id_eia=pl.lit(0), generator_id=pl.lit(dv.cat)
        #         )
        #     costs.append(a)
        #
        # costs = pl.concat(costs, how="diagonal_relaxed")

        re_specs = pl.concat(
            [
                pl.concat(
                    [
                        self["renewables"]
                        .cost_cap[(y,)]
                        .with_columns(datetime=pl.datetime(y, 1, 1), mcoe=pl.lit(None))
                        .lazy()
                        .join(
                            self.re_selected.lazy(), on=["re_site_id", "re_type"], how="right"
                        )
                        for y in self.d.opt_years
                    ]
                    + [x for t in self.es_types for x in self[f"{t}_storage"].for_econ()]
                    + new_fos_specs,
                    how="diagonal_relaxed",
                )
                .pipe(self.add_mapped_yrs)
                .with_columns(
                    fom_per_kw=pl.col("opex_raw") / 1000,
                    plant_id_eia=pl.col("re_site_id"),
                    category=pl.lit("colo_clean"),
                )
                .rename(
                    {"latitude_nrel_site": "latitude", "longitude_nrel_site": "longitude"}
                ),
            ],
            how="diagonal_relaxed",
        )

        specs = (
            pl.from_pandas(self.d["plant_data"])
            .filter((c.plant_id_eia == self.i.pid) & c.generator_id.is_in(self.i.gens))
            .sort("capacity_mw", descending=True)
            .group_by("plant_id_eia")
            .agg(
                pl.col("generator_id"),
                c.capacity_mw.sum(),
                pl.col(
                    k
                    for k in self.d["plant_data"].columns
                    if k not in ("plant_id_eia", "generator_id", "capacity_mw", "source")
                ).first(),
            )
            .with_columns(c.generator_id.list.sort().list.join(", "))
            .lazy()
        )

        id_cols = ["plant_id_eia", "generator_id", "technology_description"]
        en_cols = [
            "redispatch_mwh",
            "export_mwh",
            "load_mwh",
            "storage_charging_mwh",
            "curtailment_mwh",
            "redispatch_curt_adj_mwh",
        ]
        fin_cols = [
            "export_revenue",
            "redispatch_cost_fuel",
            "redispatch_cost_vom",
            "redispatch_cost_fom",
            "redispatch_cost_startup",
        ]

        full = (
            flows_hourly.with_columns(
                **{
                    f"export__{r}": pl.sum_horizontal(cs.matches(f"^export_.*__{r}$"))
                    for r in (*self.re_types, "storage", "fossil")
                },
            )
            .join(
                hourly.select("datetime", "baseline_sys_lambda"),
                on="datetime",
                how="left",
                validate="1:1",
            )
            .unpivot(index=["datetime", "baseline_sys_lambda"], on=cs.contains("__"))
            .with_columns(
                destination=c.variable.str.split("__").list.first(),
                source=c.variable.str.split("__").list.last(),
            )
            .join(
                pl.concat(
                    [
                        re.select(
                            pl.col("plant_id_eia").cast(pl.Int64),
                            pl.col("source").alias("generator_id"),
                            "datetime",
                            "source",
                            share=(
                                c.redispatch_mwh
                                / c.redispatch_mwh.sum().over("datetime", "source")
                            ).fill_nan(0.0),
                        ),
                        hourly.select(
                            "datetime",
                            (
                                cs.contains("discharge")
                                / pl.sum_horizontal(cs.contains("discharge"))
                            ).fill_nan(0.0),
                        )
                        .unpivot(index="datetime")
                        .select(
                            pl.when(pl.col("variable").str.contains("li"))
                            .then(pl.lit(0))
                            .when(pl.col("variable").str.contains("fe"))
                            .then(pl.lit(1))
                            .otherwise(pl.lit(None))
                            .alias("plant_id_eia")
                            .cast(pl.Int64),
                            pl.col("variable")
                            .str.replace("_discharge", "_storage")
                            .alias("generator_id"),
                            "datetime",
                            pl.lit("storage").alias("source"),
                            share=pl.col("value"),
                        ),
                    ],
                    how="vertical_relaxed",
                ),
                on=["datetime", "source"],
                how="left",
            )
            .with_columns(
                value=pl.when(
                    pl.col("generator_id").is_in(
                        ("solar", "onshore_wind", "offshore_wind", "li_storage", "fe_storage")
                    )
                )
                .then(pl.col("value") * pl.col("share").fill_null(0.0))
                .otherwise(pl.col("value")),
                plant_id_eia=pl.when(c.source == "fossil")
                .then(self.i.pid)
                .when(c.source == "new_fossil")
                .then(0)
                .when(c.source == "backup")
                .then(-1)
                .otherwise(c.plant_id_eia)
                .cast(pl.Int64),
                technology_description=c.source.replace(
                    {
                        **TECH_CODES,
                        "fossil": TECH_CODES.get(self.i.tech, self.i.tech),
                        "new_fossil": TECH_CODES.get(new_fos_tech, new_fos_tech),
                        # "backup": TECH_CODES.get(self["backup"].tech, self["backup"].tech),
                        "li": "Batteries",
                        "fe": "Batteries",
                        "storage": "Batteries",
                    }
                ),
                generator_id=pl.when(c.source == "fossil")
                .then(c.source.replace({"fossil": ", ".join(sorted(self.i.gens))}))
                .when(c.source == "new_fossil")
                .then(pl.lit(new_fos_tech))
                .when(c.source == "backup")
                .then(pl.lit("backup"))
                .otherwise(c.generator_id),
                destination=pl.when(
                    pl.col("destination").is_in(("load", "export", "storage", "curtailment"))
                )
                .then(
                    pl.concat_str(
                        pl.col("destination").replace({"storage": "storage_charging"}),
                        pl.lit("_mwh"),
                    )
                )
                .otherwise(pl.lit("del")),
            )
            .with_columns(
                export_revenue=pl.when(pl.col("destination") == "export_mwh")
                .then(pl.col("value") * pl.col("baseline_sys_lambda"))
                .otherwise(pl.lit(0.0))
            )
            .filter(pl.col("destination") != "del")
            .group_by_dynamic(
                "datetime", every="1mo", group_by=(*id_cols, "source", "destination")
            )
            .agg(pl.col("value", "export_revenue").sum())
            .collect()
            .pivot(
                on="destination",
                index=["datetime", *id_cols, "source", "export_revenue"],
                values="value",
            )
            .group_by(*id_cols, "datetime", "source")
            .agg(cs.by_dtype(pl.Float64).sum())
            .sort(*id_cols, "datetime")
            .with_columns(
                redispatch_mwh=pl.sum_horizontal(cs.contains("_mwh")),
                redispatch_curt_adj_mwh=pl.sum_horizontal(cs.contains("_mwh"))
                - pl.col("curtailment_mwh"),
            )
        )
        full = (
            full.lazy()
            .join(costs, on=["datetime", "plant_id_eia", "source"], validate="m:1", how="left")
            .with_columns(
                redispatch_cost_fuel=c.redispatch_mwh * c.fuel_per_mwh,
                redispatch_cost_vom=c.redispatch_mwh * c.vom_per_mwh,
                redispatch_cost_fom=self.i.cap * (c.fom_per_kw * 1000 / 12),
                redispatch_cost_startup=c.starts * self.i.cap * c.start_per_kw,
                redispatch_mmbtu=c.redispatch_mwh * c.heat_rate,
            )
            .with_columns(
                redispatch_co2=c.redispatch_mmbtu * c.co2_factor,
            )
            .filter(pl.col("plant_id_eia").is_not_null())
            .group_by_dynamic("datetime", every="1y", group_by=(*id_cols, "source"))
            .agg(pl.col(*en_cols, "redispatch_mmbtu", "redispatch_co2", *fin_cols).sum())
        )

        full = pl.concat(
            [
                full.filter(
                    pl.col("generator_id").is_in(self.re_types).not_()
                    & pl.col("source").is_in(("storage", "new_fossil", "backup")).not_()
                ).join(
                    specs,
                    on=["plant_id_eia", "generator_id"],
                    how="left",
                    validate="m:1",
                    suffix="__specs",
                ),
                full.filter(
                    pl.col("generator_id").is_in(self.re_types)
                    | pl.col("source").is_in(("storage", "new_fossil", "backup"))
                ).join(
                    re_specs,
                    on=["plant_id_eia", "generator_id", "datetime"],
                    how="left",
                    validate="1:1",
                    suffix="__specs",
                ),
            ],
            how="diagonal_relaxed",
        ).with_columns(
            redispatch_cost_fuel=pl.when(
                (pl.col("redispatch_cost_fuel") == 0) & pl.col("mcoe").is_not_null()
            )
            .then(pl.col("mcoe"))
            .otherwise(pl.col("redispatch_cost_fuel")),
            redispatch_cost_fom=pl.when(c.redispatch_cost_fom == 0.0)
            .then(c.fom_per_kw * c.capacity_mw * 1000)
            .otherwise(c.redispatch_cost_fom),
            operating_month=c.operating_date.dt.month(),
            operating_year=c.operating_date.dt.year(),
            retirement_month=c.retirement_date.dt.month(),
            retirement_year=c.retirement_date.dt.year(),
        )
        c_list = (
            "solar",
            "onshore_wind",
            "li_discharge",
            "fe_discharge",
            *(("export_fossil", "load_fossil") if LoadNewFossil in self else ("fossil",)),
        )
        bad = (
            full.with_columns(
                variable=pl.when(pl.col("source") == "storage")
                .then(pl.col("generator_id").str.replace("_storage", "_discharge"))
                .when(pl.col("source") == "backup")
                .then(pl.lit("fossil"))
                .otherwise(pl.col("source"))
            )
            .group_by("variable", "datetime")
            .agg(pl.col("redispatch_mwh").sum())
            .collect()
            .join(
                hourly.group_by_dynamic("datetime", every="1y")
                .agg(cs.by_name(*c_list, require_all=False).sum())
                .collect()
                .unpivot(
                    index="datetime",
                    on=cs.by_name(*c_list, require_all=False),
                )
                .with_columns(
                    variable=pl.col("variable").replace(
                        {"load_fossil": "new_fossil", "export_fossil": "fossil"}
                    )
                ),
                on=["datetime", "variable"],
            )
            .with_columns(diff=(pl.col("redispatch_mwh") - pl.col("value")).abs())
            .filter(pl.col("diff") > 1)
        )
        if not bad.is_empty():
            bad = bad.to_pandas().to_string().replace("\n", "\n\t")
            self.errors.append(
                f"energy in econ df not reconciled for following resources / years\n{bad}"
            )
            self.logger.info("%s", self.errors[-1])
        return full.select(
            list(
                dict.fromkeys(
                    [
                        *id_cols,
                        "datetime",
                        "capacity_mw",
                        *en_cols,
                        "redispatch_mmbtu",
                        "redispatch_co2",
                        *fin_cols,
                        "capacity_mw_nrel_site",
                        "area_sq_km",
                        "distance",
                    ]
                )
                | dict.fromkeys(full.collect_schema().names())
            )
        )

    def verify_hourly_df(self, hourly):
        chk_cols = (
            "datetime",
            *self.re_types,
            "es",
            "fossil",
            "load",
            "export_requirement",
            "export_clean",
            "_export_clean",
            "export",
            "_export",
            "curtailment",
            "_curtailment",
        )
        load = self.load_mw
        chk = hourly.select(*chk_cols).collect()
        ce = chk.filter((pl.col("export_clean") - pl.col("_export_clean")).abs() > 1)
        exp = chk.filter((pl.col("export") - pl.col("_export")).abs() > 1)
        cur = chk.filter((pl.col("curtailment") - pl.col("_curtailment")).abs() > 1)
        ld = chk.filter((pl.col("load") - load).abs() > 1)
        bat_curt = chk.filter((pl.col("curtailment") > 0) & (pl.col("es") > 0))
        nh = chk.shape[0]

        excess_fos = chk.filter(
            (
                pl.sum_horizontal(*self.re_types, "es", "fossil")
                > 1 + pl.sum_horizontal("load", "export_requirement")
            )
            & (pl.col("fossil") > 0)
        ).select(*self.re_types, "es", "fossil", "load", "export_requirement")
        if not excess_fos.is_empty():
            h_pct = len(excess_fos) / nh
            self.errors.append(f"fossil run for unnecessary export in {h_pct:.4%} of hours")
            self.logger.info(self.errors[-1], extra=self.extra)
        if not exp.is_empty():
            h_pct = len(exp) / nh
            self.errors.append(f"exports are incorrect for {h_pct:.4%} of hours")
            self.logger.info(self.errors[-1], extra=self.extra)
        if not ce.is_empty():
            h_pct = len(ce) / nh
            self.errors.append(f"clean export and fossil for load in {h_pct:.4%} of hours")
            if h_pct > 0.0015:
                self.logger.info(self.errors[-1], extra=self.extra)
        if not cur.is_empty():
            h_pct = len(cur) / nh
            self.errors.append(f"curtailment DVs incorrect for {h_pct:.4%} of hours")
            self.logger.info(self.errors[-1], extra=self.extra)
        if not ld.is_empty():
            h_pct = len(ld) / nh
            l_load = 1 - hourly.select("load").sum().collect().item() / (load * nh)
            self.errors.append(f"{l_load:.5%} of load not met affecting {h_pct:.4%} of hours")
            self.logger.info(self.errors[-1], extra=self.extra)
        if not bat_curt.is_empty():
            h_pct = len(bat_curt) / nh
            self.errors.append(
                f"curtailment and storage discharge both nonzero in {h_pct:.4%} of hours"
            )
            self.logger.info(self.errors[-1], extra=self.extra)
        return self

    @property
    def out_result_dict(self) -> dict:
        load = self.load_mw
        icx_df = self.d.ba_pro.select(
            pl.col("icx", "icx_hist").sum(), pl.col("icx").count().alias("n")
        ).collect()
        base = {
            "run_status": STATUS[self.status],
            "icx_shared": LoadIncumbentFossil in self,
            "load_mw": load,
            "fossil_mw": self.i.cap,
            "max_re_mw": self.i.max_re,
            "tech": self.i.tech,
            "fuel": self.i.fuel,
            "status": self.i.status,
            "regime": self.regime,
            "baseline_fossil_cf": icx_df["icx"].item() / (icx_df["n"].item() * self.i.cap),
            "historical_fossil_cf": icx_df["icx_hist"].item()
            / ((icx_df["n"].item() * self.i.cap) * self.i.cap),
            "years": str(self.i.years),
        }
        if self.selected is None or self.selected.status != "optimal":
            return nan_to_none(
                base | {"errors": "; ".join(reversed(self.errors))} | self._out_result_dict
            )
        n_sites = len(self.re_land.loc["re_area"])
        total_sqkm = self.b_land.value[:n_sites].sum().item()
        re_used_sqkm = (
            (self.re_land.iloc[:n_sites, :-2] @ self["renewables"].x_cap.value).sum().item()
        )
        sqkm_per_mw_load = self["load"].sqkm_per_mw
        base = base | {
            **{k: v for x in self for k, v in x.cap_summary.items()},
            **{
                f"{k}_max_distance_km": v[0][0]
                for k, v in self.re_selected.group_by("re_type")
                .agg(pl.col("distance").max())
                .rows_by_key("re_type")
                .items()
            },
            "total_sqkm": total_sqkm,
            "re_used_sqkm": re_used_sqkm,
            "load_used_sqkm": load * sqkm_per_mw_load,
            "unused_sqkm_pct": (total_sqkm - re_used_sqkm - load * sqkm_per_mw_load)
            / total_sqkm,
        }
        if self.status != SUCCESS:
            return nan_to_none(
                base | {"errors": "; ".join(reversed(self.errors))} | self._out_result_dict
            )
        try:
            hrly = self._dfs.get("hourly", self.hourly().pipe(self.add_mapped_yrs))
            fin_disc = to_dict(
                (fin := self.finance_df(hrly))
                .select(pl.exclude("datetime", "year", "fin_disc") * pl.col("fin_disc"))
                .sum()
            )
            en_disc = to_dict(
                (en := self.energy_df(hrly))
                .select(pl.exclude("year", "phys_disc") * pl.col("phys_disc"))
                .sum()
            )
            tot = to_dict(hrly.select(cs.numeric()).sum())
            vio = to_dict(self.gas_violation_df(hrly))
            hrly = hrly.lazy().collect()
        except Exception as exc:
            self.errors.append(repr(exc).split("Resolved plan until failure")[0])
            if self.status == SUCCESS:
                self.logger.error(
                    "out_result_dict failed %s", self.errors[-1], extra=self.extra
                )
                self.logger.info(self.errors[-1], exc_info=exc, extra=self.extra)
            return nan_to_none(
                base | {"errors": "; ".join(reversed(self.errors))} | self._out_result_dict
            )
        nh = self.life * 8760
        es_info = {t: tuple(self[f"{t}_storage"].cap_summary.values()) for t in self.es_types}
        base = base | {
            "unserved_mwh": self.load_mw * nh - tot["load"],
            "served_pct": tot["load"] / (self.load_mw * nh),
            **to_dict(
                hrly.lazy()
                .filter(pl.col("load") + 1 < self.load_mw)
                .select(unserved_pct_hrs=pl.count("load") / nh, max_unserved_mw=pl.max("load"))
            ),
            "lcoe": fin_disc["cost"] / (en_disc["export"] + en_disc["load"]),
            "ppa": (
                (fin_disc["cost"] - fin_disc["rev_clean"] - fin_disc["rev_fossil"])
                / en_disc["load"]
            ),
            "ppa_ex_fossil_export_profit": (
                (fin_disc["cost"] - fin_disc["cost_export_fossil"] - fin_disc["rev_clean"])
                / en_disc["load"]
            ),
            "attr_cost_clean": fin_disc["cost_clean"] / en_disc["load"],
            "attr_cost_load_fossil": fin_disc["cost_load_fossil"] / en_disc["load"],
            "attr_cost_export_fossil": fin_disc["cost_export_fossil"] / en_disc["load"],
            "attr_rev_export_clean": fin_disc["rev_clean"] / en_disc["load"],
            "attr_rev_export_fossil": fin_disc["rev_fossil"] / en_disc["load"],
            "attr_rev_full_ptc": fin_disc["full_ptc"] / en_disc["load"],
            "attr_cost_curtailment": fin_disc["cost_curt"] / en_disc["load"],
            "avg_cost_load_fossil": safediv(
                fin_disc["cost_load_fossil"], en_disc["load_fossil"]
            ),
            "avg_cost_export_fossil": safediv(
                fin_disc["cost_export_fossil"], en_disc["export_fossil"]
            ),
            "avg_rev_export_clean": safediv(fin_disc["rev_clean"], en_disc["export_clean"]),
            "avg_rev_export_fossil": safediv(fin_disc["rev_fossil"], en_disc["export_fossil"]),
            "ppa_ex_export_profit": (fin_disc["cost"] - fin_disc["cost_export_fossil"])
            / en_disc["load"],
            "lcoe_export": safediv(
                fin_disc["rev_clean"] + fin_disc["rev_fossil"], en_disc["export"]
            ),
            "net_capex": self.selected_cost.select(c.capacity_mw * c.capex__oc).sum().item(),
            "tx_capex": self.selected_cost.select(c.capacity_mw * c.tx_capex__oc).sum().item(),
            "pct_load_clean": (1 - tot["load_fossil"] / tot["load"]),
            "pct_load_clean_potential": tot["clean_for_load"] / tot["load"],
            "pct_export_clean": safediv(tot["export_clean"], tot["export"]),
            "clean_pct_curtailment": safediv(
                tot["curtailment"], sum(v for k, v in tot.items() if k in self.re_types)
            ),
            "fossil_cf": tot["fossil"] / (nh * self.i.cap),
            "load_fossil_cf": tot["load_fossil"] / (nh * self.i.cap),
            "fossil_co2": tot.get("export_fossil_co2", 0.0) + tot.get("load_fossil_co2", 0.0),
            "load_fossil_co2": tot.get("load_fossil_co2", 0.0),
            "redispatch_export_fossil_co2": tot.get("redispatch_export_fossil_co2", 0.0),
            "redispatch_fossil_co2": tot.get("redispatch_export_fossil_co2", 0.0)
            + tot.get("load_fossil_co2", 0.0),
            "redispatch_fossil_cf": (
                tot.get("redispatch_export_fossil", 0.0) + tot.get("load_fossil", 0.0)
            )
            / (nh * self.i.cap),
            "baseline_fossil_cf": tot["baseline_export_fossil"] / (nh * self.i.cap),
            "historical_fossil_cf": tot["historical_fossil"] / (nh * self.i.cap),
            **vio,
            **{
                k: tot[k]
                for k in ("baseline_sys_co2", "redispatch_sys_co2", "load_co2")
                if k in tot
            },
            **{
                f"{t}_pct_charge_hrs_maxed": safediv(
                    np.sum(np.isclose(hrly[f"{t}_charge"].to_numpy(), mw)),
                    np.sum(hrly[f"{t}_charge"].to_numpy() > 0),
                )
                for t, (mw, dur) in es_info.items()
                if (mw > 0) and (dur > 0)
            },
            **{
                f"{t}_pct_discharge_hrs_maxed": safediv(
                    np.sum(np.isclose(hrly[f"{t}_discharge"].to_numpy(), mw)),
                    np.sum(hrly[f"{t}_discharge"].to_numpy() > 0),
                )
                for t, (mw, dur) in es_info.items()
                if (mw > 0) and (dur > 0)
            },
            **{
                f"{t}_pct_cycles_soc_maxed": safediv(
                    np.sum(np.isclose(hrly[f"{t}_soc"].to_numpy(), mw * dur, rtol=0.001)),
                    argrelextrema(hrly[f"{t}_soc"].to_numpy(), np.greater, order=6)[0].shape[
                        0
                    ],
                )
                for t, (mw, dur) in es_info.items()
                if (mw > 0) and (dur > 0)
            },
            **{f"{k}_disc": v for k, v in fin_disc.items()},
            **{f"{k}_mwh_disc": v for k, v in en_disc.items()},
            **to_dict(fin.select(cs.exclude("year", "datetime", "fin_disc")).sum(), "_undisc"),
            **to_dict(en.select(cs.exclude("year", "phys_disc")).sum(), "_mwh_undisc"),
            "lcoe_redispatch": (
                fin_disc.get("redispatch_cost", np.nan)
                / (
                    en_disc.get("redispatch_export_fossil", np.nan)
                    + en_disc.get("export_clean", np.nan)
                    + en_disc.get("load", np.nan)
                )
            ),
            "redispatch_ppa": (
                (
                    fin_disc.get("redispatch_cost", np.nan)
                    - fin_disc.get("redispatch_rev_clean", np.nan)
                    - fin_disc.get("redispatch_rev_fossil", np.nan)
                )
                / en_disc.get("load", np.nan)
            ),
            "redispatch_ppa_ex_fossil_export_profit": (
                (
                    fin_disc.get("redispatch_cost", np.nan)
                    - fin_disc.get("redispatch_cost_export_fossil", np.nan)
                    - fin_disc.get("redispatch_rev_clean", np.nan)
                )
                / en_disc.get("load", np.nan)
            ),
            "lcoe_redispatch_export": (
                (
                    fin_disc.get("redispatch_rev_clean", np.nan)
                    + fin_disc.get("redispatch_rev_fossil", np.nan)
                )
                / (
                    en_disc.get("redispatch_export_fossil", np.nan)
                    + en_disc.get("export_clean", np.nan)
                )
            ),
            **{
                k: tot[k]
                for va in ("baseline", "redispatch")
                for rt in SIMPLE_TD_MAP.values()
                if (k := f"{va}_{rt}") in tot
            },
            "errors": "; ".join(reversed(self.errors)),
        }
        return nan_to_none(base | self.util_info | self._out_result_dict)

    @cached_property
    def util_info(self):
        self.d.load_ba_data()
        cols = [
            "reg_rank",
            "balancing_authority_code_eia",
            "utility_id_eia",
            "utility_name_eia",
            "entity_type_lse",
            "utility_name_eia_lse",
            "transmission_distribution_owner_id",
            "transmission_distribution_owner_name",
            "entity_type",
        ]
        return to_dict(
            pl.from_pandas(self.d["plant_data"])
            .filter(
                (pl.col("plant_id_eia") == self.i.pid)
                & pl.col("generator_id").is_in(self.i.gens)
            )
            .group_by("plant_id_eia")
            .agg(pl.col(cols).mode().first())
            .select(*cols)
        )

    @property
    def dfs(self) -> dict[str, pl.DataFrame]:
        out = {}
        try:
            out.update(re_selected=self.re_selected)
        except Exception:
            pass
        try:
            out.update(resource_selection_hourly=self.hourly(selection=True).collect())
        except Exception:
            pass
        try:
            out.update(
                coef_mw=pl.concat(
                    [a for dv in self if (a := dv.c_mw_all()) is not None], how="align"
                )
            )
        except Exception:
            pass
        try:
            out.update(cost_detail=self.cost_detail())
        except Exception:
            pass
        if "hourly" not in self._dfs:
            try:
                self._dfs.update(hourly=self.hourly().pipe(self.add_mapped_yrs).collect())
            except Exception:
                pass
        if "annual" not in self._dfs:
            try:
                h = self._dfs.get("hourly", None)
                self._dfs.update(
                    annual=pl.concat(
                        [self.finance_df(h).lazy(), self.energy_df(h).lazy()], how="align"
                    ).collect()
                )
            except Exception:
                pass
        return out | self._dfs

    def add_df(self, **kwargs):
        self._dfs.update(kwargs)

    def add_to_result_dict(self, **kwargs):
        self._out_result_dict.update(kwargs)

    @property
    def ppa(self):
        return (
            self.finance_df()
            .select(
                (pl.col("cost") - pl.col("rev_clean") - pl.col("rev_fossil"))
                * pl.col("fin_disc")
            )
            .sum()
            .collect()
            / self.energy_df().select(pl.col("load") * pl.col("phys_disc")).sum().collect()
        )

    @property
    def pct_load_clean(self):
        tot = self.hourly().select("load_fossil", "load", "clean_for_load").sum().collect()
        return (1 - tot["load_fossil"].item() / tot["load"].item()), tot[
            "clean_for_load"
        ].item() / tot["load"].item()

    def add_mapped_yrs[T: pl.DataFrame | pl.LazyFrame](self, df: T) -> T:
        return pl.concat([df, gen_proj(df, year_mapper=self.yr_mapper)], how="vertical")

    @property
    def extra(self) -> dict:
        return self.i.extra(self.regime, self.name)

    def __contains__(self, other: DecisionVariable | type) -> bool:
        if isinstance(other, DecisionVariable):
            return any(other is dv for dv in self)
        if isinstance(other, type):
            return any(other is dv.__class__ for dv in self)
        return False

    def __getitem__(self, index: int | str | type) -> DecisionVariable:
        if index in self.dvs:
            return self.dvs[index]
        for dv in self.dvs.values():
            if dv.__class__ is index:
                return dv
            if index.casefold() in dv.__class__.__qualname__.casefold():
                return dv
        raise KeyError(index)

    def __iter__(self) -> Iterator[DecisionVariable]:
        return iter(self.dvs.values())

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"(name={self.name}, {', '.join(repr(dv) for dv in self)})"
        )

    def _dzgetstate_(self):
        state = dict(self.__dict__)
        state["_state"].update(
            {
                "params": {},
                "solver_logs": state.pop("solver_logs", {}),
                "util_info": self.util_info,
                "results": state.pop("results", {}),
                "status": self.status,
                "dvs": self.dvs,
                "dispatchs": dict.fromkeys(self.dispatchs),
                "re_filter": self.d.meta["re_filter"],
            }
        )
        # dvs = deepcopy(state["_state"].get("dvs", {}))
        # state["dvs"] = dvs
        # for v in self.dvs.values():
        #     if hasattr(v, "x_cap") and v.x_cap is not None:
        #         state["dvs"][v.__class__.__qualname__]["x_cap"] = v.x_cap.value
        for k, v in state["_params"].items():
            state[k] = getattr(v, "value", v)
        state["data_dir"] = self.d.src_path
        delete = (
            "a",
            "d",
            "ptc_years",
            "r_disc",
            "fcr",
            "yr_mapper",
            "selected",
            "re_land",
            "b_land",
            "logger",
            "selected",
            "dispatchs",
            "_crit_hrs",
            "_params",
            "dvs",
            "re_selected",
            "_dfs",
            "_out_result_dict",
            "es_types",
            "status",
            "re_types",
            "export_profs",
        )
        del state["_state"]["params"]
        for k in delete:
            state.pop(k, None)
        return state

    def _dzsetstate_(self, state):
        other = state.pop("_state")
        data_dir = state.pop("data_dir")
        re_filter = other.pop("re_filter", None)
        if re_filter is None:
            re_filter = (pl.col("re_type") == "solar") | (
                (pl.col("re_type") == "onshore_wind") & (pl.col("distance") <= 20)
            )
        elif re_filter == "no_filter":
            re_filter = None
        else:
            re_filter = pl.Expr.deserialize(StringIO(re_filter), format="json")

        info = Info.fix(state.pop("i"))

        self.__init__(d=Data.from_dz(data_dir, info, re_filter=re_filter), i=info, **state)
        for k, v in other.items():
            setattr(self, k, v)
        for v in self.dvs.values():
            v.m = self
        # if CleanExport in self:
        #     clexp: CleanExport = self["CleanExport"]
        #     clexp.cost = {}
        #     for yr in clexp.cost_["year"].unique().sort():
        #         y_cost = clexp.cost_.filter(pl.col("year") == yr)
        #         clexp.cost[lit_eval(yr)] = -(
        #             y_cost["a"].to_numpy() * self.mkt_rev_mult - y_cost["b"].to_numpy()
        #         )

    def load_saved_x_cap(self, saved_select):
        fs = rmi_cloud_fs()
        with fs.open(f"az://patio-results/{saved_select}/colo_summary.parquet") as f:
            suma = (
                pl.scan_parquet(f)
                .filter(self.i.filter(name=self.name, regime=self.regime))
                .select(
                    "run_status",
                    "load_mw",
                    "li_mw",
                    (pl.col("li_duration") * pl.col("li_mw")).alias("li_mwh"),
                    "new_fossil_mw",
                    "fe_mw",
                    "select_time",
                    *self._params,
                )
                .collect()
            )
        if (status := suma["run_status"].item()) not in (
            "SUCCESS",
            "FAIL_DISPATCH",
            "FAIL_SERVICE",
        ):
            self.status = {v: k for k, v in STATUS.items()}[status]
            return None
        load_mw, li_mw, li_mwh, new_fossil_mw, fe_mw = (
            suma.select("load_mw", "li_mw", "li_mwh", "new_fossil_mw", "fe_mw")
            .to_numpy()
            .flatten()
        )
        with fs.open(f"az://patio-results/{saved_select}/colo_re_selected.parquet") as f:
            re_cap = (
                self.d.re_specs.join(
                    pl.scan_parquet(f)
                    .filter(self.i.filter(name=self.name, regime=self.regime))
                    .select("re_site_id", "re_type", "capacity_mw"),
                    on=["re_site_id", "re_type"],
                    how="left",
                    validate="1:1",
                    maintain_order="left",
                )
                .select(pl.col("capacity_mw").fill_null(0.0))
                .collect()
                .to_series()
                .to_numpy()
            )
        try:
            del self.re_selected
        except AttributeError:
            pass
        self["load"].x_cap.value = [load_mw]
        self["renewables"].x_cap.value = re_cap
        self["li_storage"].x_cap.value = [[li_mw], [li_mwh]]
        if FeStorage in self:
            self["fe_storage"].x_cap.value = [fe_mw]
        if LoadNewFossil in self or LoadNewFossilWithBackup in self:
            self["fossil"].x_cap.value = [new_fossil_mw]
        self.add_to_result_dict(
            select_time=suma["select_time"].item(),
            saved_select=saved_select,
            **{f"select_{k}": suma[k].item() for k in self._params},
        )
        self.status = SUCCESS
        self.selected = SimpleNamespace(status="optimal")
        self.objective(self.i.years)
        return None

    @classmethod
    def from_file(cls, path: Path | str | BytesIO, **kwargs) -> Self:
        """Recreate object fom file or buffer."""
        return DataZip.load(path, klass=cls)

    @classmethod
    def from_cloud(cls, datestr, info, regime, ix, **kwargs) -> Self:
        ba, pid, _, tech, status, *_ = info

        from etoolbox.utils.cloud import AZURE_CACHE_PATH, cached_path, rmi_cloud_fs

        file = f"patio-results/colo_{datestr}/results/{info.file(regime, ix, '.zip')}"
        f = rmi_cloud_fs().open(f"az://{file}")
        f.close()
        return Model.from_file(str(AZURE_CACHE_PATH / cached_path(file)))

    def to_file[T: Path | BytesIO | str](
        self, path: T | None = None, *, clobber=False, **kwargs
    ) -> T:
        """Write object to file or buffer."""
        if path is None:
            path = self.result_dir / self.i.file(regime=self.regime, ix=self.ix, suffix=".zip")
        DataZip.dump(self, file=path, clobber=clobber)
        return path

    @classmethod
    def from_run(cls, info, name, regime, colo_dir):
        with open(ROOT_PATH / "patio.toml", "rb") as f:
            configs = tomllib.load(f)["colo"]
        if name in configs["scenario"]:
            config = configs["scenario"][name] | {"name": name}
        else:
            config = [
                {"name": k} | v for k, v in configs["scenario"].items() if v["ix"] == name
            ][0]

        self = cls(
            config["ix"],
            config["name"],
            info,
            Data.from_dz(Path.home() / configs["project"]["data_path"], info),
            **config["params"],
            regime=regime,
            dvs=config["dv"],
        )
        summa = (
            pl.scan_parquet(colo_dir / "colo_summary.parquet")
            .filter(info.filter(regime, name))
            .collect()
            .to_dict(as_series=False)
        )
        if not summa["load_mw"]:
            raise ValueError(f"{info=}, {regime=}, {name=} does not exist")
        if len(summa["load_mw"]) > 1:
            raise ValueError(f"{info=}, {regime=}, {name=} is not unique")
        res = (
            pl.scan_parquet(colo_dir / "colo_re_selected.parquet")
            .filter(info.filter(regime, name))
            .select("re_site_id", "re_type", "capacity_mw")
        )
        for dv in self.dvs.values():
            dv.set_x_cap(summa, res)
        return self

    def write_model(self, dir_path=None):
        if dir_path is None:
            dir_path = self.d.src_path.parent
        model: gurobipy.Model = self.selected._solver_cache["GUROBI"]
        model.write(str(dir_path / self.i.file(self.regime, self.ix, ".mps")))
        model.computeIIS()
        model.write(str(dir_path / self.i.file(self.regime, self.ix, ".ilp")))

    def redispatch(self, hourly):
        self.d.load_ba_data()
        pd_dt_idx = pd.Index(hourly.select("datetime").collect().to_pandas().squeeze())
        new_profs = self.d["dispatchable_profiles"].loc[pd_dt_idx, :]
        # breaks for nuclear because not in new profs (also no export capacity)
        if (
            TECH_CODES.get(self.i.tech, self.i.tech) in OTHER_TD_MAP
            and "fossil" in self.dvs
            and isinstance(self["fossil"], LoadIncumbentFossil)
        ):
            icx_ixs = [(self.i.pid, g) for g in self.i.gens]
            colo_load_fossil = (
                hourly.select("datetime", "load_fossil")
                .collect()
                .to_pandas()
                .set_index("datetime")
                .fillna(0.0)
            ).squeeze()

            new_profs.loc[:, icx_ixs] = adjust_profiles(
                new_profs.loc[:, icx_ixs].to_numpy(),
                colo_load_fossil.to_numpy(),
                self.d["dispatchable_specs"].loc[icx_ixs, "capacity_mw"].to_numpy(),
            )
            max_check = (new_profs.loc[:, icx_ixs].sum(axis=1) + colo_load_fossil).max()
            assert max_check < self.i.cap or np.isclose(max_check, self.i.cap), (
                "fossil availability adjustment for load use of generators failed"
            )
        else:
            icx_ixs = []
        dm = DispatchModel(
            load_profile=hourly.select("datetime", "baseline_sys_load_net_of_clean_export")
            .collect()
            .to_pandas()
            .set_index("datetime")
            .fillna(0.0)
            .squeeze(),
            dispatchable_specs=self.d["dispatchable_specs"],
            dispatchable_profiles=new_profs,
            dispatchable_cost=self.d["dispatchable_cost"]
            .filter(c.datetime.dt.year() >= self.d.opt_years[0])
            .collect()
            .to_pandas()
            .set_index(["plant_id_eia", "generator_id", "datetime"]),
            storage_specs=self.d["storage_specs"],
            re_profiles=self.d["re_profiles"].loc[pd_dt_idx, :],
            re_plant_specs=self.d["re_plant_specs"],
        )()
        hly = (
            hourly.select("datetime", "export_clean", "load_fossil")
            .collect()
            .to_pandas()
            .set_index("datetime")
        )
        redispatch = pd.concat(
            [
                dm.redispatch.loc[hly.index, :]
                .sum(axis=1)
                .rename("redispatch_sys_dispatchable"),
                # rather than trying to break out colo contribution to both storage and
                # renewable all clean exports here are categorized as renewable because
                # fossil generally can't charge the battery
                (dm.re_profiles_ac.loc[hly.index, :].sum(axis=1) + hly.export_clean).rename(
                    "redispatch_sys_renewable"
                ),
                dm.storage_dispatch.loc[hly.index, :]
                .T.groupby(level=0)
                .sum()
                .T.assign(redispatch_sys_storage=lambda x: x.discharge - x.gridcharge)[
                    "redispatch_sys_storage"
                ],
            ],
            axis=1,
        )
        factors = (
            (dm.dispatchable_cost.heat_rate * dm.dispatchable_cost.co2_factor)
            .reset_index()
            .pivot(index="datetime", columns=["plant_id_eia", "generator_id"], values=0)
            .reindex_like(dm.redispatch, method="ffill")
            .loc[hly.index, :]
        )
        summary = pd.concat(
            [
                dm.system_summary_core(freq="h")
                .loc[
                    hly.index,
                    ["load_mwh", "deficit_mwh", "curtailment_mwh", "curtailment_pct"],
                ]
                .rename(
                    columns={
                        "load_mwh": "redispatch_sys_load",
                        "deficit_mwh": "redispatch_sys_deficit",
                        "curtailment_mwh": "redispatch_sys_curtailment",
                        "curtailment_pct": "redispatch_sys_curtailment_pct",
                    }
                ),
                (
                    (dm.redispatch.loc[hly.index, :] * factors).sum(axis=1)
                    # needs to never be nans even if no fossil incumbent, so above remains
                    + factors.loc[:, icx_ixs].mean(axis=1).fillna(0.0) * hly.load_fossil
                ).rename("redispatch_sys_co2"),
                calc_redispatch_cost(dm).loc[hly.index].rename("redispatch_sys_cost"),
                redispatch,
            ],
            axis=1,
        )
        hourly = (
            pl.concat(
                [
                    hourly,
                    # pl.from_pandas(
                    #     (factors.loc[:, icx_ixs].mean(axis=1) * hly.load_fossil).rename(
                    #         "load_co2"
                    #     ),
                    #     schema_overrides={"datetime": pl.Datetime()},
                    #     include_index=True,
                    # ).lazy(),
                    # hourly.select(
                    #     "datetime",
                    #     export_co2=pl.col("export_fossil")
                    #     * pl.lit(factors.loc[:, icx_ixs].mean(axis=1).to_numpy()),
                    # ),
                    pl.from_pandas(
                        (dm.redispatch.loc[hly.index, icx_ixs] * factors.loc[:, icx_ixs])
                        .sum(axis=1)
                        .rename("redispatch_export_fossil_co2"),
                        schema_overrides={"datetime": pl.Datetime()},
                        include_index=True,
                    ).lazy(),
                    pl.from_pandas(
                        dm.redispatch.loc[hly.index, icx_ixs]
                        .sum(axis=1)
                        .rename("redispatch_export_fossil"),
                        schema_overrides={"datetime": pl.Datetime()},
                        include_index=True,
                    ).lazy(),
                    pl.from_pandas(
                        dm.redispatch_lambda().loc[hly.index].rename("redispatch_sys_lambda"),
                        schema_overrides={"datetime": pl.Datetime()},
                        include_index=True,
                    ).lazy(),
                    pl.from_pandas(
                        summary,
                        schema_overrides={"datetime": pl.Datetime()},
                        include_index=True,
                    ).lazy(),
                ],
                how="align",
            )
            .with_columns(
                _rcurt=np.maximum(np.minimum(1, c.redispatch_sys_curtailment_pct), 0)
            )
            .with_columns(
                _hr_mkt_rev_red=c.redispatch_sys_lambda * self.mkt_rev_mult * (1 - c._rcurt),
            )
            .with_columns(
                redispatch_rev_clean=c.export_clean * c._hr_mkt_rev_red,
                redispatch_rev_fossil=c.redispatch_export_fossil * c._hr_mkt_rev_red,
                redispatch_cost_fossil=(c.redispatch_export_fossil + c.load_fossil)
                * c.c_export_fossil,
                redispatch_cost_export_fossil=c.redispatch_export_fossil * c.c_export_fossil,
                cost_curt=c.curtailment * c.c_curtailment
                + c.export_clean * c.assumed_curtailment_pct * c._ptc,
                redispatch_cost_curt=c.curtailment * c.c_curtailment
                + c.export_clean * c._rcurt * c._ptc,
                redispatch_system_cost=c.redispatch_sys_cost
                + c.export_clean * c._hr_mkt_rev_red,
            )
        )
        if hourly.select("redispatch_export_fossil").sum().collect().item() == 0.0:
            hourly = hourly.with_columns(redispatch_export_fossil=pl.col("export_fossil"))
        return hourly

    def econ_and_flows(self, hourly: pl.LazyFrame, run_econ):
        assert "backup" not in hourly.collect_schema().names()

        with timer() as t0:
            l_fos = "fossil" if LoadIncumbentFossil in self else "new_fossil"
            es_fos = pl.min_horizontal(
                "charge",
                "load_fossil",
                pl.max_horizontal(
                    "load_fossil",
                    pl.sum_horizontal(*self.re_types) - pl.col("load"),
                ),
            )
            re = pl.sum_horizontal(*self.re_types)
            flows = (
                hourly.lazy()
                .with_columns(
                    charge=-pl.min_horizontal(0.0, "es"),
                    discharge=pl.max_horizontal(0.0, "es"),
                    export_req__clean=pl.min_horizontal("export_requirement", "export_clean"),
                    export_addl__clean=pl.col("export") - pl.col("export_requirement"),
                )
                .with_columns(
                    es_fos.alias(f"storage__{l_fos}"),
                    (pl.col("load_fossil") - es_fos).alias(f"load__{l_fos}"),
                )
                .with_columns(
                    load__re=pl.min_horizontal(pl.col("load") - pl.col(f"load__{l_fos}"), re)
                )
                .with_columns(
                    export_addl__re=pl.min_horizontal("export_addl__clean", re - c.load__re),
                )
                .with_columns(
                    export_req__re=pl.max_horizontal(
                        0.0,
                        re
                        - pl.sum_horizontal(
                            "export_addl__re", "load__re", "charge", "curtailment"
                        ),
                    ),
                )
                .with_columns(
                    export_req__storage=pl.max_horizontal(
                        0.0, c.export_req__clean - c.export_req__re
                    ),
                    export_req__fossil=pl.max_horizontal(
                        0.0, c.export_requirement - c.export_req__clean
                    ),
                    export_addl__storage=c.export_addl__clean - c.export_addl__re,
                    load__storage=pl.min_horizontal(
                        pl.col("load") - pl.sum_horizontal(f"load__{l_fos}", "load__re"),
                        "discharge",
                    ),
                    storage__re=c.charge - es_fos,
                    curtailment__re=c.curtailment,
                )
            )
            re_ = flows.select(self.re_types).collect()
            alloc = (re_ / re_.sum_horizontal()).fill_nan(0.0)
            to_repl = ("load", "export_req", "export_addl", "storage", "curtailment")
            flows = flows.with_columns(
                **{
                    f"{part}__{t}": alloc[t] * pl.col(f"{part}__re")
                    for part in to_repl
                    for t in self.re_types
                }
            )

            flows = flows.select(
                ~cs.ends_with("_re")
                & ~cs.ends_with("_discharge")
                & ~cs.ends_with("_charge")
                & ~cs.ends_with("_clean")
                & ~cs.ends_with("_soc")
                & ~cs.starts_with("redispatch_sys_")
                & (~cs.starts_with("baseline_sys_") | cs.contains("baseline_sys_lambda"))
                & ~cs.contains("baseline")
                & ~cs.contains("redispatch")
                & ~cs.starts_with("c_")
                & ~cs.starts_with("cp_")
                & ~cs.ends_with("_co2")
                & ~cs.starts_with("rev_")
                & ~cs.starts_with("cost_")
            )
            check = (
                flows.with_columns(
                    load_check=(c.load - pl.sum_horizontal(cs.starts_with("load__"))).abs(),
                    export_check=(
                        c.export
                        - pl.sum_horizontal(cs.matches("^export_req__.*|^export_addl__.*"))
                    ).abs(),
                    export_req_check=(
                        c.export_requirement - pl.sum_horizontal(cs.matches("^export_req__.*"))
                    ).abs(),
                    fossil_check=(
                        c.fossil
                        - pl.sum_horizontal(
                            "export_req__fossil",
                            f"load__{l_fos}",
                            f"storage__{l_fos}",
                        )
                    ).abs(),
                    discharge_check=(
                        c.discharge - pl.sum_horizontal(cs.matches(".*__storage$"))
                    ).abs(),
                    charge_check=(
                        c.charge - pl.sum_horizontal(cs.matches("^storage__.*"))
                    ).abs(),
                    curtailment_check=(
                        c.curtailment - pl.sum_horizontal(cs.matches("^curtailment__.*"))
                    ).abs(),
                )
                .filter(
                    (c.load_check > 0.1)
                    | (c.export_check > 0.1)
                    | (c.export_req_check > 0.1)
                    | (c.fossil_check > 0.1)
                    | (c.discharge_check > 0.1)
                    | (c.charge_check > 0.1)
                    | (c.curtailment_check > 0.1)
                )
                .select(
                    ~cs.contains("baseline")
                    & ~cs.contains("redispatch")
                    & ~cs.starts_with("c_")
                    & ~cs.starts_with("cp_")
                    & ~cs.ends_with("_co2")
                    & ~cs.starts_with("rev_")
                    & ~cs.starts_with("cost_")
                )
                .collect()
            )
        econ_df = pl.DataFrame()
        if not check.is_empty():
            self.errors.append(
                f"flows inaccurate in {len(check) / (self.life * 8760):.6%} of hours"
            )
            self.logger.error(self.errors[-1], extra=self.extra)
        check = check.filter(((pl.col("curtailment") > 0) & (pl.col("discharge") > 0)).not_())
        if len(check) > 9 * self.life:
            self.add_to_result_dict(flows_time=t0())
            self.logger.error("check of flows failed", extra=self.extra)
            flows = pl.DataFrame()
        else:
            with timer() as t1:
                flows_hourly = flows.select(
                    cs.by_name("datetime", "starts", "baseline_sys_lambda") | cs.contains("__")
                )
                flows = (
                    flows_hourly.group_by_dynamic("datetime", every="1y")
                    .agg(cs.contains("__").sum())
                    .collect()
                )
            self.add_to_result_dict(flows_time=t0() + t1())
            if run_econ:
                with timer() as t:
                    try:
                        econ_df = self.create_df_for_econ_model(flows_hourly, hourly).collect()
                    except pl.exceptions.PolarsError as exc:
                        self.errors.append(
                            "could not create econ_df, "
                            + repr(exc).split("Resolved plan until failure")[0]
                            + "')"
                        )
                        self.logger.error(self.errors[-1], extra=self.extra)
                        self.logger.info(self.errors[-1], exc_info=exc, extra=self.extra)
                        self.to_file()
                    except Exception as exc:
                        self.errors.append(f"could not create econ_df, {exc!r}")
                        self.logger.error(self.errors[-1], extra=self.extra)
                        self.logger.info(self.errors[-1], extra=self.extra, exc_info=exc)
                        self.to_file()
                self.add_to_result_dict(econ_df_time=t())
            else:
                self.to_file()
            del flows_hourly
        return econ_df, flows

    def aeo_fuel_price(self, fuel):
        ba, state = (
            self.d.ba_data["plant_data"]
            .query("plant_id_eia == @self.i.pid")[["balancing_authority_code_eia", "state"]]
            .iloc[0]
        )
        try:
            reg = AEO_MAP.filter(
                (pl.col("balancing_authority_code_eia") == ba) & (pl.col("state") == state)
            )["electricity_market_module_region_eiaaeo"].item()
        except ValueError as exc:
            raise RuntimeError(f"{ba} {state} not in AEO MAP") from exc
        return (
            aeo(self.pudl_release)
            .filter(
                (pl.col("report_year") == self.aeo_report_year)
                & (pl.col("model_case_eiaaeo") == self.aeo_scenario)
                & (pl.col("electricity_market_module_region_eiaaeo") == reg)
                & (pl.col("fuel_type_eiaaeo") == fuel)
            )
            .select(
                datetime=pl.datetime(pl.col("projection_year"), 1, 1),
                fuel_per_mmbtu=pl.col("fuel_cost_per_mmbtu"),
            )
        )
