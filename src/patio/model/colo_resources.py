from ast import literal_eval as lit_eval
from types import NotImplementedType, SimpleNamespace
from typing import TYPE_CHECKING, Protocol, Self

from patio.constants import CARB_INTENSITY

if TYPE_CHECKING:
    from colo_lp import Model

import cvxpy as cp
import numpy as np
import polars as pl
import polars.selectors as cs
import scipy.sparse as sp
from etoolbox.utils.misc import all_logging_disabled

from patio.data.asset_data import clean_atb
from patio.model.colo_common import COSTS, ES, f_npv, f_pmt, f_pv, hstack, nt, prof, to_dict


class ParamVarLike(Protocol):
    size: int
    shape: tuple[int, ...]
    name: str
    value: np.ndarray


validators = {
    "load": lambda yr, *args, self: 8760 * len(yr),
    "critical": lambda yr, *args, self: 8760 * len(yr),
    "common_hrly_constraint": lambda yr, *args, self: 8760 * len(yr),
    "export_req": lambda yr, *args, self: 8760 * len(yr),
    "clean_export": lambda yr, *args, self: 8760 * len(yr),
    "fossil_load": lambda yr, *args, self: 1,
    "fossil_load_hrly": lambda yr, *args, self: 8760 * len(yr),
    "fossil_hist": lambda yr, *args, self: 1,
    "icx_ops": lambda yr, *args, self: 8760 * len(yr),
    "land": lambda yr, *args, self: len(self.m.re_land),
}


def check_shape():
    def decorator(func):
        validator = validators[name := func.__name__]  # noqa: F841

        def wrapper(self, yr: tuple, *args):
            out = func(self, yr, *args)
            if isinstance(out, cp.Expression):
                expected = validator(yr, *args, self=self)
                shape = out.shape[0] if out.shape else 1
                if shape not in (expected, 1):
                    raise AssertionError(
                        f"{func.__qualname__} expected to have shape {expected} or 1, got {shape}"
                    )
            else:
                print(func.__qualname__)

            return out

        wrapper._og = func

        return wrapper

    return decorator


def is_resource_selection(yr: tuple):
    return len(yr) != 1


def product(*cols, ignore_nulls=False):
    if ignore_nulls:
        return (
            pl.when(pl.all_horizontal(pl.col(*cols).is_null()))
            .then(None)
            .otherwise(pl.reduce(lambda a, x: a * x, pl.col(*cols).fill_null(1)))
        )
    return pl.reduce(lambda a, x: a * x, pl.col(*cols))


class MultiXCapShim(tuple):
    """Work with multiple cvxpy.Variable objects as if they were one."""

    @property
    def value(self):
        if self[0].value is None:
            return None
        return hstack(v.value for v in self)

    @value.setter
    def value(self, value):
        if len(value) != len(self):
            raise ValueError(
                f"Failed to set value, setting value requires {len(self)} numbers"
            )
        for i, v in enumerate(value):
            if self[i].value is None:
                self[i].value = v
            else:
                self[i].value[:] = v


class DecisionVariable:
    _var_names = ()
    _cat = ""
    _dz = {"profs": []}

    def __init__(self, m: "Model"):
        self.m: Model = m
        if self not in self.m:
            self.m.add_dv(self)
        self.x = {}
        self.cost = {}
        self.cost_cap = {}
        self.x_cap = None
        self.sqkm_per_mw = None

    @property
    def var_names(self):
        return self._var_names

    @property
    def cat(self):
        return self._cat

    @property
    def cap_names(self) -> tuple:
        return (self.cat,)

    @property
    def dur(self) -> ParamVarLike | NotImplementedType:
        return NotImplemented

    @property
    def fuels(self):
        return []

    @property
    def cap_summary(self) -> dict:
        return {}

    def cost_summary(self, every="1y", cap_grp=None, h_grp=None):
        if cap_grp is None:
            cap_grp = (pl.col("re_type").alias("type"),)
        # try:
        parts = []
        if self.cost_cap:
            ovn = {
                "capex_gross": product("capex_raw", "reg_mult", "life_adj", "capacity_mw"),
                "itc": -product("capex_raw", "reg_mult", "life_adj", "capacity_mw")
                * (1 - pl.col("itc_adj")),
                "tx_capex": product("tx_capex_raw", "distance", "capacity_mw"),
            }
            ops = {
                "fom": product("opex_raw", "capacity_mw"),
                "ptc": -product("ptc_gen", "ptc", "capacity_mw"),
            }

            parts.append(
                pl.concat(
                    [
                        v.with_columns(
                            capacity_mw=self.x_cap.value,
                            datetime=pl.datetime(
                                {self.m.i.years: min(self.m.d.opt_years)}.get(k, k[0]),
                                1,
                                1,
                            ),
                        ).with_columns(**(ops if len(k) == 1 else ovn))
                        for k, v in self.cost_cap.items()
                    ],
                    how="diagonal_relaxed",
                )
                .group_by("datetime", *cap_grp, maintain_order=True)
                .agg(pl.first("capacity_mw"), pl.col(list(ovn) + list(ops)).sum())
            )
        if self.cost:
            parts.append(
                self.hourly()
                .with_columns(**{c: product(c, f"c_{c}") for c in self.var_names})
                .group_by_dynamic("datetime", every=every, group_by=h_grp)
                .agg(pl.col(*self.var_names).sum())
                .collect()
            )
        if parts:
            out = pl.concat(parts, how="align")
            if "type" in out.columns:
                return out
            return out.with_columns(type=pl.lit(self.cat))
        return pl.DataFrame()
        #
        # except pl.exceptions.PolarsError as exc:
        #     self.m.logger.error(
        #         "%s cost summary failed %s msg=%s",
        #         self.__class__.__qualname__,
        #         exc.__class__,
        #         str(exc).split("Resolved plan until failure:")[0],
        #         extra=self.m.extra,
        #     )
        #     return pl.DataFrame()
        # except Exception as exc:
        #     self.m.logger.error(
        #         "%s cost summary failed %r",
        #         self.__class__.__qualname__,
        #         exc,
        #         extra=self.m.extra,
        #     )
        #     return pl.DataFrame()

    def set_x_cap(self, cap_summary, re_selected, *args):
        return None

    def round(self):
        return None

    def get_x(self, yr: tuple) -> cp.Variable:
        if yr in self.x:
            return self.x[yr]
        if len(self.var_names) == 1:
            self.x[yr] = cp.Variable(8760 * len(yr), name=self.var_names[0])
        else:
            self.x[yr] = tuple(
                cp.Variable(8760 * len(yr), name=name) for name in self.var_names
            )
        return self.x[yr]

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        c_hourly = self.c_hourly(selection=selection)
        hourly = self._h_core(selection, self.x, func=lambda x: x.value)
        if c_hourly is None:
            return hourly
        return hourly.join(c_hourly, on="datetime")

    def c_mw_all(self) -> pl.DataFrame | None:
        if not self.cost_cap:
            return None
        return (
            pl.DataFrame(
                [
                    v.select(
                        pl.sum_horizontal(
                            "capex__oc", "opex__oc", "tx_capex__oc", "ptc__oc"
                        ).alias(" ".join(map(str, k)))
                    ).to_series()
                    for k, v in self.cost_cap.items()
                ]
            )
            .filter(pl.any_horizontal(pl.all() > 0))
            .transpose(include_header=True, header_name="year", column_names=self.cap_names)
        )

    def c_hourly(self, selection=False) -> pl.LazyFrame | None:
        if not self.cost:
            return None
        c = self.m.i.years if selection else (self.m.d.opt_years[0],)
        f = (lambda x: x.value) if isinstance(self.cost[c], cp.Expression) else lambda x: x
        c_hourly = self._h_core(selection, self.cost, "c_", func=f)
        non_zero = (
            c_hourly.select(cs.numeric())
            .sum()
            .collect()
            .transpose(include_header=True, column_names=["values"])
            .filter(pl.col("values") != 0)["column"]
            .to_list()
        )
        if not non_zero:
            return None
        return c_hourly.select("datetime", *non_zero)

    def _h_core(self, rs, in_dict, prefix="", func=lambda x: x) -> pl.LazyFrame:
        try:
            if len(self.var_names) == 1:
                outer_func = lambda k, v: prof(self.m.d.re_pro, k, cs.datetime).with_columns(  # noqa: E731
                    **{f"{prefix}{self.var_names[0]}": func(v)}
                )
            else:
                outer_func = lambda k, v: prof(self.m.d.re_pro, k, cs.datetime).with_columns(  # noqa: E731
                    **{
                        f"{prefix}{c}": func(y)
                        for c, y in zip(self.var_names, v, strict=False)
                    }
                )
            f = (lambda k: k == self.m.i.years) if rs else lambda k: len(k) == 1
            return pl.concat(
                [outer_func(k, v) for k, v in in_dict.items() if f(k)],
                how="vertical",
            )
        except Exception as exc:
            raise RuntimeError(f"{self!r} {prefix} {exc!r}\n") from exc

    def annual_gen(self):
        return NotImplemented

    def load(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def critical(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def export_req(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def clean_export(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def icx_ops(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def incumbent_ops(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def land(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def gas_window_max(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def stored_fuel(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def backup_annual(self, yr: tuple, *args) -> cp.Expression:
        return cp.Constant(0.0)

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        return ()

    def objective(self, yr: tuple, yr_fact_map: dict) -> cp.Expression:
        return cp.Constant(0.0)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        return ()

    def ptc(self, yr: tuple) -> float:
        return self.m.ptc if is_resource_selection(yr) or yr[0] in self.m.ptc_years else 0.0

    def mk_obj_cost[T: pl.LazyFrame | pl.DataFrame](
        self, cost_df: T, yr: tuple, yr_fact_map: dict, *args
    ) -> T:
        if is_resource_selection(yr):
            return (
                cost_df.with_columns(
                    capex__oc=product("capex_raw", "reg_mult", "itc_adj", "life_adj"),
                    tx_capex__oc=product("tx_capex_raw", "distance"),
                    pver=pl.lit(sum(yr_fact_map.values())),
                    ptc_pver=yr_fact_map[yr[0]],
                )
                .with_columns(
                    opex__oc=pl.col("opex_raw") * pl.col("pver"),
                    ptc__oc=-product("ptc_gen", "ptc", "ptc_pver"),
                )
                .with_columns(
                    total__oc=pl.sum_horizontal(cs.contains("__oc")),
                )
            )
        return (
            cost_df.with_columns(fcr=pl.lit(self.m.fcr))
            .with_columns(
                capex__oc=product("capex_raw", "reg_mult", "itc_adj", "life_adj", "fcr"),
                tx_capex__oc=product("tx_capex_raw", "distance", "fcr"),
                opex__oc=pl.col("opex_raw"),
                ptc__oc=-product("ptc_gen", "ptc"),
            )
            .with_columns(
                total__oc=pl.sum_horizontal(cs.contains("__oc")),
            )
        )

    def __eq__(self, other: Self) -> bool:
        return self is other

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = f"{sum(self.x_cap.value).item():,.0f}"
        return self.__class__.__qualname__ + f"({cap})"

    def _get_prof(self, attr: str):
        func = {"x": lambda x: x.value, "cost": lambda x: x}[attr]
        if len(self.var_names) == 1:
            return pl.concat(
                pl.DataFrame({self.var_names[0]: func(v)}).with_columns(year=pl.lit(str(k)))
                for k, v in getattr(self, attr).items()
            )
        return pl.concat(
            (
                pl.DataFrame(
                    {n: func(x) for n, x in zip(self.var_names, v, strict=False)}
                ).with_columns(year=pl.lit(str(k)))
                for k, v in getattr(self, attr).items()
            ),
            how="vertical_relaxed",
        )

    def _set_prof(self, obj, attr):
        if attr == "x":

            def _get_1_prof(y, nm):
                val = obj[f"{attr}_{self.cat}"].filter(pl.col("year") == y)[nm].to_numpy()
                var = cp.Variable(len(val), name=nm)
                var.value = val
                return var
        else:

            def _get_1_prof(y, nm):
                val = obj[f"{attr}_{self.cat}"].filter(pl.col("year") == y)[nm].to_numpy()
                if val.size == 1:
                    return val.item()
                return val

        try:
            yrs = obj[f"{attr}_{self.cat}"]["year"].unique().sort()
            if len(self.var_names) == 1:
                n = self.var_names[0]
                out = {lit_eval(y): _get_1_prof(y, n) for y in yrs}
            else:
                out = {
                    lit_eval(y): tuple(_get_1_prof(y, n) for n in self.var_names) for y in yrs
                }
            setattr(self, attr, out)
        except Exception as exc:
            raise RuntimeError(f"{self!r} {attr} {exc!r}") from exc

    def _get_x_cap(self):
        if self.x_cap is None:
            return None
        if isinstance(self.x_cap, tuple):
            return [{"name": x.name(), "value": x.value, "shape": x.shape} for x in self.x_cap]
        return {
            "name": self.x_cap.name(),
            "value": self.x_cap.value,
            "shape": self.x_cap.shape,
        }

    def _set_x_cap(self, other):
        if (x_cap := other[f"x_cap_{self.cat}"]) is None:
            self.x_cap = None
        if isinstance(x_cap, dict):
            self.x_cap = cp.Variable(**x_cap)
        else:
            self.x_cap = MultiXCapShim([cp.Variable(**x) for x in x_cap])

    def _get_cost_cap(self):
        return pl.concat(
            (v.with_columns(year=pl.lit(str(k))) for k, v in self.cost_cap.items()),
            how="diagonal_relaxed",
        )

    def _set_cost_cap(self, other):
        self.cost_cap = {
            lit_eval(y): other[f"cost_cap_{self.cat}"].filter(pl.col("year") == y)
            for y in other[f"cost_cap_{self.cat}"]["year"].unique().sort()
        }

    def _dzgetstate_(self):
        state = dict(self.__dict__) | {"_state": {}}
        del state["m"]
        if "x_cap" in self._dz:
            del state["x_cap"]
            state["_state"][f"x_cap_{self.cat}"] = self._get_x_cap()
        if "cost_cap" in self._dz:
            del state["cost_cap"]
            state["_state"][f"cost_cap_{self.cat}"] = self._get_cost_cap()
        for attr in self._dz.get("profs", []):
            state["_state"][f"{attr}_{self.cat}"] = self._get_prof(attr)
            del state[attr]
        return state

    def _dzsetstate_(self, state: dict) -> None:
        other = state.pop("_state")
        self.__dict__.update(state)
        if "x_cap" in self._dz:
            self._set_x_cap(other)
        if "cost_cap" in self._dz:
            self._set_cost_cap(other)
        for attr in self._dz.get("profs", []):
            self._set_prof(other, attr)


class Load(DecisionVariable):
    _var_names = ("load",)
    _cat = "load"

    def __init__(self, m: "Model", sqkm_per_mw: float, ld_value: float) -> None:
        self.x_cap: ParamVarLike
        super().__init__(m)
        self.sqkm_per_mw = sqkm_per_mw
        self.ld_value = ld_value

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        out = self.m.d.ba_pro.select("datetime", pl.lit(self.x_cap.value.item()).alias("load"))
        if selection:
            return prof(out, self.m.i.years, cs.all)
        return out.filter(pl.col("datetime").dt.year().is_in(self.m.dispatchs))

    def common_hrly_constraint(self, yr: tuple, *args) -> NotImplementedType | cp.Expression:
        return NotImplemented

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr, *args)

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr, *args)

    @check_shape()
    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr, *args)

    @check_shape()
    def land(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return cp.multiply(
                hstack([self.sqkm_per_mw], np.zeros(len(self.m.re_land) - 1)), self.x_cap
            )
        return cp.Constant(0.0)


class ConstantLoad(Load):
    def __init__(self, m: "Model", sqkm_per_mw: float, ld_value, x_cap: float):
        """Returns:
        object:
        """
        super().__init__(m, sqkm_per_mw, ld_value)
        self.x_cap = cp.Parameter(1, name="load_cap", value=[x_cap])

    @check_shape()
    def common_hrly_constraint(self, yr: tuple, *args) -> cp.Expression:
        return -cp.Constant(self.x_cap.value.item())

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return -self.m.max_pct_fos * 8760 * len(yr) * self.x_cap.value.item()
        return -cp.Constant(8760 * len(yr) * self.x_cap.value.item())


class EndogenousLoad(Load):
    _dz = {"x_cap": True, "profs": []}

    def __init__(
        self,
        m: "Model",
        sqkm_per_mw: float = 1 / 247,
        ld_value=1000,
        x_cap: float | None = None,
    ):
        """Returns:
        object:
        """
        super().__init__(m, sqkm_per_mw, ld_value)
        self.x_cap = cp.Variable(1, name="load_cap")
        if x_cap is not None:
            self.x_cap.value = x_cap

    def round(self):
        self.x_cap.value = 25 * (self.x_cap.value // 25)

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["load_mw"]

    @check_shape()
    def common_hrly_constraint(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return -self.x_cap
        return -cp.Constant(self.x_cap.value)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return -self.m.max_pct_fos * 17_520 * self.x_cap
        return -cp.Constant(8760 * self.x_cap.value.item())

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        if is_resource_selection(yr):
            self.cost[yr] = -self.ld_value * 17_520
            return self.cost[yr] * self.x_cap
        return cp.Constant(0.0)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return self.x_cap >= 0.0, self.x_cap <= self.m.i.cap * 0.75
        return ()


class FlexLoad(Load):
    _dz = {"x_cap": True, "profs": ["x"]}

    def __init__(
        self,
        m: "Model",
        sqkm_per_mw: float = 1 / 247,
        ld_value: float = 1000,
        uptime: float = 0.5,
        min_load: float = 0.5,
        x_cap: float | None = None,
    ):
        """Returns:
        object:
        """
        super().__init__(m, sqkm_per_mw, ld_value)
        self.x_cap = cp.Variable(1, name="load_cap")
        self.uptime = cp.Parameter(1, value=[uptime], name="uptime")
        self.min_load = cp.Parameter(1, value=[min_load], name="min_load")
        if x_cap is not None:
            self.x_cap.value = x_cap

    def round(self):
        self.x_cap.value = 25 * (self.x_cap.value // 25)

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["load_mw"]

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        if selection:
            out = self.m.d.ba_pro.select(
                "datetime", pl.lit(self.x_cap.value.item()).alias("load")
            )
            return prof(out, self.m.i.years, cs.all)
        return DecisionVariable.hourly(self, selection=selection)

    @check_shape()
    def common_hrly_constraint(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return -self.x_cap
        return -self.get_x(yr)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return -self.m.max_pct_fos * 17_520 * self.x_cap
        return -cp.sum(self.get_x(yr))

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return ()
        return (cp.sum(self.get_x(yr)) >= 8760 * cp.multiply(self.uptime, self.x_cap.value),)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        if is_resource_selection(yr):
            self.cost[yr] = -self.ld_value * 17_520
            return self.cost[yr] * self.x_cap
        self.cost[yr] = -self.ld_value
        return cp.sum(self.cost[yr] * self.get_x(yr))

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return self.x_cap >= 0.0, self.x_cap <= self.m.i.cap * self.m.load_icx_max_mult
        return self.get_x(yr) >= cp.multiply(self.min_load, self.x_cap.value), self.get_x(
            yr
        ) <= self.x_cap.value


class Storage(DecisionVariable):
    sum_cols = ("capex_raw", "opex_raw", "capex__oc", "opex__oc", "total__oc")
    max_cols = (
        "dur",
        "life_adj",
        "itc_adj",
        "reg_mult",
        "tx_capex_raw",
        "distance",
        "ptc",
        "ptc_gen",
    )
    _var_names = ("discharge", "charge", "soc")
    _cat = "storage"
    _type = ""
    _dz = {"x_cap": True, "profs": ["x"], "cost_cap": True}

    def __init__(self, m: "Model", d_eff, c_eff, l_eff, pre_charge, opex_mult, min_soc_hrs):
        super().__init__(m)
        self.d_ef: float = d_eff
        self.c_ef: float = c_eff
        self.l_ef: float = l_eff
        self.pre_charge = cp.Parameter(1, name="pre_charge", value=[pre_charge])
        self.min_soc_hrs = cp.Parameter(1, name="min_soc_hrs", value=[min_soc_hrs])
        self.opex_mult: float = opex_mult
        ec = (
            (
                self.m.d.ba_data["plant_data"]
                .query("plant_id_eia == @self.m.i.pid & generator_id in @self.m.i.gens")
                .energy_community.any()
                / 10
            )
            if self.m.itc
            else 0
        )
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": [0, 0],
                "re_type": [f"{self._type}_ac", f"{self._type}_dc"],
                "capex_raw": ES.filter(pl.col("year") == self.m.build_year)[
                    "ac_" + self.m.atb_scenario, "dc_" + self.m.atb_scenario
                ]
                .to_numpy()
                .flatten(),
                "life_adj": [1.0, 1.0],
                "dur": [0, 0],
                "itc_adj": [1 - (self.m.itc + ec) * COSTS["fmv_step"]["es"]] * 2,
                "reg_mult": [1.0, 1.0],
                "tx_capex_raw": [0.0, 0.0],
                "distance": [0.0, 0.0],
                "opex_raw": [None, None],
                "ptc": [0.0, 0.0],
                "ptc_gen": [0.0, 0.0],
            },
        )

    @property
    def cat(self):
        return f"{self._type}_storage"

    @property
    def var_names(self):
        return f"{self._type}_discharge", f"{self._type}_charge", f"{self._type}_soc"

    def round(self):
        self.x_cap.value = self.x_cap.value + (25 - self.x_cap.value) % 25

    @property
    def dur(self) -> ParamVarLike:
        return self._dur

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        d, ch, _ = self.get_x(yr)
        return d - ch

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        d, ch, _ = self.get_x(yr)
        return d - ch

    @check_shape()
    def clean_export(self, yr: tuple, *args) -> cp.Expression:
        d, ch, _ = self.get_x(yr)
        return d - ch

    def opex_oc(self, yr: tuple, *args) -> np.ndarray:
        if is_resource_selection(yr):
            return f_npv(
                self.m.r_disc,
                ES.filter(pl.col("year") >= 2026)
                .head(30)["ac_" + self.m.atb_scenario, "dc_" + self.m.atb_scenario]
                .to_numpy()
                * self.opex_mult,
            )
        return (
            ES.filter(pl.col("year") == yr[0] + 2)[
                "ac_" + self.m.atb_scenario, "dc_" + self.m.atb_scenario
            ]
            .to_numpy()
            .flatten()
            * self.opex_mult
        )


class _StorageCostCapShim:
    """This allows storage duration to be a parameter while preserving the
    general interface for DecisionVariable.cap_cost
    """

    def __init__(self, dv: "FixedDurationStorage") -> None:
        self.dv = dv

    def __getitem__(self, yr):
        out = self.dv.mk_obj_cost(self.dv.cost_df, yr, self.dv.yr_fact_map)
        if is_resource_selection(yr):
            opex_oc = f_npv(
                self.dv.m.r_disc,
                ES.filter(pl.col("year") >= 2026).head(30)["ac", "dc"].to_numpy()
                * self.dv.opex_mult,
            )
            opex_raw = opex_oc / f_pv(self.dv.m.r_disc, 30, -1)
        else:
            opex_raw = opex_oc = (
                ES.filter(pl.col("year") == yr[0] + 2)["ac", "dc"].to_numpy().flatten()
                * self.dv.opex_mult
            )
        return (
            out.with_columns(dur=pl.Series([1, self.dv.dur.value]))
            .with_columns(
                re_type=pl.lit("storage"),
                capex_raw=pl.col("capex_raw") * pl.col("dur"),
                opex_raw=opex_raw * pl.col("dur"),
                opex__oc=opex_oc * pl.col("dur"),
                capex__oc=pl.col("capex__oc") * pl.col("dur"),
            )
            .with_columns(total__oc=pl.sum_horizontal(cs.contains("__oc")))
            .group_by("re_site_id", "re_type")
            .agg(
                pl.col(self.dv.sum_cols).sum(),
                pl.col(self.dv.max_cols).max(),
            )
        )

    def items(self):
        for y in [self.dv.m.i.years] + [(y,) for y in self.dv.m.d.opt_years]:
            yield y, self[y]

    def pop(self, key, default=None):
        return None


class FixedDurationStorage(Storage):
    """x_s, x_dt, x_ct, x_st"""

    _type = "li"
    _dz = {"x_cap": True, "profs": ["x"]}

    def __init__(
        self,
        m: "Model",
        d_eff: float = COSTS["eff"]["d"],
        c_eff: float = COSTS["eff"]["c"],
        l_eff: float = COSTS["eff"]["l"],
        pre_charge: float = 0.5,
        opex_mult: float = 0.8 / 30,  # ATB
        duration: int = 8,
        x_cap: float | None = None,
        min_soc_hrs: int = 0,
    ):
        super().__init__(m, d_eff, c_eff, l_eff, pre_charge, opex_mult, min_soc_hrs)
        self._dur = cp.Parameter(1, name="duration", value=[duration])
        self.yr_fact_map = {}
        self.x_cap = cp.Variable(1, name="storage_cap")
        self.cost_cap = _StorageCostCapShim(self)
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def cap_summary(self):
        return {"li_mw": self.x_cap.value.item(), "li_duration": self.dur.value}

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["li_mw"]

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        d, ch, s = self.get_x(yr)
        x = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        return (
            s[1:] == s[:-1] - d[1:] / self.d_ef + ch[1:] * self.c_ef - s[1:] * self.l_ef,
            s <= self.dur * x,
            s + ch <= self.dur * x,
            ch <= x,
            d <= x,
            d <= s,
            d + ch <= x,
            s[soc_ixs] == cp.multiply(self.dur, cp.multiply(self.pre_charge, x)),
            d[0] == 0.0,
            ch[0] == 0.0,
            s >= cp.multiply(x, self.min_soc_hrs),
        )

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        cost = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        if is_resource_selection(yr):
            self.yr_fact_map = yr_fact_map
            x_d = self.x_cap
        else:
            x_d = self.x_cap.value
        c_d, c_s = cost.with_columns(opex__oc=self.opex_oc(yr)).with_columns(
            total__oc=pl.sum_horizontal(cs.contains("__oc")),
        )["total__oc"]
        d, ch, _ = self.get_x(yr)
        return (c_d + c_s * self.dur) * x_d

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        limit = 2e4 if is_resource_selection(yr) else self.x_cap.value
        d, ch, s = self.get_x(yr)
        core = d >= 0.0, ch >= 0.0, s >= 0.0, d <= limit, ch <= limit, s <= limit * self.dur
        if is_resource_selection(yr):
            return *core, self.x_cap >= 0.0, self.x_cap <= 2e4
        return core

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = f"({self.x_cap.value.item():.0f} {self.dur.value:.1f}hr)"
        return self.__class__.__qualname__ + f"({cap})"


class FeStorage(Storage):
    """x_s, x_dt, x_ct, x_st"""

    _dz = {"x_cap": True, "profs": ["x"], "cost_cap": True}
    _type = "fe"

    def __init__(
        self,
        m: "Model",
        pre_charge: float = 0.5,
        x_cap: float | None = None,
        min_soc_hrs: int = 0,
    ):
        super().__init__(m, 0.62, 0.73, 0.0001, pre_charge, 0.0, min_soc_hrs)
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": [1],
                "re_type": ["fe_ac"],
                "capex_raw": [23 * 100 * 1000],
                "life_adj": [f_pv(self.m.r_disc, 30, f_pmt(self.m.r_disc, 20, 1))],
                "dur": [100],
                "itc_adj": [1 - COSTS["itc"] * COSTS["fmv_step"]["es"]],
                "reg_mult": [1.0],
                "tx_capex_raw": [0.0],
                "distance": [0.0],
                "opex_raw": [20 * 1000],
                "ptc": [0.0],
                "ptc_gen": [0.0],
            },
        )
        self._dur = cp.Constant([100], name="duration")
        self.x_cap = cp.Variable(1, name="fe_ac")
        self.cost_cap = {}
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def cap_summary(self):
        return {
            "fe_mw": self.x_cap.value.item(),
            "fe_duration": self.dur.value.item(),
        }

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["fe_mw"]

    def for_econ(self) -> list[pl.LazyFrame]:
        return [
            self.cost_cap[(y,)]
            .with_columns(
                datetime=pl.lit(f"{y}-01-01").str.to_datetime(),
                re_type=pl.lit("fe_storage"),
                capacity_mw=self.x_cap.value,
                generator_id=pl.lit("fe_storage"),
            )
            .lazy()
            for y in self.m.d.opt_years
        ]

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        d, ch, s = self.get_x(yr)
        if is_resource_selection(yr):
            x = self.x_cap
            ef_mod, en_cap_mod = 0.995**20, 0.98**20
        else:
            x = self.x_cap.value
            ef_mod = 0.995 ** (yr[0] - self.m.d.opt_years[0])
            en_cap_mod = 0.98 ** (yr[0] - self.m.d.opt_years[0])
        d_ef, c_ef = 1 / (self.d_ef * ef_mod), self.c_ef * ef_mod
        return (
            s[1:] == s[:-1] - d[1:] * d_ef + ch[1:] * c_ef - s[1:] * self.l_ef,
            s <= en_cap_mod * cp.multiply(self._dur, x),
            s + ch <= en_cap_mod * cp.multiply(self._dur, x),
            ch <= 2 * x,
            d <= x,
            d <= s,
            d + ch <= 2 * x,
            s[soc_ixs] == en_cap_mod * cp.multiply(self.pre_charge, cp.multiply(self._dur, x)),
            d[0] == 0.0,
            ch[0] == 0.0,
            cp.sum(d) <= 2000 * len(yr) * x,
            s >= cp.multiply(x, self.min_soc_hrs),
        )

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        self.cost_cap[yr] = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        x_d = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        c_d = self.cost_cap[yr]["total__oc"].item()
        d, ch, _ = self.get_x(yr)
        return c_d * x_d

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            limit = 2e5
            en_cap_mod = 0.98**20
        else:
            limit = self.x_cap.value.item()
            en_cap_mod = 0.98 ** (yr[0] - self.m.d.opt_years[0])
        d, ch, s = self.get_x(yr)
        core = (
            d >= 0.0,
            ch >= 0.0,
            s >= 0.0,
            d <= limit,
            ch <= 2 * limit,
            s <= self._dur * en_cap_mod * limit,
        )
        if is_resource_selection(yr):
            return *core, self.x_cap >= 0.0, self.x_cap <= 2e5
        return core


class EndogenousDurationStorage(Storage):
    """x_d, x_s, x_dt, x_ct, x_st"""

    _type = "li"
    _dz = {"x_cap": True, "profs": ["x"], "cost_cap": True}

    def __init__(
        self,
        m: "Model",
        d_eff: float = COSTS["eff"]["d"],
        c_eff: float = COSTS["eff"]["c"],
        l_eff: float = COSTS["eff"]["l"],
        pre_charge: float = 0.5,
        opex_mult: float = 0.8 / 30,  # ATB
        x_cap: float | None = None,
        min_soc_hrs: int = 0,
    ):
        super().__init__(m, d_eff, c_eff, l_eff, pre_charge, opex_mult, min_soc_hrs)
        self.x_cap = MultiXCapShim(
            (
                cp.Variable(1, name="li_ac"),
                cp.Variable(1, name="li_dc"),
            )
        )
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def cap_names(self) -> tuple:
        return "li_ac", "li_dc"

    @property
    def cap_summary(self) -> dict[str, float]:
        return {"li_mw": self.x_cap.value[0].item(), "li_duration": self.dur.value.item()}

    def set_x_cap(self, cap_summary, re_selected, *args):
        ac = cap_summary["li_mw"]
        dc = [cap_summary["li_duration"][0] * ac[0]]
        self.x_cap.value = [ac, dc]

    def for_econ(self):
        return [
            self.cost_cap[(y,)]
            .with_columns(
                datetime=pl.lit(f"{y}-01-01").str.to_datetime(),
                re_type=pl.lit("li_storage"),
                capacity_mw=hstack(self.x_cap.value[0], 0),
                generator_id=pl.lit("li_storage"),
                opex_raw=pl.col("opex_raw") * self.x_cap.value,
            )
            .group_by("re_site_id", "re_type", "datetime", "generator_id")
            .agg(pl.col("opex_raw").sum(), pl.col("capacity_mw", *Storage.max_cols).max())
            .with_columns(
                dur=self.dur.value.item(), opex_raw=pl.col("opex_raw") / pl.col("capacity_mw")
            )
            .lazy()
            for y in self.m.d.opt_years
        ]

    def round(self):
        ac, dc = self.x_cap.value
        if np.isclose(ac, 0.0) and (np.isclose(dc, 0.0) or np.isnan(dc)):
            self.x_cap.value = 0.0, 0.0
        else:
            ac = ac + (25 - ac) % 25
            dur = dc / ac
            dur = dur + (2 - dur) % 2
            self.x_cap.value = ac, ac * dur

    @property
    def dur(self) -> ParamVarLike:
        x_d, x_s = self.x_cap.value
        if np.isclose(x_d, 0):
            if not np.isclose(x_s, 0) and not np.isnan(x_s):
                raise RuntimeError(f"impossible storage capacities {x_d=}, {x_s=}")
            return SimpleNamespace(value=np.float64(0), name="duration")
        return SimpleNamespace(value=x_s / x_d, name="duration")

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        d, ch, s = self.get_x(yr)
        x_d, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        return (
            s[1:] == s[:-1] - d[1:] / self.d_ef + ch[1:] * self.c_ef - s[1:] * self.l_ef,
            s <= x_s,
            s + ch <= x_s,
            ch <= x_d,
            d <= x_d,
            d <= s,
            d + ch <= x_d,
            s[soc_ixs] == cp.multiply(self.pre_charge, x_s),
            d[0] == 0.0,
            ch[0] == 0.0,
            s >= cp.multiply(x_d, self.min_soc_hrs),
        )

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        cost = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        opex_oc = self.opex_oc(yr)
        raw_adj = f_pv(self.m.r_disc, 30, -1) if is_resource_selection(yr) else 1.0
        self.cost_cap[yr] = cost.with_columns(
            opex_raw=opex_oc / raw_adj, opex__oc=opex_oc
        ).with_columns(
            total__oc=pl.sum_horizontal(cs.contains("__oc")),
        )
        c_d, c_s = self.cost_cap[yr]["total__oc"]
        x_d, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        d, ch, _ = self.get_x(yr)
        return c_d * x_d + c_s * x_s  # + 5 * (cp.sum(d) + cp.sum(ch))

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        l_d, l_s = (2e4, 3e5) if is_resource_selection(yr) else self.x_cap.value
        d, ch, s = self.get_x(yr)
        core = d >= 0.0, ch >= 0.0, s >= 0.0, d <= l_d, ch <= l_d, s <= l_s
        if is_resource_selection(yr):
            x_d, x_s = self.x_cap
            return *core, x_d >= 0.0, x_s >= 0.0, x_d <= 2e4, x_s <= 3e5, x_s >= x_d
        return core

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = f"{self.x_cap.value[0].item():.0f} {self.dur.value:.1f}hr"
        return self.__class__.__qualname__ + f"({cap})"


class ThermalStorageLoad(Storage):
    """Charges thermal storage to meet a constant (thermal) load.

    x_d, x_c, x_s, x_ct, x_st

    """

    _type = "thermal"
    _dz = {"x_cap": True, "profs": ["x"], "cost_cap": True}

    def __init__(
        self,
        m: "Model",
        d_eff: float = 0.5,
        c_eff: float = 1.0,
        l_eff: float = 0.01,
        pre_charge: float = 0.5,
        opex_mult: float = 0.025,  # ATB
        x_cap: float | None = None,
        const_load: float = 10.0,
        min_soc_hrs: int = 0,
    ):
        super().__init__(m, d_eff, c_eff, l_eff, pre_charge, opex_mult, min_soc_hrs)
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": [9, 9, 9],
                "re_type": ["discharge_cap", "charge_cap", "storage_capacity"],
                "capex_raw": [0.0, 1.0, 0.5],
                "life_adj": [1.0, 1.0, 1.0],
                "dur": [0, 0, 0],
                "itc_adj": [1 - COSTS["itc"] * COSTS["fmv_step"]["es"]] * 3,
                "reg_mult": [1.0, 1.0, 1.0],
                "tx_capex_raw": [0.0, 0.0, 0.0],
                "distance": [0.0, 0.0, 0.0],
                "opex_raw": [-25.0, 0.0, 0.0],
                "ptc": [0.0, 0.0, 0.0],
                "ptc_gen": [0.0, 0.0, 0.0],
            },
        )
        self.const_load = const_load
        self.x_cap = MultiXCapShim(
            (
                cp.Variable(1, name="thermal_load_cap"),
                cp.Variable(1, name="thermal_charge_cap"),
                cp.Variable(1, name="thermal_soc_max"),
            )
        )
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def cap_names(self) -> tuple:
        return "thermal_load_cap", "thermal_charge_cap", "thermal_soc_max"

    @property
    def cap_summary(self):
        return {
            "thermal_load_mw": self.x_cap.value[0].item(),
            "charge_mw": self.x_cap.value[1].item(),
            "duration": self.dur.value,
        }

    def set_x_cap(self, cap_summary, re_selected, *args):
        ac = cap_summary["thermal_load_mw"]
        dc = [cap_summary["duration"][0] * ac[0]]
        self.x_cap.value = [ac, cap_summary["charge_mw"], dc]

    @property
    def dur(self) -> ParamVarLike:
        x_d, _, x_s = self.x_cap.value
        return SimpleNamespace(value=x_s / x_d, name="duration")

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        if selection:
            d, c, s = self.get_x(self.m.i.years)
            dn, cn, sn = self.var_names
            return prof(self.m.d.ba_pro, self.m.i.years, cs.datetime).select(
                "datetime",
                pl.lit(self.x_cap.value[0].item()).alias(dn),
                pl.lit(c.value).alias(cn),
                pl.lit(s.value).alias(sn),
            )
        return DecisionVariable.hourly(self, selection=selection)

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        _, ch, _ = self.get_x(yr)
        return -ch

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        _, ch, _ = self.get_x(yr)
        return -ch

    @check_shape()
    def clean_export(self, yr: tuple, *args) -> cp.Expression:
        _, ch, _ = self.get_x(yr)
        return -ch

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        d, ch, s = self.get_x(yr)
        x_d, x_c, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        if is_resource_selection(yr):
            return (
                s[1:] == s[:-1] - x_d / self.d_ef + ch[1:] * self.c_ef - s[1:] * self.l_ef,
                s <= x_s,
                s + ch <= x_s,
                ch <= x_c,
                x_d <= s,
                s[soc_ixs] == cp.multiply(self.pre_charge, x_s),
                ch[0] == 0.0,
                s >= cp.multiply(x_d, self.min_soc_hrs),
            )
        return (
            s[1:] == s[:-1] - d / self.d_ef + ch[1:] * self.c_ef - s[1:] * self.l_ef,
            s <= x_s,
            s + ch <= x_s,
            ch <= x_c,
            d <= x_d,
            d <= s,
            s[soc_ixs] == cp.multiply(self.pre_charge, x_s),
            d[0] == 0.0,
            ch[0] == 0.0,
            s >= cp.multiply(x_d, self.min_soc_hrs),
        )

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        self.cost_cap[yr] = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        x_d, x_c, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        c_d, c_c, c_s = self.cost_cap[yr]["total__oc"]
        if is_resource_selection(yr):
            d_term = len(yr) * 8760 * x_d
        else:
            d, _, _ = self.get_x(yr)
            d_term = cp.sum(d)
        return c_d * d_term + c_c * x_c + c_s * x_s

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        l_d, l_c, l_s = (2e4, 2e4, 2e6) if is_resource_selection(yr) else self.x_cap.value
        d, ch, s = self.get_x(yr)
        core = ch >= 0.0, s >= 0.0, ch <= l_c, s <= l_s
        if is_resource_selection(yr):
            x_d, x_c, x_s = self.x_cap
            return (
                *core,
                x_d >= 0.0,
                x_c >= 0.0,
                x_s >= 0.0,
                x_d <= 2e4,
                x_c <= 2e4,
                x_s <= 2e4,
            )
        return core, d <= l_d

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = f"{self.x_cap.value[0].item():.0f} {self.dur.value:.1f}hr"
        return self.__class__.__qualname__ + f"({cap})"


class DecoupledStorage(Storage):
    """x_d, x_c, x_s, x_dt, x_ct, x_st"""

    _type = "long"
    _dz = {"x_cap": True, "profs": ["x"], "cost_cap": True}

    def __init__(
        self,
        m: "Model",
        d_eff: float = 0.5,
        c_eff: float = 1.0,
        l_eff: float = 0.01,
        pre_charge: float = 0.5,
        opex_mult: float = 0.025,  # ATB
        x_cap: float | None = None,
        min_soc_hrs: int = 0,
    ):
        super().__init__(m, d_eff, c_eff, l_eff, pre_charge, opex_mult, min_soc_hrs)
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": [0, 0, 0],
                "re_type": ["discharge_cap", "charge_cap", "storage_capacity"],
                "capex_raw": [1e5, 3e4, 3e3],
                "life_adj": [1.0, 1.0, 1.0],
                "dur": [0, 0, 0],
                "itc_adj": [1 - COSTS["itc"] * COSTS["fmv_step"]["es"]] * 3,
                "reg_mult": [1.0, 1.0, 1.0],
                "tx_capex_raw": [0.0, 0.0, 0.0],
                "distance": [0.0, 0.0, 0.0],
                "opex_raw": [20.0, 20.0, 20.0],
                "ptc": [0.0, 0.0, 0.0],
                "ptc_gen": [0.0, 0.0, 0.0],
            },
        )
        self.x_cap = MultiXCapShim(
            (
                cp.Variable(1, name="discharge_cap"),
                cp.Variable(1, name="charge_cap"),
                cp.Variable(1, name="max_soc"),
            )
        )
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def cap_names(self) -> tuple:
        return "discharge_cap", "charge_cap", "storage_capacity"

    @property
    def cap_summary(self):
        return {
            "storage_mw": self.x_cap.value[0].item(),
            "charge_mw": self.x_cap.value[1].item(),
            "duration": self.dur.value,
        }

    @property
    def dur(self) -> ParamVarLike:
        x_d, _, x_s = self.x_cap.value
        return SimpleNamespace(value=x_s / x_d, name="duration")

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        d, ch, s = self.get_x(yr)
        x_d, x_c, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        return (
            s[1:] == s[:-1] - d[1:] / self.d_ef + ch[1:] * self.c_ef - s[1:] * self.l_ef,
            d <= x_d,
            ch <= x_c,
            s <= x_s,
            d <= s,
            s + ch <= x_s,
            s[soc_ixs] == cp.multiply(self.pre_charge, x_s),
            d[0] == 0.0,
            ch[0] == 0.0,
            s >= cp.multiply(x_d, self.min_soc_hrs),
        )

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        self.cost_cap[yr] = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        x_d, x_c, x_s = self.x_cap if is_resource_selection(yr) else self.x_cap.value
        c_d, c_c, c_s = self.cost_cap[yr]["total__oc"]
        d, ch, _ = self.get_x(yr)
        return c_d * x_d + c_c * x_c + c_s * x_s

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        l_d, l_c, l_s = (2e4, 2e4, 2e4) if is_resource_selection(yr) else self.x_cap.value
        d, ch, s = self.get_x(yr)
        core = d >= 0.0, ch >= 0.0, s >= 0.0, d <= l_d, ch <= l_c, s <= l_s
        if is_resource_selection(yr):
            x_d, x_c, x_s = self.x_cap
            return (
                *core,
                x_d >= 0.0,
                x_c >= 0.0,
                x_s >= 0.0,
                x_d <= 2e4,
                x_c <= 2e4,
                x_s <= 2e4,
            )
        return core

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = f"({self.x_cap.value[0].item():.0f} {self.dur.value:.1f}hr)"
        return self.__class__.__qualname__ + f"({cap})"


class Renewables(DecisionVariable):
    """x_j"""

    _cat = "renewables"
    _dz = {"x_cap": True, "profs": [], "cost_cap": True}

    def __init__(self, m, x_cap: np.ndarray | None = None):
        super().__init__(m)
        self.x_cap = cp.Variable(len(self.m.d.re_ids), name="renewables_cap")
        if x_cap is not None:
            self.x_cap.value = x_cap
        re_specs = self.m.d.re_specs.collect()
        with all_logging_disabled():
            atb = clean_atb(
                scenario=self.m.atb_scenario,
                report_year=2024,
                pudl_release=self.m.pudl_release,
            ).filter(
                (pl.col("projection_year") == self.m.build_year)
                & pl.col("technology_description").is_in(re_specs["technology_description"])
            )
        self.cost_df = re_specs.join(
            atb,
            on=["technology_description", "class_atb"],
            validate="m:1",
            how="left",
        ).select(
            "re_site_id",
            "re_type",
            "reg_mult",
            "distance",
            "combi_id",
            "energy_community",
            capex_raw=(
                pl.sum_horizontal(
                    "capex_overnight_per_kw", "capex_construction_finance_factor"
                )
            )
            * 1000,  # Fixme
            life_adj=pl.lit(1.0),
            dur=pl.lit(0),
            tx_capex_raw=pl.lit(COSTS["capex"]["tx"]),
            opex_raw=pl.col("fom_per_kw") * 1000,
            itc_adj=pl.lit(1.0),
        )

    def round(self):
        self.x_cap.value = self.x_cap.value + (10 - self.x_cap.value) % 10

    @property
    def cap_summary(self) -> dict:
        return {
            f"{k}_mw": v[0][0]
            for k, v in self.m.re_selected.group_by("re_type")
            .agg(pl.col("capacity_mw").sum())
            .rows_by_key("re_type")
            .items()
        }

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = (
            self.m.d.re_specs.select("re_site_id", "re_type")
            .join(
                re_selected.select("re_site_id", "re_type", "capacity_mw"),
                on=["re_site_id", "re_type"],
                how="left",
                maintain_order="left",
            )
            .select("capacity_mw")
            .fill_null(0.0)
            .collect()
            .to_series()
            .to_numpy()
        )

    @property
    def cap_names(self) -> tuple:
        return self.m.d.re_ids

    def get_x(self, yr: tuple) -> cp.Variable:
        return self.x_cap

    def hourly(self, *, selection=False, by_type=True) -> pl.LazyFrame:
        out = self.m.d.re_pro.select(
            "datetime",
            *[pl.col(rid).mul(self.x_cap.value[i]) for i, rid in enumerate(self.m.d.re_ids)],
        )
        if by_type:
            out = out.select(
                "datetime",
                *[pl.sum_horizontal(cs.contains(rt)).alias(rt) for rt in self.m.re_types],
            )
        if selection:
            return prof(out, self.m.i.years, cs.all)
        return out.filter(pl.col("datetime").dt.year().is_in(self.m.dispatchs))

    def annual_gen(self) -> cp.Expression:
        return (
            self.m.d.re_pro.group_by_dynamic("datetime", every="1y")
            .agg(pl.all().sum())
            .select(self.m.d.re_ids)
            .collect()
            .to_numpy()
            @ self.x_cap
        )

    def common_hrly_constraint(self, yr):
        if is_resource_selection(yr):
            perf_mult = (1 - self.m.solar_degrade_per_year) ** 20
            pro = (
                prof(self.m.d.re_pro.with_columns(cs.contains("solar") * perf_mult), yr)
                .collect()
                .to_numpy()
            )
            return pro @ self.x_cap
        perf_mult = (1 - self.m.solar_degrade_per_year) ** (yr[0] - self.m.d.opt_years[0])
        pro = (
            prof(self.m.d.re_pro.with_columns(cs.contains("solar") * perf_mult), yr)
            .collect()
            .to_numpy()
        )
        return cp.Constant(pro @ self.x_cap.value)

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def clean_export(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def land(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            # pl.from_pandas(self.m.re_land).select(cs.exclude(cs.contains("rhs")))
            return self.m.re_land.to_numpy(nt)[:, :-2] @ self.x_cap
        return cp.Constant(0.0)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        cost_df = (
            self.cost_df.join(
                # doesn't this formula work for a single year??
                prof(self.m.d.re_pro, (yr[0],))
                .sum()
                .collect()
                .transpose(
                    include_header=True, header_name="combi_id", column_names=["ptc_gen"]
                ),
                on="combi_id",
            )
            .with_columns(
                ptc=pl.lit(self.ptc(yr))
                * pl.col("energy_community").replace_strict(
                    {True: 1.1, False: 1.0}, default=1.0
                )
            )
            .select(self.m.cost_cols)
        )
        self.cost_cap[yr] = self.mk_obj_cost(cost_df, yr, yr_fact_map)
        out = self.cost_cap[yr]["total__oc"].to_numpy()
        if is_resource_selection(yr):
            return out @ self.x_cap
        return cp.Constant(out @ self.x_cap.value)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return self.x_cap >= 0.0, self.x_cap <= 2e4
        return ()

    def __repr__(self) -> str:
        cap = ""
        if self.x_cap is not None and self.x_cap.value is not None:
            cap = " ".join(
                f"{k}={v:.0f}"
                for k, v in self.m.re_selected.group_by("re_type")
                .agg(pl.sum("capacity_mw"))
                .iter_rows()
            )
        return self.__class__.__qualname__ + f"({cap})"


class Curtailment(DecisionVariable):
    _var_names = ("curtailment",)
    _cat = "curtailment"
    _dz = {"profs": ["x", "cost"]}

    def __init__(self, m, min_cost=0.0) -> None:
        super().__init__(m)
        self.min_cost = min_cost

    def c_hourly(self, selection=False) -> pl.LazyFrame | None:
        return self._h_core(selection, self.cost, "c_", func=lambda x: x)

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return -self.get_x(yr)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        if is_resource_selection(yr):
            self.cost[yr] = (
                prof(self.m.d.ba_pro, yr, cs.datetime)
                .select(
                    pl.when(pl.col("datetime").dt.year() == yr[0])
                    .then(self.ptc(yr) * yr_fact_map[yr[0]])
                    .otherwise(0)
                )
                .collect()
                .to_series()
                .to_numpy()
            )
            # self.cost[yr] = self.ptc(yr) * np.where(
            #     self.m.dt_rng(yr).year == yr[0], yr_fact_map[yr[0]], 0
            # )
            return np.maximum(self.min_cost, self.cost[yr]) @ self.get_x(yr)
        self.cost[yr] = self.ptc(yr)
        return cp.sum(np.maximum(self.min_cost, self.cost[yr]) * self.get_x(yr))

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        return self.get_x(yr) >= 0.0, self.get_x(yr) <= 2e5


class CleanExport(DecisionVariable):
    _var_names = ("export_clean",)
    _cat = "export"
    _dz = {"profs": ["x", "cost"]}

    def __init__(self, m) -> None:
        super().__init__(m)
        self.cost_ = []

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return -self.get_x(yr)

    @check_shape()
    def export_req(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def clean_export(self, yr: tuple, *args) -> cp.Expression:
        return -self.get_x(yr)

    @check_shape()
    def icx_ops(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        cost_ = (
            pl.concat(
                [
                    prof(self.m.export_profs, yr, cs.all),
                    prof(self.m.d.ba_pro, yr, cs.by_name("datetime", "baseline_sys_lambda")),
                ],
                how="align",
            )
            .join(
                self.m.d.mcoe.upsample("datetime", every="1h").lazy(),
                on="datetime",
                how="left",
            )
            .with_columns(
                pl.col("total_var_mwh").forward_fill(),
                crit=self.m._crit_hrs[yr].value,
                hrly_fact=pl.col("datetime")
                .dt.year()
                .replace_strict(yr_fact_map, default=1.0),
                ptc_ar=pl.when(pl.col("datetime").dt.year() == yr[0])
                .then(self.ptc(yr) * yr_fact_map.get(yr[0], 1))
                .otherwise(0),
            )
            .select(
                year=pl.lit(str(yr)),
                a=pl.when(pl.sum_horizontal("export_requirement", "crit") > 0)
                .then(pl.min_horizontal("baseline_sys_lambda", "total_var_mwh"))
                .otherwise(
                    pl.min_horizontal(
                        pl.col("baseline_sys_lambda") * self.m.mkt_rev_mult,
                        "total_var_mwh",
                    )
                )
                * pl.col("hrly_fact")
                * (1 - pl.col("curtailment_pct")),
                b=pl.col("ptc_ar") * pl.col("curtailment_pct"),
            )
            .collect()
        )

        self.cost_.append(cost_)
        self.cost[yr] = cost_.select(-pl.col("a") + pl.col("b")).to_series().to_numpy()
        return self.cost[yr] @ self.get_x(yr)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        return self.get_x(yr) >= 0.0, self.get_x(yr) <= self.m.i.cap


class Fossil(DecisionVariable):
    def __init__(self, m, mcoe: pl.DataFrame | None = None):
        super().__init__(m)
        if (
            "export_fossil" in self._var_names
            and "icx_hist" not in m.d.ba_pro.collect_schema().names()
        ):
            raise ValueError(
                f"PowerCouple.icx_histpro cannot be None, with a {self.__class__.__qualname__} resource."
            )
        if mcoe is None:
            mcoe = m.d.mcoe
        self.mcoe = mcoe
        self.for_co2 = (
            self.m.d.ba_data["dispatchable_cost"]
            .filter(
                (pl.col("plant_id_eia") == self.m.i.pid)
                & (pl.col("generator_id").is_in(self.m.i.gens))
                & (pl.col("datetime").dt.year() >= 2024)
            )
            .group_by("datetime")
            .agg(pl.col("co2_factor", "heat_rate").mean())
            .sort("datetime")
            .collect()
            .upsample(time_column="datetime", every="1h")
            .lazy()
            .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
            .fill_null(strategy="forward")
            .select("datetime", "co2_factor", "heat_rate")
        )
        self.cost_w_penalty = {}

    @property
    def tech(self):
        return self.m.i.tech

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        hourly = super().hourly(selection=selection, **kwargs)
        if selection:
            return hourly
        if self.cost_w_penalty:
            hourly = hourly.join(
                self._h_core(selection, self.cost_w_penalty, "cp_"), on="datetime"
            )
        return hourly.join(self.for_co2, on="datetime").select(
            *hourly.collect_schema().names(),
            *[
                product(k, "co2_factor", "heat_rate").alias(f"{k}_co2")
                for k in self._var_names
            ],
        )


class LoadIncumbentFossil(Fossil):
    """x_lt x_ft"""

    _var_names = ("load_fossil",)
    _cat = "fossil"
    _dz = {"profs": ["x", "cost"]}

    @property
    def fuels(self):
        return [self.m.i.fuel]

    @property
    def cap_summary(self) -> dict:
        if (2030,) not in self.x:
            return {}
        return to_dict(
            self.m.add_mapped_yrs(self.hourly()).select(
                **{
                    f"{k}_cf": pl.col(k).sum() / (pl.col(k).count() * self.m.i.cap)
                    for k in self._var_names
                }
            )
        )

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.get_x(yr))

    @check_shape()
    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.get_x(yr))

    def incumbent_ops(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    def gas_window_max(self, yr: tuple, *args) -> cp.Expression:
        return sum(
            sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.gas_window_max_hrs)
        ) @ self.get_x(yr)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        mcoe = (
            prof(
                self.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        mult = self.m.fos_load_cost_mult
        if is_resource_selection(yr):
            mcoe = (
                mcoe
                * prof(self.m.d.ba_pro.select("datetime"), yr, cs.first)
                .collect()
                .to_series()
                .dt.year()
                .replace_strict(yr_fact_map)
                .to_numpy()
            )
            mult = 1.0
        self.cost[yr] = mcoe
        self.cost_w_penalty[yr] = mult * mcoe
        return mult * mcoe @ self.get_x(yr)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        l = self.get_x(yr)
        return l >= 0.0, l <= self.m.i.cap

    def __repr__(self) -> str:
        cap = f"{self.m.i.cap:.0f}"
        try:
            cap = (
                cap
                + f" {sum([v[0].value.sum() + v[1].value.sum() for k, v in self.x.items() if len(k) == 1]) / ((len(self.x) - 1) * 8760 * self.m.i.cap):.0%}"
            )
        except Exception:
            pass
        return self.__class__.__qualname__ + f"({cap})"


class ExportIncumbentFossil(Fossil):
    """x_ft"""

    _var_names = ("export_fossil",)
    _cat = "export_fossil"
    _dz = {"profs": ["x", "cost"]}

    @property
    def fuels(self):
        return [self.m.i.fuel]

    @property
    def cap_summary(self) -> dict:
        if (2030,) not in self.x:
            return {}
        return {
            "export_fossil_cf": self.m.add_mapped_yrs(self.hourly())
            .select(
                pl.col("export_fossil").sum() / (pl.col("datetime").count() * self.m.i.cap),
            )
            .collect()
            .item()
        }

    @check_shape()
    def export_req(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.get_x(yr))

    @check_shape()
    def icx_ops(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    def incumbent_ops(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    def gas_window_max(self, yr: tuple, *args) -> cp.Expression:
        return sum(
            sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.gas_window_max_hrs)
        ) @ self.get_x(yr)

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        mcoe = (
            prof(
                self.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        if is_resource_selection(yr):
            mcoe = (
                mcoe
                * prof(self.m.d.ba_pro.select("datetime"), yr, cs.first)
                .collect()
                .to_series()
                .dt.year()
                .replace_strict(yr_fact_map)
                .to_numpy()
            )
        self.cost[yr] = mcoe
        return mcoe @ self.get_x(yr)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        return self.get_x(yr) >= 0.0, self.get_x(yr) <= self.m.i.cap


class LoadNewFossil(Fossil):
    """x_f x_lt"""

    _var_names = ("load_fossil",)
    _cat = "fossil"
    _dz = {"x_cap": True, "profs": ["x", "cost"], "cost_cap": True}

    def __init__(
        self,
        m,
        mcoe: pl.DataFrame | None = None,
        tech="nggt",
        primary_fuel: str = "natural_gas",
        x_cap: float | None = None,
    ):
        self._tech = tech
        self.primary_fuel = primary_fuel
        d_cost = m.d.ba_data["dispatchable_cost"].collect()
        if mcoe is None and self.primary_fuel != m.i.fuel:
            if self.primary_fuel in d_cost["fuel_group"].unique():
                mcoe = (
                    d_cost.filter(pl.col("fuel_group") == self.primary_fuel)
                    .group_by("datetime", maintain_order=True)
                    .agg(pl.mean("fuel_per_mmbtu"))
                )
            else:
                mcoe = m.aeo_fuel_price(self.primary_fuel)
            mcoe = mcoe.select(
                "datetime",
                total_var_mwh=pl.col("fuel_per_mmbtu") * COSTS["eff"][self._tech]
                + COSTS["vom"][self._tech],
            )
        super().__init__(m, mcoe)
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": 0,
                "re_type": "load_fossil",
                "generator_id": self._tech,
                "tech": self._tech,
                "fuel": self.fuels[0],
                "capex_raw": COSTS["capex"][self._tech],
                "life_adj": f_pv(
                    self.m.r_disc,
                    self.m.life,
                    f_pmt(self.m.r_disc, COSTS["life"][self._tech], 1),
                ),
                "dur": 0,
                "itc_adj": 1.0,
                "reg_mult": 1.0,
                "tx_capex_raw": 0.0,
                "distance": 0.0,
                "opex_raw": COSTS["opex"][self._tech],
                "ptc": 0.0,
                "ptc_gen": 0.0,
            },
        )
        self.x_cap = cp.Variable(1, name="fossil_cap")
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def fuels(self):
        return [self.primary_fuel]

    @property
    def tech(self):
        return self._tech

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        hourly = super().hourly(selection=selection, **kwargs)
        if selection:
            return hourly
        return hourly.join(self.for_co2, on="datetime").select(
            *cs.expand_selector(hourly, ~cs.contains("co2")),
            load_fossil_co2=product("load_fossil", "co2_factor") * COSTS["eff"][self._tech],
        )

    @property
    def cap_summary(self) -> dict:
        cap = self.x_cap.value.item()
        out = {"new_fossil_mw": cap}
        if (2030,) not in self.x:
            return out
        return out | {
            "load_fossil_cf": self.m.add_mapped_yrs(self.hourly())
            .select(
                pl.col("load_fossil").sum() / (pl.col("datetime").count() * cap),
            )
            .collect()
            .item(),
        }

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["new_fossil_mw"]

    def round(self):
        self.x_cap.value = self.x_cap.value + (25 - self.x_cap.value) % 25

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.get_x(yr))

    @check_shape()
    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return self.get_x(yr)

    @check_shape()
    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.get_x(yr))

    def gas_window_max(self, yr: tuple, *args) -> cp.Expression:
        return sum(
            sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.gas_window_max_hrs)
        ) @ self.get_x(yr)

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return (self.get_x(yr) <= self.x_cap,)
        return ()

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        mcoe = (
            prof(
                self.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        self.cost_cap[yr] = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        fixed_cost = self.cost_cap[yr]["total__oc"].to_numpy().item()
        if is_resource_selection(yr):
            mcoe = (
                mcoe
                * prof(self.m.d.ba_pro.select("datetime"), yr, cs.first)
                .collect()
                .to_series()
                .dt.year()
                .replace_strict(yr_fact_map)
                .to_numpy()
            )
            self.cost_w_penalty[yr] = mcoe
            self.cost[yr] = mcoe
            return fixed_cost * self.x_cap + mcoe @ self.get_x(yr)
        self.cost[yr] = mcoe
        self.cost_w_penalty[yr] = self.m.fos_load_cost_mult * mcoe
        return cp.Constant(
            fixed_cost * self.x_cap.value
        ) + self.m.fos_load_cost_mult * mcoe @ self.get_x(yr)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return (
                self.get_x(yr) >= 0.0,
                self.get_x(yr) <= 2e4,
                self.x_cap >= 0.0,
                self.x_cap <= 2e4,
            )
        return self.get_x(yr) >= 0.0, self.get_x(yr) <= self.x_cap.value


class LoadNewFossilWithBackup(Fossil):
    """x_f x_lt"""

    _var_names = ("load_fossil", "load_stored_fuel")
    _cat = "fossil"
    _dz = {"x_cap": True, "profs": ["x", "cost"], "cost_cap": True}

    def __init__(
        self,
        m,
        mcoe: pl.DataFrame | None = None,
        tech="nggt",
        primary_fuel: str = "natural_gas",
        backup_fuel: str = "distillate_fuel_oil",
        x_cap: float | None = None,
    ):
        self._tech = tech
        self.primary_fuel = primary_fuel
        self.backup_fuel = backup_fuel
        d_cost = m.d.ba_data["dispatchable_cost"].collect()
        if mcoe is None and self.primary_fuel != m.i.fuel:
            if self.primary_fuel in d_cost["fuel_group"].unique():
                mcoe = (
                    d_cost.filter(pl.col("fuel_group") == self.primary_fuel)
                    .group_by("datetime", maintain_order=True)
                    .agg(pl.mean("fuel_per_mmbtu"))
                )
            else:
                mcoe = m.aeo_fuel_price(self.primary_fuel)
            mcoe = mcoe.select(
                "datetime",
                total_var_mwh=pl.col("fuel_per_mmbtu") * COSTS["eff"][self._tech]
                + COSTS["vom"][self._tech],
            )
        if self.backup_fuel in d_cost["fuel_group"].unique():
            bmcoe = (
                d_cost.filter(pl.col("fuel_group") == self.backup_fuel)
                .group_by("datetime", maintain_order=True)
                .agg(pl.mean("fuel_per_mmbtu"))
            )
        else:
            bmcoe = m.aeo_fuel_price(self.backup_fuel)
        self.backup_mcoe = bmcoe.select(
            "datetime",
            total_var_mwh=pl.col("fuel_per_mmbtu") * COSTS["eff"][self._tech]
            + COSTS["vom"][self._tech],
        )

        super().__init__(m, mcoe)
        self.cost_df = pl.DataFrame(
            {
                "re_site_id": 0,
                "re_type": self._tech,
                "capex_raw": COSTS["capex"][self._tech],
                "life_adj": f_pv(
                    self.m.r_disc,
                    self.m.life,
                    f_pmt(self.m.r_disc, COSTS["life"][self._tech], 1),
                ),
                "dur": 0,
                "itc_adj": 1.0,
                "reg_mult": 1.0,
                "tx_capex_raw": 0.0,
                "distance": 0.0,
                "opex_raw": COSTS["opex"][self._tech],
                "ptc": 0.0,
                "ptc_gen": 0.0,
            },
        )
        self.x_cap = cp.Variable(1, name="fossil_cap")
        if x_cap is not None:
            self.x_cap.value = x_cap

    @property
    def fuels(self):
        return [self.primary_fuel, self.backup_fuel]

    @property
    def tech(self):
        return self._tech

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        hourly = (
            super()
            .hourly(selection=selection, **kwargs)
            .with_columns(load_fossil=pl.sum_horizontal(*self._var_names))
        )
        if selection:
            return hourly
        return hourly.join(self.for_co2, on="datetime").select(
            *cs.expand_selector(hourly, ~cs.contains("co2")),
            load_fossil_co2=product("load_fossil", "co2_factor") * COSTS["eff"][self._tech]
            + pl.col("load_stored_fuel")
            * CARB_INTENSITY["Petroleum Liquids"]
            * COSTS["eff"][self._tech],
        )

    @property
    def cap_summary(self) -> dict:
        cap = self.x_cap.value.item()
        out = {"new_fossil_mw": cap}
        if (2030,) not in self.x:
            return out
        return out | to_dict(
            self.m.add_mapped_yrs(self.hourly()).select(
                load_fossil_cf=pl.sum_horizontal("load_fossil", "load_stored_fuel").sum()
                / (pl.col("datetime").count() * cap),
                load_gas_cf=pl.col("load_fossil").sum() / (pl.col("datetime").count() * cap),
                load_stored_fuel_cf=pl.col("load_stored_fuel").sum()
                / (pl.col("datetime").count() * cap),
            )
        )

    def set_x_cap(self, cap_summary, re_selected, *args):
        self.x_cap.value = cap_summary["new_fossil_mw"]

    def round(self):
        self.x_cap.value = self.x_cap.value + (25 - self.x_cap.value) % 25

    def common_hrly_constraint(self, yr):
        l, b = self.get_x(yr)
        if is_resource_selection(yr):
            return l
        return cp.sum([l, b])

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.common_hrly_constraint(yr))

    @check_shape()
    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.common_hrly_constraint(yr))

    def gas_window_max(self, yr: tuple, *args) -> cp.Expression:
        l, b = self.get_x(yr)
        return (
            sum(sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.gas_window_max_hrs))
            @ l
        )

    def stored_fuel(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return cp.Constant(0.0)
        l, b = self.get_x(yr)
        return (
            sum(sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.stored_fuel_hrs)) @ b
        )

    # def backup_annual(self, yr: tuple, *args) -> cp.Expression:
    #     if is_resource_selection(yr):
    #         return cp.Constant(0.0)
    #     l, b = self.get_x(yr)
    #     return cp.sum(b)

    def single_dv(self, yr: tuple, soc_ixs, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return (self.common_hrly_constraint(yr) <= self.x_cap,)
        return ()

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        mcoe = (
            prof(
                self.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        self.cost_cap[yr] = self.mk_obj_cost(self.cost_df, yr, yr_fact_map)
        fixed_cost = self.cost_cap[yr]["total__oc"].to_numpy().item()
        l, b = self.get_x(yr)
        if is_resource_selection(yr):
            mcoe = (
                mcoe
                * prof(self.m.d.ba_pro.select("datetime"), yr, cs.first)
                .collect()
                .to_series()
                .dt.year()
                .replace_strict(yr_fact_map)
                .to_numpy()
            )
            self.cost[yr] = (mcoe, np.zeros(8760 * 2))
            self.cost_w_penalty[yr] = (mcoe, np.zeros(8760 * 2))
            return fixed_cost * self.x_cap + mcoe @ l
        stored_mcoe = (
            prof(
                self.backup_mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        self.cost[yr] = (mcoe, stored_mcoe)
        mcoe_ = self.m.fos_load_cost_mult * mcoe
        stored_mcoe_ = np.maximum(stored_mcoe, mcoe_ + 5)
        self.cost_w_penalty[yr] = (mcoe_, stored_mcoe_)
        return cp.Constant(fixed_cost * self.x_cap.value) + mcoe_ @ l + stored_mcoe_ @ b

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        l, b = self.get_x(yr)
        if is_resource_selection(yr):
            return (l >= 0.0, l <= 2e4, self.x_cap >= 0.0, self.x_cap <= 2e4)
        return (
            l >= 0.0,
            b >= 0.0,
            l <= self.x_cap.value,
            b <= self.x_cap.value,
            l + b <= self.x_cap.value,
        )


class DCBackupFossil(Fossil):
    """x_f x_lt"""

    _var_names = ("backup",)
    _cat = "backup"
    _dz = {"profs": ["x", "cost"]}

    def __init__(
        self,
        m,
        mcoe: pl.DataFrame | None = None,
        tech="rice",
        primary_fuel="distillate_fuel_oil",
    ):
        self._tech = tech
        self.primary_fuel = primary_fuel
        d_cost = m.d.ba_data["dispatchable_cost"].collect()
        if mcoe is None:
            if self.primary_fuel in d_cost["fuel_group"].unique():
                mcoe = (
                    d_cost.filter(pl.col("fuel_group") == self.primary_fuel)
                    .group_by("datetime", maintain_order=True)
                    .agg(pl.mean("fuel_per_mmbtu"))
                )
            else:
                mcoe = m.aeo_fuel_price(self.primary_fuel)
            mcoe = mcoe.select(
                "datetime",
                total_var_mwh=pl.col("fuel_per_mmbtu") * COSTS["eff"][self._tech]
                + COSTS["vom"][self._tech],
            )
        super().__init__(m, mcoe)

    @property
    def fuels(self):
        return [self.primary_fuel]

    @property
    def tech(self):
        return self._tech

    def hourly(self, *, selection=False, **kwargs) -> pl.LazyFrame:
        hourly = super().hourly(selection=selection, **kwargs)
        if selection:
            return hourly
        return hourly.select(
            *cs.expand_selector(hourly, ~cs.contains("co2")),
            backup_co2=pl.col("backup")
            * CARB_INTENSITY["Petroleum Liquids"]
            * COSTS["eff"][self._tech],
        )

    @property
    def cap_summary(self) -> dict:
        if (2030,) not in self.x:
            return {}
        return {
            "dc_backup_cf": self.m.add_mapped_yrs(self.hourly())
            .select(
                pl.col("backup").sum() / (pl.col("datetime").count() * self.m.load_mw),
            )
            .collect()
            .item(),
        }

    def common_hrly_constraint(self, yr):
        if is_resource_selection(yr):
            return cp.Constant(0.0)
        return self.get_x(yr)

    @check_shape()
    def load(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def critical(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def fossil_load(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.common_hrly_constraint(yr))

    @check_shape()
    def fossil_load_hrly(self, yr: tuple, *args) -> cp.Expression:
        return self.common_hrly_constraint(yr)

    @check_shape()
    def fossil_hist(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.common_hrly_constraint(yr))

    def stored_fuel(self, yr: tuple, *args) -> cp.Expression:
        if is_resource_selection(yr):
            return cp.Constant(0.0)
        return sum(
            sp.eye_array(8760 * len(yr), k=-n) for n in range(self.m.stored_fuel_hrs)
        ) @ self.get_x(yr)

    def backup_annual(self, yr: tuple, *args) -> cp.Expression:
        return cp.sum(self.common_hrly_constraint(yr))

    def objective(self, yr: tuple, yr_fact_map, *args) -> cp.Expression:
        if is_resource_selection(yr):
            self.cost[yr] = np.zeros(8760 * 2)
            return cp.Constant(0.0)
        stored_mcoe = (
            prof(
                self.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        mcoe = (
            prof(
                self.m.d.mcoe.upsample("datetime", every="1h")
                .lazy()
                .join(self.m.d.ba_pro.select("datetime"), on="datetime", how="right")
                .select(pl.all().forward_fill()),
                yr,
                cs.numeric,
            )
            .collect()
            .to_numpy()
            .flatten()
        )
        penalty_cost = np.maximum(stored_mcoe, self.m.fos_load_cost_mult * mcoe + 5)
        self.cost[yr] = stored_mcoe
        self.cost_w_penalty[yr] = penalty_cost
        return penalty_cost @ self.get_x(yr)

    def bounds(self, yr: tuple, *args) -> tuple[cp.Constraint, ...]:
        if is_resource_selection(yr):
            return ()
        return self.get_x(yr) >= 0.0, self.get_x(yr) <= self.m.load_mw
