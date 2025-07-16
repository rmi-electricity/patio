from __future__ import annotations

import logging
import operator
import os
import sys
import tempfile
import threading
import time
from collections.abc import Generator, Sequence  # noqa: TC003
from contextlib import contextmanager
from functools import reduce, singledispatch
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import scipy.sparse as sp  # noqa: TC002
from etoolbox.utils.cloud import rmi_cloud_fs
from etoolbox.utils.pudl import pl_read_pudl, pl_scan_pudl
from numba.cuda.cudadrv.devicearray import lru_cache
from plotly.subplots import make_subplots
from polars.polars import ColumnNotFoundError
from tqdm.auto import tqdm

from patio.constants import REGION_MAP, TECH_CODES
from patio.data.asset_data import clean_atb

if TYPE_CHECKING:
    from patio.model.colo_lp import Info

logger = logging.getLogger("patio")


class _LPProblem(NamedTuple):
    c: np.ndarray
    A_ub: sp.csc_array
    b_ub: np.ndarray
    A_eq: sp.csc_array
    b_eq: np.ndarray
    bounds: np.ndarray
    x0: np.ndarray | None = None
    integrality: np.ndarray | None = None

    def check(self):
        dv_aligned = (
            self.c.shape[0] == self.A_ub.shape[1] == self.A_eq.shape[1] == self.bounds.shape[0]
        )
        ub_aligned = self.A_ub.shape[0] == self.b_ub.shape[0]
        eq_aligned = self.A_eq.shape[0] == self.b_eq.shape[0]
        if not all((dv_aligned, ub_aligned, eq_aligned)):
            raise AssertionError(
                f"LP input shapes do not match:\n\n"
                f"{self.c.shape=}\n{self.A_ub.shape=}\n{self.b_ub.shape=}"
                f"\n{self.A_eq.shape=}\n{self.b_eq.shape=}\n{self.bounds.shape=}"
            )
        return self

    def as_cvxpy(self):
        x = cp.Variable(len(self.c))
        constraints = [
            self.A_ub @ x <= self.b_ub,
            self.A_eq @ x == self.b_eq,
            x >= self.bounds[:, 0],
            x <= self.bounds[:, 1],
        ]
        objective = cp.Minimize(self.c @ x)
        return cp.Problem(objective, constraints)


SUCCESS = 0
FAIL_SMALL = 1
FAIL_SELECT = 2
FAIL_DISPATCH = 3
INFEASIBLE = 4
FAIL_SERVICE = 5
STATUS = {
    0: "SUCCESS",
    1: "FAIL_SMALL",
    2: "FAIL_SELECT",
    3: "FAIL_DISPATCH",
    4: "INFEASIBLE",
    5: "FAIL_SERVICE",
}
nt = np.float32
WACC = 0.085
MACRS = np.array([0.2, 0.32, 0.192, 0.1152, 0.1152, 0.0576])
TR = 0.257
ES = pl.DataFrame(
    [
        (2022, 390.93, 390.93, 390.93, 363.01, 363.01, 363.01),
        (2023, 386.92, 386.92, 386.92, 359.28, 359.28, 359.28),
        (2024, 319.43, 355.75, 408.28, 296.61, 346.86, 379.25),
        (2025, 252.16, 309.93, 402.31, 233.06, 311.36, 374.56),
        (2026, 241.37, 297.14, 385.52, 223.53, 312.35, 367.44),
        (2027, 230.59, 284.32, 368.74, 213.93, 313.44, 360.24),
        (2028, 219.79, 271.48, 352.0, 204.44, 314.63, 352.93),
        (2029, 208.98, 258.6, 335.28, 194.97, 315.94, 345.53),
        (2030, 198.51, 245.69, 318.58, 184.18, 317.38, 338.01),
        (2031, 195.07, 241.22, 316.4, 180.78, 315.23, 335.69),
        (2032, 191.63, 236.75, 314.22, 177.39, 313.06, 333.38),
        (2033, 187.86, 232.28, 312.04, 175.28, 310.87, 331.06),
        (2034, 184.42, 227.82, 309.86, 171.89, 308.66, 328.75),
        (2035, 181.33, 223.36, 307.68, 167.11, 306.44, 326.43),
        (2036, 177.57, 218.91, 305.5, 165.01, 304.2, 324.12),
        (2037, 174.13, 214.47, 303.32, 161.62, 301.94, 321.81),
        (2038, 170.69, 210.02, 301.13, 158.22, 299.65, 319.49),
        (2039, 167.25, 205.59, 298.95, 154.82, 297.35, 317.18),
        (2040, 163.48, 201.16, 296.77, 152.74, 295.03, 314.86),
        (2041, 160.04, 196.73, 294.59, 149.34, 292.68, 312.55),
        (2042, 156.96, 192.32, 292.41, 144.53, 290.3, 310.23),
        (2043, 153.52, 187.91, 290.23, 141.13, 287.91, 307.92),
        (2044, 149.75, 183.5, 288.05, 139.06, 285.48, 305.61),
        (2045, 146.31, 179.1, 285.87, 135.66, 283.03, 303.29),
        (2046, 142.87, 174.71, 283.68, 132.26, 280.55, 300.98),
        (2047, 139.44, 170.33, 281.5, 128.85, 278.04, 298.66),
        (2048, 135.66, 165.96, 279.32, 126.8, 275.5, 296.35),
        (2049, 132.22, 161.59, 277.14, 123.4, 272.92, 294.04),
        (2050, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
        (2051, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
        (2052, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
        (2053, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
        (2054, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
        (2055, 129.15, 157.24, 274.96, 118.54, 270.31, 291.72),
    ],
    schema=[
        "year",
        "dc_Advanced",
        "dc_Moderate",
        "dc_Conservative",
        "ac_Advanced",
        "ac_Moderate",
        "ac_Conservative",
    ],
    orient="row",
).with_columns(cs.contains("_") * 1000)
"""ATB 2024 downloaded April 4. 2025, moderate case
Year, DC, AC, extended to 2055
"""
COSTS = {
    # https://data.openei.org/files/6006/2024%20v2%20Annual%20Technology%20Baseline%20Workbook%20Errata%207-19-2024.xlsx
    "opex": {
        # "solar": 2e4,  # 2026 mod
        # "onshore_wind": 3.1e4,  # 2026 c6t1 mod
        # "offshore_wind": 8.2e4,  # 2026 fixed c4
        "nggt": 2.56e4,  # 2026 mod
        "rice": 2.56e4 * 1.5,
    },
    "vom": {
        "nggt": 6.94,  # 2026 mod
        "rice": 6.94 * 1.5,
    },
    "capex": {
        # "solar": 1.27e6,  # 2026 mod
        # "onshore_wind": 1.35e6,  # 2026 c6t1 mod
        # "offshore_wind": 4.3e6,  # 2026 fixed c4
        # "es_ac": 3.12e5,  # 2026 mod
        # "es_dc": 2.97e5,  # 2026 mod
        "nggt": 1.31e6,  # 2026 mod
        "rice": 1.31e6 * 1.5,
        "tx": 1867,
    },
    "discount": WACC,
    "inflation": 0.02,
    "eff": {
        "d": 0.922,  # discharge
        "c": 0.922,  # charge
        "l": 0.0001,  # hourly self discharge / loss
        "nggt": 9.72,  # 2026 mod HR
        "rice": 8.5,
        "linear": 7.4,
    },
    # "ptc": 17.0 * 1.1,  # mostly EC
    "ptc": 17.0,  # ptc bonus now based on EC data
    # "itc": 0.3 + 0.1,  # mostly EC
    "itc": 0.3,  # itc bonus now based on incumbent EC status
    "pff": (1 - TR * ((1 / (1 + WACC)) ** np.arange(1, 7) * MACRS).sum()) / (1 - TR),
    "fmv_step": {
        "solar": 1.1,
        "onshore_wind": 1.1,
        "offshore_wind": 1.1,
        "es": 1.2,
    },
    "life": {
        "nggt": 20,  # 2026 mod
        "rice": 20,  # 2026 mod
    },
}
SUM_COL_ORDER = [
    "run_status",
    "balancing_authority_code_eia",
    "icx_id",
    "status",
    "tech",
    "name",
    "regime",
    "icx_shared",
    "good",
    "best_at_site",
    "land_available",
    "fossil_mw",
    "reg_rank",
    "plant_name_eia",
    "utility_name_eia",
    "utility_name_eia_lse",
    "state",
    "county",
    "load_mw",
    "served_pct",
    "unserved_mwh",
    "unserved_pct_hrs",
    "max_unserved_mw",
    "pct_load_clean",
    "pct_land_used",
    "pct_overbuild",
    "ppa",
    "ppa_ex_fossil_export_profit",
    "attr_cost_clean",
    "attr_cost_curtailment",
    "attr_cost_export_fossil",
    "attr_cost_load_fossil",
    "attr_rev_export_clean",
    "attr_rev_export_fossil",
    "attr_rev_full_ptc",
    "avg_cost_export_fossil",
    "avg_cost_load_fossil",
    "avg_rev_export_clean",
    "avg_rev_export_fossil",
    "com_rate",
    "ind_rate",
    "cc_lcoe",
    "solar_mw",
    "onshore_wind_mw",
    "offshore_wind_mw",
    "li_mw",
    "fe_mw",
    "new_fossil_mw",
    "li_duration",
    "fe_duration",
    "solar_max_distance_km",
    "onshore_wind_max_distance_km",
    "offshore_wind_max_distance_km",
    "fossil_cf",
    "load_fossil_cf",
    "load_stored_fuel_cf",
    "dc_backup_cf",
    "export_fossil_cf",
    "historical_fossil_cf",
    "baseline_fossil_cf",
    "redispatch_fossil_cf",
    "not_largest_violation",
    "longest_violation",
    "total_capacity_violation",
    "total_rolling_violation",
    "max_capacity_violation_pct",
    "max_rolling_violation_pct",
    "fossil_co2",
    "load_fossil_co2",
    "export_fossil_co2",
    "redispatch_export_fossil_co2",
    "redispatch_fossil_co2",
    "pct_export_clean",
    "pct_load_clean_potential",
    "clean_pct_curtailment",
    # "re_used_sqkm",
    # "load_used_sqkm",
    # "unused_sqkm_pct",
    # "total_sqkm",
    "required_acres",
    "total_acres",
    "buildable_acres",
    "num_clusters",
    "num_owners",
    "num_parcels",
    "num_crossings",
    "siting_notes",
    "cf_hist_ratio",
    "max_re_mw",
    "lcoe",
    "lcoe_export",
    "lcoe_redispatch",
    "lcoe_redispatch_export",
    "redispatch_ppa",
    "redispatch_ppa_ex_fossil_export_profit",
    "li_pct_charge_hrs_maxed",
    "li_pct_cycles_soc_maxed",
    "li_pct_discharge_hrs_maxed",
    "fe_pct_charge_hrs_maxed",
    "fe_pct_cycles_soc_maxed",
    "fe_pct_discharge_hrs_maxed",
    "load_mwh_disc",
    "load_fossil_mwh_disc",
    "historical_fossil_mwh_disc",
    "fossil_mwh_disc",
    "curtailment_mwh_disc",
    "export_clean_mwh_disc",
    "export_fossil_mwh_disc",
    "export_mwh_disc",
    "net_capex",
    "tx_capex",
    "func_disc",
    "func_undisc",
    "rev_clean_disc",
    "rev_clean_undisc",
    "rev_fossil_disc",
    "rev_fossil_undisc",
    "cost_clean_disc",
    "cost_clean_undisc",
    "cost_undisc",
    "cost_disc",
    "cost_export_fossil_disc",
    "cost_export_fossil_undisc",
    "cost_fossil_disc",
    "cost_fossil_undisc",
    "cost_load_fossil_disc",
    "cost_load_fossil_undisc",
    "full_ptc_disc",
    "full_ptc_undisc",
    "cost_curt_disc",
    "cost_curt_undisc",
    "ptc_disc",
    "ptc_undisc",
    "baseline_export_fossil_mwh_disc",
    "baseline_rev_fossil_disc",
    "baseline_rev_fossil_undisc",
    "baseline_sys_co2",
    "baseline_sys_co2_mwh_disc",
    "baseline_sys_cost_disc",
    "baseline_sys_cost_undisc",
    "baseline_sys_deficit_mwh_disc",
    "baseline_sys_load_mwh_disc",
    "baseline_sys_load_net_of_clean_export_mwh_disc",
    "redispatch_cost_curt_disc",
    "redispatch_cost_curt_undisc",
    "redispatch_cost_disc",
    "redispatch_cost_export_fossil_disc",
    "redispatch_cost_export_fossil_undisc",
    "redispatch_cost_fossil_disc",
    "redispatch_cost_fossil_undisc",
    "redispatch_cost_undisc",
    "redispatch_export_fossil_mwh_disc",
    "redispatch_ptc_disc",
    "redispatch_ptc_undisc",
    "redispatch_rev_clean_disc",
    "redispatch_rev_clean_undisc",
    "redispatch_rev_fossil_disc",
    "redispatch_rev_fossil_undisc",
    "redispatch_sys_co2",
    "redispatch_sys_co2_mwh_disc",
    "redispatch_sys_cost_disc",
    "redispatch_sys_cost_undisc",
    "redispatch_sys_deficit_mwh_disc",
    "pct_chg_system_co2",
    "pct_chg_system_cost",
    "icx_gen",
    "ba_code",
    "utility_id_eia",
    "utility_name_eia",
    "entity_type",
    "entity_type_lse",
    "owner_utility_id_eia",
    "owner_utility_name_eia",
    "fraction_owned",
    "single_owner",
    "single_parent",
    "parent_name",
    "transmission_distribution_owner_id",
    "transmission_distribution_owner_name",
    "latitude",
    "longitude",
    "ix",
    "max_pct_fos",
    "max_pct_hist_fos",
    "num_crit_hrs",
    "fos_load_cost_mult",
    "mkt_rev_mult",
    "life",
    "years",
    "total_time",
    "select_time",
    "dispatch_time",
    "redispatch_time",
    "flows_time",
    "econ_df_time",
    "errors",
]
FANCY_COLS = {
    "icx_id": "Plant ID",
    "icx_gen": "Generator IDs",
    "tech": "Technology",
    "plant_name_eia": "Plant Name",
    "fossil_mw": "Capacity MW",
    "utility_name_eia": "Operator Name",
    "owner_utility_name_eia": "Owner Name",
    "utility_name_eia_lse": "LSE Name",
    "state": "State",
    "county": "County",
    "reg_rank": "Regulatory Rank",
    "single_owner": "Single Owner",
    "balancing_authority_code_eia": "Balancing Authority",
    "name": "Scenario Name",
    "regime": "Siting Regime",
    "icx_shared": "Share Existing",
    "best_at_site": "Best At Site",
    "land_available": "Land Confirmed",
    "load_mw": "Load MW",
    "ppa_ex_fossil_export_profit": "PPA $/MWh",
    "attr_rev_export_clean": "Contribution from Clean Export $/MWh",
    "avg_rev_export_clean": "Average Revenue of Clean Export $/MWh",
    "com_rate": "Average Commercial Rate $/MWh",
    "ind_rate": "Average Industrial Rate $/MWh",
    "cc_lcoe": "Estimated Combined Cycle LCOE",
    "solar_mw": "Solar MW",
    "onshore_wind_mw": "Onshore Wind MW",
    "li_mw": "Li Battery MW",
    "fe_mw": "Fe Battery MW",
    "new_fossil_mw": "New Fossil MW",
    "li_duration": "Li Battery Hours",
    "fe_duration": "Fe Battery Hours",
    "served_pct": "Load Served %",
    "unserved_mwh": "Unserved MWh",
    "unserved_pct_hrs": "Unserved Hrs %",
    "pct_load_clean": "Hourly Matched Clean %",
    "fossil_cf": "Fossil Combined CF",
    "load_fossil_cf": "Fossil Load CF",
    "export_fossil_cf": "Fossil Export CF",
    "historical_fossil_cf": "Historical Fossil CF",
    "clean_pct_curtailment": "Curtailment %",
    "required_acres": "Required Acres",
    "total_acres": "Total Confirmed Acres",
    "buildable_acres": "Buildable Confirmed Acres",
}


def hstack(*args) -> np.ndarray:
    if len(args) == 1 and isinstance(args[0], Generator):
        return np.hstack(tuple(args[0]))
    return np.hstack(args)


def vstack(*args) -> np.ndarray:
    if len(args) == 1 and isinstance(args[0], Generator):
        return np.vstack(tuple(args[0]))
    return np.vstack(args)


def safediv(a, b):
    try:
        return np.divide(a, b, where=~np.isclose(b, 0)).item()
    except Exception as exc:
        logger.warning(exc)
        return np.nan


def add_missing_col[T: pl.LazyFrame | pl.DataFrame](df: T, col: str) -> T:
    if col not in df.lazy().collect_schema().names():
        df = df.with_columns(pl.lit(0.0).alias(col))
    return df


def prof[T: pl.DataFrame | pl.LazyFrame](
    df: T, years: tuple[int, ...], selection=cs.numeric
) -> T:
    # if len(years) == 1:
    #     return df.filter(pl.col("datetime").dt.year() == years[0]).select(selection())
    return pl.concat(df.filter(pl.col("datetime").dt.year() == y) for y in years).select(
        selection() if callable(selection) else selection
    )


def to_dict[T: pl.LazyFrame | pl.DataFrame](d: T, suf="") -> dict[str, int | float]:
    return {k + suf: v[0] for k, v in d.lazy().collect().to_dict(as_series=False).items()}


def _when_to_num(when):
    _when_to_num_ = {
        "end": 0,
        "begin": 1,
        "e": 0,
        "b": 1,
        0: 0,
        1: 1,
        "beginning": 1,
        "start": 1,
        "finish": 0,
    }
    try:
        return _when_to_num_[when]
    except (KeyError, TypeError):
        return [_when_to_num_[x] for x in when]


def f_pmt(rate, nper, pv, fv=0, when="end"):
    """Pmt method lifted from numpy/numpy-financial:

    Compute the payment against loan principal plus interest.

    :rtype: object
    :param rate: Rate of interest (per period)
    :type rate: array_like
    :param nper: Number of compounding periods
    :type nper: array_like
    :param pv: Present value
    :type pv: array_like
    :param fv: Future value (default = 0)
    :type fv: array_like
    :param when: When payments are due ('begin' (1) or 'end' (0))
    :type when: {{'begin', 1}, {'end', 0}}, {string, int}
    :return: Payment against loan plus interest. If all input is scalar, returns a
        scalar float. If any input is array_like, returns payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.
    :rtype: ndarray

    .. [N] numpy-financial (2019, Nov 13)
       github.com/numpy/numpy-financial/blob/master/numpy_financial/_financial.py
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php
       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt
    """
    when = _when_to_num(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate) ** nper
    mask = rate == 0
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper, (1 + masked_rate * when) * (temp - 1) / masked_rate)
    return -(fv + pv * temp) / fact


def f_pv(rate, nper, pmt, fv=0, when="end"):
    """Pv method lifted from numpy/numpy-financial:

    Compute the present value.

    :param rate: Rate of interest (per period)
    :type rate: array_like
    :param nper: Number of compounding periods
    :type nper: array_like
    :param pmt: Payment
    :type pmt: array_like
    :param fv: Future value
    :type fv: array_like
    :param when: When payments are due ('begin' (1) or 'end' (0))
    :type when: {{'begin', 1}, {'end', 0}}, {string, int}
    :return: Present value of a series of payments or investments.
    :rtype: ndarray, float

    .. [N] numpy-financial (2019, Nov 13)
       github.com/numpy/numpy-financial/blob/master/numpy_financial/_financial.py
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php
       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt
    """
    when = _when_to_num(when)
    (rate, nper, pmt, fv, when) = map(np.asarray, [rate, nper, pmt, fv, when])
    temp = (1 + rate) ** nper
    fact = np.where(rate == 0, nper, (1 + rate * when) * (temp - 1) / rate)
    return -(fv + pmt * fact) / temp


def f_npv(
    rate: float, values: list | pd.Series | np.array | pl.DataFrame
) -> float | np.array | pl.DataFrame:
    """Npv method lifted from numpy/numpy-financial:

    Returns the NPV (Net Present Value) of a cash flow series.
    Modified to behave the same as the excel function, i.e. the first value is
    now discounted. i=1 -> len(values) + 1 rather than i=0 -> len(values)

    :param rate: The discount rate.
    :type rate: float
    :param values: The values of the time
        series of cash flows.  The (fixed) time
        interval between cash flow "events" must be the same as that for
        which `rate` is given (i.e., if `rate` is per year, then precisely
        a year is understood to elapse between each cash flow event).  By
        convention, investments or "deposits" are negative, income or
        "withdrawals" are positive; `values` must begin with the initial
        investment, thus `values[0]` will typically be negative.
    :type values: array_like
    :return: The NPV of the input cash flow series `values` at the discount
        `rate`.
    :rtype: float

    .. [N] numpy-financial (2019, Nov 13)
       github.com/numpy/numpy-financial/blob/master/numpy_financial/_financial.py
    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
       Addison-Wesley, 2003, pg. 346.
    """
    if isinstance(values, np.ndarray) and len(values.shape) == 2:
        return hstack(f_npv(rate, values[:, i]) for i in range(values.shape[1]))
    if isinstance(values, pl.DataFrame):
        return pl.DataFrame({c: f_npv(rate, values[c]) for c in values.columns})
    values = np.asarray(values)
    return (values / (1 + rate) ** np.arange(1, 1 + len(values))).sum(axis=0)


def pl_filter(**kwargs):
    return reduce(
        operator.and_,
        [pl.col(k) == v for k, v in kwargs.items()],
    )


class SeriesLike:
    def __init__(self, in_df: pl.DataFrame | None = None, **kwargs):
        if in_df is not None:
            assert in_df.shape[0] == 1
            for k, v in in_df.to_dict(as_series=False).items():
                setattr(self, k, v[0])
        self.__dict__.update(kwargs)

    def items(self):
        for k, v in self.__dict__.items():  # noqa: UP028
            yield k, v


class OptResult(NamedTuple):
    status: int
    result: dict
    dfs: dict
    bad_year: int | None = None
    max_re: float | None = None

    @classmethod
    def infeasible(cls, info: Info, regime: str):
        return cls(
            INFEASIBLE,
            {
                "fossil_mw": info.cap,
                "load_mw": None,
                "max_re_mw": info.max_re,
                "tech": info.tech,
                "status": info.status,
                "regime": regime,
                "errors": [],
            },
            {},
        )

    def fmt_result(self):
        return self.result | {
            "errors": "; ".join(reversed(self.result.get("errors", []))),
            "export_req_src": ", ".join(
                f"{k}: {v}" for k, v in self.result.get("export_req_src", {}).items()
            ),
        }

    def fmt_status(self):
        return {
            0: "SUCCESS",
            1: "FAIL_SMALL",
            2: "FAIL_SELECT",
            3: "FAIL_DISPATCH",
            4: "INFEASIBLE",
            5: "FAIL_SERVICE",
        }[self.status]


class OptimizationData(NamedTuple):
    l_re: int
    yrs: Sequence
    lp_in: dict
    result: dict
    new_clean: dict
    x_idx: dict
    dt_idx: dict
    ub_idx: dict
    dvs: list = [
        "discharge",
        "charge",
        "soc",
        "export_fossil",
        "export_clean",
        "curtailment",
        "load_fossil",
    ]

    @classmethod
    def new(cls, l_re, yrs):
        return cls(l_re, yrs, {}, {}, {}, {}, {}, {})

    def add_year(
        self,
        year,
        result,
        lp_in,
        new_clean,
        x_idx,
        dt_idx,
        ub_idx,
        **kwargs,
    ):
        self.lp_in[year] = lp_in
        self.result[year] = result
        self.new_clean[year] = new_clean
        self.x_idx[year] = x_idx
        self.dt_idx[year] = dt_idx
        self.ub_idx[year] = ub_idx
        return self

    def clean_cost(self) -> dict:
        pre = next(iter(self.lp_in))
        f_yr = self.yrs[0]
        clean_cost = (
            self.new_clean[pre][
                [
                    "capex__oc",
                    "opex__oc",
                    "tx_capex__oc",
                ]
            ].sum(axis=1)
            @ self.result[pre].x[: self.l_re + 1]
        )
        return {yr: (clean_cost if yr == f_yr else 0) for yr in self.yrs}

    def full_ptc(self) -> dict:
        return {
            yr: -self.new_clean[yr].ptc__oc @ self.result[yr].x[: self.l_re + 1]
            for yr in self.yrs
        }

    def c_mw_all(self):
        return pd.concat(
            [
                pd.Series(
                    self.lp_in[yr].c[: self.l_re + 1],
                    index=self.x_idx[yr][: self.l_re + 1],
                    name=str(yr),
                )
                for yr in self.lp_in
            ],
            axis=1,
        ).T

    def c_ts_all(self):
        return pd.concat({yr: self.c_ts(yr) for yr in self.lp_in})

    def c_ts(self, year) -> pd.DataFrame:
        return pd.DataFrame(
            np.reshape(
                self.lp_in[year].c[self.l_re + 2 :],
                (len(self.dt_idx[year]), 7),
                order="F",
            ),
            index=self.dt_idx[year],
            columns=self.dvs,
        )

    def x_ts(self, year) -> pd.DataFrame:
        return pd.DataFrame(
            np.reshape(
                self.result[year].x[self.l_re + 2 :],
                (len(self.dt_idx[year]), 7),
                order="F",
            ),
            index=self.dt_idx[year],
            columns=self.dvs,
        )

    def ineqlin(
        self, year, kind: Literal["marginals", "residual"] = "marginals"
    ) -> pd.DataFrame:
        return (
            pd.Series(self.result[year].ineqlin[kind], index=self.ub_idx[year], name=kind)
            .reset_index()
            .pivot(index="sub", columns="cat", values=kind)
        )

    def hourly(self, year):
        c_cols = ["load_fossil", "export_fossil", "export_clean", "curtailment"]
        ub_cols = ["export_requirement"]
        return pd.concat(
            [
                self.x_ts(year),
                self.c_ts(year)[c_cols].rename(columns={k: "c_" + k for k in c_cols}),
                -pd.Series(
                    self.lp_in[year].b_ub,
                    index=self.ub_idx[year],
                    name="export_requirement",
                ).loc["export_requirement"],
                pd.Series(
                    self.result[year].eqlin.marginals[: len(self.dt_idx[year])],
                    index=self.dt_idx[year],
                    name="shadowpx_load",
                ),
                pd.Series(
                    self.result[year].ineqlin.marginals,
                    index=self.ub_idx[year],
                    name="marginals",
                )
                .loc[ub_cols]
                .reset_index()
                .pivot(index="sub", columns="cat", values="marginals")
                .rename(columns={k: "shadowpx_" + k for k in ub_cols}),
            ],
            axis=1,
        )


class OutputGrabber:
    """Class used to grab standard output or another stream."""

    escape_char = "\b"

    def __init__(self, stream, threaded=True):
        self.origstream = stream
        self.threaded = threaded
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        """Start capturing the stream data."""
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.read_output)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)
        return self

    def __exit__(self, type, value, traceback):  # noqa: A002
        """Stop capturing the stream data and save the text in `capturedtext`."""
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.read_output()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def read_output(self):
        """Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char


@contextmanager
def no_capture_stdout():
    yield ""


@contextmanager
def capture_stdout():
    """Save targetfd descriptor, and open a new
    temporary file there.  If no tmpfile is
    specified a tempfile.Tempfile() will be opened
    in text mode.
    """
    f = tempfile.TemporaryFile("wb+")  # noqa: SIM115
    # return a new open file object that's a duplicate of f
    #
    # mode is duplicated if not given, 'buffering' controls
    # buffer size (defaulting to no buffering)
    tmpfile = os.fdopen(
        os.dup(f.fileno()), f.mode.replace("b", ""), True, "UTF-8", closefd=True
    )
    f.close()
    _savefd = os.dup(1)
    _oldsys = sys.stdout
    try:
        os.fstat(_savefd)
    except OSError:
        raise ValueError("saved filedescriptor not valid, did you call start() twice?")  # noqa: B904
    os.dup2(tmpfile.fileno(), 1)
    sys.stdout = tmpfile
    yielded = StringIO()
    yield yielded
    # unpatch and clean up, returns the self.tmpfile (file object)
    try:
        os.dup2(_savefd, 1)
        os.close(_savefd)
        tmpfile.seek(0)
        yielded.write(tmpfile.read())
    finally:
        sys.stdout = _oldsys
        tmpfile.close()


@contextmanager
def capture_stderr():
    """Save targetfd descriptor, and open a new
    temporary file there.  If no tmpfile is
    specified a tempfile.Tempfile() will be opened
    in text mode.
    """
    f = tempfile.TemporaryFile("wb+")  # noqa: SIM115
    # return a new open file object that's a duplicate of f
    #
    # mode is duplicated if not given, 'buffering' controls
    # buffer size (defaulting to no buffering)
    tmpfile = os.fdopen(
        os.dup(f.fileno()), f.mode.replace("b", ""), True, "UTF-8", closefd=True
    )
    f.close()
    _savefd = os.dup(2)
    _oldsys = sys.stderr
    try:
        os.fstat(_savefd)
    except OSError:
        raise ValueError("saved filedescriptor not valid, did you call start() twice?")  # noqa: B904
    os.dup2(tmpfile.fileno(), 2)
    sys.stderr = tmpfile
    yielded = StringIO()
    yield yielded
    # unpatch and clean up, returns the self.tmpfile (file object)
    try:
        os.dup2(_savefd, 2)
        os.close(_savefd)
        tmpfile.seek(0)
        yielded.write(tmpfile.read())
    finally:
        sys.stderr = _oldsys
        tmpfile.close()


@contextmanager
def timer():
    t1 = t2 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()


class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


@singledispatch
def order_columns(df):
    raise NotImplementedError


@order_columns.register(pd.DataFrame)
def _(df: pd.DataFrame):
    out_cols = list(
        dict.fromkeys([co for co in SUM_COL_ORDER if co in df.columns])
        | dict.fromkeys(df.columns)
    )
    return df[out_cols]


@order_columns.register(pl.DataFrame)
def _(df: pl.DataFrame):
    out_cols = list(
        dict.fromkeys([co for co in SUM_COL_ORDER if co in df.columns])
        | dict.fromkeys(df.columns)
    )
    return df.select(*out_cols)


@order_columns.register(pl.LazyFrame)
def _(df: pl.LazyFrame):
    in_cols = df.collect_schema().names()
    out_cols = list(
        dict.fromkeys([co for co in SUM_COL_ORDER if co in in_cols]) | dict.fromkeys(in_cols)
    )
    return df.select(*out_cols)


def keep_nonzero_cols(df, keep=()):
    cols = dict.fromkeys(
        df.lazy()
        .fill_null(0.0)
        .sum()
        .collect()
        .transpose(include_header=True)
        .filter(pl.col("column_0") != 0)["column"]
    )
    old_cols = df.lazy().collect_schema().names()
    keep = dict.fromkeys(c for c in {"datetime", *keep} if c in old_cols)
    return df.select(*list(keep | cols))


def get_summary(run, ad, pudl_release=None, fs=None):
    if fs is None:
        fs = rmi_cloud_fs()
    with fs.open(f"az://patio-results/{run}/colo_summary.parquet") as f:
        colo_ = pl.scan_parquet(f)  # .with_columns(tech=pl.lit("CC"))
    try:
        temp = colo_.select("pudl_release").unique().collect().item()
    except ColumnNotFoundError:
        if pudl_release is None:
            raise ValueError("pudl_release not set and not in summary data")  # noqa: B904
    else:
        pudl_release = temp
    atb = (
        clean_atb(
            case="Market", scenario="Moderate", report_year=2024, pudl_release=pudl_release
        )
        .filter(
            (pl.col("technology_description") == "Natural Gas Fired Combined Cycle")
            & (pl.col("projection_year") == 2026)
        )
        .select("heat_rate_mmbtu_per_mwh", "fom_per_kw", "vom_per_mwh")
        .to_dict(as_series=False)
    )
    r_disc = ((1 + COSTS["discount"]) / (1 + COSTS["inflation"])) - 1
    id_cols = ("ba_code", "icx_id", "icx_gen", "regime", "name")
    try:
        with fs.open(f"az://patio-results/{run}/colo_re_selected.parquet") as f:
            re_select = (
                pl.scan_parquet(f)
                .select(
                    *id_cols,
                    "re_site_id",
                    "re_type",
                    "area_per_mw",
                    "capacity_mw",
                    used_acres=pl.col("capacity_mw") * pl.col("area_per_mw") * 247,
                )
                .group_by(*id_cols, "re_type")
                .agg(pl.col("used_acres", "capacity_mw").sum())
                .with_columns(
                    mw_per_acre=pl.col("capacity_mw") / pl.col("used_acres"),
                    acre_per_mw=pl.col("used_acres") / pl.col("capacity_mw"),
                )
                .collect()
                .pivot(on="re_type", index=id_cols, values="used_acres")
                .select(
                    *id_cols,
                    pl.col("solar").alias("solar_acres"),
                    pl.col("onshore_wind").alias("onshore_wind_acres"),
                )
                .lazy()
            )
    except:  # noqa: E722
        pass
    else:
        colo_ = colo_.join(re_select, on=id_cols, how="left", validate="1:1")

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
        "latitude",
        "longitude",
        "plant_name_eia",
        "state",
        "county",
    ]

    utils = (
        colo_.select(
            pl.col("icx_id"),
            pl.col("icx_id").alias("plant_id_eia"),
            "icx_gen",
            pl.col("icx_gen").str.split(",").alias("generator_id"),
        )
        .explode("generator_id")
        .unique()
        .collect()
        .join(
            pl.from_pandas(ad.own),
            on=["plant_id_eia", "generator_id"],
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
            single_owner=pl.col("owner_utility_id_eia").n_unique().over("icx_id", "icx_gen")
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

    almost = (
        colo_.join(
            pl.from_pandas(ad.gens)
            .lazy()
            .with_columns(
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
            utils.lazy(), on=["icx_id", "icx_gen"], validate="m:1", how="left", suffix="_utils"
        )
        .select(~cs.contains("_utils"))
    )
    out = (
        almost.collect()
        .to_pandas()
        .rename(
            columns={
                "redispatch_co2": "redispatch_sys_co2",
                "baseline_co2": "baseline_sys_co2",
                "redispatch_system_cost_disc": "redispatch_sys_cost_disc",
                "baseline_system_cost_disc": "baseline_sys_cost_disc",
            }
        )
        .assign(
            params=lambda x: "c_hrs="
            + x.num_crit_hrs.astype(str)
            + ", mx_fs="
            + x.max_pct_fos.astype(str)
            + ", mx_hst="
            + x.max_pct_hist_fos.astype(str),
            cf_hist_ratio=lambda x: x.fossil_cf / x.historical_fossil_cf,
            percent_clean=lambda x: pd.cut(x.pct_load_clean, bins=[0.25, 0.5, 0.8, 0.95, 1.0])
            .astype("string")
            .replace(
                {
                    "(0.25, 0.5]": ">25%",
                    "(0.5, 0.8]": ">50%",
                    "(0.8, 0.95]": ">80%",
                    "(0.95, 1.0]": ">95%",
                }
            ),
            region=lambda x: x.ba_code.map(REGION_MAP),
            pct_land_used=lambda y: 1 - y.unused_sqkm_pct,
            pct_chg_system_co2=lambda x: x.redispatch_sys_co2 / x.baseline_sys_co2 - 1,
            pct_chg_system_cost=lambda x: x.redispatch_sys_cost_disc / x.baseline_sys_cost_disc
            - 1,
            pct_overbuild=lambda x: x[
                [
                    c
                    for c in (
                        "solar_mw",
                        "onshore_wind_mw",
                        "offshore_wind_mw",
                        "storage_mw",
                        "li_mw",
                        "fe_mw",
                        "new_fossil_mw",
                    )
                    if c in x
                ]
            ].sum(axis=1)
            / x.load_mw,
            regimes=lambda y: y.groupby(["icx_id", "icx_gen", "params"]).regime.transform(
                "nunique"
            ),
            own_short=lambda x: x["owner_utility_name_eia"].str.split(" ").str[0]
            + " "
            + x["owner_utility_name_eia"].str.split(" ").str[1].fillna(""),
            plant_name_short=lambda x: x["plant_name_eia"].str.split(" ").str[0]
            + " "
            + x["plant_name_eia"].str.split(" ").str[1].fillna(""),
            utility_name_eia_lse_=lambda x: x["utility_name_eia_lse"].str.split("|").str[0],
        )
        .replace({"balancing_authority_code_eia": {"CPLE": "DUK"}})
        .merge(
            pl_scan_pudl("core_eia861__yearly_sales", pudl_release)
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
            .to_pandas(),
            on="utility_name_eia_lse_",
            how="left",
            validate="m:1",
        )
        .drop(columns=["utility_name_eia_lse_"])
        .merge(
            pl_read_pudl(
                "core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type",
                pudl_release,
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
            .to_pandas(),
            on=["balancing_authority_code_eia", "state"],
            how="left",
            validate="m:1",
        )
    )
    try:
        out = (
            out.assign(
                required_acres=lambda x: x[
                    ["solar_acres", "onshore_wind_acres", "load_mw"]
                ].sum(axis=1)
            )
            .merge(LAND.to_pandas(), on="icx_id", how="left", validate="m:1")
            .assign(
                land_available=lambda x: x["buildable_acres"].fillna(0.0) * 1.5
                >= x["required_acres"]
            )
        )
    except KeyError:
        pass
    id_cols = [
        x for x in ("icx_id", "status", "tech", "name", "regime", "params") if x in out.columns
    ]
    out = (
        out.merge(
            best_at_site(out.query("required_acres <= 1.5 * buildable_acres"))[id_cols],
            on=id_cols,
            how="left",
            validate="m:1",
            indicator="best_at_site",
        )
        .astype({"best_at_site": str})
        .replace({"best_at_site": {"left_only": False, "both": True}})
    )
    assert out.groupby(["icx_id", "icx_gen"]).best_at_site.sum().max() == 1
    out_cols = list(  # noqa: F841
        dict.fromkeys([co for co in SUM_COL_ORDER if co in out.columns])
        | dict.fromkeys(out.columns)
    )
    out = out.merge(
        good_screen(out)[["icx_id", "icx_gen", "name", "regime"]],
        on=["icx_id", "icx_gen", "name", "regime"],
        how="left",
        indicator="good",
    ).assign(good=lambda x: x.good == "both")
    print("good" in out)
    return out.pipe(order_columns).sort_values(["state", "utility_name_eia_lse", "icx_id"])


def good_screen[T: pl.DataFrame | pl.LazyFrame | pd.DataFrame](df: T) -> T:
    polars = True
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
        polars = False
    out = df.filter(
        (pl.col("run_status") == "SUCCESS")
        & (pl.col("served_pct") >= 0.999)
        & (pl.col("pct_load_clean") >= 0.6)
        & (pl.col("ppa_ex_fossil_export_profit") <= 200)
        & (pl.col("reg_rank") <= 9)
    ).filter(
        (pl.col("new_fossil_mw").fill_null(0.0) == 0.0)
        | (
            (pl.col("new_fossil_mw") > 0.0)
            # & (pl.col("max_capacity_violation_pct") <= 0.25)
            & (pl.col("max_rolling_violation_pct") <= 0.25)
        )
    )
    if polars:
        return out
    return out.to_pandas()


def best_at_site(colo_):
    colo_cols = colo_.columns
    try:
        return (
            pl.from_pandas(colo_)
            .pipe(good_screen)
            # .filter(
            #     (pl.col("run_status") == "SUCCESS")
            #     & (pl.col("served_pct") >= 0.99)
            #     & (pl.col("pct_load_clean") >= 0.6)
            #     & (pl.col("ppa_ex_fossil_export_profit") <= 200)
            #     & (pl.col("reg_rank") <= 7)
            # )
            # .filter(
            #     (pl.col("new_fossil_mw").fill_null(0.0) == 0.0)
            #     | (
            #         (pl.col("new_fossil_mw") > 0.0)
            #         # & (pl.col("max_capacity_violation_pct") <= 0.25)
            #         & (pl.col("max_rolling_violation_pct") <= 0.25)
            #     )
            # )
            .sort("load_mw", descending=True)
            .group_by("icx_id", "icx_gen")
            .agg(pl.all().first())
            .select(*colo_cols)
            .sort("state", "utility_name_eia", "plant_name_eia")
            .to_pandas()
        )
    except Exception:
        print("using backup")
        return (
            pl.from_pandas(colo_)
            .filter(
                pl.col("ppa").is_not_null()
                & (pl.col("num_crit_hrs") == 25)
                & (pl.col("cf_hist_ratio") < 2)
                & (pl.col("pct_load_clean") >= 0.6)
                & (pl.col("ppa") <= 200)
            )
            .sort("icx_id", "tech", "ppa", "load_mw", descending=[False, False, False, True])
            .group_by("icx_id", "icx_gen")
            .agg(pl.all().first())
            .select(*colo_cols)
            .to_pandas()
        )


def get_flows(run, fs=None):
    if fs is None:
        fs = rmi_cloud_fs()
    with fs.open(f"az://patio-results/{run}/colo_flows.parquet") as f:
        flows_ = pl.read_parquet(f)
    repl = {"export_req": "Required Export", "export_addl": "Surplus Export"}
    idx_cols = cs.expand_selector(flows_, ~cs.contains("__") & ~cs.datetime())

    return (
        flows_.group_by(*idx_cols)
        .agg(cs.contains("__").sum())
        .unpivot(on=cs.contains("__"), index=idx_cols)
        .select(
            *idx_cols,
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


def sankey(h3):
    import plotly.graph_objects as go
    from dispatch.constants import COLOR_MAP

    ixs = {
        k: i
        for i, k in enumerate(
            sorted(
                set(h3.select("source_").to_series().to_list())
                | set(h3.select("target_").to_series().to_list()),
                # key=lambda x: key.get(x, x),
            )
        )
    }
    h3 = h3.with_columns(
        source=pl.col("source_").replace_strict(ixs),
        target=pl.col("target_").replace_strict(ixs),
    )
    nums = pl.concat(
        [
            h3.filter(pl.col("source_") != "Storage")
            .group_by(pl.col("source_").alias("label"))
            .agg(pl.sum("value")),
            h3.filter(pl.col("target_") != "Storage")
            .group_by(pl.col("target_").alias("label"))
            .agg(pl.sum("value")),
        ]
    ).rows_by_key("label")
    labels = []
    for k in ixs:
        if k not in nums:
            labels.append(k)
        elif (num := nums[k][0][0]) > 1e6:
            labels.append(f"{k} {num / 1e6:,.0f} TWh")
        else:
            labels.append(f"{k} {num / 1e3:,.0f} GWh")

    colors = COLOR_MAP | {
        "Baseline Fossil": COLOR_MAP["Gas CC"],
        "New Fossil": COLOR_MAP["Gas CC"],
        "Backup": COLOR_MAP["Gas CC"],
        "Fossil": COLOR_MAP["Gas CC"],
        "Export Fossil": COLOR_MAP["Gas CT"],
        "Curtailment": "#eec7b7",
        "Load": "#58585b",
        "Required Export": "#a0a0a0",
        "Baseline Export": "#a0a0a0",
        "Export": "#a0a0a0",
        "Surplus Export": "#a0a0a0",
    }
    san = go.Sankey(
        # arrangement="perpendicular",
        node=dict(  # noqa: C408
            pad=30,
            thickness=20,
            line_width=0,
            label=labels,
            color=[colors[k] for k in ixs],
        ),
        link={k: h3.select(k).to_series().to_list() for k in ("target", "source", "value")},
    )
    return san
    # return go.Figure(
    #     data=[
    #         san
    #     ]
    # ).update_layout(
    #     font_family="Roboto",
    #     template="ggplot2",
    #     width=int(773 * 0.6),
    #     height=int(400 * 0.6),
    #     margin=dict(l=0, r=0, b=10, t=35, pad=5),
    #     #     plot_bgcolor="#EBF0F8",
    #     # paper_bgcolor="rgba(0,0,0,0)",
    # )


flow_cols = [
    "load_fossil",
    "load_storage",
    "export_req_storage",
    "export_addl_storage",
    "export_req_fossil",
    "baseline_export_fossil",
    "load_solar",
    "load_onshore_wind",
    "export_req_solar",
    "export_req_onshore_wind",
    "export_addl_solar",
    "export_addl_onshore_wind",
    "storage_solar",
    "storage_onshore_wind",
    "curtailment_solar",
    "curtailment_onshore_wind",
    "load_offshore_wind",
    "export_req_offshore_wind",
    "export_addl_offshore_wind",
    "storage_offshore_wind",
    "curtailment_offshore_wind",
]
flow_id_cols = ["ba_code", "icx_id", "icx_gen", "params", "regime", "tech"]


def old_flows(run, ad, fs=None):
    _colo = get_summary(run, ad)
    _best = best_at_site(_colo)
    if fs is None:
        fs = rmi_cloud_fs()
    with fs.open(f"az://patio-results/{run}/colo_flows.parquet") as f:
        flows_ = pl.read_parquet(f)
    if "regime" not in flows_.columns:
        k_cols = ["icx_id", "icx_gen", "num_crit_hrs", "max_pct_fos", "max_pct_hist_fos"]
        flows_ = flows_.with_columns(
            og_counter=pl.col("icx_id").cum_count().over(*k_cols, "datetime")
        )
        flows_ = flows_.join(
            flows_.group_by(["ba_code", *k_cols, "og_counter"])
            .agg(pl.col(flow_cols).sum())
            .sort(*k_cols, "og_counter")
            .with_columns(
                t_load=pl.sum_horizontal(*[x for x in flow_cols if x.startswith("load_")]),
            )
            .sort(
                "icx_id",
                "icx_gen",
                "num_crit_hrs",
                "max_pct_fos",
                "max_pct_hist_fos",
                "t_load",
            )
            .with_columns(counter=pl.col("icx_id").cum_count().over(*k_cols))
            .join(
                pl.from_pandas(
                    _colo.query("ppa.notna()")
                    .sort_values([*k_cols, "load_mw"])[[*k_cols, "regime"]]
                    .assign(
                        counter=lambda x: 1 + x.groupby(k_cols).icx_id.transform("cumcount")
                    )
                ),
                on=[*k_cols, "counter"],
            )
            .select(*k_cols, "regime", "og_counter"),
            on=[*k_cols, "og_counter"],
        )
    flows = flows_.join(
        pl.from_pandas(
            _best[
                # remove and add to on when flows has regime
                # .query("regimes == 1")
                [
                    "icx_id",
                    "icx_gen",
                    "num_crit_hrs",
                    "max_pct_fos",
                    "max_pct_hist_fos",
                    "tech",
                    "regime",
                    "params",
                ]
            ]
        ),
        on=["icx_id", "icx_gen", "num_crit_hrs", "max_pct_fos", "max_pct_hist_fos"],
        how="inner",
    )
    return (
        flows.group_by(flow_id_cols)
        .agg(pl.col(flow_cols).sum())
        .melt(id_vars=flow_id_cols, value_vars=flow_cols)
        .with_columns(
            target_=pl.when(pl.col("variable").str.starts_with("load"))
            .then(pl.lit("Load"))
            .when(pl.col("variable").str.starts_with("export_req"))
            .then(pl.lit("Required Export"))
            .when(pl.col("variable").str.starts_with("export_addl"))
            .then(pl.lit("Surplus Export"))
            .when(pl.col("variable").str.starts_with("storage"))
            .then(pl.lit("Storage"))
            .when(pl.col("variable").str.starts_with("curtailment"))
            .then(pl.lit("Curtailment"))
            .when(pl.col("variable").str.starts_with("baseline_export"))
            .then(pl.lit("Baseline Export")),
            source_=pl.when(pl.col("variable").str.ends_with("baseline_export_fossil"))
            .then(pl.lit("Baseline Fossil"))
            .when(pl.col("variable").str.ends_with("storage"))
            .then(pl.lit("Storage"))
            .when(pl.col("variable").str.ends_with("solar"))
            .then(pl.lit("Solar"))
            .when(pl.col("variable").str.ends_with("onshore_wind"))
            .then(pl.lit("Onshore Wind"))
            .when(pl.col("variable").str.ends_with("offshore_wind"))
            .then(pl.lit("Offshore Wind"))
            .when(pl.col("variable").str.ends_with("fossil"))
            .then(pl.lit("Fossil")),
        )
    )


@lru_cache
def aeo(release):
    return pl_read_pudl(
        "core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type", release
    )


AEO_MAP = pl.from_records(
    [
        ("AECI", "AR", "midcontinent_south"),
        ("AECI", "IA", "midcontinent_west"),
        ("AECI", "MO", "midcontinent_central"),
        ("AECI", "OK", "southwest_power_pool_south"),
        ("AVA", "ID", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("AVA", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("AVA", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("AVRN", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("AVRN", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("AZPS", "AZ", "western_electricity_coordinating_council_southwest"),
        ("AZPS", "NM", "western_electricity_coordinating_council_southwest"),
        ("BANC", "CA", "western_electricity_coordinating_council_california_north"),
        ("BPAT", "ID", "western_electricity_coordinating_council_basin"),
        ("BPAT", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("BPAT", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("BPAT", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("BPAT", "WY", "western_electricity_coordinating_council_basin"),
        ("CHPD", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("CISO", "AZ", "western_electricity_coordinating_council_southwest"),
        ("CISO", "CA", "western_electricity_coordinating_council_california_south"),
        ("CISO", "NV", "western_electricity_coordinating_council_basin"),
        ("CISO", "WY", "western_electricity_coordinating_council_basin"),
        ("CPLE", "NC", "serc_reliability_corporation_east"),
        ("CPLE", "SC", "serc_reliability_corporation_east"),
        ("CPLW", "NC", "serc_reliability_corporation_east"),
        ("DEAA", "AZ", "western_electricity_coordinating_council_southwest"),
        ("DOPD", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("DUK", "GA", "serc_reliability_corporation_southeastern"),
        ("DUK", "NC", "serc_reliability_corporation_east"),
        ("DUK", "SC", "serc_reliability_corporation_east"),
        ("EPE", "NM", "western_electricity_coordinating_council_southwest"),
        ("EPE", "TX", "western_electricity_coordinating_council_southwest"),
        ("ERCO", "OK", "southwest_power_pool_south"),
        ("ERCO", "TX", "texas_reliability_entity"),
        ("FMPP", "FL", "florida_reliability_coordinating_council"),
        ("FPC", "FL", "florida_reliability_coordinating_council"),
        ("FPL", "FL", "florida_reliability_coordinating_council"),
        ("GCPD", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("GRID", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("GRIF", "AZ", "western_electricity_coordinating_council_southwest"),
        ("GRIS", "NM", "western_electricity_coordinating_council_southwest"),
        ("GRMA", "AZ", "western_electricity_coordinating_council_southwest"),
        ("GVL", "FL", "florida_reliability_coordinating_council"),
        ("GWA", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("HGMA", "AZ", "western_electricity_coordinating_council_southwest"),
        ("HST", "FL", "florida_reliability_coordinating_council"),
        ("IID", "CA", "western_electricity_coordinating_council_california_south"),
        ("IPCO", "ID", "western_electricity_coordinating_council_basin"),
        ("IPCO", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("IPCO", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("ISNE", "CT", "northeast_power_coordinating_council_new_england"),
        ("ISNE", "MA", "northeast_power_coordinating_council_new_england"),
        ("ISNE", "ME", "northeast_power_coordinating_council_new_england"),
        ("ISNE", "NH", "northeast_power_coordinating_council_new_england"),
        ("ISNE", "NY", "northeast_power_coordinating_council_upstate_new_york"),
        ("ISNE", "RI", "northeast_power_coordinating_council_new_england"),
        ("ISNE", "VT", "northeast_power_coordinating_council_new_england"),
        ("JEA", "FL", "florida_reliability_coordinating_council"),
        ("LDWP", "CA", "western_electricity_coordinating_council_california_south"),
        ("LDWP", "NV", "western_electricity_coordinating_council_basin"),
        ("LDWP", "UT", "western_electricity_coordinating_council_basin"),
        ("LGEE", "KY", "serc_reliability_corporation_central"),
        ("LGEE", "VA", "pjm_west"),
        ("MISO", "AR", "midcontinent_south"),
        ("MISO", "IA", "midcontinent_west"),
        ("MISO", "IL", "midcontinent_central"),
        ("MISO", "IN", "midcontinent_central"),
        ("MISO", "KY", "midcontinent_central"),
        ("MISO", "LA", "midcontinent_south"),
        ("MISO", "MI", "midcontinent_east"),
        ("MISO", "MN", "midcontinent_west"),
        ("MISO", "MO", "midcontinent_central"),
        ("MISO", "MS", "midcontinent_south"),
        ("MISO", "MT", "midcontinent_west"),
        ("MISO", "ND", "midcontinent_west"),
        ("MISO", "SD", "midcontinent_west"),
        ("MISO", "TX", "midcontinent_south"),
        ("MISO", "WI", "midcontinent_west"),
        ("NBSO", "ME", "northeast_power_coordinating_council_new_england"),
        ("NEVP", "CA", "western_electricity_coordinating_council_california_south"),
        ("NEVP", "NV", "western_electricity_coordinating_council_basin"),
        ("NWMT", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("NWMT", "WY", "western_electricity_coordinating_council_basin"),
        ("NYIS", "NJ", "pjm_east"),
        ("NYIS", "NY", "northeast_power_coordinating_council_upstate_new_york"),
        ("NYIS", "PA", "pjm_east"),
        ("PACE", "CO", "western_electricity_coordinating_council_rockies"),
        ("PACE", "ID", "western_electricity_coordinating_council_basin"),
        ("PACE", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PACE", "UT", "western_electricity_coordinating_council_basin"),
        ("PACE", "WY", "western_electricity_coordinating_council_basin"),
        ("PACW", "CA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PACW", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PACW", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PGE", "OR", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PGE", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PJM", "DC", "pjm_east"),
        ("PJM", "DE", "pjm_east"),
        ("PJM", "IL", "pjm_commonwealth_edison"),
        ("PJM", "IN", "pjm_west"),
        ("PJM", "KY", "pjm_west"),
        ("PJM", "MD", "pjm_east"),
        ("PJM", "MI", "pjm_west"),
        ("PJM", "MN", "pjm_west"),
        ("PJM", "NC", "pjm_dominion"),
        ("PJM", "NJ", "pjm_east"),
        ("PJM", "OH", "pjm_west"),
        ("PJM", "PA", "pjm_east"),
        ("PJM", "TN", "pjm_west"),
        ("PJM", "VA", "pjm_dominion"),
        ("PJM", "WV", "pjm_west"),
        ("PNM", "AZ", "western_electricity_coordinating_council_southwest"),
        ("PNM", "NM", "western_electricity_coordinating_council_southwest"),
        ("PNM", "TX", "western_electricity_coordinating_council_southwest"),
        ("PSCO", "CO", "western_electricity_coordinating_council_rockies"),
        ("PSEI", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("PSEI", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("SC", "SC", "serc_reliability_corporation_east"),
        ("SCEG", "GA", "serc_reliability_corporation_southeastern"),
        ("SCEG", "SC", "serc_reliability_corporation_east"),
        ("SCL", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("SEC", "FL", "florida_reliability_coordinating_council"),
        ("SEPA", "GA", "serc_reliability_corporation_southeastern"),
        ("SEPA", "NC", "serc_reliability_corporation_east"),
        ("SEPA", "SC", "serc_reliability_corporation_east"),
        ("SOCO", "AL", "serc_reliability_corporation_southeastern"),
        ("SOCO", "FL", "florida_reliability_coordinating_council"),
        ("SOCO", "GA", "serc_reliability_corporation_southeastern"),
        ("SOCO", "MA", "serc_reliability_corporation_southeastern"),
        ("SOCO", "MS", "serc_reliability_corporation_southeastern"),
        ("SPA", "AR", "southwest_power_pool_south"),
        ("SPA", "MO", "southwest_power_pool_central"),
        ("SPA", "OK", "southwest_power_pool_south"),
        ("SRP", "AZ", "western_electricity_coordinating_council_southwest"),
        ("SWPP", "AR", "southwest_power_pool_south"),
        ("SWPP", "CO", "southwest_power_pool_central"),
        ("SWPP", "IA", "southwest_power_pool_north"),
        ("SWPP", "KS", "southwest_power_pool_central"),
        ("SWPP", "LA", "southwest_power_pool_south"),
        ("SWPP", "MN", "southwest_power_pool_north"),
        ("SWPP", "MO", "southwest_power_pool_central"),
        ("SWPP", "MT", "southwest_power_pool_north"),
        ("SWPP", "ND", "southwest_power_pool_north"),
        ("SWPP", "NE", "southwest_power_pool_north"),
        ("SWPP", "NM", "southwest_power_pool_south"),
        ("SWPP", "OK", "southwest_power_pool_south"),
        ("SWPP", "SD", "southwest_power_pool_north"),
        ("SWPP", "TX", "southwest_power_pool_south"),
        ("TAL", "FL", "florida_reliability_coordinating_council"),
        ("TEC", "FL", "florida_reliability_coordinating_council"),
        ("TEPC", "AZ", "western_electricity_coordinating_council_southwest"),
        ("TEPC", "NM", "western_electricity_coordinating_council_southwest"),
        ("TIDC", "CA", "western_electricity_coordinating_council_california_north"),
        ("TPWR", "WA", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("TVA", "AL", "serc_reliability_corporation_central"),
        ("TVA", "GA", "serc_reliability_corporation_central"),
        ("TVA", "KY", "serc_reliability_corporation_central"),
        ("TVA", "MS", "serc_reliability_corporation_central"),
        ("TVA", "NC", "serc_reliability_corporation_east"),
        ("TVA", "TN", "serc_reliability_corporation_central"),
        ("WACM", "CO", "western_electricity_coordinating_council_rockies"),
        ("WACM", "KS", "southwest_power_pool_central"),
        ("WACM", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("WACM", "ND", "southwest_power_pool_north"),
        ("WACM", "NE", "western_electricity_coordinating_council_rockies"),
        ("WACM", "NM", "western_electricity_coordinating_council_rockies"),
        ("WACM", "SD", "western_electricity_coordinating_council_rockies"),
        ("WACM", "UT", "western_electricity_coordinating_council_basin"),
        ("WACM", "WY", "western_electricity_coordinating_council_rockies"),
        ("WALC", "AZ", "western_electricity_coordinating_council_southwest"),
        ("WALC", "CA", "western_electricity_coordinating_council_southwest"),
        ("WALC", "NM", "western_electricity_coordinating_council_southwest"),
        ("WALC", "NV", "western_electricity_coordinating_council_basin"),
        ("WAUW", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("WAUW", "UT", "western_electricity_coordinating_council_basin"),
        ("WAUW", "WY", "western_electricity_coordinating_council_basin"),
        ("WWA", "MT", "western_electricity_coordinating_council_northwest_power_pool_area"),
        ("YAD", "NC", "serc_reliability_corporation_east"),
    ],
    schema=[
        "balancing_authority_code_eia",
        "state",
        "electricity_market_module_region_eiaaeo",
    ],
    orient="row",
)


# From Transect analysis
# https://rockmtnins.sharepoint.com/:x:/s/UTF/EWhBbejQ5HxMnPPKZV5hYGYBCyFLEdVxshBkJeikr_3NIg?e=NQJ2Sc
# based on colo_202504230013
LAND = pl.from_records(
    [
        (3, "timber companies", 79000.0, 45000.0, 2.0, 262.0, 27.0, None),
        (55409, "no", None, None, None, None, None, None),
        (55440, "no", None, None, None, None, None, None),
        (10, "yes", 124000.0, 87000.0, 7.0, 766.0, 172.0, None),
        (55411, "no", None, None, None, None, None, None),
        (7721, "no", None, None, None, None, None, None),
        (55340, "yes", 51000.0, 41000.0, 2.0, 304.0, 35.0, None),
        (55455, "yes", 112000.0, 78000.0, 1.0, 388.0, 22.0, None),
        (
            117,
            "5 miles through suburbs + 8 to big land 30k with unkown owner, not parcels?",
            None,
            30000.0,
            2.0,
            1.0,
            None,
            40.0,
        ),
        (399, "lol no", None, None, None, None, None, None),
        (56298, "yes", 10000.0, 8000.0, 2.0, 55.0, 15.0, 5.0),
        (404, "lol no", None, None, None, None, None, None),
        (55200, "urban", None, None, None, None, None, None),
        (55645, "yes", 60000.0, 39000.0, 2.0, 142.0, 29.0, 5.0),
        (469, "urban", None, None, None, None, None, None),
        (
            6112,
            "2 miles to 6k, 6 miles to 10k, 16 miles to 20k",
            140000.0,
            135000.0,
            2.0,
            41.0,
            38.0,
            15.0,
        ),
        (55453, "yes", 15000.0, 10000.0, 2.0, 21.0, 6.0, 5.0),
        (55505, "adjacent", 6000.0, 5000.0, 1.0, 42.0, 5.0, 0.0),
        (55504, "adjacent, other options", 35000.0, 29000.0, 1.0, 22.0, 7.0, 0.0),
        (
            55127,
            "colo coal plant, increases export capacity",
            89000.0,
            75000.0,
            1.0,
            202.0,
            12.0,
            0.0,
        ),
        (
            6761,
            "colo coal plant, increases export capacity",
            102000.0,
            66000.0,
            1.0,
            202.0,
            30.0,
            5.0,
        ),
        (55835, "may have conflict", 64000.0, 53000.0, 1.0, 224.0, 8.0, 15.0),
        (56445, "yes", 10000.0, 8700.0, 1.0, None, 4.0, 20.0),
        (
            7846,
            "6 miles to 5.8k 8 miles to 16k acres, another 12 miles to 20k acres, multiple clusters of 640 acre chunks closer",
            92000.0,
            55000.0,
            6.0,
            32.0,
            20.0,
            15.0,
        ),
        (612, "suburban, shame cuz lots of land", None, None, None, None, None, None),
        (56799, "urban", None, None, None, None, None, None),
        (7302, "near Tampa", 29000.0, 17000.0, 2.0, 145.0, 4.0, 30.0),
        (8049, "suburban", None, None, None, None, None, None),
        (666, "urban", None, None, None, None, None, None),
        (613, "urban", None, None, None, None, None, None),
        (6042, " near Tampa ", 31000.0, 21000.0, 1.0, 63.0, 6.0, 5.0),
        (55286, "no", None, None, None, None, None, None),
        (54365, "conflict", None, None, None, None, None, None),
        (7242, "conflict", None, None, None, None, None, None),
        (564, "no", None, None, None, None, None, None),
        (7699, "conflict", None, None, None, None, None, None),
        (56400, "yes", 47000.0, 28000.0, 1.0, 92.0, 17.0, 30.0),
        (55672, "so many it broke transect", 154000.0, 101000.0, 5.0, 0.0, 0.0, 0.0),
        (7917, "no", None, None, None, None, None, None),
        (55244, "no", None, None, None, None, None, None),
        (55141, "no", None, None, None, None, None, None),
        (56150, None, 61000.0, 41000.0, 4.0, 115.0, 35.0, None),
        (7764, "no", None, None, None, None, None, None),
        (7813, "no", None, None, None, None, None, None),
        (7829, "no", None, None, None, None, None, None),
        (55382, "no", None, None, None, None, None, None),
        (8031, "no", None, None, None, None, None, None),
        (7985, "no", None, None, None, None, None, None),
        (58236, "no", None, None, None, None, None, None),
        (55733, "yes", 42000.0, 33000.0, 6.0, 45.0, 30.0, 25.0),
        (7953, "yes", 42000.0, 33000.0, 6.0, 45.0, 30.0, 25.0),
        (57028, "not great", 30000.0, 18000.0, 12.0, 207.0, 22.0, 50.0),
        (7759, "urban", None, None, None, None, None, None),
        (990, "no", None, None, None, None, None, None),
        (7763, "yes", 8000.0, 6000.0, 7.0, 163.0, 18.0, 35.0),
        (1007, "marginal", 14000.0, 11000.0, 5.0, 257.0, 41.0, 100.0),
        (55224, "hard", None, None, None, None, None, None),
        (
            56502,
            "Two clusters nearly adjacent, the others 18 and 11 miles away ",
            86000.0,
            62000.0,
            4.0,
            283.0,
            85.0,
            15.0,
        ),
        (1240, "a long way for not much", 9000.0, 7000.0, 2.0, 37.0, 12.0, 50.0),
        (7929, "too fragmented", None, None, None, None, None, None),
        (1363, "no", None, None, None, None, None, None),
        (1355, "very hard", None, None, None, None, None, None),
        (1366, "urban and too fragmented", None, None, None, None, None, None),
        (6071, " no ", None, None, None, None, None, None),
        (55173, None, 72000.0, 40000.0, 3.0, 150.0, 22.0, None),
        (1396, None, 89000.0, 56000.0, 1.0, 481.0, 51.0, 5.0),
        (56565, "urban, shame cuz lots of land", None, None, None, None, None, None),
        (6035, " too fragmented ", None, None, None, None, None, None),
        (55270, "no", None, None, None, None, None, None),
        (55402, "hard", None, None, None, None, None, None),
        (55087, "no", None, None, None, None, None, None),
        (1904, "suburban", None, None, None, None, None, None),
        (8027, "suburban", None, None, None, None, None, None),
        (7848, "marginal", 30000.0, 23000.0, 4.0, 200.0, 64.0, 35.0),
        (7754, "not enough land", None, None, None, None, None, None),
        (7964, "hard", None, None, None, None, None, None),
        (55063, "hard", None, 8600.0, 1.0, None, 15.0, 25.0),
        (57037, "Weyerhaeuser and other timber", 106000.0, 70000.0, 3.0, 357.0, 37.0, None),
        (6073, "no", None, None, None, None, None, None),
        (56249, "same as Hamlet", None, None, None, None, None, None),
        (2720, "no", None, None, None, None, None, None),
        (2723, "no", None, None, None, None, None, None),
        (56292, None, 16950.0, 10000.0, 10.0, 105.0, 4.0, 30.0),
        (59325, "no", None, None, None, None, None, None),
        (7277, "no", None, None, None, None, None, None),
        (55116, "no", None, None, None, None, None, None),
        (55343, "mostly state land", 141000.0, 101000.0, 1.0, 58.0, 13.0, 3.0),
        (
            7975,
            "adjacent option is 20k acre cattle ranch, other options within 20 miles fed and state land, also sufficient contiguous 640 acre lots but unkown owner",
            108000.0,
            88000.0,
            4.0,
            106.0,
            16.0,
            5.0,
        ),
        (
            55802,
            "N: 12 miles to 12k state and ranches, 14 miles to 61k Isleta Indian Reservation, S: 16 miles to 30k private",
            0.0,
            103000.0,
            4.0,
            0.0,
            7.0,
            15.0,
        ),
        (
            55322,
            "Usa Moapa Indian Reservation border within 6 miles, already has solar farm, land inbetween owned by USA",
            71000.0,
            50000.0,
            1.0,
            25.0,
            10.0,
            1.0,
        ),
        (2322, "urban", None, None, None, None, None, None),
        (7082, "same as chuck lenzie", None, None, None, None, None, None),
        (55687, "no", None, None, None, None, None, None),
        (10761, "urban", None, None, None, None, None, None),
        (55841, "same as chuck lenzie", None, None, None, None, None, None),
        (2336, "no", None, None, None, None, None, None),
        (7757, None, 52000.0, 31000.0, 7.0, 249.0, 29.0, None),
        (2963, None, 46000.0, 32000.0, 2.0, 135.0, 11.0, None),
        (4940, " suburban ", None, None, None, None, None, None),
        (2964, "too fragmented", None, None, None, None, None, None),
        (55166, "no", None, None, None, None, None, None),
        (55386, "no", None, None, None, None, None, None),
        (55927, None, 32000.0, 21000.0, 1.0, 38.0, 14.0, None),
        (7981, "no", None, None, None, None, None, None),
        (
            3295,
            "10 miles to AEC land another 7 to 310 sq mile Savannah river site offered by DOE",
            None,
            100000.0,
            None,
            None,
            None,
            None,
        ),
        (3264, "no", None, None, None, None, None, None),
        (7237, "too fragmented", None, None, None, None, None, None),
        (61144, None, 73000.0, 58000.0, 5.0, 535.0, 117.0, None),
        (7512, "20 miles from plant", 50000.0, 32000.0, 1.0, 143.0, 15.0, None),
        (60264, "suburban", None, None, None, None, None, None),
        (55168, None, 49000.0, 33000.0, 3.0, 319.0, 52.0, None),
        (55172, None, 79000.0, 50000.0, 2.0, 333.0, 27.0, 5.0),
        (56350, "close to Houston", 72000.0, 46000.0, 5.0, 250.0, 65.0, None),
        (
            58471,
            "lots of land but current use is o&g",
            552000.0,
            460000.0,
            1.0,
            784.0,
            17.0,
            0.0,
        ),
        (55365, "urban", None, None, None, None, None, None),
        (55480, "suburban", None, None, None, None, None, None),
        (55153, "clusters far apart", 51000.0, 39000.0, 4.0, 213.0, 37.0, None),
        (55545, None, 422000.0, 250000.0, 1.0, 24.0, 7.0, None),
        (63335, "no", None, None, None, None, None, None),
        (55230, None, 130000.0, 86000.0, 4.0, 863.0, 113.0, None),
        (
            54817,
            "multiple options, some exurban around plant",
            21000.0,
            15000.0,
            2.0,
            102.0,
            10.0,
            25.0,
        ),
        (
            3482,
            "13 miles to main cluster, others adjacent",
            234000.0,
            153000.0,
            2.0,
            555.0,
            37.0,
            8.0,
        ),
        (55097, None, 74000.0, 54000.0, 8.0, 432.0, 54.0, 50.0),
        (
            3609,
            "average size of 350, mix of city and other owners across south edge of San Antonio, multiple interstates must be crossed",
            None,
            4900.0,
            8.0,
            0.0,
            8.0,
            30.0,
        ),
        (55154, "possible conflict with Bastrop", 42000.0, 28000.0, 1.0, 196.0, 10.0, 15.0),
        (58562, "larger options further away", 28000.0, 22000.0, 3.0, 69.0, 14.0, 20.0),
        (
            3456,
            "conflict with Montana TX, close land protected",
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (55215, "oil and gas", 82000.0, 71000.0, 2.0, 139.0, 5.0, 10.0),
        (
            56349,
            "lots of land but current use is o&g",
            256000.0,
            226000.0,
            1.0,
            693.0,
            54.0,
            0.0,
        ),
        (58005, "too fragmented", None, None, None, None, None, None),
        (3631, "lots of further options", 18000.0, 5000.0, 1.0, 41.0, 7.0, 1.0),
        (7900, "conflict with Bastrop", None, None, None, None, None, None),
        (58001, None, 49000.0, 37000.0, 6.0, 129.0, 39.0, None),
        (4937, None, 52000.0, 34000.0, 3.0, 161.0, 25.0, None),
        (63688, "suburban", None, None, None, None, None, None),
        (3612, "conflicts with Arthur von R", None, None, None, None, None, None),
        (
            3443,
            "lots of options, this one is 11 miles away",
            16000.0,
            11000.0,
            1.0,
            11.0,
            2.0,
            None,
        ),
        (56674, None, 17000.0, 11000.0, 1.0, 64.0, 10.0, None),
        (56807, "no", None, None, None, None, None, None),
        (
            58260,
            "Lots of small parcels connecting bigger ones",
            137000.0,
            91000.0,
            6.0,
            722.0,
            172.0,
            0.0,
        ),
        (59913, "same as Brunswick", None, None, None, None, None, None),
        (7838, "expensive", None, None, None, None, None, None),
        (55029, "urban", None, None, None, None, None, None),
        (7270, "msft / foxconn", 8000.0, 6000.0, 9.0, 140.0, 16.0, None),
        (4040, " urban ", None, None, None, None, None, None),
        (55641, None, 37000.0, 30000.0, 11.0, 478.0, 76.0, None),
        (7991, "urban", None, None, None, None, None, None),
        (55011, "conflicts with riverside", None, None, None, None, None, None),
    ],
    schema=[
        "icx_id",
        "siting_notes",
        "total_acres",
        "buildable_acres",
        "num_clusters",
        "num_parcels",
        "num_owners",
        "num_crossing",
    ],
    orient="row",
)


def compare(a: pd.DataFrame, b: pd.DataFrame, cols=None, **kwargs) -> pd.DataFrame:
    ix = ["ba_code", "icx_id", "status", "tech", "name", "regime"]
    if cols is None:
        cols = [
            "run_status",
            "load_mw",
            "served_pct",
            "unserved_mwh",
            "unserved_pct_hrs",
            "pct_load_clean",
            "pct_land_used",
            "pct_overbuild",
            "ppa_ex_fossil_export_profit",
            "attr_cost_clean",
            "attr_cost_load_fossil",
            "attr_rev_export_clean",
            "solar_mw",
            "onshore_wind_mw",
            "li_mw",
            "fe_mw",
            "new_fossil_mw",
            "li_duration",
            "fossil_cf",
            "load_fossil_cf",
            "export_fossil_cf",
            "historical_fossil_cf",
            "baseline_fossil_cf",
            "longest_violation",
            "total_capacity_violation",
            "total_rolling_violation",
            "max_capacity_violation_pct",
            "max_rolling_violation_pct",
            "fossil_co2",
            "clean_pct_curtailment",
        ]
    a = a.set_index(ix)
    b = b.set_index(ix)
    common_ix = a.index.intersection(b.index)
    common_cols = a.columns.intersection(b.columns).intersection(cols)
    o = (
        a.loc[common_ix, common_cols]
        .compare(b.loc[common_ix, common_cols], **kwargs)
        .sort_index()
        .reorder_levels([1, 0], axis=1)
    )
    o.columns = map("|".join, o.columns)
    return o.reset_index()


def find_violations():
    id_cols = ("icx_id", "icx_gen", "regime", "ix", "datetime")
    base_cols = ("export_fossil", "load_fossil", "historical_fossil", "export_requirement")

    def violations(file):
        base = (
            pl.scan_parquet(file, schema=id_cols | base_cols)
            .select(*id_cols, *base_cols)
            .with_columns(
                rolling_max=pl.sum("export_requirement")
                .rolling("datetime", period="24h")
                .max()
                .cast(pl.Float32)
            )
            .with_columns(
                capacity_vio=pl.max_horizontal(
                    pl.sum_horizontal("export_fossil", "load_fossil")
                    - pl.col("historical_fossil").max(),
                    0.0,
                ).cast(pl.Float32),
                rolling_vio=pl.max_horizontal(
                    pl.sum_horizontal("export_fossil", "load_fossil")
                    .sum()
                    .rolling("datetime", period="24h")
                    - pl.col("rolling_max"),
                    0.0,
                ).cast(pl.Float32),
            )
            .with_columns(
                rolling_rel=(pl.col("rolling_vio") / pl.col("rolling_max")).cast(pl.Float32),
                capacity_rel=(pl.col("capacity_vio") / pl.col("historical_fossil").max()).cast(
                    pl.Float32
                ),
            )
        )
        base.unpivot(
            on=cs.contains("_rel"),
            index=[*id_cols],
            variable_name="violation",
            value_name="relative",
        ).with_columns(pl.col("violation").str.replace("_rel", "").cast(pl.Categorical)).join(
            base.unpivot(
                on=cs.contains("_vio"),
                index=[*id_cols],
                variable_name="violation",
                value_name="absolute",
            ).with_columns(pl.col("violation").str.replace("_vio", "").cast(pl.Categorical)),
            on=[*id_cols, "violation"],
        ).sink_parquet(file.parents[1] / "violations" / file.name)

    for file in tqdm(Path.home().glob("patio_data/colo_202505062157/hourly/*_10.parquet")):
        violations(file)
    for file in tqdm(Path.home().glob("patio_data/colo_202505062157/hourly/*_13.parquet")):
        violations(file)


def text_position(scen_df, pid):
    if scen_df["pct_load_clean"] == pid["pct_load_clean"].min():
        h = "left"
    elif scen_df["pct_load_clean"] == pid["pct_load_clean"].max():
        h = "right"
    else:
        h = "center"
    if scen_df["ppa_ex_fossil_export_profit"] == pid["ppa_ex_fossil_export_profit"].min():
        v = "bottom"
    elif scen_df["ppa_ex_fossil_export_profit"] == pid["ppa_ex_fossil_export_profit"].max():
        v = "top"
    else:
        v = "middle"
    position = f"{v} {h}"
    return position


def make_case_sheets(colo, flows, *, land_screen=True, test=False):
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
    if land_screen:
        colo = colo.query("required_acres <= 1.5 * buildable_acres")
    good = good_screen(colo)
    pl_good = (
        pl.from_pandas(colo.query("run_status == 'SUCCESS'"))
        # pl.from_pandas(good_screen(colo))
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
            colors,
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
        zip(cs.expand_selector(pl_good, cs.contains("attr_")), colors, strict=False)
    )
    good = pl.from_pandas(good).select(
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
        flows.with_columns(
            full_name=pl.concat_str("name", "regime", separator="<br>"),
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
    pl_good_parts = pl_good.sort("icx_id").partition_by("icx_id", "icx_gen", as_dict=True)
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
    subplots = []

    def name_adder(n_):
        # if "<b>" in n_:
        #     return n_ + "<br>Load Served ≥ 99.9% & Hourly Matched Clean ≥ 60% & PPA ≤ $200/MWh"
        return n_

    for j, (keys, pid) in enumerate(tqdm(pl_good_parts.items())):
        pid = pid.sort("name", "regime")
        names = pid.select("full_name").to_series()
        rows = max(8, len(pid))
        plt = (
            make_subplots(
                rows=rows,
                cols=2,
                specs=[[{"type": "scatter"}, {"type": "sankey"}]] * rows,
                subplot_titles=[k for n in names for k in ["", name_adder(n)]],
                vertical_spacing=0.025,
                horizontal_spacing=0.02,
            )
            .update_layout(
                title=("** " if keys[0] in good["icx_id"].to_numpy() else "")
                + pid.select(
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
                .item(),
                template="ggplot2",
                font_family="Roboto",
                height=50 + 380 * rows,
                width=1400,
                xaxis=dict(  # noqa: C408
                    tickformat=".0%", range=[0.5, 1.05], title="% clean", title_standoff=1
                ),
                yaxis=dict(range=[0, 350], tickformat="$.0f", title="ppa", title_standoff=1),  # noqa: C408
                yaxis2=dict(tickprefix="$", title="ppa attribution", title_standoff=1),  # noqa: C408
                yaxis7=dict(tickprefix="$", title="capex/MW load", title_standoff=1),  # noqa: C408
                yaxis3=dict(tickformat=".0%", title="% of load capacity", title_standoff=1),  # noqa: C408
                yaxis4=dict(tickformat=".0%", title="% of load energy", title_standoff=1),  # noqa: C408
                yaxis5=dict(  # noqa: C408
                    tickformat=".0%",
                    title="% of generation",
                    range=[0, 1.19],
                    title_standoff=1,
                ),
                yaxis6=dict(  # noqa: C408
                    tickformat=".0%", title="fossil capacity factor", title_standoff=1
                ),
                yaxis8=dict(title="acres", title_standoff=1),  # noqa: C408
                barmode="stack",
                legend=dict(  # noqa: C408
                    title="Scenario name", xref="paper", yref="paper", x=1.025, y=0.97
                ),
                margin_r=200,
            )
            .add_annotation(
                text="<b>Bolded</b> Scenario name means <br> Load Served ≥ 99.9%"
                "<br> & Hourly Matched Clean ≥ 60%<br> & PPA ≤ $200/MWh<br>Land confirmed",
                xref="paper",
                yref="paper",
                align="left",
                showarrow=False,
                font=dict(size=12),  # noqa: C408
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
            pl.col("source_").is_in(("Solar", "Wind")) & pl.col("target_").is_in(("Load",))
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
        costs = {"attr_cost_clean": "Clean Cost", "attr_cost_load_fossil": "Fossil Cost"}
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
                            "solar_mw", "li_mw", "onshore_wind_mw", "fe_mw", "new_fossil_mw"
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
        for (target,), part in source_flows.partition_by("source_", as_dict=True).items():
            if target == "Solar":
                plt.add_scatter(
                    x=part["full_name"],
                    y=[
                        source_flows.group_by("full_name").agg(pl.sum("share"))["share"].max()
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
        for target in ("Required Export", "Load", "Storage", "Surplus Export", "Curtailment"):
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
                plt.add_scatter(y=[com_rate] * 3, text=["", "commercial<br>rate", ""], **rkwgs)
            if (ind_rate := pid["ind_rate"][0]) is not None:
                plt.add_scatter(y=[ind_rate] * 3, text=["", "industrial<br>rate", ""], **rkwgs)
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
    return subplots


def flows_w_allocated_storage(run, fs=None):
    idc = ["ba_code", "icx_id", "icx_gen", "regime", "name", "ix"]
    repl = {"Surplus Export": "Export", "Required Export": "Export"}
    for_util = (
        get_flows(run, fs)
        .group_by(*idc, "source_", pl.col("target_").replace(repl))
        .agg(pl.sum("value"))
    )
    via_storage = (
        for_util.filter(pl.col("source_") == "Storage")
        .select(*idc, "target_", "value")
        .join(
            for_util.filter(pl.col("target_") == "Storage").select(
                *idc,
                "source_",
                es=pl.sum("value").over(*idc, "source_") / pl.sum("value").over(*idc),
            ),
            on=idc,
            how="left",
        )
        .select(*idc, "source_", "target_", es=pl.col("value") * pl.col("es"))
    )
    out = (
        for_util.join(via_storage, on=[*idc, "source_", "target_"], how="left")
        .with_columns(value=pl.sum_horizontal("value", "es"))
        .filter((pl.col("target_") != "Storage") & (pl.col("source_") != "Storage"))
        .pivot(on="target_", index=[*idc, "source_"], values="value")
        .sort("icx_id", "regime", "ix", "source_")
    )
    assert (
        out.group_by(*idc)
        .agg(pl.col("Load", "Export").sum())
        .join(
            for_util.filter(pl.col("target_").is_in(("Export", "Load"))).pivot(
                on="target_", index=idc, values="value", aggregate_function="sum"
            ),
            on=idc,
            validate="1:1",
            how="left",
        )
        .filter(
            ((pl.col("Load").fill_nan(None) - pl.col("Load_right")).abs() > 1)
            | ((pl.col("Export").fill_nan(None) - pl.col("Export_right")).abs() > 1)
        )
        .is_empty()
    )
    return out


def parse_log(run: str, *, local: bool = False) -> pl.DataFrame:
    run = "colo_" + run.removeprefix("colo_")
    if local:
        return pl.read_ndjson(Path.home() / f"patio_data/{run}/log.jsonl")
    else:
        fs = rmi_cloud_fs()
        with fs.open(f"patio-results/{run}/log.jsonl") as f:
            return pl.read_ndjson(f)
