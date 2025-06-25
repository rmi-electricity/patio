from __future__ import annotations

import itertools
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import signature
from typing import TYPE_CHECKING, Literal, NamedTuple

import cvxpy as cp
import numpy as np
import pandas as pd
from dispatch import zero_profiles_outside_operating_dates
from numba import NumbaPerformanceWarning, njit
from plotly import express as px
from plotly import graph_objs as go  # noqa: TC002
from scipy.optimize import LinearConstraint, linprog, minimize

from patio.constants import COLORS, MTDF, PERMUTATIONS
from patio.helpers import agg_profile, solver

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from dispatch import DispatchModel

LOGGER = logging.getLogger("patio")
__all__ = ["BaseProfileMatch", "equal_capacity", "equal_energy"]


class ScenarioConfig(NamedTuple):
    re_energy: float
    storage_li_pct: float = 0.0  # as % fossil capacity
    storage_fe_pct: float = 0.0  # as % fossil capacity
    storage_h2_pct: float = 0.0  # as % fossil capacity
    nuclear_scen: int = 0  # which of the baseload retirement plant sets to use
    ccs_scen: int = 0
    excl_or_moth: Literal["mothball", "exclude"] = "mothball"

    @property
    def re_share_of_total(self):
        return self.re_energy

    def is_re_child(self, other: ScenarioConfig) -> bool:
        """Determine if ``other`` is a child of this scenario."""
        if hasattr(other, "config") and isinstance(other.config, ScenarioConfig):
            other = other.config
        if not isinstance(other, ScenarioConfig):
            return NotImplemented
        return all(
            (
                other.re_energy > self.re_energy,
                # other.storage_li_pct >= self.storage_li_pct,
                # other.storage_fe_pct >= self.storage_fe_pct,
                # other.storage_h2_pct >= self.storage_h2_pct,
                other.ccs_scen >= self.ccs_scen,
                other.nuclear_scen >= self.nuclear_scen,
                other.excl_or_moth == self.excl_or_moth,
            )
        )

    def is_li_child(self, other: ScenarioConfig) -> bool:
        """Determine if ``other`` is a child of this scenario."""
        if hasattr(other, "config") and isinstance(other.config, ScenarioConfig):
            other = other.config
        if not isinstance(other, ScenarioConfig):
            return NotImplemented
        return all(
            (
                other.re_energy >= self.re_energy,
                other.storage_li_pct >= self.storage_li_pct,
                other.storage_fe_pct >= self.storage_fe_pct,
                other.storage_h2_pct >= self.storage_h2_pct,
                other.ccs_scen >= self.ccs_scen,
                other.nuclear_scen >= self.nuclear_scen,
                other.excl_or_moth == self.excl_or_moth,
            )
        )

    def __str__(self):
        return (
            f"({', '.join(f'{k}={v}' for k, v in self._asdict().items() if any((v, '_li_' in k)))})".replace(
                "storage_", ""
            )
            .replace("_scen", "")
            .replace("_pct", "")
            .replace("excl_or_moth=", "")
        )

    @property
    def for_idx(self):
        return (
            str(self)
            .removeprefix("(")
            .removesuffix(")")
            .replace("mothball", "")
            .replace(" , ", " ")
            .replace("_energy", "")
            .replace("ear=1", "ear")
            .replace("ccs=-1", "ccs_ptc")
            .replace("ccs=1", "ccs_no_ptc")
            .removesuffix(", ")
        )

    @property
    def storage_cap_pct_array(self):
        return np.array(
            [self.storage_li_pct, self.storage_fe_pct, self.storage_h2_pct],
            dtype=np.float64,
        )

    def split_ccs(self) -> tuple:
        if self.ccs_scen == 0:
            return (self,)
        return self, self._replace(ccs_scen=-self.ccs_scen)

    def storage_specs(
        self,
        capacity: float,
        operating_date: pd.Timestamp | datetime | None = None,
        eff: tuple[float] = (0.9, 0.75, 0.5),
        ids: tuple[int] = (-1, -2, -3),
        **kwargs,
    ) -> pd.DataFrame:
        if operating_date is None:
            operating_date = pd.to_datetime(2020, format="%Y")
        return pd.DataFrame(
            {
                "plant_id_eia": list(ids),
                "generator_id": ["es"] * 3,
                "capacity_mw": capacity * self.storage_cap_pct_array,
                "duration_hrs": [4, 100, 8760],
                "roundtrip_eff": list(eff),
                "reserve": [0.0, 0.1, 0.1],
                "technology_description": ["Batteries", "Batteries", "H2 Storage"],
                "operating_date": [operating_date] * 3,
            }
            | {k: [v] * 3 for k, v in kwargs.items()}
        ).set_index(["plant_id_eia", "generator_id"])

    @staticmethod
    def from_sweeps(
        re: Sequence[float],
        li: Sequence[float] = (0.0,),
        fe: Sequence[float] = (0.0,),
        h2: Sequence[float] = (0.0,),
        nuke: Sequence[int] = (0,),
        ccs: Sequence[int] = (0,),
        excl_or_moth: Sequence[str] = ("mothball",),
        no_limit_prime: Sequence[str] = ("GT",),
    ):
        return [
            ScenarioConfig(*x)
            for x in itertools.product(re, li, fe, h2, nuke, ccs, excl_or_moth)
        ]


@dataclass
class BaseProfileMatch:
    capacity: float = field(repr=False, compare=False)
    req_profile: pd.Series | pd.DataFrame = field(repr=False, compare=False)
    re_profiles: pd.DataFrame = field(repr=False, compare=False)
    fuel_profile: pd.DataFrame = field(default=None, repr=False, compare=False)
    _re_cap: None | np.ndarray = field(default=None, repr=False, compare=False)
    _en_cap: None | np.ndarray = field(default=None, repr=False, compare=False)
    _errors: list = field(default_factory=list, repr=False, compare=False)
    _resources: tuple = field(default_factory=tuple, repr=False, compare=False)
    _cap: bool = field(default=False, repr=False, compare=False)
    _co2_profile: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False, compare=False)

    def __post_init__(self):
        self._resources = tuple(self.re_profiles.columns)
        self.req_profile = self.req_profile.fillna(0.0)
        assert self.req_profile.notna().all(), "profiles has nans"
        assert self.re_profiles.notna().all(axis=None), "re_profiles has nans"

    @classmethod
    def kw(cls, **kwargs):
        return cls(**{k: v for k, v in kwargs.items() if k in signature(cls).parameters})

    @property
    def resources(self) -> tuple:
        try:
            return self._resources
        except AttributeError:
            return tuple(self.re_profiles.columns)

    def by_inverse(self) -> None:
        """Find the solar and wind capacity that most closely matches
        the fossil plant's output using a generalized inverse

        :return: None

        .. math::
            \\mathbf{G} = \begin{bmatrix}
            \\mathbf{daily\\:solar} & \\mathbf{daily\\:wind}
            \\end{bmatrix}\\
            \\mathbf{d} = [\\mathbf{daily\\:output}]

            \begin{bmatrix}
            solar\\:cap & wind\\:cap
            \\end{bmatrix} = (\\mathbf{G}^{T}\\mathbf{G})^{-1}\\mathbf{G}^{T}\\mathbf{d}
        """  # noqa: D301
        re_cap = np.linalg.pinv(self.re_profiles.to_numpy()) @ self.req_profile.to_numpy()
        assert np.all(re_cap >= 0), "inverse method produced negative capacity"
        self._re_cap = re_cap

    def optimize(self, try_capacity=True) -> None:
        self.by_equal_energy()
        if self.try_capacity(try_capacity):
            self.by_equal_capacity()
        assert self.test_offshore(), "offshore selected but no resource"
        assert self.test_onshore(), "onshore selected but no resource"
        return None

    def try_capacity(self, try_capacity):
        return try_capacity and self._re_cap.sum() < self.capacity

    def by_equal_energy(self, inc=0.1) -> None:
        """Find the solar and wind capacity that produces the same energy
        while avoiding as much fossil generation as possible

        :param inc: increment to use in search
        :return: None
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
            self._re_cap = equal_energy(
                self.req_profile.to_numpy(),
                self.re_profiles.to_numpy(),
                inc,
            )
        return None

    def by_equal_capacity(self, inc=0.1) -> None:
        """Find the solar and wind capacity equal to fossil capacity
        that avoids as much fossil generation as possible

        :param inc: increment to use in search
        :return: None
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
            result = equal_capacity(
                self.capacity,
                self.req_profile.to_numpy(),
                self.re_profiles.to_numpy(),
                inc,
            )
            if np.sum(result) > 0:
                self._en_cap, self._re_cap = self._re_cap, result
                assert self.avoided_test(self._re_cap) > self.avoided_test(self._en_cap), (
                    "less fossil avoided by capacity method"
                )
                self._cap = True
                # print("c", end="", flush=True)
                # LOGGER.info("c")
        return None

    def to_pd(self, item, name=None, prefix="", suffix="") -> pd.Series | pd.DataFrame:
        if isinstance(item, pd.DataFrame | pd.Series):
            return item.fillna(0.0)
        if isinstance(item, np.ndarray):
            if item.shape == (len(self.req_profile),) or item.shape == (
                len(self.req_profile),
                1,
            ):
                return pd.Series(item, index=self.req_profile.index, name=name).fillna(0.0)
            if item.shape == (len(self.req_profile), len(self.resources)):
                return pd.DataFrame(
                    item,
                    index=self.req_profile.index,
                    columns=[f"{prefix}{r}{suffix}" for r in self.resources],
                ).fillna(0.0)

    @property
    def re_cap(self) -> dict:
        return dict(zip(self.resources, self._re_cap, strict=False))

    @property
    def re(self) -> np.ndarray:
        """Total renewable generation"""
        return self.re_profiles.to_numpy() @ self._re_cap

    @property
    def re_shares(self) -> np.ndarray:
        a, b = self.re_by_type, self.re.reshape((len(self.re), 1))
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    @property
    def re_by_type(self) -> np.ndarray:
        """Total renewable generation"""
        return self.re_profiles.to_numpy() @ np.diag(self._re_cap)

    @property
    def avoided(self) -> np.ndarray:
        """Avoided fossil generation"""
        return np.nan_to_num(
            np.minimum(
                self.req_profile.to_numpy(),
                self.re,
            )
        )

    def avoided_test(self, re):
        return np.nansum(
            np.minimum(
                self.req_profile.to_numpy(),
                self.re_profiles.to_numpy() @ re,
            )
        )

    @property
    def avoided_fuel(self) -> np.ndarray:
        """Avoided fossil generation"""
        return np.nan_to_num(
            self.fuel_profile.to_numpy() * self.avoided / self.req_profile.to_numpy()
        )

    @property
    def excess_re(self) -> np.ndarray:
        """Renewable generation that exceeds fossil requirement"""
        return np.nan_to_num(np.maximum(self.re - self.req_profile.to_numpy(), 0))

    @property
    def excess_by_type(self) -> np.ndarray:
        return self.excess_re.reshape((len(self.excess_re), 1)) * self.re_shares

    @property
    def ix(self):
        return NotImplemented

    @property
    def ix_names(self):
        return NotImplemented

    @property
    def carbon_intensity(self):
        return NotImplemented

    def _output_incr(self, share: float):
        re_cap = dict(zip(self.re_profiles.columns, self._re_cap * share, strict=False))
        re_by_type = self.re_profiles.to_numpy() @ np.diag(self._re_cap * share)
        avoided = np.nan_to_num(
            np.minimum(
                self.req_profile.to_numpy(),
                re_by_type.sum(axis=1),
            )
        )
        avoided_fuel = (self.fuel_profile * avoided / self.req_profile.to_numpy()).fillna(0)
        excess_re = np.nan_to_num(
            np.maximum(re_by_type.sum(axis=1) - self.req_profile.to_numpy(), 0)
        )
        excess_re_by_type = excess_re.reshape((len(excess_re), 1)) * self.re_shares
        if self.req_profile.sum() == 0:  # noqa: SIM108
            replace = 0
        else:
            replace = avoided.sum() / self.req_profile.sum()
        try:
            df = pd.DataFrame.from_dict(
                {
                    (*self.ix, self.fmt_share(share)): {
                        **re_cap,
                        "avoided_fossil_mwh": avoided.sum(),
                        "avoided_fuel_mmbtu": avoided_fuel.sum(),
                        "avoided_carbon": avoided_fuel.sum() * self.carbon_intensity,
                        **{
                            f"{r}_mwh": re_by_type[:, i].sum()
                            for i, r in enumerate(self.resources)
                        },
                        # "excess_re_mwh": excess_re.sum(),
                        **{
                            f"excess_{r}_mwh": excess_re_by_type[:, i].sum()
                            for i, r in enumerate(self.resources)
                        },
                        **self.fixed_out_metr,
                        "%_fossil_replaced": replace,
                    }
                },
                orient="index",
            )
            df.index = df.index.set_names([*self.ix_names, "pct_of_re"])
            return df
        except Exception as exc:
            LOGGER.info("%r %r", self, exc)
            return MTDF.copy()

    def _output_incr_freq(self, share: float, freq="MS"):
        re_by_type = np.nan_to_num(self.re_profiles.to_numpy() @ np.diag(self._re_cap * share))
        avoided = np.nan_to_num(
            np.minimum(
                self.req_profile.to_numpy(),
                re_by_type.sum(axis=1),
            )
        )
        avoided_fuel = (self.fuel_profile * avoided / self.req_profile.to_numpy()).fillna(0)
        excess_re = np.nan_to_num(
            np.maximum(re_by_type.sum(axis=1) - self.req_profile.to_numpy(), 0)
        )
        to_df = [
            self.to_pd(avoided, name="avoided_fossil_mwh"),
            avoided_fuel.rename("avoided_fuel_mmbtu"),
            (avoided_fuel * self.carbon_intensity).rename("avoided_carbon"),
            self.to_pd(re_by_type, suffix="_mwh"),
            self.to_pd(
                excess_re.reshape((len(excess_re), 1)) * self.re_shares,
                prefix="excess_",
                suffix="_mwh",
            ),
            self.req_profile.rename("total_fossil_mwh"),
            self.fuel_profile.rename("total_fuel_mmbtu"),
            self.fuel_profile.rename("total_carbon") * self.carbon_intensity,
        ]
        try:
            df = pd.concat(
                [
                    agg_profile(
                        pd.concat(
                            to_df,
                            axis=1,
                        ).rename_axis("report_date"),
                        freq=freq,
                    )
                ],
                keys=[
                    (
                        *self.ix,
                        self.fmt_share(share),
                        *(self._re_cap * share).tolist(),
                        self.capacity,
                    )
                ],
                names=[*self.ix_names, "pct_of_re", *self.resources, "capacity"],
            ).reset_index(level=[*self.resources, "capacity"])
            df["%_fossil_replaced"] = (
                df["avoided_fossil_mwh"] / df["total_fossil_mwh"]
            ).fillna(0.0)
            return df
        except Exception as exc:
            LOGGER.info("%r %r", self, exc)
            return MTDF.copy()

    @staticmethod
    def fmt_share(share):
        return int(share * 100)

    def output_incr(self, step=0.1, freq="T"):
        if freq == "T":
            return [self._output_incr(x) for x in np.arange(step, 1 + step, step)]
        return [self._output_incr_freq(x, freq) for x in np.arange(step, 1 + step, step)]

    def output_incr_df(self, step=0.1, freq="T"):
        return pd.concat(self.output_incr(step, freq))

    @property
    def out_metr(self):
        return {
            "avoided_fossil_mwh": self.avoided.sum(),
            "avoided_fuel_mmbtu": self.avoided_fuel.sum(),
            "avoided_carbon": self.avoided_fuel.sum() * self.carbon_intensity,
            "re_mwh": self.re.sum() * self.carbon_intensity,
            **{f"{r}_mwh": self.re_by_type[:, i].sum() for i, r in enumerate(self.resources)},
            "excess_re_mwh": self.excess_re.sum(),
            **{
                f"excess_{r}_mwh": self.excess_by_type[:, i].sum()
                for i, r in enumerate(self.resources)
            },
            "%_fossil_replaced": self.avoided.sum() / self.req_profile.sum(),
        }

    @property
    def fixed_out_metr(self):
        return {
            "total_fossil_mwh": self.req_profile.sum(),
            "total_fuel_mmbtu": self.fuel_profile.sum(),
            "total_carbon": self.fuel_profile.sum() * self.carbon_intensity,
            # "capacity_mw": self.capacity,
            # "start_date": self.profiles.index[0],
            # "end_date": self.profiles.index[-1],
        }

    def make_fig(self, starting_year=None) -> go.Figure:
        """Create bar figure showing results"""
        df = pd.concat(
            [
                self.re_profiles * self._re_cap,
                self.req_profile - self.avoided,
            ],
            axis=1,
        ).rename(columns={0: "fossil"})
        if starting_year is not None:
            df = df.loc[f"{starting_year}-01-01" :, :]
        df = df.stack()  # noqa: PD013
        return (
            px.bar(
                df.reset_index().replace({"net_generation_mwh": "fossil"}),
                x="level_0",
                y=0,
                color="level_1",
                color_discrete_map=COLORS,
                title=repr(self),
            )
            .update_layout(xaxis_title=None, yaxis_title="MWh", legend_title=None)
            .update_traces(marker_line_width=0)
        )

    def copy(self):
        return deepcopy(self)

    @property
    def for_cap_func(self):
        """cap, req_prof, re_profs = self.for_cap_func
        Returns:

        """
        return self.capacity, self.req_profile.to_numpy(), self.re_profiles.to_numpy()

    def test_offshore(self):
        return self.re_profiles.sum()["offshore_wind"] >= self.re_cap["offshore_wind"]

    def test_onshore(self):
        return self.re_profiles.sum()["onshore_wind"] >= self.re_cap["onshore_wind"]

    @staticmethod
    def grouper(df, prefix="", suffix="", freq=None, group_cols=True, total=False):
        if group_cols:
            df = df.groupby(level=[1, 2], axis=1).sum()
            df.columns = [f"{prefix}{'_'.join(tup)}{suffix}" for tup in df.columns]
        else:
            df.columns = [f"{prefix}{tup}{suffix}" for tup in df.columns]
        if total:
            df[f"{prefix}total{suffix}"] = df.sum(axis=1)
        if freq is None:
            return df
        return df.groupby(pd.Grouper(freq=freq)).sum()

    @staticmethod
    def stacker(df, name=None, freq="YS"):
        df = (
            df.groupby(pd.Grouper(freq=freq))  # noqa: PD013
            .sum()
            .stack([0, 1])
            .reorder_levels([1, 2, 0])
            .sort_index()
        )
        df.name = name
        return df


@njit(error_model="numpy", cache=True)
def equal_energy(
    req_prof: np.ndarray, re_profs: np.ndarray, inc=0.1, pct_to_replace=1.0
) -> np.ndarray:
    """Find the combination of wind and solar that produces the same energy
    while avoiding as much fossil generation as possible.

    :param req_prof: the fossil profile we are trying to match
    :param re_profs: 2 or 3 cols of re CFs
    :param inc: increment to use in search, only available with 2 profs
    :param pct_to_replace: percentage of energy to actually replace
    :return: re_capacity
    """
    assert len(req_prof.shape) == 1, "`req_prof` shape is not a vector, try np.squeeze"
    b = np.append(re_profs.sum(axis=0), 1.0)
    res = re_profs.shape[1]
    # assert pct_to_replace <= 1.0, "cannot replace more than 100% of energy"
    if res == 2:
        a = (
            pct_to_replace
            * np.nansum(req_prof)
            * np.stack(
                (
                    np.arange(0, 1 + inc, inc),
                    1 - np.arange(0, 1 + inc, inc),
                    np.zeros((int((1 + inc) / inc),)),
                ),
                axis=1,
            )
        )
    else:
        a = (
            pct_to_replace
            * np.nansum(req_prof)
            * np.hstack(
                (
                    PERMUTATIONS.copy(),
                    np.zeros((len(PERMUTATIONS), 1)),
                ),
            )
        )
    data = np.divide(
        a,
        b,
    )
    data = np.where(np.isfinite(data), data, 0.0)
    for i, ar in enumerate(data[:, :-1]):
        data[i, res] = np.nansum(
            np.minimum(
                req_prof,
                re_profs @ ar,
            )
        )
    return data[np.argmax(data[:, res]), :res]


@njit(error_model="numpy", cache=True)
def fuel_auc(edges: np.ndarray, curve: np.ndarray):  # noqa: D417
    """Compute the area under a supply curve

    Args:
        edges: generators with start and end cumulative mmbtus,
            ie the bounds for their area under the curve
        curve:

    Returns:

    """  # noqa: D414
    bill = np.zeros(len(edges))
    for i, (dt_id, start, end) in enumerate(edges):
        xy = curve[(curve[:, 0] == dt_id) & (curve[:, 1] >= start) & (curve[:, 1] <= end), 1:]
        bill[i] = np.trapz(y=xy[:, 1], x=xy[:, 0].flatten())
    return bill


@njit(error_model="numpy", cache=True)
def objective(re_cap: np.ndarray, re_prof: np.ndarray, target: np.ndarray):
    return -np.nansum(np.minimum(target, re_prof @ re_cap))


def optimize_equal_energy(re_profs, target, Ab_ub, Ab_eq):  # noqa: N803
    solver_ = solver()
    options = {
        cp.GUROBI: {
            "TimeLimit": 60,
            "Threads": 8,
            "reoptimize": True,
        },
        cp.HIGHS: {"time_limit": 60, "parallel": "on"},
        cp.COPT: {"TimeLimit": 60},
    }[solver_]

    Ab_ub = Ab_ub.fillna(0.0).to_numpy()
    Ab_eq = Ab_eq.fillna(0.0).to_numpy()
    x = cp.Variable(re_profs.shape[1])
    p = cp.Problem(
        cp.Maximize(
            cp.sum(
                cp.minimum(re_profs.fillna(0.0).to_numpy() @ x, np.nan_to_num(target, nan=0.0))
            )
        ),
        [
            Ab_ub[:, :-1] @ x <= Ab_ub[:, -1],
            Ab_eq[:, :-1] @ x >= Ab_eq[:, -1] * 0.99,
            Ab_eq[:, :-1] @ x <= Ab_eq[:, -1] * 1.01,
            x >= 0.0,
        ],
    )
    p.solve(solver_, **options)
    if not p.status == cp.OPTIMAL:  # noqa: SIM201
        return False, p.status
    return True, pd.Series(x.value, index=list(re_profs))


def optimize(
    req_prof: np.ndarray,
    re_profs: np.ndarray,
    max_re_cap: np.ndarray,
    pct_to_replace=float,
):
    """Find the combination of wind and solar that produces no more
    than the desired energy while avoiding as much fossil generation as possible.

    :param req_prof: the fossil profile we are trying to match
    :param re_profs: 2 or 3 cols of re CFs
    :param pct_to_replace: percentage of energy to actually replace
    :param max_re_cap: maximum renewable capacity by type

    :return: re_capacity
    """
    energy = np.nansum(req_prof) * pct_to_replace

    A = np.vstack(
        (
            np.sum(re_profs, axis=0),  # energy
            np.eye(len(max_re_cap)),  # lt max_re_cap
            -np.eye(len(max_re_cap)),  # gt zero
        )
    )
    b = np.hstack(
        (
            np.array(energy),  # energy
            max_re_cap,  # lt max_re_cap
            np.zeros_like(max_re_cap),  # gt zero
        )
    )

    result = minimize(
        fun=objective,
        x0=np.zeros_like(max_re_cap),
        args=(re_profs, req_prof),
        constraints=LinearConstraint(A=A, ub=b),  # type: ignore  # noqa: PGH003
        # SLSQP is faster but too unreliable, at least given the settings I tried
        method="trust-constr",
        # options={"ftol": 5e-2}
    )
    if not result.success:
        con_vals = A @ result.x
        abs(1 - con_vals[0] / energy)
        raise RuntimeError(f"{pct_to_replace=} {result.message} {result.status=}")
    selected_pct_of_max = np.nan_to_num(result.x / max_re_cap, posinf=0.0, neginf=0.0)
    selected_mwh = re_profs.dot(result.x).sum()
    max_mwh = re_profs.dot(max_re_cap).sum()
    if np.any(np.abs(selected_pct_of_max) > 1e-2) and (1 - selected_mwh / max_mwh) > 0.01:
        assert abs(1 - selected_mwh / energy) < 1e-5, (
            "we didn't saturate our RE capacity and didn't hit our energy target"
        )
    return np.round(result.x)


@njit(cache=True)
def np_all_axis1(x: np.ndarray):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


@njit(error_model="numpy")
def equal_capacity(
    cap: float, req_prof: np.ndarray, re_profs: np.ndarray, inc=0.1, sent=False
) -> np.ndarray:
    """Find the combination of wind and solar capacity that equals fossil capacity
    while avoiding as much fossil generation as possible.

    :param cap: capacity of the fossil plant
    :param req_prof: the fossil profile we are trying to match
    :param re_profs: 2 or 3 cols of re CFs
    :param inc: increment to use in search, only available with 2 profs
    :param sent: sentinel to avoid recursion
    :return: re_capacity
    """
    res = re_profs.shape[1]
    if res == 2:
        data = cap * np.stack(
            (
                np.arange(0, 1 + inc, inc),
                1 - np.arange(0, 1 + inc, inc),
                np.zeros((int((1 + inc) / inc),)),
            ),
            axis=1,
        )
    else:
        data = cap * np.hstack(
            (
                PERMUTATIONS.copy(),
                np.zeros((len(PERMUTATIONS), 1)),
            ),
        )
    for i, ar in enumerate(data):
        data[i, res] = np.nansum(
            np.minimum(
                req_prof,
                re_profs @ ar[:res],
            )
        )
    out = data[np.argmax(data[:, res]), :res]
    if np.all(out <= re_profs.sum(axis=0)) or sent:
        return out
    return np.append(equal_capacity(cap, req_prof, re_profs[:, :2], inc, True), 0.0)


def rolling_re_match(load, re_profiles, obj, window_days=4):
    re_rolling_day = (
        re_profiles.groupby(pd.Grouper(freq="D"))
        .sum()
        .rolling(pd.Timedelta(days=window_days))
        .sum()
        .dropna()
    )
    load_rolling_day = (
        load.groupby(pd.Grouper(freq="D"))
        .sum()
        .rolling(pd.Timedelta(days=window_days))
        .sum()
        .dropna()
    )
    assert len(re_rolling_day) == len(load_rolling_day)
    r = linprog(
        A_ub=-1 * re_rolling_day.to_numpy(),
        b_ub=-1 * load_rolling_day.to_numpy(),
        c=obj.to_numpy(),
        bounds=(0, None),
        method="highs",
    )
    return r


def to_nuclear(
    plant_id_eia,
    re_profiles,
    re_plant_specs,
    window_days=8,
    cap_override=None,
    **kwargs,
):
    a = re_plant_specs.query("fos_id == @plant_id_eia")
    ids = list(
        a[["plant_id_prof_site", "re_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    cost = a.groupby(["plant_id_prof_site", "re_type"]).lcoe.mean().loc[ids]
    pro = re_profiles.loc[:, ids]
    cap = a.drop_duplicates(subset=["fos_id", "fos_gen"]).capacity_mw.sum()
    load = pd.Series(cap, index=pro.index)
    r = rolling_re_match(
        load, re_profiles=pro, obj=(pro * cost).sum(), window_days=window_days
    )
    return r.x / cap


@njit(cache=True)
def adjust_profiles(hist_dispatch: np.ndarray, for_load: np.ndarray, capacities: np.ndarray):
    assert hist_dispatch.shape[1] == capacities.shape[0]
    assert hist_dispatch.shape[0] == for_load.shape[0]

    out = np.zeros_like(hist_dispatch)
    total_cap = capacities.sum()

    for i, load in enumerate(for_load):
        total_hist_hour = hist_dispatch[i, :].sum()
        if total_hist_hour + load <= total_cap:
            out[i, :] = hist_dispatch[i, :]
        elif np.isclose(total_cap, load):
            out[i, :] = 0.0
        elif np.all(np.isclose(hist_dispatch[i, :], capacities)):
            # if historical dispatch is all nameplate capacity, effectively no limit on
            # dispatch, we don't want to reduce each generator proportionally to serve
            # colo load we want to concentrate that effect and apply to the first
            # generator first
            disp: float
            for j, disp in enumerate(hist_dispatch[i, :]):
                if np.isclose(load, 0.0):
                    out[i, j] = disp
                elif load >= disp:
                    out[i, j] = 0.0
                    load = load - disp
                else:
                    out[i, j] = disp - load
                    load = 0.0
        else:
            available_for_out = total_cap - load
            out[i, :] = available_for_out * hist_dispatch[i, :] / total_hist_hour
    return out


def calc_redispatch_cost(dm: DispatchModel):
    var = (
        dm.dispatchable_cost[["fuel_per_mwh", "vom_per_mwh"]]
        .sum(axis=1)
        .unstack([0, 1])[dm.redispatch.columns]
        .to_numpy()
    )
    start = dm.dispatchable_cost.startup_cost.unstack([0, 1])[dm.redispatch.columns].to_numpy()
    fom = (
        zero_profiles_outside_operating_dates(
            (
                dm.dispatchable_cost.fom
                / dm.load_profile.groupby(pd.Grouper(freq="YS"))
                .transform("count")
                .groupby(pd.Grouper(freq="MS"))
                .first()
            ).unstack([0, 1])[dm.redispatch.columns],
            dm.dispatchable_specs.operating_date.apply(lambda x: x.replace(day=1, month=1)),
            dm.dispatchable_specs.retirement_date.apply(lambda x: x.replace(day=1, month=1)),
        )
        .sum(axis=1)
        .to_numpy()
    )
    m_index = (
        dm.dispatchable_cost.reset_index()[["datetime"]]
        .drop_duplicates()
        .reset_index()
        .set_index("datetime")
        .reindex(index=dm.redispatch.index, method="ffill")
        .squeeze()
        .to_numpy()
    )

    return pd.Series(
        redispatch_cost_loop(m_index, dm.redispatch.to_numpy(), fom, var, start),
        index=dm.redispatch.index,
        name="system_cost",
    )


@njit(cache=True)
def redispatch_cost_loop(
    m_index: np.ndarray,
    dispatch: np.ndarray,
    fom: np.ndarray,
    var: np.ndarray,
    start: np.ndarray,
):
    out = np.zeros_like(m_index)
    out[0] = fom[0] + var[0, :] @ dispatch[0, :]
    for h, m in enumerate(m_index[1:], 1):
        out[h] = (
            fom[m]
            + var[m, :] @ dispatch[h, :]
            + start[m, (dispatch[h - 1, :] == 0) & (dispatch[h, :] > 0)].sum()
        )
    return out
