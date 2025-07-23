import logging
from dataclasses import dataclass
from functools import lru_cache
from traceback import TracebackException

import numpy as np
import pandas as pd
from etoolbox.datazip import DataZip
from etoolbox.utils.pudl_helpers import simplify_columns
from numba import njit
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm

from patio.constants import (
    CEMS_CARB_INTENSITY,
    CEMS_COL_MAP,
    CROSSWALK_COL_MAP,
    CROSSWALK_DTYPES,
    EGRID_COL_MAP,
    EGRID_DTYPES,
    FOSSIL_PRIME_MOVER_MAP,
    ROOT_PATH,
    STATE_BOUNDS,
    UDAY_FOSSIL_FUEL_MAP,
)
from patio.helpers import agg_profile, fix_cems_datetime, lstrip0

__all__ = [
    "CEMS",
    "all_cems_daily",
    "all_cems_normalized_daily",
    "all_cems_starts",
    "all_state_cems_normalized_daily",
    "clean_all_cems",
    "combine_crosswalk",
    "compare_crosswalks",
    "egrid_crosswalk",
    "epa_eia_crosswalk",
    "make_ba_cems",
    "make_eia_cems",
]
LOGGER = logging.getLogger("patio")


@lru_cache
def emission_factors():
    """Data is in million tonnes / mmbtu so convert to tonnes / mmbtu, blank or null fuel
    should have no emission factor
    """
    return (
        pd.read_excel(
            io=ROOT_PATH / "notebooks/inputs/emissions_factors.xlsx",
            header=0,
            index_col=0,
            engine="openpyxl",
        ).energy_fuel_group_co2
        * 1e6
    ).to_dict() | {np.nan: 0.0, "": 0.0}


@njit
def fuel_split(heat_co2: np.ndarray, co2_factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Args:
        heat_co2:       (0) heat in mmbtu
                        (1) co2 tonnes
        co2_factors:    (0) emission factor for fuel 1
                        (1) emission factor for fuel 2

    Returns: mmbtu fuel 1, mmbtu fuel 2

    """
    fuel_mmbtu_1 = np.where(
        # if no second fuel all fuel is the first fuel, also if the emission
        # factors are the same, this needs to be fixed later since that is
        # wrong but requires 923
        (co2_factors[:, 1] == 0) | (co2_factors[:, 0] == co2_factors[:, 1]),
        heat_co2[:, 0],
        # if there is a second fuel, first need to make sure the result is >0
        np.maximum(
            0,
            # then we need to make sure that our calculation doesn't give a
            # an amount for the first fuel greater than the total
            np.minimum(
                heat_co2[:, 0],
                # now we have the actual fuel split calculation
                (heat_co2[:, 1] - heat_co2[:, 0] * co2_factors[:, 1])
                / (co2_factors[:, 0] - co2_factors[:, 1]),
            ),
        ),
    )
    assert np.all(fuel_mmbtu_1 <= heat_co2[:, 0] * (1 + 1e4)), "fuel 1 exceeds total"
    assert np.all(fuel_mmbtu_1 >= 0.0), "fuel 1 has negatives"

    fuel_mmbtu_2 = heat_co2[:, 0] - fuel_mmbtu_1
    assert np.all(fuel_mmbtu_2 <= heat_co2[:, 0] * (1 + 1e4)), "fuel 2 exceeds total"
    assert np.all(fuel_mmbtu_2 >= 0.0), "fuel 2 has negatives"
    return fuel_mmbtu_1, fuel_mmbtu_2


@dataclass
class CEMS:
    """object to contain, process, clean and transform CEMS data for a state"""

    state: str
    years: tuple[int, int] = (2006, 2020)
    _cems: pd.DataFrame | None = None
    _crosswalk: pd.DataFrame | None = None
    _eia_cems: bool = False

    def __post_init__(self):
        self.state = self.state.upper()
        if self._crosswalk is None:
            self._crosswalk = new_crosswalk().query(f"state == '{self.state}'")
        if self._cems is None:
            self._cems = self._clean_cems()

    @classmethod
    def from_p(cls, state):
        try:
            with DataZip(ROOT_PATH / "state_cems", "r") as z:
                return cls(
                    state=state,
                    _cems=z[f"cems_{state.casefold()}"],
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Put `state_cems.zip` in `patio` dir, you can download it from from "
                "https://rockmtnins.sharepoint.com/:f:/s/UTF/EmfCaHhei6NMtDuXhMhu_NsB_3GF_tM4FW546-2M1JxRZg?e=nUX0y3"
            ) from exc

    @classmethod
    def eia_from_p(cls, state):
        try:
            with DataZip(ROOT_PATH / "eia_cems", "r") as z:
                return cls(
                    state=state,
                    _cems=z[f"cems_{state.casefold()}"],
                    _eia_cems=True,
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Put `eia_cems.zip` in `patio` dir, you can download it from from "
                "https://rockmtnins.sharepoint.com/:f:/s/UTF/EmfCaHhei6NMtDuXhMhu_NsB_3GF_tM4FW546-2M1JxRZg?e=nUX0y3"
            ) from exc

    def to_p(self):
        if not (ROOT_PATH / "cems").exists():
            (ROOT_PATH / "cems").mkdir()
        self._cems.to_parquet(ROOT_PATH / f"cems/cems_{self.state.casefold()}.parquet")

    def cross_test(self, cross=None):
        if cross is None:
            cross = self._crosswalk
        try:
            return (
                self._cems.groupby(["plant_id_cems", "unit_id_cems"])
                .count()
                .reset_index()
                .merge(
                    cross,
                    on=["plant_id_cems", "unit_id_cems"],
                    how="left",
                    validate="1:1",
                )
            )
        except pd.errors.MergeError:
            out = (
                self._cems.groupby(["plant_id_cems", "unit_id_cems"])
                .count()
                .reset_index()
                .merge(
                    cross.groupby(["plant_id_cems", "unit_id_cems"]).agg(list).reset_index(),
                    on=["plant_id_cems", "unit_id_cems"],
                    how="left",
                    validate="1:1",
                )
            )
            for i in range(out.shape[0]):
                for j in range(6, out.shape[1]):
                    item = out.iloc[i, j]
                    if isinstance(item, list) and len(item) == 1:
                        out.iloc[i, j] = item[0]
            return out

    def monthly_unit_fuel_by_type(
        self, plant_id: int, unit_id: str | int, fuels: tuple[str, str]
    ):
        """Use emission factors to split a unit's generation between a
        primary and secondary fuel
        """
        unit_id = str(unit_id)
        i0, i1 = CEMS_CARB_INTENSITY[fuels[0]], CEMS_CARB_INTENSITY[fuels[1]]
        return pd.concat(
            {
                (plant_id, unit_id): self.raw_cems.set_index("datetime")
                .query("plant_id_cems == @plant_id & unit_id_cems == @unit_id")
                .assign(
                    f0=lambda x: (x.co2_tons - x.heat_in_mmbtu * i1) / (i0 - i1),
                    f1=lambda x: x.heat_in_mmbtu - x.f0,
                )
                .rename(columns={"f0": fuels[0], "f1": fuels[1]})
                .groupby([pd.Grouper(freq="MS")])[
                    ["gross_gen", "heat_in_mmbtu", "co2_tons", "op_time", *fuels]
                ]
                .sum()
            },
            names=["plant_id_cems", "unit_id_cems", "report_date"],
        )

    def fuel_by_type(self, freq="MS"):
        """WIP do the primary/secondary fuel split for all plants/units
        works because all energy is allocated to the first fuel if
        secondary intensity is 0
        """
        if not self._eia_cems:
            raise RuntimeError("we only support EIA CEMS")
        em = emission_factors()
        # probably want to try and map over to
        merge_on = ["plant_id_eia", "generator_id", "capacity_year"]
        cems = self._cems.reset_index().assign(capacity_year=lambda x: x.datetime.dt.year)
        cw = (
            self._crosswalk[
                [
                    *merge_on,
                    "eia_fuel_type_1_hist",
                    "eia_fuel_type_2_hist",
                    "eia_unit_type_hist",
                ]
            ]
            .assign(
                co2_factor_1=lambda x: x.eia_fuel_type_1_hist.replace(em),
                co2_factor_2=lambda x: x.eia_fuel_type_2_hist.replace(em),
            )
            .drop_duplicates(subset=merge_on)
        )
        merged = cems.merge(
            cw,
            on=[x for x in merge_on if x in cw and x in cems],
            how="left",
            validate="m:1",
        )
        fuel_mmbtu_1, fuel_mmbtu_2 = fuel_split(
            heat_co2=merged[["heat_in_mmbtu", "co2_tons"]].to_numpy().astype(np.float64),
            co2_factors=merged[["co2_factor_1", "co2_factor_2"]].to_numpy().astype(np.float64),
        )
        return (
            merged.assign(
                fuel_1_mmbtu=fuel_mmbtu_1,
                fuel_2_mmbtu=fuel_mmbtu_2,
            )
            .groupby(["plant_id_eia", "generator_id", pd.Grouper(key="datetime", freq=freq)])
            .agg(
                {
                    "eia_fuel_type_1_hist": "first",
                    "eia_fuel_type_2_hist": "first",
                    "eia_unit_type_hist": "first",
                    "gross_gen": np.sum,
                    "heat_in_mmbtu": np.sum,
                    "co2_tons": np.sum,
                    "fuel_1_mmbtu": np.sum,
                    "fuel_2_mmbtu": np.sum,
                    "co2_factor_1": "first",
                    "co2_factor_2": "first",
                }
            )
            .assign(cems_factor=lambda x: x.co2_tons / x.heat_in_mmbtu)
        )

    def count_unit_starts(self, freq="YS"):
        """Identify each time that a unit starts generating electricity
        and count the number of starts for each increment of time specified
        in ``freq``
        """
        pid, uid = "plant_id_cems", "unit_id_cems"
        if self._eia_cems:
            pid, uid = "plant_id_eia", "generator_id"
        out = (
            self._cems.reset_index()
            .assign(
                id=lambda x: (x[pid] == np.roll(x[pid], -1)) & (x[uid] == np.roll(x[uid], -1)),
                gen_starts=lambda x: (
                    ((x.gross_gen == 0) | x.gross_gen.isna())
                    & (np.roll(x.gross_gen, -1) > 0)
                    & x.id
                ),
                fuel_starts=lambda x: (
                    ((x.heat_in_mmbtu == 0) | x.heat_in_mmbtu.isna())
                    & (np.roll(x.heat_in_mmbtu, -1) > 0)
                    & x.id
                ),
            )
            .groupby([pid, uid, pd.Grouper(key="datetime", freq=freq)])
            .agg(
                {
                    "gen_starts": ["sum"],
                    "fuel_starts": ["sum"],
                    "gross_gen": ["sum", "max"],
                    "heat_in_mmbtu": ["sum", "max"],
                    "co2_tons": ["sum", "max"],
                }
            )
        )
        return out.set_axis(
            [(a if b == "sum" else f"{a}_{b}") for a, b in out.columns], axis=1
        )

    def fuel_by_type_and_starts(self, freq="MS"):
        return self.fuel_by_type(freq=freq).merge(
            self.count_unit_starts(freq=freq),
            left_index=True,
            right_index=True,
            validate="1:1",
        )

    def annual_summary(self):
        return self._cems.groupby(
            ["plant_id_cems", "unit_id_cems", pd.Grouper(key="datetime", freq="YS")]
        ).sum()

    def daily_cems(self):
        """Aggregates cems by prime_mover and fuel then converts to daily
        for utility-level analysis, does not normalize
        """
        print(".", end="", flush=True)
        df = self._cems.fillna(0.0).merge(
            self._crosswalk,
            on=["plant_id_cems", "unit_id_cems"],
            how="inner",
            validate="m:1",
        )
        df = (
            df.groupby(
                [
                    "plant_id_cems",
                    "prime_mover_code",
                    "fuel_type_code_pudl",
                    df.datetime.dt.date,
                ]
            )
            .agg({"gross_gen": np.sum})
            .reset_index()
        )
        df["state"] = self.state
        return df

    def normalized_daily_state_cems(self):
        """Normalize cems monthly then convert aggregate to days"""
        print(".", end="", flush=True)
        df = (
            self._cems.fillna(0)
            .merge(
                self._crosswalk,
                on=["plant_id_cems", "unit_id_cems"],
                how="inner",
                validate="m:1",
            )
            .groupby(
                [
                    "prime_mover_code",
                    "fuel_code",
                    "datetime",
                ]
            )
            .agg({"gross_gen": np.sum})
        )
        print(".", end="", flush=True)
        df = (
            df.reset_index()
            .pivot(
                index="datetime",
                columns=["prime_mover_code", "fuel_code"],
                values="gross_gen",
            )
            .fillna(0)
        )
        print(".", end="", flush=True)
        df = (
            agg_profile(  # noqa: PD013
                df / df.groupby([df.index.year, df.index.month]).transform("sum"),
                freq="D",
            )
            .stack([0, 1])
            .reorder_levels([1, 2, 0])
            .sort_index(axis=0)
            .reset_index()
            .rename(
                columns={
                    0: "gross_gen",
                }
            )
        )
        df.insert(0, "state", self.state)
        return df

    def normalized_daily_cems(self):
        """Normalize cems monthly then convert aggregate to days"""
        print(".", end="", flush=True)
        df = self.agg_to_plant_prime_fuel()
        print(".", end="", flush=True)
        df = (
            df.reset_index()
            .pivot(
                index="datetime",
                columns=["plant_id_cems", "prime_mover_code", "fuel_code"],
                values="gross_gen",
            )
            .fillna(0)
        )
        print(".", end="", flush=True)
        return (
            agg_profile(  # noqa: PD013
                df / df.groupby([df.index.year, df.index.month]).transform("sum"),
                freq="D",
            )
            .stack([0, 1, 2])
            .reorder_levels([1, 2, 3, 0])
            .sort_index(axis=0)
            .reset_index()
            .rename(
                columns={
                    "level_0": "plant_id_cems",
                    "level_1": "prime_mover_code",
                    "level_2": "fuel_code",
                    0: "gross_gen",
                }
            )
        )

    def agg_to_plant_prime_fuel(self, agg_dict=None):
        if agg_dict is None:
            agg_dict = {"gross_gen": np.sum}
        return (
            self._cems.fillna(0)
            .merge(
                self._crosswalk,
                on=["plant_id_cems", "unit_id_cems"],
                how="inner",
                validate="m:1",
            )
            .groupby(
                [
                    "plant_id_cems",
                    "prime_mover_code",
                    "fuel_code",
                    "datetime",
                ]
            )
            .agg(agg_dict)
        )

    def write_daily(self):
        self.daily_cems().to_parquet(ROOT_PATH / f"cems/daily_{self.state.casefold()}.parquet")

    def write_normalized_daily(self):
        self.normalized_daily_cems().to_parquet(
            ROOT_PATH / f"cems/daily_{self.state.casefold()}.parquet"
        )

    def write_normalized_daily_state(self):
        self.normalized_daily_state_cems().to_parquet(
            ROOT_PATH / f"cems/daily_{self.state.casefold()}.parquet"
        )

    def apply_crosswalk(self, validate=False):
        cems_m = self._cems.assign(capacity_year=lambda x: x.datetime.dt.year).merge(
            new_crosswalk()[
                [
                    "plant_id_cems",
                    "unit_id_cems",
                    "plant_id_eia",
                    "generator_id",
                    "capacity_year",
                    "allocation_factor",
                ]
            ],
            on=["plant_id_cems", "unit_id_cems", "capacity_year"],
            how="left",
            indicator=True,
        )
        missing = (
            cems_m.query("_merge == 'left_only'")
            .groupby(["plant_id_cems", "unit_id_cems", "capacity_year"])
            .datetime.count()
        )
        matched = cems_m.query("_merge == 'both'")

        allocation_factor_check = matched.groupby(
            ["plant_id_cems", "unit_id_cems", "datetime"]
        ).allocation_factor.sum()

        assert np.all(allocation_factor_check <= 1.00001), (
            "allocation_factors for plant_id_cems, unit_id_cems, and datetime sum to more than 1 at least once"
        )
        out = (
            matched.assign(
                gross_gen=lambda x: x.gross_gen * x.allocation_factor,
                heat_in_mmbtu=lambda x: x.heat_in_mmbtu * x.allocation_factor,
                co2_tons=lambda x: x.co2_tons * x.allocation_factor,
            )
            .groupby(["plant_id_eia", "generator_id", "datetime"])[
                ["gross_gen", "heat_in_mmbtu", "co2_tons"]
            ]
            .sum()
        )
        if validate:
            # this check is very slow
            match_plant_total_check = (
                self._cems.set_index(["plant_id_cems", "unit_id_cems", "datetime"])
                .loc[
                    matched.set_index(
                        ["plant_id_cems", "unit_id_cems", "datetime"]
                    ).index.unique(),
                    ["gross_gen", "heat_in_mmbtu", "co2_tons"],
                ]
                .sum()
            )
            assert np.isclose(
                match_plant_total_check.gross_gen, out.sum().gross_gen, rtol=1e-3
            ), "check against pre-aggregation gross_gen total failed"
        return out, missing

    def write_eia_matched_cems(self):
        out, missing = self.apply_crosswalk()
        out.to_parquet(ROOT_PATH / f"cems/eia_cems/cems_{self.state.casefold()}.parquet")
        missing.to_frame().to_parquet(
            ROOT_PATH / f"cems/eia_cems/missing_{self.state.casefold()}.parquet"
        )

    def _clean_cems(self):
        try:
            with DataZip(ROOT_PATH / "raw_cems", "r") as z:
                c = z[f"CEMS_{self.state.casefold()}"]
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Put `raw_cems.zip` in `patio` dir, you can download it from from "
                "https://rockmtnins.sharepoint.com/:f:/s/UTF/EmfCaHhei6NMtDuXhMhu_NsB_3GF_tM4FW546-2M1JxRZg?e=nUX0y3"
            ) from exc
        else:
            df = (
                c
                # .pipe(lstrip0, col="UNITID")
                .pipe(fix_cems_datetime)[[*list(CEMS_COL_MAP), "datetime"]]
                .rename(columns=CEMS_COL_MAP)
                .astype({"unit_id_cems": str})
                # .pipe(
                #     combine_cols,
                #     col1="plant_id_cems",
                #     col2="unit_id_cems",
                #     out_col="plant_unit_id",
                # )
            )

            # only keep plants that operated in 2020
            # pid20 = list(df[df.datetime.dt.year == 2020].plant_unit_id.unique())
            # print(".", end="", flush=True)
            # return df.query("plant_unit_id in @pid20").drop(["plant_unit_id"], axis=1)
            return df

    @property
    def raw_cems(self):
        return self._cems.copy()

    @property
    def cems_eia_plant_ids(self):
        if self._eia_cems:
            return self._cems
        return self._cems.replace({"plant_id_cems": cems_to_eia_plant_id_dict()}).rename(
            columns={"plant_id_cems": "plant_id_eia"}
        )

    def count_plant_prime_fuel_starts(self, freq="YS"):
        """Identify each time that a plant/prime/fuel starts generating electricity
        and count the number of starts for each increment of time specified
        in ``freq``
        """
        return (
            self.agg_to_plant_prime_fuel()
            .reset_index()
            .assign(
                starts=lambda x: (
                    ((x.gross_gen == 0) | x.gross_gen.isna())
                    & (np.roll(x.gross_gen, -1) > 0)
                    & (x.plant_id_cems == np.roll(x.plant_id_cems, -1))
                    & (x.prime_mover_code == np.roll(x.prime_mover_code, -1))
                    & (x.fuel_code == np.roll(x.fuel_code, -1))
                )
            )
            .groupby(
                [
                    "plant_id_cems",
                    "prime_mover_code",
                    "fuel_code",
                    pd.Grouper(key="datetime", freq=freq),
                ]
            )
            .agg({"starts": np.sum, "gross_gen": np.max})
            .rename(columns={"gross_gen": "max_gross_gen"})
        )

    def units_with_missing_rows(self):
        return (
            self._cems.fillna({"gross_gen": 0.0})
            .groupby(["plant_id_cems", "unit_id_cems", pd.Grouper(key="datetime", freq="YS")])
            .count()
            .query("gross_gen < 8760")
        )


def make_eia_cems(mp=True):
    """Combine CEMS data from all relevant states into parquets
    of hourly CEMS data for each BA

    Args:
        mp: bool use multiprocessing

    Returns: None, produces BA CEMS parquets

    """
    eia_cems_dir = ROOT_PATH / "cems/eia_cems"
    if not eia_cems_dir.exists():
        eia_cems_dir.mkdir(parents=True)
    with logging_redirect_tqdm():
        if mp:
            process_map(_eia_cems_mp_helper, STATE_BOUNDS["xmin"], max_workers=6)
        else:
            for ba_tuple in tqdm(STATE_BOUNDS["xmin"]):
                _eia_cems_mp_helper(ba_tuple)


def _eia_cems_mp_helper(state):
    if (ROOT_PATH / f"cems/eia_cems/cems_{state.casefold()}.parquet").exists():
        return None
    try:
        CEMS(state).write_eia_matched_cems()
    except FileNotFoundError:
        pass
    except Exception as exc:
        LOGGER.error("%s %r", state, exc)


def make_ba_cems(ba_dict: dict[str | int, pd.DataFrame], mp=True):
    """Combine CEMS data from all relevant states into parquets
    of hourly CEMS data for each BA

    Args:
        ba_dict: dict of ba plant_data
        mp: bool use multiprocessing

    Returns: None, produces BA CEMS parquets

    """
    ba_cems_dir = ROOT_PATH / "cems/ba_cems"
    if not ba_cems_dir.exists():
        ba_cems_dir.mkdir(parents=True)
    with logging_redirect_tqdm():
        if mp:
            process_map(_ba_cems_mp_helper, ba_dict.items(), max_workers=4)
        else:
            for ba_tuple in tqdm(ba_dict.items()):
                _ba_cems_mp_helper(ba_tuple)


def _ba_cems_mp_helper(ba_tuple):
    ba, v = ba_tuple
    if (ROOT_PATH / f"cems/ba_cems/{ba}_cems.parquet").exists():
        return None
    state = None
    try:
        state_cem_blocks = []
        # get the list of plant_ids in the ba
        sorted(set(v.index.get_level_values(level=0)))
        # for each state, open the state cems parquet, pull out the relevant plants
        for state in v.state.unique():
            cems = CEMS.eia_from_p(state)
            state_cem_blocks.append(
                cems.cems_eia_plant_ids.query("plant_id_eia in @plant_list")
            )
    except Exception as exc:
        LOGGER.error("%s %s %r", ba, state, exc)
    else:
        pd.concat(state_cem_blocks, axis=0).astype(np.float32).to_parquet(
            ROOT_PATH / f"cems/ba_cems/{ba}_cems.parquet"
        )


def clean_all_cems(mp=False):
    if not ROOT_PATH.joinpath("cems").exists():
        ROOT_PATH.joinpath("cems").mkdir()
    if not ROOT_PATH.joinpath("raw_cems").exists():
        raise RuntimeError("process requires raw_cems data, contact Alex or Uday")
    if mp:
        process_map(_clean_cems_mp_helper, STATE_BOUNDS["xmin"])
    else:
        for state in tqdm(STATE_BOUNDS["xmin"]):
            _clean_cems_mp_helper(state)


def _clean_cems_mp_helper(state):
    if (
        not (ROOT_PATH / f"cems/cems_{state.casefold()}.parquet").exists()
        and (ROOT_PATH / f"raw_cems/CEMS_{state.casefold()}.parquet").exists()
    ):
        try:
            CEMS(state).to_p()
        except Exception as exc:
            LOGGER.error("%s %r", state, exc)
        else:
            LOGGER.info("%s", state)


def all_cems_starts(by="prime_fuel", freq="YS", raw=False):
    starts = []
    errors = []
    assert "prime" in by or "fuel" in by or "unit" in by, "`by` must be 'prime_fuel' or 'unit'"
    with logging_redirect_tqdm():
        for state in tqdm(STATE_BOUNDS["xmin"]):
            try:
                cems = CEMS(state) if raw else CEMS.from_p(state)
                if "prime" in by or "fuel" in by:
                    starts.append(cems.count_plant_prime_fuel_starts(freq=freq))
                elif "unit" in by:
                    starts.append(cems.count_unit_starts(freq=freq))
            except FileNotFoundError:
                pass
            except Exception as exc:
                LOGGER.error("%s %r", state, exc)
                errors.append((state, TracebackException.from_exception(exc)))
    return (
        pd.concat(starts, axis=0)
        .sort_index()
        .reset_index()
        .rename(columns={"datetime": "report_date"})
        .assign(
            year=lambda x: x.report_date.dt.year,
            month=lambda x: x.report_date.dt.month,
            gross_gen_max=lambda x: x.gross_gen_max.fillna(
                x.groupby(
                    [
                        "plant_id_cems",
                        "unit_id_cems",
                        pd.Grouper(freq="YS", key="report_date"),
                    ]
                ).gross_gen_max.transform("max")
            )
            .fillna(
                x.groupby(["plant_id_cems", "unit_id_cems"]).gross_gen_max.transform("max")
            )
            .fillna(0.0),
        ),
        errors,
    )


def all_cems_normalized_daily():
    """Create daily normlized cems data for patio model"""
    if not ROOT_PATH.joinpath("cems").exists():
        raise RuntimeError("run `clean_all_cems()` first")
    for state in tqdm(STATE_BOUNDS["xmin"]):
        if (
            not (ROOT_PATH / f"cems/daily_{state.casefold()}.parquet").exists()
            and (ROOT_PATH / f"cems/cems_{state.casefold()}.parquet").exists()
        ):
            try:
                CEMS.from_p(state).write_normalized_daily()
            except Exception as exc:
                LOGGER.error("%s %r", state, exc)
            else:
                LOGGER.info("%s", state)
    dfs = []
    for f in tqdm(ROOT_PATH.glob("cems/daily_*.parquet")):
        dfs.append(pd.read_parquet(f).assign(state=f.stem.partition("_")[2].upper()))
    pd.concat(dfs, axis=0).rename(columns={"plant_id_cems": "plant_id_eia"}).set_index(
        ["plant_id_eia", "prime_mover_code", "fuel_code", "datetime"]
    ).sort_index().to_parquet(ROOT_PATH / "patio_data/normalized_daily_cems.parquet")
    if (ROOT_PATH / "patio_data/normalized_daily_cems.parquet").exists():
        for f in ROOT_PATH.glob("cems/daily_*.parquet"):
            f.unlink()


def all_state_cems_normalized_daily():
    """Create daily normlized cems data by state for patio model"""
    if not ROOT_PATH.joinpath("cems").exists():
        raise RuntimeError("run `clean_all_cems()` first")
    for state in STATE_BOUNDS["xmin"]:
        if (
            not (ROOT_PATH / f"cems/daily_{state.casefold()}.parquet").exists()
            and (ROOT_PATH / f"cems/cems_{state.casefold()}.parquet").exists()
        ):
            print(state, end="", flush=True)
            try:
                CEMS.from_p(state).write_normalized_daily_state()
            except Exception as exc:
                print(f" {exc!r}.", end=" ", flush=True)
            else:
                print(" done.", end=" ", flush=True)
    dfs = []
    for f in ROOT_PATH.glob("cems/daily_*.parquet"):
        dfs.append(pd.read_parquet(f).assign(state=f.stem.partition("_")[2].upper()))
        print(".", end="", flush=True)
    pd.concat(dfs, axis=0).set_index(
        ["state", "prime_mover_code", "fuel_code", "datetime"]
    ).sort_index().to_parquet(ROOT_PATH / "patio_data/daily_state_cems.parquet")
    if (ROOT_PATH / "patio_data/daily_state_cems.parquet").exists():
        for f in ROOT_PATH.glob("cems/daily_*.parquet"):
            f.unlink()


def all_cems_daily():
    """Create non-normalized daily CEMS for utility-level analysis"""
    if not ROOT_PATH.joinpath("cems").exists():
        raise RuntimeError("run `clean_all_cems()` first")
    for state in STATE_BOUNDS["xmin"]:
        if (
            not (ROOT_PATH / f"cems/daily_{state.casefold()}.parquet").exists()
            and (ROOT_PATH / f"cems/cems_{state.casefold()}.parquet").exists()
        ):
            print(state, end="", flush=True)
            try:
                CEMS.from_p(state).write_daily()
            except Exception as exc:
                print(f" {exc!r}.", end=" ", flush=True)
            else:
                print(" done.", end=" ", flush=True)
    dfs = []
    for f in ROOT_PATH.glob("cems/daily_*.parquet"):
        print(".", end="", flush=True)
        dfs.append(pd.read_parquet(f))
    df = (
        pd.concat(dfs, axis=0)
        .rename(columns={"plant_id_cems": "plant_id_eia"})
        .set_index(["plant_id_eia", "prime_mover_code", "fuel_type_code_pudl", "datetime"])
        .sort_index()
        .reset_index()
    )
    df["id_pm_fuel"] = df.plant_id_eia.astype(str).str.cat(
        [df.prime_mover_code, df.fuel_type_code_pudl], sep="_"
    )
    df.set_index(["id_pm_fuel", "datetime"]).to_parquet(
        ROOT_PATH / "patio_data/daily_cems.parquet"
    )
    if (ROOT_PATH / "patio_data/daily_cems.parquet").exists():
        for f in ROOT_PATH.glob("cems/daily_*.parquet"):
            f.unlink()


# ================= CROSSWALKS =================


def epa_eia_crosswalk():
    try:
        return (
            pd.read_parquet(ROOT_PATH / "patio_data/epa_eia_crosswalk.parquet")
            .rename(columns=CROSSWALK_COL_MAP)
            .pipe(lstrip0, col="generator_id")
            .replace({"prime_mover_code": FOSSIL_PRIME_MOVER_MAP})
            .query("EIA_RETIRE_YEAR == 0 & plant_id_eia.notnull()", engine="python")[
                list(CROSSWALK_COL_MAP.values())
            ]
        )
    except FileNotFoundError:
        pd.read_csv(
            "https://raw.githubusercontent.com/USEPA/camd-eia-crosswalk/master/epa_eia_crosswalk.csv",
            index_col=0,
            header=0,
            dtype=CROSSWALK_DTYPES,
        ).to_parquet(ROOT_PATH / "patio_data/epa_eia_crosswalk.parquet")
        return epa_eia_crosswalk()


def egrid_crosswalk() -> pd.DataFrame:
    try:
        return (
            pd.read_parquet(ROOT_PATH / "patio_data/egrid.parquet")
            .rename(columns=EGRID_COL_MAP)
            .pipe(lstrip0, col="unit_id_cems")
            .query(
                "prime_mover_code in @FOSSIL_PRIME_MOVER_MAP & op_status == 'OP' "
                "& energy_source_code_1 in @UDAY_FOSSIL_FUEL_MAP"
            )
            .replace({"prime_mover_code": FOSSIL_PRIME_MOVER_MAP})
        )
    except FileNotFoundError:
        pd.read_excel(
            "https://www.epa.gov/system/files/documents/2022-01/egrid2020_data.xlsx",
            sheet_name="UNT20",
            header=1,
            dtype=EGRID_DTYPES,
            usecols=[
                "PSTATABB",
                "PNAME",
                "ORISPL",
                "UNITID",
                "PRMVR",
                "UNTOPST",
                "FUELU1",
                "HTIANSRC",
            ],
        ).to_parquet(ROOT_PATH / "patio_data/egrid.parquet")
        return egrid_crosswalk()


@lru_cache
def cems_to_eia_plant_id_dict():
    return (
        epa_eia_crosswalk()
        .query("plant_id_change == 1")[["plant_id_eia", "plant_id_cems"]]
        .drop_duplicates()
        .set_index(["plant_id_cems"])
        .to_dict()["plant_id_eia"]
    )


def compare_crosswalks() -> pd.DataFrame:
    cw = (
        epa_eia_crosswalk()
        .groupby(["state", "plant_id_cems", "unit_id_cems"])
        .agg(
            {
                "prime_mover_code": ["first", list],
                "energy_source_code_1": ["first", list],
                "plant_id_eia": ["first", list],
            }
        )
    )
    cw.columns = [f"{two}_{one}" for one, two in cw.columns]
    return egrid_crosswalk()[
        [
            "plant_id_cems",
            "unit_id_cems",
            "state",
            "prime_mover_code",
            "energy_source_code_1",
        ]
    ].merge(
        cw.reset_index(),
        on=["state", "plant_id_cems", "unit_id_cems"],
        how="outer",
        validate="1:1",
        suffixes=("_eg", "_cw"),
    )


@lru_cache
def combine_crosswalk() -> pd.DataFrame:
    """Combine egrid and epa crosswalks"""
    try:
        return pd.read_parquet(ROOT_PATH / "patio_data/crosswalk.parquet")
    except FileNotFoundError:
        comp = compare_crosswalks()
        comb = comp.fillna(
            {
                "prime_mover_code": comp.first_prime_mover_code,
                "energy_source_code_1": comp.first_energy_source_code_1,
            }
        ).drop(
            [
                "first_prime_mover_code",
                "list_prime_mover_code",
                "first_energy_source_code_1",
                "list_energy_source_code_1",
            ],
            axis=1,
        )
        comb.query("energy_source_code_1 in @UDAY_FOSSIL_FUEL_MAP").assign(
            fuel_code=comb.energy_source_code_1.replace(UDAY_FOSSIL_FUEL_MAP),
            plant_id_eia=lambda x: x.first_plant_id_eia.fillna(x.plant_id_cems),
        ).drop(columns=["first_plant_id_eia", "list_plant_id_eia"]).to_parquet(
            ROOT_PATH / "patio_data/crosswalk.parquet"
        )
        return combine_crosswalk()


@lru_cache
def new_crosswalk():
    cw = (
        pd.read_parquet(ROOT_PATH / "patio_data/camd_eia_crosswalk_by_year.parquet")
        .rename(
            columns=CROSSWALK_COL_MAP
            | {"camd_unit_to_eia_gen_allocation_factor": "allocation_factor"}
        )
        .pipe(simplify_columns)
        .astype({"plant_id_cems": int, "plant_id_eia": "Int64"})
        .assign(
            # fuel codes are missing from the year a generator retires so this
            # carries the previous fuel code forward to fill in those blanks
            eia_unit_type_hist=lambda x: x.eia_unit_type_hist.fillna(
                pd.Series(np.roll(x.eia_unit_type_hist, 1), index=x.index)
            ),
            eia_fuel_type_1_hist=lambda x: x.eia_fuel_type_1_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_1_hist, 1), index=x.index)
            ),
            eia_fuel_type_2_hist=lambda x: x.eia_fuel_type_2_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_2_hist, 1), index=x.index)
            ),
            eia_fuel_type_3_hist=lambda x: x.eia_fuel_type_3_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_3_hist, 1), index=x.index)
            ),
            eia_fuel_type_4_hist=lambda x: x.eia_fuel_type_4_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_4_hist, 1), index=x.index)
            ),
            eia_fuel_type_5_hist=lambda x: x.eia_fuel_type_5_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_5_hist, 1), index=x.index)
            ),
            eia_fuel_type_6_hist=lambda x: x.eia_fuel_type_6_hist.fillna(
                pd.Series(np.roll(x.eia_fuel_type_6_hist, 1), index=x.index)
            ),
        )[
            [
                "plant_id_cems",
                "unit_id_cems",
                "capacity_year",
                "plant_id_eia",
                "generator_id",
                "allocation_factor",
                "plant_name_cems",
                "generator_id_cems",
                "capacity_mw_cems",
                "fuel_code_cems",
                "camd_status",
                "state",
                "plant_name_eia",
                "capacity_mw",
                "boiler_id",
                "prime_mover_code",
                "energy_source_code_1",
                "camd_status_year",
                "capacity_in_year",
                "sum_capacity_in_year",
                "normalized_capacity_in_year",
                "sum_normalized_capacity_in_year",
                "eia_unit_type_hist",
                "eia_nameplate_capacity_hist",
                "eia_fuel_type_1_hist",
                "eia_fuel_type_2_hist",
                "eia_fuel_type_3_hist",
                "eia_fuel_type_4_hist",
                "eia_fuel_type_5_hist",
                "eia_fuel_type_6_hist",
                "eia_retire_year",
            ]
        ]
    )
    # want to make sure that allocation_factors for duplicated CEMS keys sum
    # to one but need to acknowledge floating point errors hence 1.00001
    assert np.all(
        cw[cw[["plant_id_cems", "unit_id_cems", "capacity_year"]].duplicated(keep=False)]
        .groupby(["plant_id_cems", "unit_id_cems", "capacity_year"])
        .allocation_factor.sum()
        <= 1.00001,
    ), "allocation_factors for duplicated CEMS keys do not sum to 1"
    return cw
