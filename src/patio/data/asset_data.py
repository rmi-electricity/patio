"""a model I first heard of while on a patio

This module contains the objects and specific methods to set up and manage data.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from etoolbox.datazip import DataZip
from etoolbox.utils.cloud import get, rmi_cloud_fs
from etoolbox.utils.misc import download
from etoolbox.utils.pudl import generator_ownership, pd_read_pudl, pl_scan_pudl
from etoolbox.utils.pudl_helpers import (
    month_year_to_date,
    simplify_columns,
)

# from gencost.atb import get_atb as get_atb
# from gencost.constants import (
#     CARB_INTENSITY,
#     COST_FLOOR,
#     FUEL_GROUP_MAP,
#     UDAY_FOSSIL_FUEL_MAP,
# )
# from gencost.waterfall import fix_cc_in_prime
from platformdirs import user_cache_path, user_data_path
from tqdm import tqdm

from patio.constants import (
    CARB_INTENSITY,
    COST_FLOOR,
    ES_TECHS,
    FOSSIL_PRIME_MOVER_MAP,
    FUEL_GROUP_MAP,
    MIN_HEAT_RATE,
    OTHER_TD_MAP,
    PATIO_DATA_AZURE_URLS,
    PATIO_DATA_RELEASE,
    PATIO_PUDL_RELEASE,
    RE_TECH,
    ROOT_PATH,
    UDAY_FOSSIL_FUEL_MAP,
)
from patio.data.entity_ids import add_ba_code
from patio.helpers import (
    check_lat_lons,
    pl_df_product,
    pl_distance,
    round_coordinates,
)

LOGGER = logging.getLogger("patio")
__all__ = ["AssetData"]
# setup_gencost_data()

_ = OTHER_TD_MAP


def fix_cc_in_prime(df, old_col="prime_mover_code"):
    """Add prime_mover call with CCs rolled together."""
    return df.assign(prime_mover=lambda x: x[old_col].replace(FOSSIL_PRIME_MOVER_MAP))


CACHE_PATH = user_cache_path("patio", ensure_exists=True)
USER_DATA_PATH = user_data_path("patio", "rmi", PATIO_DATA_RELEASE, ensure_exists=True)


_ = ES_TECHS  # so it doesn't get removed by ruff
r_map = {
    "Biopower_Dedicated": "Wood/Wood Waste Biomass",
    "Coal_FE_newAvgCF2ndGen": "Conventional Steam Coal",
    "Geothermal_HydroFlash": "Geothermal",
    "Onshore Wind Turbine_Class1": "Onshore Wind Turbine",
    "NaturalGas_FE_CCAvgCF": "Natural Gas Fired Combined Cycle",
    "NaturalGas_FE_CTAvgCF": "Natural Gas Fired Combustion Turbine",
    "Nuclear_Nuclear": "Nuclear",
    "Offshore Wind Turbine_Class1": "Offshore Wind Turbine",
    "Solar Photovoltaic_Class1": "Solar Photovoltaic",
}


def clean_atb(
    case="Market",
    scenario="Moderate",
    report_year=2022,
    crp=30,
    pudl_release=PATIO_PUDL_RELEASE,
):
    return (
        pl_scan_pudl("core_nrelatb__yearly_projected_cost_performance", release=pudl_release)
        .filter(
            (pl.col("model_case_nrelatb") == case)
            & (pl.col("scenario_atb") == scenario)
            & (pl.col("cost_recovery_period_years") == crp)
            & (pl.col("report_year") == report_year)
            & pl.col("technology_description").is_in(
                (
                    "NaturalGas_FE",
                    # "Hydropower",
                    "Utility-Scale Battery Storage",
                    # "NaturalGas_Retrofits",
                    "Biopower",
                    "Coal_FE",
                    # "Coal_Retrofits",
                    "UtilityPV",
                    "Utility-Scale PV-Plus-Battery",
                    "Nuclear",
                    "LandbasedWind",
                    "Geothermal",
                    "OffShoreWind",
                )
            )
            # remove exotic geothermal
            & pl.col("technology_description_detail_1")
            .is_in(
                ("DeepEGSBinary", "DeepEGSFlash", "HydroBinary", "NFEGSBinary", "NFEGSFlash")
            )
            .not_()
            & (
                (pl.col("technology_description") == "Nuclear")
                & pl.col("technology_description_detail_1").is_in(
                    ("NuclearSMR", "Nuclear - Small")
                )
            ).not_()
        )
        .rename(
            {
                "opex_fixed_per_kw": "fom_per_kw",
                "fuel_cost_per_mwh": "fuel_per_mwh",
                "opex_variable_per_mwh": "vom_per_mwh",
                "capex_per_kw": "capex_per_kw",
            }
        )
        .with_columns(
            class_atb=pl.when(
                pl.col("technology_description").is_in(
                    (
                        "UtilityPV",
                        "Utility-Scale PV-Plus-Battery",
                        "LandbasedWind",
                        "OffShoreWind",
                    )
                )
            )
            .then(pl.col("technology_description_detail_1").str.split("Class").list.last())
            .otherwise(pl.lit(-1))
            .cast(pl.Int64),
            duration_hrs=pl.when(
                pl.col("technology_description") == "Utility-Scale Battery Storage"
            )
            .then(pl.col("technology_description_detail_1").str.split("Hr B").list.first())
            .when(pl.col("technology_description") == "Utility-Scale PV-Plus-Battery")
            .then(pl.lit(4))
            .otherwise(None)
            .cast(pl.Int64),
            technology_description_detail_1=pl.when(
                pl.col("technology_description").is_in(("NaturalGas_FE", "Coal_FE"))
            )
            .then(
                pl.col("technology_description_detail_1")
                .fill_null(pl.col("technology_description_detail_2"))
                .cast(pl.Utf8)
                .replace_strict(
                    {
                        "CCAvgCF": "Combined Cycle",
                        "F-Frame CC": "Combined Cycle",
                        "NG 2-on-1 Combined Cycle (F-Frame)": "Combined Cycle",
                        "CTAvgCF": "Combustion Turbine",
                        "F-Frame CT": "Combustion Turbine",
                        "NG Combustion Turbine (F-Frame)": "Combustion Turbine",
                        "newAvgCF2ndGen": "Coal",
                        "Coal-new": "Coal",
                    },
                    default=None,
                )
            )
            .otherwise(pl.col("technology_description_detail_1")),
            technology_description=pl.col("technology_description")
            .cast(pl.Utf8)
            .replace(
                {
                    "LandbasedWind": "Onshore Wind Turbine",
                    "OffShoreWind": "Offshore Wind Turbine",
                    "UtilityPV": "Solar Photovoltaic",
                    "Utility-Scale Battery Storage": "Batteries",
                    "NaturalGas_FE": "Natural Gas Fired",
                    "Biopower": "Wood/Wood Waste Biomass",
                    "Coal_FE": "Conventional Steam",
                }
            ),
        )
        .filter(
            (
                pl.col("technology_description").is_in(
                    ("Natural Gas Fired", "Conventional Steam")
                )
                & pl.col("technology_description_detail_1").is_null()
            ).not_()
        )
        .with_columns(
            technology_description=pl.when(
                pl.col("technology_description").is_in(
                    ("Natural Gas Fired", "Conventional Steam")
                )
                & pl.col("technology_description_detail_1").is_not_null()
            )
            .then(
                pl.concat_str(
                    pl.col("technology_description"),
                    pl.col("technology_description_detail_1"),
                    separator=" ",
                )
            )
            .otherwise(pl.col("technology_description")),
        )
        .collect()
    )


def get_atb(
    df: pd.DataFrame,
    years: tuple = (2008, 2021),
    how="inner",
    operating_date_col="operating_date",
    pudl_release: str = PATIO_PUDL_RELEASE,
):
    atb = (
        clean_atb(
            case="Market", scenario="Moderate", report_year=2022, pudl_release=pudl_release
        )
        .select(
            pl.col("projection_year").alias("year"),
            "technology_description",
            "class_atb",
            "fom_per_kw",
            pl.col("fuel_per_mwh")
            .fill_null(
                pl.col("technology_description").replace_strict(
                    COST_FLOOR.set_index("technology_description").fuel_floor.to_dict(),
                    default=None,
                )
            )
            .alias("fuel_per_mwh"),
            pl.col("vom_per_mwh").fill_null(0.0),
            "capex_per_kw",
            pl.lit(0).alias("start_per_kw"),
            pl.col("technology_description")
            .replace_strict(
                {
                    "Wood/Wood Waste Biomass": 13.5,  # ATB 2022 v2 Biopower Dedicated
                    "Conventional Steam Coal": 8.48894,  # ATB 2022 v2 Coal-new
                    "Natural Gas Fired Combined Cycle": 6.363,  # ATB 2022 v2 NG F-Frame CC
                    "Natural Gas Fired Combustion Turbine": 9.717,  # ATB 2022 v2 NG F-Frame CT
                    "Nuclear": 10.461,  # ATB 2022 v2
                    # convention heat rates, not really meaningfull
                    "Solar Photovoltaic": 10.0,
                    "Onshore Wind Turbine": 10.0,
                    "Offshore Wind Turbine": 10.0,
                    "Geothermal": 10.0,
                },
                default=None,
            )
            .alias("heat_rate"),
            pl.col("technology_description")
            .replace_strict(CARB_INTENSITY, default=None)
            .alias("co2_factor"),
        )
        .filter(
            pl.col("technology_description").is_not_null()
            & pl.col("technology_description")
            .is_in(("Batteries", "Utility-Scale PV-Plus-Battery"))
            .not_()
        )
        .to_pandas()
    )
    if missing := {"technology_description", operating_date_col} - set(df):
        raise ValueError(f"`df` is missing required columns or index levels: {missing}")
    if "Offshore Wind Turbine" in df.technology_description:
        LOGGER.warning(
            "data for `technology_description` == Offshore Wind Turbine is not "
            "reliable because it is not wind class-specific"
        )
    atb = (
        df.assign(
            year_=lambda x: x[operating_date_col].dt.year,
            year=lambda x: x.year_.mask(x.year_ < 2020, 2020),
        )
        .merge(
            atb,
            on=["technology_description", "year", "class_atb"],
            how=how,
            validate="m:1",
        )
        .drop(columns=["year", "year_"])
    )
    dt_range = pd.to_datetime(range(*years), format="%Y")
    return pd.concat(atb.assign(datetime=dt) for dt in dt_range)


def generator_ownership(year: int | None = None, release: str = "nightly") -> pl.DataFrame:  # noqa: F811
    """Generator ownership.

    Args:
        year: year of report date to use
        release: ``nightly``, ``stable`` or versioned, use :func:`.pudl_list` to
            see releases.

    Examples:
    --------
    >>> from etoolbox.utils.pudl import generator_ownership
    >>>
    >>> generator_ownership(year=2023, release="v2024.10.0").sort("plant_id_eia").select(
    ...     "plant_id_eia", "generator_id", "owner_utility_id_eia"
    ... ).head()
    shape: (5, 3)
    ┌──────────────┬──────────────┬──────────────────────┐
    │ plant_id_eia ┆ generator_id ┆ owner_utility_id_eia │
    │ ---          ┆ ---          ┆ ---                  │
    │ i64          ┆ str          ┆ i64                  │
    ╞══════════════╪══════════════╪══════════════════════╡
    │ 1            ┆ 1            ┆ 63560                │
    │ 1            ┆ 2            ┆ 63560                │
    │ 1            ┆ 3            ┆ 63560                │
    │ 1            ┆ 5.1          ┆ 63560                │
    │ 1            ┆ WT1          ┆ 63560                │
    └──────────────┴──────────────┴──────────────────────┘

    """
    year = (
        (
            pl_scan_pudl("core_eia860__scd_ownership", release=release)
            .filter(pl.col("data_maturity") == "final")
            .select("report_date")
            .unique()
            .max()
            .collect()
            .to_series()
            .item()
            .year
        )
        if year is None
        else year
    )
    return (
        pl_scan_pudl("_out_eia__yearly_generators", release=release)
        .filter(
            (pl.col("data_maturity") == "final")
            & (pl.col("report_date").dt.year() == year)
            & (pl.col("operational_status") == "existing")
        )
        .select(
            "plant_id_eia",
            "generator_id",
            "plant_name_eia",
            "utility_id_eia",
            "utility_name_eia",
            "capacity_mw",
        )
        .join(
            pl_scan_pudl("core_eia860__scd_ownership", release=release)
            .filter(
                (pl.col("data_maturity") == "final")
                & (pl.col("report_date").dt.year() == year)
            )
            .select(
                "plant_id_eia",
                "generator_id",
                "owner_utility_id_eia",
                "owner_utility_name_eia",
                "fraction_owned",
            ),
            on=["plant_id_eia", "generator_id"],
            how="left",
            validate="1:m",
        )
        .select(
            pl.col("plant_id_eia").cast(pl.Int64),
            "generator_id",
            "plant_name_eia",
            "capacity_mw",
            pl.col("owner_utility_id_eia").fill_null(pl.col("utility_id_eia")).cast(pl.Int64),
            pl.col("owner_utility_name_eia").fill_null(pl.col("utility_name_eia")),
            pl.col("fraction_owned").fill_null(1.0),
        )
        .collect()
    )


@dataclass
class AssetData:
    years: tuple[int, int] = (2008, 2022)
    cutoff: float = 0.99
    state_cutoff: float = 0.95
    max_re_sites: int = 2
    addl_fuels: bool = (
        False  # augments the 860 gen list to include duplicate plant/primes with 2nd/3rd fuels
    )
    overrides: dict[str, dict[int, tuple]] | None = (
        None  # {new_ba_code: {plant_id_eia: (generator_id, ...)}}
    )
    pudl_release: str = PATIO_PUDL_RELEASE
    _dfs: dict[str, pd.DataFrame | dict] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        LOGGER.info("info log test")
        self.fos_fuels = (
            "coal",
            "natural_gas",
            "petroleum",
            "petroleum_coke",
            "other_gas",
        )
        techs = (
            "Solar Photovoltaic",
            "Batteries",
            "Geothermal",
            "Onshore Wind Turbine",
            # "All Other",
            "Nuclear",
            # "Other Waste Biomass",
            # "Other Natural Gas",
            # "Conventional Hydroelectric",
            "Hydroelectric Pumped Storage",
            # "Natural Gas Steam Turbine",
            # "Wood/Wood Waste Biomass",
            "Offshore Wind Turbine",
        )
        j1 = datetime(self.years[0], 1, 1, 0)

        offsets = (
            pd_read_pudl("core_eia__entity_plants", release=self.pudl_release)[
                ["plant_id_eia", "timezone"]
            ]
            .drop_duplicates()
            .dropna()
        )
        offsets["utc_offset"] = offsets["timezone"].apply(
            lambda tz: j1.replace(tzinfo=ZoneInfo(tz)).utcoffset()
        )
        offsets = offsets[offsets.utc_offset.dt.total_seconds() < 0]
        offsets = offsets.set_index("plant_id_eia").utc_offset.to_dict()
        rules = pd.read_csv(ROOT_PATH / "patio/package_data/ba_rules.csv").drop_duplicates()
        LOGGER.warning("WE DROP ALL OF AK AND HI.")

        def retirement_fixes(df):
            df_ = df.copy()  # noqa: F841
            new = pd.read_csv(ROOT_PATH / "patio/package_data/retirement_fixes.csv").rename(
                columns={"planned_retirement_date": "planned_generator_retirement_date"}
            )
            in_cols = df.columns
            cols = (
                "generator_retirement_date",
                # "retirement_year",
                # "retirement_month",
                # "planned_retirement_date",
                "planned_generator_retirement_date",
                # "planned_retirement_year",
                # "planned_retirement_month",
                "operational_status",
                "operational_status_code",
            )
            df = df.merge(
                new,
                on=["plant_id_eia", "generator_id"],
                how="left",
                suffixes=("", "_new"),
                validate="1:1",
            )
            for col in cols:
                if "date" in col:
                    df[col] = pd.to_datetime(df[col + "_new"].fillna(df[col]))
                else:
                    df[col] = df[col + "_new"].fillna(df[col])
            return df[[*in_cols, "sc_cat"]]

        def fix_op_status(df):
            # ERCO Fails because no 923 data for existing plant built in 2023, odd
            # that it only break ERCO but that's we we only change status of
            # single generator
            # df.loc[(df.plant_id_eia == 63233) & (
            #         df.generator_id == 'UNIT1'), 'operating_date'] = pd.Timestamp('2023-07-01 00:00:00')
            df.loc[
                (df.plant_id_eia == 63233) & (df.generator_id == "UNIT1"),
                "operational_status",
            ] = "proposed"
            return df

        def fix_ba_codes(df):
            df = df.query("plant_id_eia not in (58396, 59000, 60203, 60977)")  # canceled
            test2 = df.query(
                "balancing_authority_code_eia in (None,) & operational_status != 'retired' & state not in ('AK', 'HI')"
            )
            expected = {6567, 66833, 66923, 66924, 67905, 68054, 68271, 68391}
            if test2.empty:
                return df
            elif set(test2.plant_id_eia.unique()) <= expected:
                return df.assign(
                    balancing_authority_code_eia=lambda x: x.balancing_authority_code_eia.mask(
                        x.plant_id_eia.isin((6567, 66833)), "ISNE"
                    )
                    .mask(x.plant_id_eia.isin((66923, 66924)), "PNM")
                    .mask(x.plant_id_eia.isin((67905,)), "TEPC")
                    .mask(x.plant_id_eia.isin((68054,)), "MISO")
                    .mask(x.plant_id_eia.isin((68391, 68271)), "PJM")
                )
            else:
                raise AssertionError(
                    f"additional non-retired plants beyond {expected} are missing BA codes"
                )

        plant_cols = [
            "transmission_distribution_owner_id",
            "transmission_distribution_owner_name",
        ]

        self.gens = (
            pd_read_pudl("core_eia860m__changelog_generators", release=self.pudl_release)
            .sort_values("report_date", ascending=True)
            .groupby(["plant_id_eia", "generator_id"], as_index=False)
            .last(skipna=False)
            .merge(
                pd_read_pudl("out_eia__yearly_plants", release=self.pudl_release)
                .query("report_date.dt.year >= @self.years[0]")
                .sort_values("report_date")
                .groupby("plant_id_eia", as_index=False)[
                    ["balancing_authority_code_eia", *plant_cols]
                ]
                .last(),
                on="plant_id_eia",
                how="left",
                validate="m:1",
                suffixes=(None, "_plant"),
            )
            .merge(
                pd_read_pudl("core_eia860__scd_utilities", release=self.pudl_release)
                .groupby("utility_id_eia", as_index=False)
                .entity_type.last()
                .replace(
                    {
                        "entity_type": {
                            "C": "Cooperative",
                            "I": "Investor Owned",
                            "Q": "IPP",
                            "M": "Municipal",
                            "P": "Political Subdivision",
                            "F": "Federal",
                            "S": "State",
                        }
                    }
                ),
                on="utility_id_eia",
                how="left",
                validate="m:1",
            )
            .assign(
                balancing_authority_code_eia=lambda x: x.balancing_authority_code_eia.fillna(
                    x.balancing_authority_code_eia_plant
                )
            )
            .drop(
                columns=[
                    "valid_until_date",
                    "data_maturity",
                    "balancing_authority_code_eia_plant",
                ]
            )
            .pipe(fix_ba_codes)
            .assign(
                report_date=lambda x: pd.to_datetime(x.report_date.max()),
                capacity_mw=lambda x: x.capacity_mw.fillna(x.summer_capacity_mw),
                current_planned_generator_operating_date=lambda x: pd.to_datetime(
                    x.current_planned_generator_operating_date
                ),
                generator_retirement_date=lambda x: pd.to_datetime(
                    x.generator_retirement_date
                ),
                planned_generator_retirement_date=lambda x: pd.to_datetime(
                    x.planned_generator_retirement_date
                ),
                planned_uprate_date=lambda x: pd.to_datetime(x.planned_uprate_date),
            )
            .pipe(retirement_fixes)
            .pipe(fix_cc_in_prime)
            .pipe(fix_op_status)
            .rename(columns={"energy_source_code_1": "energy_source_code_860m"})
            .assign(
                fuel_group=lambda x: x.energy_source_code_860m.map(FUEL_GROUP_MAP),
                operating_date=lambda x: x.generator_operating_date.fillna(
                    x.current_planned_generator_operating_date
                ),
                retirement_date=lambda x: x.generator_retirement_date.fillna(
                    x.planned_generator_retirement_date
                ),
                respondent_name=lambda x: x.utility_name_eia,
                category=lambda x: x.operational_status.str.cat(
                    np.where(
                        x.prime_mover.isin(("CC", "GT", "ST", "IC"))
                        & x.fuel_group.isin(self.fos_fuels),
                        "fossil",
                        np.where(x.technology_description.isin(techs), "clean", "other"),
                    ),
                    sep="_",
                ),
                utcoffset=lambda x: x.plant_id_eia.map(offsets),
                source="860m",
            )
            .pipe(add_ba_code)
            .astype({"plant_id_eia": int})
            .pipe(add_plant_role)
            .pipe(self._ba_overrides)
            .pipe(self.proposed_atb_class)
            .pipe(add_ec_flag)
            .merge(rules, on="balancing_authority_code_eia", validate="m:1", how="left")
            .assign(
                cr_eligible=lambda x: (x.category == "existing_fossil")
                & (x.sc_cat != "convert")
                & (
                    (x.surplus & (x.retirement_date.dt.year.fillna(2100) > 2035))
                    # TODO replacement should be constrained by retirement date
                    | (x.replacement & x.retirement_date.notna())
                    # we want to keep ERCO and NYIS around and can always remove completely later
                    | x.balancing_authority_code_eia.isin(("NYIS", "ERCO"))
                ),
            )
            .query("state not in ('AK', 'HI')")
            .pipe(self.add_regulatory_ranking)
        )
        print(
            self.gens.query("technology_description == 'Natural Gas Fired Combustion Turbine'")
            .groupby("reg_rank", dropna=False)
            .capacity_mw.sum()
        )
        ids = (  # noqa: F841
            self.gens.query(
                "(retirement_date.isna() | retirement_date.dt.year > 2050) & reg_rank < 8 &"
                "((operating_date.dt.year >= 2000 & technology_description in @OTHER_TD_MAP) | technology_description not in @OTHER_TD_MAP)"
            )
            .groupby(["plant_id_eia", "generator_id"], as_index=False)[
                [
                    "technology_description",
                    "capacity_mw",
                    "operational_status",
                    "balancing_authority_code_eia",
                    "final_ba_code",
                    "reg_rank",
                ]
            ]
            .first()
            .groupby(
                ["plant_id_eia", "technology_description", "operational_status"],
                as_index=False,
            )
            .agg(
                {
                    "capacity_mw": "sum",
                    "generator_id": tuple,
                    "balancing_authority_code_eia": "first",
                    "final_ba_code": "first",
                    "reg_rank": "first",
                }
            )
            .query(
                "technology_description in ('Natural Gas Fired Combustion Turbine', 'Natural Gas Fired Combined Cycle') & capacity_mw >= 150"
            )
        )

        self.ba_offsets = (
            self.gens.groupby(["final_ba_code", "utcoffset"])
            .capacity_mw.sum()
            .reset_index()
            .sort_values(["final_ba_code", "capacity_mw"], ascending=[True, False])
            .groupby("final_ba_code")
            .utcoffset.first()
            .to_frame(name="td_offset")
            .assign(
                hr_offset=lambda x: x.td_offset.dt.total_seconds().astype(int) // 3600,
                tz_code=lambda x: x.hr_offset.map(
                    # this sign change seems crazy but is in fact correct
                    {-i: f"Etc/GMT+{i}" for i in range(13)}
                    | {i: f"Etc/GMT{-i}" for i in range(15)}
                ),
            )
        )

        self.cw = (
            pl_scan_pudl("core_epa__assn_eia_epacamd_subplant_ids", self.pudl_release)
            .select("plant_id_eia", "subplant_id", "emissions_unit_id_epa", "generator_id")
            .drop_nulls()
            .unique()
            .collect()
            .to_pandas()
            .merge(
                self.gens[self.gens.operational_status != "proposed"][
                    ["plant_id_eia", "generator_id", "capacity_mw"]
                ],
                on=["plant_id_eia", "generator_id"],
                how="inner",
                validate="m:1",
            )
        )

        self.en_com = (
            pd.read_parquet(ROOT_PATH / "patio_data/re_sites_w_energy_community.parquet")
            .reset_index()[["plant_id_eia", "energy_community"]]
            .drop_duplicates(subset="plant_id_eia")
        )
        self.df923m_clean = (
            pl_scan_pudl("out_eia923__monthly_generation_fuel_by_generator", self.pudl_release)
            .select(
                "report_date",
                pl.col("plant_id_eia").cast(pl.Int64),
                "generator_id",
                "net_generation_mwh",
            )
            .filter(pl.col("report_date").dt.year() >= self.years[0])
            .join(
                pl.from_pandas(self.gens)
                .lazy()
                .filter(pl.col("technology_description").is_in(RE_TECH.keys()))
                .select("plant_id_eia", "generator_id", "final_ba_code"),
                on=["plant_id_eia", "generator_id"],
                how="inner",
            )
            .with_columns(pl.col("net_generation_mwh").fill_null(0.0))
        )
        self.df923 = (
            pl_scan_pudl(
                "out_eia923__yearly_generation_fuel_by_generator",
                self.pudl_release,
            )
            .filter(pl.col("report_date").dt.year() >= self.years[0])
            .with_columns(pl.col("plant_id_eia").cast(pl.Int64))
            .rename({"report_date": "datetime"})
            .join(
                self.df923m_clean.rename({"report_date": "datetime"})
                .group_by_dynamic(
                    index_column="datetime",
                    every="1y",
                    period="1y",
                    by=["plant_id_eia", "generator_id"],
                )
                .agg(pl.sum("net_generation_mwh").alias("mo_mwh")),
                on=["plant_id_eia", "generator_id", "datetime"],
                how="left",
            )
            .with_columns(
                net_generation_mwh=pl.when(
                    pl.col("mo_mwh").is_not_null()
                    & pl.col("net_generation_mwh").is_not_null()
                    & (pl.col("mo_mwh") != pl.col("net_generation_mwh"))
                )
                .then(pl.col("mo_mwh"))
                .otherwise(pl.col("net_generation_mwh"))
            )
            .drop("mo_mwh")
            .sort("plant_id_eia", "generator_id", "datetime")
            .collect()
            .to_pandas()
            .merge(
                self.gens[["plant_id_eia", "generator_id", "final_ba_code"]],
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="m:1",
            )
        )

        c_hist = pd.read_parquet(ROOT_PATH / "r_data/python_inputs_data_hist.parquet")
        c_cfl = pd.read_parquet(ROOT_PATH / "r_data/python_inputs_data.parquet")
        act_cost = read_ferc860()[["Plant", "Prime", "report_year", "capex_per_kW"]].rename(
            columns={"Plant": "plant_id_eia"}
        )
        cost_idx = ["plant_id_eia", "generator_id", "report_year", "Prime"]
        self.capex = (
            c_hist[[*cost_idx, "capex_per_kW_no_cum_starts_est"]]
            .drop_duplicates()
            .merge(
                c_cfl[
                    [*cost_idx, "real_maint_capex_per_kW_no_cum_starts_est"]
                ].drop_duplicates(),
                on=cost_idx,
                validate="1:1",
                how="left",
            )
            .merge(
                act_cost.drop_duplicates(subset=["plant_id_eia", "Prime", "report_year"]),
                on=["plant_id_eia", "Prime", "report_year"],
                how="left",
                validate="m:1",
            )
            .assign(
                capex_per_kW=lambda x: x.capex_per_kW.fillna(x.capex_per_kW_no_cum_starts_est),
                report_month=1,
            )
            .pipe(month_year_to_date)
            .pipe(simplify_columns)
            .astype({"plant_id_eia": int})
            .rename(
                columns={
                    "real_maint_capex_per_kw_no_cum_starts_est": "real_maint_capex_per_kw",
                    "report_date": "datetime",
                }
            )[
                [
                    "plant_id_eia",
                    "generator_id",
                    "datetime",
                    "capex_per_kw",
                    "real_maint_capex_per_kw",
                ]
            ]
        )
        cols = [
            "plant_id_eia",
            "generator_id",
            "technology_description",
            "capacity_mw",
            "datetime",
            "utility_id_eia",
        ]
        LOGGER.critical("WE ARE DROPPING DUPLICATE ROWS IN COSTS, THIS NEEDS TO BE FIXED")
        costs = (
            c_cfl.pipe(cost_helper, kind="")
            # FIXME don't like this dropping, GH-36
            .drop_duplicates(subset=cols)
            .merge(
                self.gens[
                    [
                        "plant_id_eia",
                        "generator_id",
                        "final_ba_code",
                        "prime_mover",
                        "fuel_group",
                    ]
                ].drop_duplicates(),
                on=["plant_id_eia", "generator_id"],
                how="left",
                validate="m:1",
            )
            .replace({np.inf: np.nan})
            .merge(COST_FLOOR, on="technology_description", how="left", validate="m:1")
            .fillna({"vom_floor": 0.0, "fom_floor": 0.0, "som_floor": 0.0})
            .assign(year=lambda x: x.datetime.dt.year)
        )
        warnings.warn("2021 and 2022 costs are escalated from 2020", UserWarning)  # noqa: B028
        cost_cols = [
            "vom_per_mwh",
            "fom_per_kw",
            "start_per_kw",
        ]
        last_year = costs.datetime.dt.year.max()
        new = []
        for yr in range(last_year + 1, self.years[1] + 1):
            new_ = costs.query("datetime.dt.year == @last_year").copy()
            # yr = 2021
            new_.loc[:, cost_cols] = new_.loc[:, cost_cols] * 1.03 ** (yr - last_year)
            new.append(
                new_.assign(
                    datetime=lambda x: pd.to_datetime(
                        x.datetime.dt.strftime(f"{yr}-%m-%d %H:%M:%S")  # noqa: B023
                    ),
                    year=yr,
                )
            )

        costs = (
            pd.concat([costs, *new])
            .sort_values(["plant_id_eia", "generator_id", "datetime"])
            .merge(
                self._fuel_cost_setup(),
                on=["plant_id_eia", "generator_id", "year"],
                how="inner",
                suffixes=(None, "_"),
                validate="1:m",
            )
            .assign(datetime=lambda x: x.datetime_)
        )

        curve = (
            pl.read_parquet(ROOT_PATH / "r_data/fuel_supply_curve.parquet")
            .filter(
                (pl.col("fuel_group_code") == "natural_gas")
                & (pl.col("fuel_consumed_mmbtu_cumsum") <= pl.col("top_percent_mmbtu"))
            )
            .select(
                "ba_code",
                pl.col("fuel_group_code").alias("fuel_group"),
                pl.col("report_date").cast(pl.Datetime).alias("datetime"),
                pl.col("final_fuel_cost_per_mmbtu").alias("cost_per_mmbtu"),
                pl.col("fuel_consumed_mmbtu").alias("consumed_mmbtu"),
                pl.col("fuel_consumed_mmbtu_cumsum").alias("fuel_mmbtu_cumsum"),
                pl.col("top_percent_mmbtu").alias("fuel_mmbtu_max"),
                pl.col("final_fuel_price_increase_per_mmbtu").alias("slope"),
            )
            .sort(["ba_code", "datetime", "cost_per_mmbtu"])
        )

        assert curve.filter(pl.col("slope").is_null()).is_empty()

        self.curve = (
            pl.concat(
                [
                    curve,
                    curve.group_by("ba_code", "fuel_group", "datetime")
                    .agg(pl.all().first())
                    .with_columns(fuel_mmbtu_cumsum=pl.lit(0.0)),
                ]
            )
            .sort("ba_code", "datetime", "fuel_mmbtu_cumsum")
            .select("ba_code", "datetime", "fuel_mmbtu_cumsum", "cost_per_mmbtu")
        )

        costs = costs.merge(
            curve.group_by(
                pl.col("ba_code").alias("final_ba_code"),
                "datetime",
                "fuel_group",
                maintain_order=True,
            )
            .agg(
                pl.max("cost_per_mmbtu").alias("intercept"),
                pl.first("slope"),
                pl.max("fuel_mmbtu_max"),
            )
            .to_pandas(),
            on=["final_ba_code", "fuel_group", "datetime"],
            how="left",
            validate="m:1",
            indicator=True,
        )
        if not costs.query(
            "slope.isna() & fuel_group == 'natural_gas' "
            "& final_ba_code.notna() & plant_id_eia != 1260",
            engine="python",
        ).empty:
            LOGGER.warning("MISSING SOME NATURAL GAS SLOPES")
        techs = [
            "Petroleum Liquids",
            "Natural Gas Steam Turbine",
            "Conventional Steam Coal",
            "Natural Gas Fired Combined Cycle",
            "Natural Gas Fired Combustion Turbine",
            "Natural Gas Internal Combustion Engine",
            "Coal Integrated Gasification Combined Cycle",
            "Other Gases",
            "Petroleum Coke",
            "Wood/Wood Waste Biomass",
            "Other Waste Biomass",
            "Landfill Gas",
            "Municipal Solid Waste",
            "All Other",
        ]
        LOGGER.critical("WE ARE REMOVING NON-FOSSIL COSTS")
        costs = costs.query("technology_description in @techs")
        assert not costs.duplicated(subset=["plant_id_eia", "generator_id", "datetime"]).any()
        cost_out_cols = [
            "final_ba_code",
            "technology_description",
            "fuel_group",
            # "state",
            # "atb_class",
            "vom_per_mwh",
            "fuel_per_mwh",
            "total_var_mwh",
            "fom_per_kw",
            "start_per_kw",
            "heat_rate",
            "co2_factor",
            "fuel_per_mmbtu",
            "fuel_mmbtu_max",
            "slope",
            "intercept",
        ]
        warnings.warn("bracketing heat rates", UserWarning)  # noqa: B028
        warnings.warn("bracketing fuel costs for CCs and GTs at 99th percentile", UserWarning)  # noqa: B028
        warnings.warn("co2_factor is from EIA, see constants.CARB_INTENSITY", UserWarning)  # noqa: B028

        self.costs = costs.assign(
            vom_per_mwh=lambda x: np.maximum(x.vom_per_mwh.fillna(x.vom_floor), x.vom_floor),
            fom_per_kw=lambda x: np.maximum(x.fom_per_kw.fillna(x.fom_floor), x.fom_floor),
            start_per_kw=lambda x: np.maximum(x.start_per_kw.fillna(x.som_floor), x.som_floor),
            heat_rate=lambda x: np.minimum(
                np.maximum(
                    x.fuel_mmbtu / x.fossil_mwh,
                    x.technology_description.replace(MIN_HEAT_RATE),
                ),
                20.0,
            ),
            fuel_per_mmbtu=lambda x: x.final_fuel_cost_per_mmbtu,
            fuel_per_mwh=lambda x: x.fuel_per_mmbtu * x.heat_rate,
            total_var_mwh=lambda x: x[["vom_per_mwh", "fuel_per_mwh"]].sum(axis=1),
            # emission intensity from cost data not usable
            co2_factor=lambda x: x.technology_description.replace(CARB_INTENSITY),
        ).set_index(["plant_id_eia", "generator_id", "datetime"])[cost_out_cols]

        assert self.costs.index.is_unique

        with rmi_cloud_fs().open(
            f"az://patio-data/{PATIO_DATA_RELEASE}/utility_ids.parquet"
        ) as f:
            parent = (
                pl.read_parquet(f)
                .filter(pl.col("utility_id_eia").is_not_null())
                .select(
                    pl.col("utility_id_eia").alias("owner_utility_id_eia"),
                    pl.col("utility_type_rmi").replace_strict(
                        {
                            "Cooperative": "coop",
                            "Vertically Integrated": "viu",
                            "Municipal": "muni",
                            "Independent Power Producer": "ipp",
                            "Unknown": "ipp",
                            "Restructured": "viu",
                            "Federal": "gov",
                            "Political Subdivision": "gov",
                            "Investor Owned": "viu",
                            "State": "muni",
                            "Industrial": "ipp",
                            "Commercial": "ipp",
                        },
                        default="unk",
                    ),
                    "parent_lei",
                    "parent_name",
                    pl.col("ticker").alias("parent_ticker"),
                )
                .unique()
                .group_by("owner_utility_id_eia")
                .agg(
                    pl.col("utility_type_rmi", "parent_ticker").drop_nulls().first(),
                    pl.col("parent_name").drop_nulls().first(),
                )
            )

        self.own = (
            pl.from_pandas(self.gens)
            .join(
                generator_ownership(release=self.pudl_release).join(
                    parent,
                    on="owner_utility_id_eia",
                    how="left",
                    validate="m:1",
                    suffix="_p",
                ),
                on=["plant_id_eia", "generator_id"],
                suffix="_o",
                how="left",
                validate="1:m",
            )
            .select(
                "plant_id_eia",
                "generator_id",
                "technology_description",
                "utility_id_eia",
                "utility_name_eia",
                (
                    pl.col("capacity_mw") * pl.col("fraction_owned").fill_null(pl.lit(1.0))
                ).alias("owned_capacity_mw"),
                "final_ba_code",
                "balancing_authority_code_eia",
                pl.col("owner_utility_id_eia")
                .fill_null(pl.col("utility_id_eia"))
                .cast(pl.Int64),
                pl.col("owner_utility_name_eia").fill_null(pl.col("utility_name_eia")),
                pl.col("fraction_owned").fill_null(pl.lit(1.0)),
                pl.col("parent_name").fill_null(
                    pl.col("owner_utility_name_eia").fill_null(pl.col("utility_name_eia"))
                ),
                pl.col("utility_type_rmi").fill_null(pl.lit("unk")),
                "parent_ticker",
            )
            .to_pandas()
        )

    def add_regulatory_ranking(self, gens):
        if not (file := USER_DATA_PATH / "Electric_Retail_Service_Territories.kml").exists():
            if not file.with_suffix(".kml.zip").exists():
                download(PATIO_DATA_AZURE_URLS["lse"], file.with_suffix(".kml.zip"))
            shutil.unpack_archive(file.with_suffix(".kml.zip"), USER_DATA_PATH)
            file.with_suffix(".kml.zip").unlink()
        utils = (
            pd_read_pudl(
                "core_eia861__yearly_operational_data_misc", release=self.pudl_release
            )
            .query("retail_sales_mwh.notna() & retail_sales_mwh > 0 & data_observed")
            .groupby("utility_id_eia", as_index=False)
            .last()
            .rename(
                columns={
                    "utility_id_eia": "id",
                    "entity_type": "entity_type_lse",
                    "utility_name_eia": "utility_name_eia_lse",
                }
            )[["id", "utility_name_eia_lse", "entity_type_lse"]]
        )
        lse = (
            gpd.read_file(file)
            .query(
                "~id.str.contains('NA') & TYPE in ('MUNICIPAL', 'INVESTOR OWNED', 'COOPERATIVE','POLITICAL SUBDIVISION', 'STATE',)"
            )
            .astype({"id": int})
            .merge(utils, on="id", how="inner", validate="m:1")
        )
        non_geo_cols = [
            "plant_id_eia",
            "generator_id",
            "utility_name_eia",
            "utility_id_eia",
            "entity_type",
            "transmission_distribution_owner_id",
            "transmission_distribution_owner_name",
            "balancing_authority_code_eia",
            "technology_description",
        ]
        isos = ["PJM", "MISO", "SWPP", "CISO", "ERCO", "ISNE", "NYIS"]
        monopoly = [
            "MISO",
            "SOCO",
            "SWPP",
            "LGEE",
            "AECI",
            "AZPS",
            "GRIF",
            "DEAA",
            "SRP",
            "WALC",
            "IID",
            "TIDC",
            "BANC",
            "DUK",
            "EPE",
            "FMPP",
            "FPC",
            "TAL",
            "FPL",
            "JEA",
            "LDWP",
            "NEVP",
            "PACE",
            "PNM",
            "BPAT",
            "IPCO",
            "AVRN",
            "AVA",
            "GRID",
            "PSEI",
            "PGE",
            "NWMT",
            "PSCO",
            "SC",
            "SCEG",
            "SEC",
            "HGMA",
            "TEC",
            "TEPC",
            "WACM",
        ]
        non_profit = ("Cooperative", "State", "Municipal", "Political Subdivision")
        for_geo = gens[[*non_geo_cols, "latitude", "longitude"]].drop_duplicates()
        gen_lse = (
            pl.from_pandas(
                pd.DataFrame(
                    gpd.sjoin(
                        gpd.GeoDataFrame(
                            for_geo[non_geo_cols],
                            geometry=gpd.points_from_xy(
                                for_geo["longitude"], for_geo["latitude"]
                            ),
                        ),
                        lse,
                        "left",
                        "within",
                    )
                ).drop(columns=["geometry"])
            )
            .select(
                *non_geo_cols,
                pl.col("id").cast(pl.Int32),
                "entity_type_lse",
                "utility_name_eia_lse",
                "Name",
                "SOURCE",
                "REGULATED",
                "HOLDING_CO",
            )
            .with_columns(
                n_matches_o=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
                _id_match=pl.col("id") == pl.col("utility_id_eia"),
                id_match=(pl.col("id") == pl.col("utility_id_eia"))
                .any()
                .over("plant_id_eia", "generator_id"),
            )
            .filter(
                (pl.col("_id_match") & pl.col("id_match"))
                | pl.col("id_match").not_()
                | (pl.col("n_matches_o") == 1)
            )
            .with_columns(
                n_matches=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
                _coop_match=(pl.col("entity_type") == "Cooperative")
                & (pl.col("entity_type_lse") == "Cooperative"),
                coop_match=(
                    (pl.col("entity_type") == "Cooperative")
                    & (pl.col("entity_type_lse") == "Cooperative")
                )
                .any()
                .over("plant_id_eia", "generator_id"),
            )
            .filter(
                (pl.col("_coop_match") & pl.col("coop_match"))
                | (pl.col("n_matches") == 1)
                | (pl.col("entity_type") != "Cooperative")
            )
            .with_columns(
                n_matches=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
                n_types=pl.col("entity_type_lse")
                .replace(
                    ["State", "Federal", "Municipal"],
                    ["Political Subdivision"] * 3,
                )
                .n_unique()
                .over("plant_id_eia", "generator_id"),
            )
            .with_columns(
                reg_rank=pl.when(
                    (pl.col("entity_type_lse") == "Investor Owned")
                    & pl.col("balancing_authority_code_eia").is_in(monopoly)
                    & (pl.col("utility_id_eia") == pl.col("id"))
                    & pl.col("balancing_authority_code_eia").is_in(isos).not_()
                )
                .then(1)
                .when(
                    (pl.col("entity_type_lse") == "Investor Owned")
                    & (pl.col("entity_type") == "IPP")
                    & pl.col("balancing_authority_code_eia").is_in(monopoly)
                    & pl.col("balancing_authority_code_eia").is_in(isos).not_()
                )
                .then(2)
                .when(
                    (pl.col("entity_type_lse") == "Investor Owned")
                    & (pl.col("utility_id_eia") == pl.col("id"))
                    & (
                        pl.col("balancing_authority_code_eia").is_in(("MISO", "SWPP"))
                        | (pl.col("utility_name_eia") == "Virginia Electric & Power Co")
                    )
                )
                .then(3)
                .when(
                    pl.col("entity_type_lse").is_in(non_profit).not_()
                    & (pl.col("balancing_authority_code_eia") == "ERCO")
                    & (pl.col("entity_type") == "IPP")
                )
                .then(4)
                .when(
                    pl.col("entity_type_lse").is_in(non_profit)
                    & (pl.col("balancing_authority_code_eia") == "ERCO")
                    & pl.col("entity_type").is_in(non_profit)
                )
                .then(5)
                .when(
                    pl.col("entity_type_lse").is_in(non_profit)
                    & (pl.col("balancing_authority_code_eia") == "ERCO")
                    & (pl.col("entity_type") == "IPP")
                )
                .then(6)
                .when(
                    pl.col("entity_type_lse").is_in(("Cooperative", "Municipal"))
                    & pl.col("balancing_authority_code_eia").is_in(isos).not_()
                )
                .then(7)
                .otherwise(9)
            )
            .with_columns(
                n_reg_ranks=pl.col("reg_rank").n_unique().over("plant_id_eia", "generator_id")
            )
            .select(
                *non_geo_cols,
                "id",
                "entity_type_lse",
                "utility_name_eia_lse",
                "reg_rank",
                "n_reg_ranks",
                "n_matches_o",
                "n_matches",
                "n_types",
            )
        )
        mess = (
            gen_lse.filter(
                (pl.col("n_matches") != 1)
                & (pl.col("n_types") != 1)
                & (pl.col("n_reg_ranks") != 1)
            )
            .with_columns(
                _id_match=pl.col("id") == pl.col("transmission_distribution_owner_id"),
                id_match=(pl.col("id") == pl.col("transmission_distribution_owner_id"))
                .any()
                .over("plant_id_eia", "generator_id"),
            )
            .filter(
                (pl.col("n_matches") == 1)
                | (pl.col("_id_match") & pl.col("id_match"))
                | pl.col("id_match").not_()
            )
            .with_columns(
                n_matches=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
                all_good_ranks=(pl.col("reg_rank") < 7)
                .all()
                .over("plant_id_eia", "generator_id"),
            )
        )
        done = pl.concat(
            [
                gen_lse.filter(pl.col("n_matches") == 1).select(
                    "plant_id_eia",
                    "generator_id",
                    "reg_rank",
                    "entity_type",
                    "entity_type_lse",
                    "utility_name_eia_lse",
                    "balancing_authority_code_eia",
                    "id",
                ),
                gen_lse.filter((pl.col("n_matches") != 1) & (pl.col("n_types") == 1))
                .group_by("plant_id_eia", "generator_id")
                .agg(
                    pl.col("reg_rank").first(),
                    pl.col("entity_type").first(),
                    pl.col("entity_type_lse").unique().sort().str.join("|"),
                    pl.col("utility_name_eia_lse").unique().sort().str.join("|"),
                    pl.col("balancing_authority_code_eia").first(),
                ),
                gen_lse.filter(
                    (pl.col("n_matches") != 1)
                    & (pl.col("n_types") != 1)
                    & (pl.col("n_reg_ranks") == 1)
                )
                .group_by("plant_id_eia", "generator_id")
                .agg(
                    pl.col("reg_rank").first(),
                    pl.col("entity_type").first(),
                    pl.col("entity_type_lse").unique().sort().str.join("|"),
                    pl.col("utility_name_eia_lse").unique().sort().str.join("|"),
                    pl.col("balancing_authority_code_eia").first(),
                ),
                mess.filter(pl.col("n_matches") == 1).select(
                    "plant_id_eia",
                    "generator_id",
                    "reg_rank",
                    "entity_type",
                    "entity_type_lse",
                    "utility_name_eia_lse",
                    "balancing_authority_code_eia",
                    "id",
                ),
                mess.filter((pl.col("n_matches") != 1) & pl.col("all_good_ranks"))
                .group_by("plant_id_eia", "generator_id")
                .agg(
                    pl.col("reg_rank").max(),
                    pl.col("entity_type").first(),
                    pl.col("entity_type_lse").unique().sort().str.join("|"),
                    pl.col("utility_name_eia_lse").unique().sort().str.join("|"),
                    pl.col("balancing_authority_code_eia").first(),
                ),
                mess.filter(
                    (pl.col("n_matches") != 1)
                    & (pl.col("n_types") != 1)
                    & pl.col("all_good_ranks").not_()
                    & pl.col("utility_name_eia_lse").str.contains("WAPA").not_()
                )
                .with_columns(
                    n_matches=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
                )
                .filter(pl.col("n_matches") == 1)
                .select(
                    "plant_id_eia",
                    "generator_id",
                    "reg_rank",
                    "entity_type",
                    "entity_type_lse",
                    "utility_name_eia_lse",
                    "balancing_authority_code_eia",
                    "id",
                ),
            ],
            how="diagonal",
        )
        assert done[["plant_id_eia", "generator_id"]].is_unique().all()
        fmess = (  # noqa: F841
            mess.filter(
                (pl.col("n_matches") != 1)
                & (pl.col("n_types") != 1)
                & pl.col("all_good_ranks").not_()
                & pl.col("utility_name_eia_lse").str.contains("WAPA").not_()
            )
            .with_columns(
                n_matches=pl.col("id").n_unique().over("plant_id_eia", "generator_id"),
            )
            .filter(pl.col("n_matches") != 1)
        )
        return gens.merge(
            done.select(
                "plant_id_eia",
                "generator_id",
                "reg_rank",
                "entity_type_lse",
                "utility_name_eia_lse",
            ).to_pandas(),
            on=["plant_id_eia", "generator_id"],
            how="left",
            validate="1:1",
        )

    def _fuel_cost_setup(self):
        fuel = (
            pl.scan_parquet(ROOT_PATH / "r_data/final_fuel_costs.parquet")
            .select(
                pl.col("plant_id_eia").cast(int),
                pl.col("prime_mover_code").alias("prime_mover"),
                pl.col("fuel_group_code").alias("fuel_group"),
                pl.col("report_date").alias("datetime").dt.replace_time_zone(None),
                "final_fuel_cost_per_mmbtu",
                pl.col("year").cast(int),
            )
            .collect()
            .to_pandas()
        )
        assert not fuel.duplicated(
            ["plant_id_eia", "prime_mover", "fuel_group", "datetime"]
        ).any()
        # 3 plants that use other_gas do not have recoverable cost data for that fuel
        # for many months, so we fill it in with natural gas prices
        replace = (
            fuel.query(
                "plant_id_eia in (1391, 50627, 55088) & final_fuel_cost_per_mmbtu.isna()"
            )[["plant_id_eia", "prime_mover", "datetime", "year", "fuel_group"]]
            .merge(
                fuel.query(
                    "plant_id_eia in (1391, 50627, 55088) & fuel_group == 'natural_gas'"
                ),
                on=["plant_id_eia", "prime_mover", "datetime", "year"],
                how="inner",
                validate="1:1",
                suffixes=(None, "_"),
            )
            .set_index(["plant_id_eia", "prime_mover", "fuel_group", "datetime"])[
                ["final_fuel_cost_per_mmbtu", "year"]
            ]
        )
        fuel = fuel.set_index(
            ["plant_id_eia", "prime_mover", "fuel_group", "datetime"]
        ).combine_first(replace)
        # we do the merge in two steps, first get the data by generator,
        # than switch to monthly on the big merge
        fuel_by_gen = (
            self.gens[["plant_id_eia", "generator_id", "prime_mover", "fuel_group"]]
            .drop_duplicates()
            .dropna()
            .merge(
                fuel.reset_index(),
                on=["plant_id_eia", "fuel_group", "prime_mover"],
                how="inner",
            )
            .sort_values(["plant_id_eia", "generator_id", "datetime"])
        )
        assert not fuel_by_gen.duplicated(["plant_id_eia", "generator_id", "datetime"]).any()
        LOGGER.warning("ALL COAL PRICES ARE 2022")
        fuel_by_gen = fuel_by_gen.assign(month=lambda x: x.datetime.dt.month)
        fuel_by_gen = fuel_by_gen.merge(
            fuel_by_gen.query("year == 2022")[
                ["plant_id_eia", "generator_id", "month", "final_fuel_cost_per_mmbtu"]
            ],
            on=["plant_id_eia", "generator_id", "month"],
            validate="m:1",
            how="left",
            suffixes=(None, "_"),
        ).assign(
            final_fuel_cost_per_mmbtu=lambda x: x.final_fuel_cost_per_mmbtu.mask(
                (x.fuel_group == "coal") & x.final_fuel_cost_per_mmbtu_.notna(),
                x.final_fuel_cost_per_mmbtu_,
            )
        )[
            [
                "plant_id_eia",
                "generator_id",
                "prime_mover",
                "fuel_group",
                "datetime",
                "final_fuel_cost_per_mmbtu",
                "year",
            ]
        ]
        return fuel_by_gen

    def raw_curves(self, atb_year, regime: Literal["reference", "limited"]):
        links = {
            "reference": {
                "onshore_wind": "https://www.nrel.gov/gis/assets/docs/reference-access-siting-regime-atb-mid-turbine-fy21.csv",
                "offshore_wind": "https://www.nrel.gov/gis/assets/docs/offshore-wind-open-access-siting-regime-atb-mid-turbine-fy21.csv",
                "solar": "https://drive.google.com/uc?id=1OSN44kDafr6iYyg39gxIR5AZMmQ_Bvd2",
            },
            "limited": {
                "onshore_wind": "https://www.nrel.gov/gis/assets/docs/limited-access-siting-regime-atb-mid-turbine-fy21.csv",
                "offshore_wind": "https://www.nrel.gov/gis/assets/docs/offshore-wind-limited-access-siting-regime-atb-mid-turbine-fy21.csv",
                "solar": "https://drive.google.com/uc?id=1pGK77iM8qad3L66ya0OT3zgOV2VJk-uJ",
            },
        }[regime]
        links = {
            "reference": {
                "onshore_wind": "https://data.openei.org/files/6119/reference_access_2030_moderate_115hh_170rd_supply-curve%20(1).csv",
                "offshore_wind": "https://data.openei.org/files/6189/reference_supply-curve_post_proc%202.csv",
                "solar": "https://data.openei.org/files/6001/reference_access_2030_moderate_supply-curve.csv",
            },
            "limited": {
                "onshore_wind": "https://data.openei.org/files/6119/limited_access_2030_moderate_115hh_170rd_supply-curve.csv",
                "offshore_wind": "https://data.openei.org/files/6189/limited_supply-curve_post_proc.csv",
                "solar": "https://data.openei.org/files/6001/limited_access_2030_moderate_supply-curve.csv",
            },
        }[regime]

        re_mapper = {
            "Onshore Wind Turbine": "onshore_wind",
            "Offshore Wind Turbine": "offshore_wind",
            "Solar Photovoltaic": "solar",
        }
        atb = (
            clean_atb(
                case="Market",
                scenario="Moderate",
                crp=30,
                report_year=2023,
                pudl_release=self.pudl_release,
            )
            .filter(
                (pl.col("technology_description").is_in(list(re_mapper)))
                & (pl.col("projection_year") == atb_year)
            )
            .select(
                pl.col("technology_description")
                .replace_strict(re_mapper, default=None)
                .alias("technology"),
                "class_atb",
                pl.col("capex_per_kw").alias("capex_atb").cast(pl.Float64),
                pl.col("capacity_factor").alias("cf_atb").cast(pl.Float64),
                pl.col("fom_per_kw").alias("fom_atb").cast(pl.Float64),
                pl.col("levelized_cost_of_energy_per_mwh").alias("loce_atb").cast(pl.Float64),
            )
            .to_pandas()
        )
        # https://atb.nrel.gov/electricity/2023/utility-scale_pv
        # solar_path = CACHE_PATH / f"pv_{regime}_2020.csv"
        # if not solar_path.exists():
        #     gdown.download(links["solar"], temp := BytesIO())
        #     ZipFile(temp).extract(solar_path.name, path=CACHE_PATH)
        rn = {
            "mean_res": "global_horizontal_irradiance",
            "capacity_mw": "capacity_mw_ac",
            "mean_cf": "mean_cf_ac",
            "reinforcement_cost_per_mw": "reinforcement_cost_per_mw_ac",
            "trans_cap_cost_per_mw": "trans_cap_cost_per_mw_ac",
            "capacity_ac_mw": "capacity_mw_ac",
            "area_developable_sq_km": "area_sq_km",
            "resource": "windspeed_m_per_s",
            "dist_spur_km": "distance_to_transmission_km",
            "dist_km": "distance_to_transmission_km",
            # "ncf_2035": "mean_cf_ac",
            "cost_reinforcement_usd_per_mw": "reinforcement_cost_per_mw_ac",
            "cost_total_trans_usd_per_mw": "trans_cap_cost_per_mw_ac",
            "lcot_usd_per_mwh": "lcot",
            "dist_reinforcement_km": "reinforcement_dist_km",
            "all_in_lcoe_2035": "total_lcoe",
            "site_lcoe_usd_per_mwh_2035": "mean_lcoe",
            "mean_depth": "elevation",
            "offtake_state": "state",
            "technology": "offshore_tech",
        }
        solar_path = CACHE_PATH / links["solar"].rsplit("/")[-1]
        if not solar_path.exists():
            download(links["solar"], solar_path)
        solar_curve = (
            pd.read_csv(solar_path)
            .rename(columns=rn)
            .assign(
                class_atb=lambda x: pd.cut(
                    x.global_horizontal_irradiance,
                    bins=[0, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 12],
                    labels=range(10, 0, -1),
                ),
            )
            .merge(
                atb.query("technology == 'solar'").drop(columns=["technology"]),
                on="class_atb",
                validate="m:1",
            )
        )
        # https://atb.nrel.gov/electricity/2023/land-based_wind
        wind_curve_path = CACHE_PATH / links["onshore_wind"].rsplit("/")[-1]
        if not wind_curve_path.exists():
            download(links["onshore_wind"], wind_curve_path)
        wind_curve = (
            pd.read_csv(wind_curve_path)
            .rename(columns=rn)
            .assign(
                class_atb=lambda x: pd.cut(
                    x.windspeed_m_per_s,
                    bins=[0, 5.9, 6.53, 7.1, 7.62, 8.07, 8.35, 8.57, 8.77, 9.01, 20],
                    labels=range(10, 0, -1),
                ).astype(int),
            )
            .merge(
                atb.query("technology == 'onshore_wind'").drop(columns=["technology"]),
                on="class_atb",
                validate="m:1",
            )
        )
        off_cf = atb.query("technology == 'offshore_wind'").drop(columns=["technology"])
        _off_path = CACHE_PATH / links["offshore_wind"].rsplit("/")[-1]
        if not _off_path.exists():
            download(links["offshore_wind"], _off_path)
        _off = (
            pd.read_csv(_off_path).rename(columns=rn).assign(elevation=lambda x: -x.elevation)
        )
        off_curve = pd.concat(
            [
                _off[_off.offshore_tech == "floating"]
                .assign(
                    class_atb=lambda x: pd.cut(
                        x.windspeed_m_per_s,
                        bins=[0, 7.07, 7.93, 8.85, 9.13, 9.31, 9.98, 20],
                        labels=range(7, 0, -1),
                    ).astype(int),
                )
                .merge(off_cf, on="class_atb", validate="m:1"),
                _off[_off.offshore_tech == "fixed"]
                .assign(
                    class_atb=lambda x: pd.cut(
                        x.windspeed_m_per_s,
                        bins=[0, 7.43, 8.84, 9.6, 10.01, 10.18, 10.3, 20],
                        labels=range(14, 7, -1),
                    ).astype(int)
                )
                .merge(off_cf, on="class_atb", validate="m:1"),
            ]
        )
        return solar_curve, wind_curve, off_curve

    def proposed_atb_class(self, gens):
        cache_path = CACHE_PATH / "proposed_atb.parquet"
        if not cache_path.exists():
            pl.read_csv(ROOT_PATH / "patio/package_data/proposed_atb.csv").write_parquet(
                cache_path
            )

        re_codes = {
            "Solar Photovoltaic": 1,
            "Onshore Wind Turbine": 2,
            "Offshore Wind Turbine": 3,
        }
        prop = pl.from_pandas(
            gens.query("technology_description in @re_codes").assign(
                re_code=lambda x: x.technology_description.map(re_codes)
            )
        )
        full = pl.read_parquet(cache_path)
        test = prop.join(full, on=["plant_id_eia", "technology_description"], how="inner")
        if len(test) == len(prop):
            return (
                gens.merge(
                    full.to_pandas(),
                    on=["plant_id_eia", "technology_description"],
                    how="left",
                    validate="m:1",
                )
                .fillna({"class_atb": -1})
                .astype({"class_atb": int})
            )
        LOGGER.warning("rebuilding proposed ATB classes, this can take a while")

        s, on, off = self.raw_curves(2025, regime="reference")
        curve_sites = (
            pl.from_pandas(
                pd.concat([s.assign(re_code=1), on.assign(re_code=2), off.assign(re_code=3)])
            )
            .select("latitude", "longitude", "class_atb", "cf_atb", "re_code")
            .lazy()
        )
        prop_lite = (
            # prop.select("plant_id_eia", "re_code", "latitude", "longitude").unique().lazy()
            prop.join(full, on=["plant_id_eia", "technology_description"], how="left")
            .filter(pl.col("class_atb").is_null())
            .select("plant_id_eia", "re_code", "latitude", "longitude")
            .unique()
            .lazy()
        )
        to_cat = [full]
        for code in tqdm((1, 2, 3)):
            to_cat.append(
                prop_lite.filter(pl.col("re_code") == code)
                .join(
                    curve_sites.filter(pl.col("re_code") == code),
                    how="cross",
                    suffix="_site",
                )
                .pipe(
                    pl_distance,
                    lat1="latitude",
                    lat2="latitude_site",
                    lon1="longitude",
                    lon2="longitude_site",
                )
                .sort("plant_id_eia", "distance", descending=False)
                .group_by("plant_id_eia")
                .agg(pl.col("re_code", "class_atb", "cf_atb").first())
                .select(
                    "plant_id_eia",
                    pl.col("re_code")
                    .replace_strict({v: k for k, v in re_codes.items()})
                    .alias("technology_description"),
                    "class_atb",
                    "cf_atb",
                )
                .collect()
            )
        full = pl.concat(to_cat)
        full.write_parquet(CACHE_PATH / "proposed_atb.parquet")
        return (
            gens.merge(
                full.to_pandas(),
                on=["plant_id_eia", "technology_description"],
                how="left",
                validate="m:1",
            )
            .fillna({"class_atb": -1})
            .astype({"class_atb": int})
        )

    def re_sites_for_encom(self):
        curve_sites = (
            pd.concat(self.raw_curves(2025, regime="reference"))[["latitude", "longitude"]]
            .drop_duplicates()
            .assign(site_type="curve")
        )
        df = pd.concat(
            [
                curve_sites,
                self.gens.query("category == 'proposed_clean'").assign(site_type="proposed"),
            ]
        )[["site_type", "latitude", "longitude"]].sort_values(
            ["site_type", "latitude", "longitude"]
        )
        df.to_csv(Path.home() / "patio_datare_sites_to_tag.csv", index=False)
        return df

    def curve_for_site_selection(
        self,
        atb_year,
        assumed_ilr,
        regime: Literal["reference", "limited"] | None = None,
    ):
        solar_curve_r, wind_curve_r, off_curve_r = self.raw_curves(
            atb_year, regime="reference"
        )
        LOGGER.warning("reference curves setup.")
        solar_curve_l, wind_curve_l, off_curve_l = self.raw_curves(atb_year, regime="limited")
        LOGGER.warning("limited curves setup.")
        common = [
            "sc_point_gid",
            "latitude",
            "longitude",
            "distance_to_transmission_km",
            "elevation",
            "state",
            "area_sq_km",
            "capacity_mw_ac",
            "mean_cf_ac",
            "mean_lcoe",
            "lcot",
            "total_lcoe",
            "trans_cap_cost_per_mw_ac",
            "class_atb",
            "capex_atb",
            "cf_atb",
            "fom_atb",
            "reinforcement_cost_per_mw_ac",
            "reinforcement_dist_km",
            "area_sq_km_lim",
            "capacity_mw_ac_lim",
        ]
        solar_curve = solar_curve_r.merge(
            solar_curve_l,
            on=["latitude", "longitude"],
            how="left",
            validate="1:1",
            suffixes=(None, "_lim"),
        ).assign(
            distance_to_transmission_km=lambda x: np.maximum(
                x.distance_to_transmission_km,
                x.distance_to_transmission_km_lim.fillna(0.0),
            ),
        )[
            [
                *common,
                "cnty_fips",
                "timezone",
                "reg_mult",
                "capacity_mw_dc",
                "capacity_mw_dc_lim",
                "mean_cf_dc",
                "global_horizontal_irradiance",
            ]
        ]
        LOGGER.warning("solar curves combined.")
        wind_curve = wind_curve_r.merge(
            wind_curve_l,
            on=["latitude", "longitude"],
            how="left",
            validate="1:1",
            suffixes=(None, "_lim"),
        ).assign(
            distance_to_transmission_km=lambda x: np.maximum(
                x.distance_to_transmission_km,
                x.distance_to_transmission_km_lim.fillna(0.0),
            ),
        )[
            [
                *common,
                "cnty_fips",
                "timezone",
                "reg_mult",
                "n_turbines",
                "turbine_capacity",
                "hub_height",
                "windspeed_m_per_s",
            ]
        ]
        LOGGER.warning("onshore curves combined.")
        off_curve = off_curve_r.merge(
            off_curve_l,
            on=["latitude", "longitude"],
            how="left",
            validate="1:1",
            suffixes=(None, "_lim"),
        ).assign(
            distance_to_transmission_km=lambda x: np.maximum(
                x.distance_to_transmission_km,
                x.distance_to_transmission_km_lim.fillna(0.0),
            ),
        )[
            [
                *common,
                "offshore_tech",
                "windspeed_m_per_s",
                "mean_dist_p_to_s",
                "mean_wake_losses-means",
                "dist_to_coast_offshore",
                "dist_export_km",
                "cost_spur_usd_per_mw",
                "cost_poi_usd_per_mw",
                "cost_export_usd_per_mw",
                "cost_occ_2035_usd_per_mw",
                "cost_opex_2035_usd_per_mw",
            ]
        ]
        LOGGER.warning("offshore curves combined.")
        del (
            solar_curve_r,
            wind_curve_r,
            off_curve_r,
            solar_curve_l,
            wind_curve_l,
            off_curve_l,
        )
        if not (
            all_re_path := USER_DATA_PATH.parent / "all_re_new_too_big_tabled.parquet"
        ).exists():
            get("raw-data/all_re_new_too_big_tabled.parquet", all_re_path)
        re_profs = (
            pl.scan_parquet(all_re_path)
            .with_columns(
                cf_ilr_prof_site=pl.when(pl.col("re_type") == "solar")
                .then(pl.min_horizontal(pl.col("generation") * assumed_ilr, 1.0))
                .otherwise(pl.col("generation"))
            )
            .group_by("plant_id_eia", "re_type")
            .agg(pl.mean("generation"), pl.mean("cf_ilr_prof_site"))
            .select(
                pl.col("plant_id_eia").cast(pl.Int64),
                "re_type",
                pl.col("generation").alias("cf_prof_site"),
                "cf_ilr_prof_site",
            )
            .collect()
        )
        all_meta = pl.concat(
            [
                pl.from_pandas(self.re().drop(columns=["all_plant_ids"])).select(
                    pl.col("plant_id_eia").cast(pl.Int64),
                    pl.col("technology_description").alias("re_type"),
                    "latitude",
                    "longitude",
                ),
                pl.read_csv(
                    ROOT_PATH / "patio/package_data/re_site_ids.csv",
                    dtypes={
                        "plant_id_eia": pl.Int64,
                        "re_type": pl.Utf8,
                        "latitude": pl.Float64,
                        "longitude": pl.Float64,
                    },
                ),
            ]
        ).unique()

        sites = (
            all_meta.lazy()
            .join(re_profs.lazy(), on=["plant_id_eia", "re_type"], how="inner")
            .select(
                pl.col("plant_id_eia").alias("plant_id_prof_site"),
                "re_type",
                "latitude",
                "longitude",
                "cf_prof_site",
                "cf_ilr_prof_site",
            )
            .collect()
            .lazy()
        )
        LOGGER.warning("profile sites loaded")
        solar_curve = (
            pl.from_pandas(solar_curve)
            .filter(pl.col("capacity_mw_ac") >= 25)
            .select("sc_point_gid", "latitude", "longitude")
            .lazy()
            .join(
                sites.filter(pl.col("re_type") == "solar"),
                how="cross",
                suffix="_prof_site",
            )
            .pipe(
                pl_distance,
                lat1="latitude",
                lat2="latitude_prof_site",
                lon1="longitude",
                lon2="longitude_prof_site",
            )
            .sort("sc_point_gid", "distance", descending=False)
            .group_by("sc_point_gid")
            .agg(pl.all().first())
            .rename({"distance": "distance_prof_site"})
            .join(
                pl.from_pandas(solar_curve).drop(["latitude", "longitude"]).lazy(),
                on="sc_point_gid",
                how="inner",
                validate="1:1",
            )
            .with_columns(pl.col("capacity_mw_ac").cast(pl.Float64))
            .sort("sc_point_gid", descending=False)
            .collect()
        )
        LOGGER.warning("profile assigned to solar")
        wind_curve = (
            pl.from_pandas(wind_curve)
            .filter(pl.col("capacity_mw_ac") >= 25)
            .select("sc_point_gid", "latitude", "longitude")
            .lazy()
            .join(
                sites.filter(pl.col("re_type") == "onshore_wind"),
                how="cross",
                suffix="_prof_site",
            )
            .pipe(
                pl_distance,
                lat1="latitude",
                lat2="latitude_prof_site",
                lon1="longitude",
                lon2="longitude_prof_site",
            )
            .sort("sc_point_gid", "distance", descending=False)
            .group_by("sc_point_gid")
            .agg(pl.all().first())
            .rename({"distance": "distance_prof_site"})
            .join(
                pl.from_pandas(wind_curve).drop(["latitude", "longitude"]).lazy(),
                on="sc_point_gid",
                how="inner",
                validate="1:1",
            )
            .with_columns(pl.col("capacity_mw_ac").cast(pl.Float64))
            .sort("sc_point_gid", descending=False)
            .collect()
        )
        LOGGER.warning("profile assigned to onshore wind")
        off_curve = (
            pl.from_pandas(off_curve)
            .filter(pl.col("capacity_mw_ac") >= 25)
            .lazy()
            .join(
                sites.filter(pl.col("re_type") == "offshore_wind"),
                how="cross",
                suffix="_prof_site",
            )
            .pipe(
                pl_distance,
                lat1="latitude",
                lat2="latitude_prof_site",
                lon1="longitude",
                lon2="longitude_prof_site",
            )
            .sort("sc_point_gid", "distance", descending=False)
            .group_by("sc_point_gid")
            .agg(pl.all().first())
            .rename({"distance": "distance_prof_site"})
            .with_columns(pl.col("capacity_mw_ac").cast(pl.Float64))
            .collect()
        )
        LOGGER.warning("profile assigned to offshore wind")

        re_site_id = pl.from_pandas(
            pl.concat(
                [
                    solar_curve.select("latitude", "longitude"),
                    wind_curve.select("latitude", "longitude"),
                    off_curve.select("latitude", "longitude"),
                ]
            )
            .unique()
            .sort("latitude", "longitude")
            .with_row_index("re_site_id", 500000)
            .to_pandas()
            .pipe(add_ec_flag, key_col="re_site_id")
        ).rename({"latitude": "latitude_nrel_site", "longitude": "longitude_nrel_site"})

        fossil = (
            pl.concat(
                [
                    pl.from_pandas(self.modelable_generators())
                    .filter(pl.col("category").is_in(("existing_fossil", "proposed_fossil")))
                    .join(
                        pl.from_pandas(self.counterfactual_cost().reset_index())
                        .filter(
                            (pl.col("total_var_mwh") > 0)
                            & pl.col("total_var_mwh").is_not_nan()
                        )
                        .group_by("plant_id_eia", "generator_id")
                        .agg(pl.mean("total_var_mwh")),
                        on=["plant_id_eia", "generator_id"],
                        how="left",
                    ),
                    pl.from_pandas(
                        self.gens.query(
                            "final_ba_code.notna() & final_ba_code != '<NA>' "
                            "& category in ('existing_clean', 'proposed_clean')"
                            "& capacity_mw > 250"
                        )
                    ),
                ],
                how="diagonal_relaxed",
            )
            .select(
                "plant_id_eia",
                "generator_id",
                "latitude",
                "longitude",
                "technology_description",
                "capacity_mw",
                "final_ba_code",
                "total_var_mwh",
            )
            .with_row_count("icx_row_num")
        )
        l_fossil = fossil.select("icx_row_num", "latitude", "longitude").lazy()

        to_cat = []
        for re_type, re_curve in (
            ("solar", solar_curve),
            ("onshore_wind", wind_curve),
            ("offshore_wind", off_curve),
        ):
            re = (
                l_fossil.join(
                    re_curve.select("sc_point_gid", "latitude", "longitude").lazy(),
                    how="cross",
                    suffix="_nrel_site",
                )
                .pipe(
                    pl_distance,
                    lat1="latitude",
                    lat2="latitude_nrel_site",
                    lon1="longitude",
                    lon2="longitude_nrel_site",
                )
                .filter(pl.col("distance") < 50)
                .collect()
            )
            LOGGER.warning("%s to fossil distances calculated", re_type)
            re = (
                re.join(
                    fossil.drop("latitude", "longitude"),
                    on=["icx_row_num"],
                    suffix="_fossil_site",
                )
                .join(
                    re_curve.drop("latitude", "longitude"),
                    on=["sc_point_gid"],
                    suffix="_nrel_site",
                )
                .drop("icx_row_num")
            )
            LOGGER.warning("%s data rejoined", re_type)
            re.write_parquet(CACHE_PATH / f"{re_type}_curve_to_fossil.parquet")
            to_cat.append(re)
        crf = (0.08 * 1.08**30) / (1.08**30 - 1)
        out = (
            pl.concat(to_cat, how="diagonal")
            .join(re_site_id, on=["latitude_nrel_site", "longitude_nrel_site"])
            .with_columns(
                itc=pl.when(
                    (pl.col("re_type") == "offshore_wind") & pl.col("energy_community")
                )
                .then(pl.lit(0.4))
                .when(pl.col("re_type") == "offshore_wind")
                .then(pl.lit(0.3))
                .otherwise(pl.lit(0.0)),
                ptc=pl.when((pl.col("re_type") == "onshore_wind") & pl.col("energy_community"))
                .then(pl.lit(17.99 * 1.1))
                .when(pl.col("re_type") == "onshore_wind")
                .then(pl.lit(17.99))
                .when((pl.col("re_type") == "solar") & pl.col("energy_community"))
                .then(pl.lit(16.63 * 1.1))
                .when(pl.col("re_type") == "solar")
                .then(pl.lit(16.63))
                .otherwise(pl.lit(0.0)),
            )
            .with_columns(
                lcoe=(
                    (pl.col("capex_atb") * (1 - pl.col("itc")) + 1.854 * pl.col("distance"))
                    * crf
                    + pl.col("fom_atb")
                )
                / (8.76 * pl.col("cf_atb"))
                - pl.col("ptc"),
            )
            .rename(
                {
                    "plant_id_eia": "icx_id",
                    "generator_id": "icx_gen",
                    "final_ba_code": "ba_code",
                    "capacity_mw_ac": "capacity_mw_nrel_site",
                    "capacity_mw_ac_lim": "capacity_mw_nrel_site_lim",
                    "technology_description": "icx_tech",
                    "capacity_mw": "icx_capacity",
                }
            )
            .sort(
                "icx_id",
                "icx_gen",
                "re_type",
                "lcoe",
                descending=False,
            )
        )
        out.write_parquet(CACHE_PATH / "re_curve_to_fossil.parquet")

        return out

    def re_curve(self, atb_year, assumed_ilr):
        solar_curve, wind_curve, off_curve = self.raw_curves(atb_year, regime="reference")

        re_profs = (
            pl.scan_parquet(Path.home() / "patio_data/all_re.parquet")
            .with_columns(
                cf_rninja_ilr=pl.when(pl.col("re_type") == "solar")
                .then(pl.min_horizontal(pl.col("generation") * assumed_ilr, 1.0))
                .otherwise(pl.col("generation"))
            )
            .group_by("plant_id_eia", "re_type")
            .agg(pl.mean("generation"), pl.mean("cf_rninja_ilr"))
            .select(
                pl.col("plant_id_eia").cast(pl.Int64),
                pl.col("re_type").alias("technology_description"),
                pl.col("generation").alias("cf_rninja"),
                "cf_rninja_ilr",
            )
        )
        sites = (
            pl.from_pandas(self.re().drop(columns=["all_plant_ids"]))
            .lazy()
            .join(re_profs, on=["plant_id_eia", "technology_description"], how="inner")
        )
        return (
            pl.concat(
                [
                    pl_df_product(
                        pl.from_pandas(_curve).lazy(),
                        sites.filter(pl.col("technology_description") == _type),
                        suffix="_rninja",
                    )
                    .pipe(
                        pl_distance,
                        lat1="latitude",
                        lat2="latitude_rninja",
                        lon1="longitude",
                        lon2="longitude_rninja",
                    )
                    .filter(pl.col("distance") < 200)
                    .sort("distance")
                    .group_by("plant_id_eia", "technology_description")
                    .agg(pl.all().first())
                    for (_curve, _type) in (
                        (solar_curve, "solar"),
                        (wind_curve, "onshore_wind"),
                        (off_curve, "offshore_wind"),
                    )
                ],
                how="diagonal",
            )
            .rename({"capacity_factor": "cf_curve"})
            .with_columns(
                cf_mult=pl.max_horizontal(pl.col("cf_atb") / pl.col("cf_rninja_ilr"), 1.0)
            )
            .collect()
        )

    def temp_nuke_replacements(self):
        """Screen for plants with high CF that could be replaced by nuclear."""
        if "temp_nuke_replacements" not in self._dfs:
            if not (file := USER_DATA_PATH / "camd_starts_ms.parquet").exists():
                download(PATIO_DATA_AZURE_URLS["camd_starts_ms"], file)
            plants = (
                self.modelable_generators()
                .query("category == 'existing_fossil'")
                .groupby(["plant_id_eia"])
                .agg(
                    {
                        "prime_mover_code": set,
                        "fuel_group": set,
                        "capacity_mw": "sum",
                        "final_ba_code": "first",
                    }
                )
                .merge(
                    self.historical_cost().groupby(["plant_id_eia"]).total_var_mwh.max(),
                    on=["plant_id_eia"],
                    validate="1:1",
                )
                .merge(
                    pd.read_parquet(file).groupby(["plant_id_eia"]).gross_generation_mwh.sum(),
                    on=["plant_id_eia"],
                    validate="1:1",
                )
                .assign(
                    cf=lambda xx: xx.gross_generation_mwh / (xx.capacity_mw * 8760 * 12),
                    ba_cap=lambda x: x.groupby(["final_ba_code"]).capacity_mw.transform("sum"),
                )
                .reset_index()
                .astype({"plant_id_eia": int})
            )
            out = (
                plants.query("cf > 0.6 & capacity_mw > 300")
                .sort_values(["final_ba_code", "total_var_mwh"], ascending=[True, False])
                .assign(
                    cum_cap=lambda x: x.groupby(["final_ba_code"]).capacity_mw.transform(
                        "cumsum"
                    ),
                    cum_share=lambda x: x.cum_cap / x.ba_cap,
                )
                .query("cum_share < 0.2")
                .set_index(["final_ba_code", "plant_id_eia"])
                # .sort_index()
            )
            self._dfs["temp_nuke_replacements"] = {
                k: {1: []} for k in self.modelable_generators().final_ba_code.unique()
            } | {
                k: {1: list(out.loc[k].index)}
                for k in out.index.get_level_values("final_ba_code").unique()
            }
        return self._dfs["temp_nuke_replacements"]

    def temp_ccs_conversion(self):
        """Screen for plants with high CF that could be replaced by nuclear."""
        if "temp_ccs_conversion" not in self._dfs:
            plants = self.modelable_generators().merge(
                self.historical_cost()
                .groupby(["plant_id_eia", "generator_id"])
                .total_var_mwh.min(),
                on=["plant_id_eia", "generator_id"],
                validate="1:1",
            )

            out = (
                plants.query(
                    "technology_description == 'Conventional Steam Coal' "
                    "& (retirement_date.isna() | retirement_date > '2040') "
                )
                .sort_values(["final_ba_code", "total_var_mwh"])
                .merge(
                    plants.groupby(["final_ba_code"]).capacity_mw.sum().reset_index(),
                    on="final_ba_code",
                    validate="m:1",
                    suffixes=(None, "_ba"),
                )
                .assign(
                    cum_cap=lambda x: x.groupby(["final_ba_code"]).capacity_mw.transform(
                        "cumsum"
                    ),
                    cum_share=lambda x: x.cum_cap / x.capacity_mw_ba,
                )
                .query("cum_share < 0.15")
                .set_index("final_ba_code")[["plant_id_eia", "generator_id"]]
                .sort_values(["plant_id_eia", "generator_id"])
                .astype({"plant_id_eia": int})
            )
            self._dfs["temp_ccs_conversion"] = {
                k: {1: []} for k in self.modelable_generators().final_ba_code.unique()
            } | {
                k: {1: list(out.loc[[k]].itertuples(index=False, name=None))}
                for k in out.index.unique()
            }
        return self._dfs["temp_ccs_conversion"]

    def _ba_overrides(self, df):
        if self.overrides is None:
            return df
        for ba, d in self.overrides.items():
            for pid, gens in d.items():
                df.loc[
                    df[(df.plant_id_eia == pid) & df.generator_id.isin(gens)].index,
                    "final_ba_code",
                ] = ba
        return df

    def historical_cost(self):
        return self.costs.copy()
        # if "historical_cost" not in self._dfs:
        #     self._dfs["historical_cost"] = self._cost("historical")
        # return self._dfs["historical_cost"]

    def counterfactual_cost(self):
        return self.costs.copy()
        # if "counterfactual_cost" not in self._dfs:
        #     self._dfs["counterfactual_cost"] = self._cost("counterfactual")
        # return self._dfs["counterfactual_cost"]

    def problem_cost_explorer(self):
        """Code for finding weird cost data."""
        df = self.costs.merge(
            self.modelable_generators()[["plant_id_eia", "generator_id", "operating_date"]],
            on=["plant_id_eia", "generator_id"],
            how="right",
            validate="m:1",
        ).query("datetime >= operating_date")
        df[
            df[["counterfactual_fuel_per_mwh", "counterfactual_vom_per_mwh"]].sum(axis=1) <= 0
        ].drop_duplicates(subset=["plant_id_eia", "generator_id"])
        df[
            df[["historical_fuel_per_mwh", "historical_vom_per_mwh"]].sum(axis=1) <= 0
        ].drop_duplicates(subset=["plant_id_eia", "generator_id"])

    def unsafe_plants(self):
        """Plants whose lat/lon is outside state bounds"""
        df = check_lat_lons(self.gens)
        return (
            df[~df.safe_lat_lon]
            .drop(["report_date", "generator_id"], axis=1)
            .drop_duplicates()
        )

    def utility_codes(self):
        return self.gens[
            ["final_ba_code", "respondent_name", "balancing_authority_code_eia"]
        ].drop_duplicates()

    def get_missing_generators(self, ba_code, plant_data):
        all_gens = self.gens.query(
            "final_ba_code == @ba_code & operational_status == 'existing'"
        ).set_index(plant_data.index.names)
        cur = plant_data.query("operational_status == 'existing'")
        return self.df923.merge(
            all_gens.loc[
                [x for x in all_gens.index if x not in cur.index],
                [
                    x
                    for x in all_gens
                    if x
                    not in [
                        *all_gens.columns.intersection(self.df923.columns),
                        "report_date",
                    ]
                ],
            ].reset_index(),
            on=["plant_id_eia", "generator_id"],
            how="inner",
            validate="m:1",
        )

    def modelable_generators(self):
        if not (file := USER_DATA_PATH / "irp.parquet").exists():
            download(PATIO_DATA_AZURE_URLS["irp"], file)
        irp = pd.read_parquet(file).astype({"plant_id_eia": int}).pipe(add_plant_role)
        if "utility_id_eia" not in irp:
            irp = irp.assign(
                utility_id_eia=lambda x: x.final_ba_code.map(
                    {
                        "186": 19876,
                        "EPE": 5701,
                        "ETR": -999,
                        "PAC": 14354,
                        "SC": 17543,
                        "TEPC": 24211,
                        "TVA": 18642,
                        "WACM": 30151,
                    }
                ).astype(int),
                # final_ba_code=lambda x: x.final_ba_code.replace(
                #     {"186": "PJM", "ETR": "MISO"}
                # ),
            )
        return (
            pd.concat([self.gens, irp]).query(
                "final_ba_code.notna() & final_ba_code != '<NA>' "
                "& (category in ('existing_fossil', 'proposed_fossil', 'proposed_clean')"
                # "| technology_description in @ES_TECHS)"
                ")"
            )
            # .astype(
            #     {
            #         # "final_ba_code": str,
            #         "state": "category",
            #         "respondent_name": "category",
            #         "balancing_authority_code_eia": "category",
            #         "prime_mover_code": "category",
            #         "prime_mover": "category",
            #         "technology_description": "category",
            #         # "energy_source_code_860m": "category",
            #         # "fuel_group_energy_source_code_860m": "category",
            #         "energy_source_code_860m": "category",
            #         # "rmi_energy_source_code_2": "category",
            #         # "rmi_energy_source_code_3": "category",
            #         "fuel_group": "category",
            #         # "fuel_group_rmi_energy_source_code_2": "category",
            #         # "fuel_group_rmi_energy_source_code_3": "category",
            #         # "status_860m": "category",
            #         "operational_status": "category",
            #         "plant_role": "category",
            #         "final_ba_code": "category",
            #     }
            # )
        )

    # def all_modelable_bas(self) -> dict[str, pd.DataFrame]:
    #     if "all_modelable_bas" not in self._dfs:
    #         df = self.modelable_generators().set_index(["plant_id_eia", "generator_id"])
    #         self._dfs["all_modelable_bas"] = {
    #             k: df.query("final_ba_code == @k")
    #             for k in sorted(df.final_ba_code.unique())
    #         }
    #     return self._dfs["all_modelable_bas"]

    def all_modelable_generators(self) -> pd.DataFrame:
        if "all_modelable_generators" not in self._dfs:
            self._dfs["all_modelable_generators"] = (
                self.modelable_generators()
                .sort_values(["final_ba_code", "plant_id_eia", "generator_id"])
                .set_index(["plant_id_eia", "generator_id"])
            )
        return self._dfs["all_modelable_generators"]

    def get_generator_data(self, ba_code, cost_type, solar_ilr):
        plant_data = (
            self.all_modelable_generators()
            .query("final_ba_code == @ba_code")
            .assign(
                ilr=lambda x: np.where(
                    x.technology_description == "Solar Photovoltaic", solar_ilr, 1.0
                ),
                roundtrip_eff=lambda x: x.technology_description.map(
                    {"Batteries": 0.9, "Hydroelectric Pumped Storage": 0.8}
                ),
                duration_hrs=lambda x: (x.energy_storage_capacity_mwh / x.capacity_mw).fillna(
                    x.technology_description.map(
                        {"Batteries": 4, "Hydroelectric Pumped Storage": 12}
                    )
                ),
            )
        )
        if not (no_capacity := plant_data.query("capacity_mw.isna()")).empty:
            LOGGER.info(
                "%s counts of generators missing capacity that will be excluded  \n%s",
                ba_code,
                no_capacity.groupby(["operational_status", "technology_description"])
                .plant_id_eia.count()
                .to_string()
                .replace("\n", "\n\t"),
            )
        plant_data = plant_data.query("capacity_mw.notna()")

        cost_data = (
            getattr(self, f"{cost_type}_cost")()
            .query("final_ba_code == @ba_code")
            .sort_index()
        )
        def_cost = (
            get_atb(
                plant_data.reset_index(),
                years=(self.years[0], self.years[1] + 1),
                pudl_release=self.pudl_release,
            )
            # .set_index(cost_data.index.names)
            # .sort_index()
            .assign(
                total_var_mwh=np.nan,
                year=lambda x: x.datetime.dt.year,
            )
            .merge(
                cost_data.reset_index()[["datetime"]]
                .drop_duplicates()
                .dropna()
                .assign(year=lambda x: x.datetime.dt.year),
                on="year",
                suffixes=("_", None),
            )
            # TODO check that this is working for proposed generators so they get the
            #  curve data they need
            .merge(
                cost_data.query("fuel_group == 'natural_gas'")
                .reset_index()[
                    [
                        "datetime",
                        "fuel_group",
                        "fuel_mmbtu_max",
                        "slope",
                        "intercept",
                    ]
                ]
                .drop_duplicates(),
                on=["datetime", "fuel_group"],
                how="left",
                validate="m:1",
            )
            .set_index(cost_data.index.names)
            .sort_index()
        )
        def_cost = def_cost[[c for c in cost_data if c in def_cost]]
        cost_data = (
            cost_data
            # fill in missing data and get cost data for proposed generators
            .combine_first(def_cost).assign(
                total_var_mwh=lambda x: x[["vom_per_mwh", "fuel_per_mwh"]].sum(axis=1),
                fuel_per_mmbtu=lambda x: x.fuel_per_mmbtu.fillna(x.fuel_per_mwh / x.heat_rate),
            )
        )
        msg = f"{ba_code} some natural gas generators are missing fuel"
        if not cost_data.query("fuel_group == 'natural_gas' & slope.isna()").empty:
            raise AssertionError(f"{msg} slopes")
        if not cost_data.query("fuel_group == 'natural_gas' & fuel_per_mmbtu.isna()").empty:
            raise AssertionError(f"{msg} cost")
        # we want to add clean FOM to plant_data
        ids = plant_data.query("category == 'proposed_clean'").category
        clean_fom = (
            cost_data.reset_index(level="datetime")
            .merge(ids, left_index=True, right_index=True, how="inner", validate="m:1")
            .fillna(0.0)
            .assign(fom_per_kw=lambda x: x.fom_per_kw + x.total_var_mwh * 8.76)
            .fom_per_kw.drop_duplicates()
        )
        return (
            plant_data.query("technology_description not in @ES_TECHS").merge(
                clean_fom, left_index=True, right_index=True, how="left", validate="1:1"
            ),
            cost_data,
            plant_data.query("technology_description in @ES_TECHS"),
        )

    def pl_counterfactual_re(self):
        if "pl_counterfactual_re" not in self._dfs:
            self._dfs["pl_counterfactual_re"] = pl.from_pandas(
                self.counterfactual_re()
                .columns.to_frame()
                .reset_index(drop=True)
                .rename(
                    columns={
                        "technology_description": "re_type",
                        "final_ba_code": "ba_code",
                        "plant_id_eia": "plant_id_prof_site",
                    }
                )
            )
        return self._dfs["pl_counterfactual_re"]

    def counterfactual_re(self):
        if "counterfactual_re" not in self._dfs:
            # yr range is inclusive
            fyr, lyr = self.years
            id_cols = [
                "technology_description",
                "latitude",
                "longitude",
            ]
            re = (
                self.gens.query(
                    "technology_description in @RE_TECH "
                    "& operating_date.dt.year >= @fyr & operating_date.dt.year < @lyr"
                )
                .pipe(check_lat_lons)
                .query("safe_lat_lon | technology_description == 'offshore_wind'")
                .copy()
                .replace({"technology_description": RE_TECH})
                .pipe(round_coordinates, tol=0.5)
                .groupby([*id_cols, "final_ba_code", "operating_date"])
                .agg({"capacity_mw": np.sum, "state": "first"})
                .sort_index()
                .reset_index()
                .merge(
                    self.re()[[*id_cols, "plant_id_eia"]],
                    on=id_cols,
                    how="left",
                    validate="m:1",
                )
                .assign(
                    cum_sum=lambda x: x.groupby(
                        ["final_ba_code", "technology_description", "plant_id_eia"]
                    ).capacity_mw.transform("cumsum")
                )
                .pivot(
                    index="operating_date",
                    columns=["final_ba_code", "plant_id_eia", "technology_description"],
                    values="cum_sum",
                )
                .sort_index(axis=1)
                .fillna(method="ffill", axis=0)
                .fillna(0.0)
            )
            self._dfs["counterfactual_re"] = re.iloc[-1, :].to_numpy() - re
        return self._dfs["counterfactual_re"]

    def re(self):
        """All operating renewable generators that would generate unique
        profiles from renewables.ninja
        """
        if "re" not in self._dfs:
            re = (
                pd.concat(
                    [
                        self.gens.query("technology_description in @RE_TECH").copy(),
                        pd.read_csv(ROOT_PATH / "patio_data/re_queue.csv", header=0),
                    ]
                )
                .replace({"technology_description": RE_TECH})
                .pipe(round_coordinates, tol=0.5)
            )
            re["all_plant_ids"] = re["plant_id_eia"]
            re = (
                re.sort_values(["capacity_mw"], ascending=False)
                .groupby(["technology_description", "latitude", "longitude"])
                .agg(
                    {
                        "capacity_mw": np.sum,
                        "plant_id_eia": "first",
                        "all_plant_ids": set,
                        # "timezone": "first",
                        "state": "first",
                    }
                )
                .reset_index()
                .pipe(check_lat_lons)
            )
            self._dfs["re"] = re[
                re.safe_lat_lon | (re.technology_description == "offshore_wind")
            ]
            self._dfs["unsafe_re"] = re[
                ~re.safe_lat_lon & (re.technology_description != "offshore_wind")
            ]
        return self._dfs["re"]

    # def re_dist(self):
    #     if "re_dist" not in self._dfs:
    #         cols = ["plant_id_eia", "latitude", "longitude"]
    #
    #         re_cf = (
    #             pl.scan_parquet(PATIO_DOC_PATH / "data/all_re.parquet")
    #             .groupby(["plant_id_eia", "re_type"])
    #             .agg(pl.mean("generation").alias("cf"))
    #             .select(
    #                 "plant_id_eia",
    #                 pl.col("re_type").alias("technology_description"),
    #                 "cf",
    #             )
    #             .collect()
    #             .to_pandas()
    #         )
    #         crf = (0.08 * 1.08**30) / (1.08**30 - 1)
    #         re = get_atb(
    #             self.re()
    #             .copy()
    #             .merge(
    #                 re_cf,
    #                 on=["plant_id_eia", "technology_description"],
    #                 how="left",
    #                 validate="1:1",
    #             )
    #             .merge(self.en_com, on="plant_id_eia", how="left", validate="m:1")
    #             .assign(
    #                 technology_description_=lambda x: x.technology_description,
    #                 technology_description=lambda x: x.technology_description.map(
    #                     RE_TECH_R
    #                 ),
    #                 operating_date=pd.to_datetime("2025-01-01"),
    #             ),
    #             (2028, 2029),
    #         ).assign(technology_description=lambda x: x.technology_description_)
    #
    #         df = df_product(
    #             self.modelable_generators()
    #             .dropna(subset=cols, how="any")
    #             .groupby("plant_id_eia")[["latitude", "longitude"]]
    #             .first()
    #             .reset_index(),
    #             re[
    #                 [
    #                     *cols,
    #                     "technology_description",
    #                     "capacity_mw",
    #                     "state",
    #                     "cf",
    #                     "energy_community",
    #                     "capex_per_kw",
    #                     "fom_per_kw",
    #                 ]
    #             ].copy(),
    #         )
    #         df["distance"] = distance_arrays(
    #             df[["latitude_l", "longitude_l"]].to_numpy(),
    #             df[["latitude_r", "longitude_r"]].to_numpy(),
    #         )
    #         self._dfs["re_dist"] = (
    #             df.query("distance < 1000")
    #             .fillna({"energy_community_r": False})
    #             .assign(
    #                 itc=lambda x: x[["technology_description_r", "energy_community_r"]]
    #                 .astype(str)
    #                 .agg("".join, axis=1)
    #                 .map({"offshore_windTrue": 0.4, "offshore_windFalse": 0.3})
    #                 .fillna(0.0),
    #                 ptc=lambda x: x.technology_description_r.map(
    #                     {"onshore_wind": 17.99, "solar": 16.63}
    #                 ).fillna(0.0)
    #                 * np.where(x.energy_community_r, 1.1, 1.0),
    #                 lcoe=lambda x: (
    #                     (x.capex_per_kw_r * (1 - x.itc) + 1.854 * x.distance) * crf
    #                     + x.fom_per_kw_r
    #                 )
    #                 / (x.cf_r * 8.76)
    #                 - x.ptc,
    #             )
    #             .sort_values(["plant_id_eia_l", "technology_description_r", "lcoe"])
    #         )
    #     return self._dfs["re_dist"]

    # def close_re_ba_by_gen(self, ba_code, drop_lcoe_na):
    #     plants = (
    #         self.all_modelable_bas()[ba_code]
    #         .query("plant_id_eia > 0")
    #         .capacity_mw.dropna()
    #     )
    #     t_dict = []
    #     for (pid, gen), cap in plants.items():
    #         try:
    #             t_dict.append(
    #                 self.close_re(pid, drop_lcoe_na=drop_lcoe_na).assign(
    #                     fos_id=pid, fos_gen=gen, capacity_mw=cap
    #                 )
    #             )
    #         except Exception as exc:
    #             if ba_code != "Alaska":
    #                 LOGGER.error("%s %s %r", ba_code, cap, exc)
    #     cols = list(
    #         dict.fromkeys(("fos_id", "fos_gen", "capacity_mw"))
    #         | dict.fromkeys(t_dict[0])
    #     )
    #     return (
    #         pd.concat(t_dict)[cols]
    #         .assign(
    #             capacity_for_weight=lambda x: x.groupby(
    #                 ["technology_description", "fos_id", "fos_gen"]
    #             ).capacity_mw.transform("sum")
    #             / x.groupby(["technology_description"]).capacity_mw.transform("sum"),
    #             ba_weight=lambda x: x.capacity_for_weight * x.weight,
    #             ba_distance=lambda x: x.ba_weight * x.distance,
    #         )
    #         # .drop(columns="level_3")
    #         .sort_values(["technology_description", "plant_id_eia", "ba_weight"])
    #         .reset_index(drop=True)
    #     )

    # def close_re_meta(self, drop_lcoe_na):
    #     if "re_meta" not in self._dfs:
    #         # try:
    #         #     self._dfs["re_meta"] = pd.read_parquet(CACHE_PATH / "re_meta.parquet")
    #         # except FileNotFoundError:
    #         #     with logging_redirect_tqdm():
    #         out = pd.concat(
    #             (
    #                 self.close_re_ba_by_gen(k, drop_lcoe_na=drop_lcoe_na).assign(
    #                     ba_code=k
    #                 )
    #                 for k in tqdm(self.all_modelable_bas(), desc="close_re_meta")
    #             ),
    #             axis=0,
    #         )
    #         # out.to_parquet(CACHE_PATH / "re_meta.parquet")
    #         self._dfs["re_meta"] = out
    #     return self._dfs["re_meta"]

    # def close_re(self, plant_id, drop_lcoe_na):
    #     """find re resources close to ``plant_id``, trying 250, 500, and then 1000 km
    #     until solar and wind are found, returns the resources and their weights"""
    #     # if plant_id not in self.gens_to_model.plant_id_eia:
    #     #     raise RuntimeError(f"{plant_id} not in set of fossil plants")
    #     full = self.re_dist().query("plant_id_eia_l == @plant_id")
    #     if drop_lcoe_na:
    #         full = full[full.lcoe.notna()]
    #     close = (
    #         full.query("distance <= 250")
    #         .copy()
    #         .pipe(self._close_re_helper, dist=250, max_num=self.max_re_sites)
    #     )
    #     techs = close.technology_description.unique()
    #     if "solar" in techs and ("onshore_wind" in techs or "offshore_wind" in techs):
    #         return close
    #     missing_techs = list({"solar", "onshore_wind", "offshore_wind"} - set(techs))
    #     missing = (
    #         full.query("technology_description_r in @missing_techs & distance <= 500")
    #         .copy()
    #         .pipe(self._close_re_helper, dist=500, max_num=self.max_re_sites)
    #     )
    #     if missing.empty:
    #         missing = full.query("technology_description_r in @missing_techs")
    #         min_dist = missing.distance.min()
    #         missing = (
    #             missing.query("distance == @min_dist")
    #             .copy()
    #             .pipe(self._close_re_helper, dist=min_dist, max_num=self.max_re_sites)
    #         )
    #         if missing.empty:
    #             raise NoREData(f"{plant_id} has no {missing_techs} within 1000 km")
    #         LOGGER.info(
    #             "%s closest %s is %s km",
    #             plant_id,
    #             missing.technology_description[0],
    #             min_dist,
    #         )
    #     if not all(
    #         [close[close.latitude.isna()].empty, missing[missing.latitude.isna()].empty]
    #     ):
    #         raise AssertionError(f"missing RE latitudes for {plant_id}")
    #     return (
    #         pd.concat([close, missing], axis=0, ignore_index=True)
    #         .groupby(["technology_description", "plant_id_eia"])
    #         .agg(
    #             {
    #                 "distance": "first",
    #                 "weight": np.sum,
    #                 "latitude": "first",
    #                 "longitude": "first",
    #                 "state": "first",
    #                 "lcoe": "first",
    #             }
    #         )
    #         .sort_index()
    #         .reset_index()
    #     )

    # def all_re_dists(self, plant_ids=None) -> pd.DataFrame:
    #     """the weighted distance of wind and solar for each fossil plant in ``plant_ids``"""
    #     if plant_ids is None:
    #         plant_ids = sorted({v["plant_id"] for v in self.all_modelable.values()})
    #     dfs = {}
    #     for pid in plant_ids:
    #         try:
    #             dfs.update({pid: self.close_re(pid, drop_lcoe_na=True)})
    #         except NoREData as exc:
    #             LOGGER.error("%r", exc)
    #     return (
    #         pd.concat(dfs, axis=0, names=["plant_id_eia"])
    #         .drop(["plant_id_eia"], axis=1)
    #         .assign(wd=lambda d: d.weight * d.distance)
    #         .reset_index()
    #         .groupby(["plant_id_eia", "technology_description"])
    #         .agg({"wd": np.sum})
    #     )

    @staticmethod
    def _close_re_helper(close, dist, max_num):
        if max_num is not None:
            close = close.loc[
                [
                    y
                    for x in ("solar", "onshore_wind", "offshore_wind")
                    for y in close[close.technology_description_r == x].head(max_num).index
                ],
                :,
            ]
        return (
            close.assign(
                lw=lambda x: 1 / x.lcoe,
                weight=lambda x: x.lw
                / x.groupby("technology_description_r").lw.transform("sum"),
            )[
                [
                    "plant_id_eia_r",
                    "technology_description_r",
                    "distance",
                    "weight",
                    "latitude_r",
                    "longitude_r",
                    "state_r",
                    "lcoe",
                ]
            ]
            .reset_index(drop=True)
            .rename(
                columns={
                    "plant_id_eia_r": "plant_id_eia",
                    "technology_description_r": "technology_description",
                    "latitude_r": "latitude",
                    "longitude_r": "longitude",
                    "state_r": "state",
                }
            )
        )

    # def gens_by_cr_cat(self):
    #     df860 = (
    #         self.pdl.gens_eia860()
    #         .merge(
    #             self.pdl.plants_eia860()[
    #                 ["plant_id_eia", "report_date", "balancing_authority_code_eia"]
    #             ],
    #             on=["plant_id_eia", "report_date"],
    #             validate="m:1",
    #         )
    #         .assign(fuel_group=lambda x: x.energy_source_code_1.map(FUEL_GROUP_MAP))
    #     )
    #     df923 = (
    #         self.pdl.gen_fuel_by_generator_eia923()
    #         .groupby(
    #             [
    #                 "plant_id_eia",
    #                 "generator_id",
    #                 pd.Grouper(key="report_date", freq="YS"),
    #             ]
    #         )
    #         .net_generation_mwh.sum()
    #         .reset_index()
    #     )
    #     df923 = df923[df923.report_date.dt.year == 2021]
    #     return (
    #         # df860[
    #         #     df860.prime_mover_code.isin(("CC", "CS", "CA", "CT", "GT", "ST", "IC"))
    #         #     & df860.fuel_group.isin(self.fos_fuels)
    #         #     & df860.operational_status.isin(["existing", "retired"])
    #         #     & (df860.report_date.dt.year == 2021)
    #         # ]
    #         self.gens[self.gens.category.str.contains("fossil")]
    #         .drop(columns=["category"])
    #         .merge(
    #             df923,
    #             on=["plant_id_eia", "generator_id"],
    #             validate="1:1",
    #             how="left",
    #             suffixes=("", "_923"),
    #         )
    #         .assign(
    #             # cf=lambda x:  x.net_generation_mwh / (x.capacity_mw * 8760),
    #             category=lambda x: np.where(
    #                 x.operational_status == "retired",
    #                 "retired",
    #                 np.where(
    #                     x.retirement_date.notnull(),
    #                     "retiring",
    #                     np.where(
    #                         (x.net_generation_mwh / (x.capacity_mw * 8760)) < 0.25,
    #                         "low_cf",
    #                         "high_cf",
    #                     ),
    #                 ),
    #             ),
    #             ba_code=lambda x: x.final_ba_code,
    #         )[
    #             [
    #                 "ba_code",
    #                 "balancing_authority_code_eia",
    #                 "plant_id_eia",
    #                 "generator_id",
    #                 "category",
    #                 "capacity_mw",
    #                 "utility_id_eia",
    #                 "utility_name_eia",
    #             ]
    #         ]
    #     )


@lru_cache
def load_ec():
    files = (
        (
            "Coal_Closure_Energy_Communities_SHP_2023v2/Coal_Closure_Energy_Communities_SHP_2023v2.shp",
            "coal_mine_closure_or_plant_retirement",
            lambda x: x,
        ),
        (
            "MSA_NMSA_FEE_EC_Status_SHP_2023v2/MSA_NMSA_FEE_EC_Status_2023v2.shp",
            "fossil_employment",
            lambda x: x.query('EC_qual_st == "Yes"'),
        ),
    )
    for file_, *_ in files:
        parent, *_ = file_.partition("/")
        if not (file := USER_DATA_PATH / parent).exists():
            if not file.with_suffix(".zip").exists():
                download(PATIO_DATA_AZURE_URLS[parent], file.with_suffix(".zip"))
            shutil.unpack_archive(file.with_suffix(".zip"), USER_DATA_PATH)
            file.with_suffix(".zip").unlink()
            if (mf := USER_DATA_PATH / "__MACOSX").exists():
                shutil.rmtree(mf)
    return pd.concat(
        gpd.read_file(USER_DATA_PATH / file).assign(criteria=criteria).pipe(func)
        for file, criteria, func in files
    )


def add_ec_flag(
    df,
    key_col: str = "plant_id_eia",
    lat_lon_cols: tuple[str, str] = ("latitude", "longitude"),
    ec_col: str = "energy_community",
):
    lat, lon = lat_lon_cols
    sites = df.groupby(key_col, as_index=False)[[lat, lon]].first()
    gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites[lon], sites[lat]))

    # Perform spatial join to match points and polygons
    # - identify which sites are in energy communities
    in_polys = gpd.tools.sjoin(
        gdf,
        gpd.GeoDataFrame(load_ec(), geometry="geometry", crs=gdf.crs),
        predicate="intersects",
        how="inner",
    )
    df[ec_col] = df[key_col].isin(in_polys[key_col])
    return df


def cost_comparison():
    def problem(df):
        return (
            (df.om_per_mwh.isna() | (df.om_per_mwh < 5))
            | (df.fom.isna() | (df.fom < 1) | (df.fom > 200))
            | (df.vom.isna() | (df.vom < 0) | (df.vom > 50))
            | (df.som.isna() | (df.som < 0) | (df.som > 50))
        )

    # if not (pudl_sql := Path(get_pudl_sql_url().removeprefix("sqlite:///"))).exists():
    #     if not pudl_sql.parent.exists():
    #         pudl_sql.parent.mkdir(parents=True)
    #     get("az://raw-data/pudl.sqlite.gz", f"{pudl_sql}.gz")
    #     ungzip(Path(f"{pudl_sql}.gz"))
    from etoolbox.utils.pudl import PretendPudlTablCore

    pdl = DataZip.load(
        user_cache_path("gencost", "rmi") / "pdltbl",
        klass=PretendPudlTablCore,
    )
    gens = (
        pdl.gens_eia860m()
        .query("report_date == '2023-03-01'")
        .pipe(fix_cc_in_prime)
        .rename(columns={"energy_source_code_1": "energy_source_code_860m"})
        .assign(
            fuel_group=lambda x: x.energy_source_code_860m.map(FUEL_GROUP_MAP),
            operating_date=lambda x: x.generator_operating_date.fillna(
                x.current_planned_generator_operating_date
            ),
            retirement_date=lambda x: x.generator_retirement_date.fillna(
                x.planned_retirement_date
            ),
        )
        .drop_duplicates(subset=["plant_id_eia", "generator_id"], keep="first")
    )
    cols = [
        "plant_id_eia",
        "generator_id",
        "technology_description",
        "capacity_mw",
        "datetime",
        "utility_id_eia",
    ]
    ix_cols = ["plant_id_eia", "generator_id", "datetime"]
    comp_cols = [
        "num_issue",
        "technology_description",
        "prime_mover",
        "fuel_group",
        "operating_date",
        "retirement_date",
        "capacity_mw",
        "som",
        "vom",
        "fom",
        "om_per_mwh",
    ]
    c_cfl = pd.read_parquet(ROOT_PATH / "r_data/python_inputs_data.parquet")

    gencost = (
        pd.read_parquet(ROOT_PATH / "patio/package_data/epd_w_vom_fom_som.parquet")
        .rename(columns={"report_date": "datetime"})
        .query("technology_description in @FOSSIL_TECH & datetime.dt.year >= 2008")
        .merge(
            gens[["plant_id_eia", "generator_id", "operating_date", "retirement_date"]],
            on=["plant_id_eia", "generator_id"],
            how="left",
        )
        .assign(
            fuel_group=lambda x: x.filter(like="fraction")
            .idxmax(axis=1)
            .str.split("_fraction", expand=True)[0],
            num_issue=lambda x: problem(x),
        )
    )
    patio_cost = (
        c_cfl.rename(
            columns={
                "prime_mover_code": "prime_mover",
                "Technology_1": "technology_description",
                "operational_capacity_in_report_year": "capacity_mw",
                "real_opex_per_kW_start": "som",
                "real_variable_opex_per_MWh_est": "vom",
                "real_fixed_opex_per_kW_no_starts_est": "fom",
                "Operating_Date": "operating_date",
            }
        )
        .query("technology_description in @FOSSIL_TECH")
        .assign(
            day=1,
            fuel_group=lambda x: x.plant_prime_fuel_1.str.split("|", expand=True)[1],
            retirement_date=lambda x: pd.to_datetime(
                x.rename(columns={"Retirement_Month": "month", "Retirement_Year": "year"})[
                    ["year", "month", "day"]
                ]
            ),
            om_per_mwh=lambda x: (
                (x.som * (x.starts_adj / x.capacity_adj) + x.fom) * 1000 * x.capacity
            )
            / x.total_fossil_mwh
            + x.vom,
            datetime=lambda x: pd.to_datetime(x.report_year, format="%Y"),
            num_issue=lambda x: problem(x),
        )
        .drop_duplicates(subset=cols)
        .astype({"plant_id_eia": int})
    )
    compm = (
        patio_cost[ix_cols + comp_cols]
        .merge(
            gencost[ix_cols + comp_cols],
            on=ix_cols,
            how="outer",
            suffixes=("_patio", "_gencost"),
            indicator=True,
        )
        .sort_values(ix_cols)
        .replace({"_merge": {"right_only": "gencost_only", "left_only": "patio_only"}})
    )
    return compm.merge(
        compm[["plant_id_eia", "generator_id", "_merge"]]
        .drop_duplicates()
        .query("_merge == 'patio_only'")
        .assign(missing_gencost=True)[["plant_id_eia", "generator_id", "issue_in_gencost"]],
        on=["plant_id_eia", "generator_id"],
        how="left",
    ).fillna({"missing_gencost": False})[
        [
            *ix_cols,
            "_merge",
            "missing_gencost",
            *[x + y for x in comp_cols for y in ("_patio", "_gencost")],
        ]
    ]


def cost_helper(df, kind):
    cost_cols = [
        "plant_id_eia",
        "generator_id",
        "Technology_1",
        "datetime",
        "utility_id_eia",
        "operational_capacity_in_report_year",
        "real_fuel_costs_per_MWh",
        "real_opex_per_kW_start",
        "real_variable_opex_per_MWh_est",
        "real_fixed_opex_per_kW_no_starts_est",
        "total_fuel_mmbtu",
        "total_fossil_mwh",
        # "emissions_intensity",
    ]
    kind = kind + "_" if kind else ""
    return (
        df.astype({"plant_id_eia": "Int64", "utility_id_eia": int})
        .assign(datetime=lambda x: pd.to_datetime(x.report_year, format="%Y"))[cost_cols]
        .rename(
            columns={
                "Technology_1": "technology_description",
                "operational_capacity_in_report_year": "capacity_mw",
                "real_fuel_costs_per_MWh": f"{kind}fuel_per_mwh",
                "real_opex_per_kW_start": f"{kind}start_per_kw",
                "real_variable_opex_per_MWh_est": f"{kind}vom_per_mwh",
                "real_fixed_opex_per_kW_no_starts_est": f"{kind}fom_per_kw",
                "total_fuel_mmbtu": f"{kind}fuel_mmbtu",
                "total_fossil_mwh": f"{kind}fossil_mwh",
                # "emissions_intensity": f"{kind}_emissions_intensity",
            }
        )
    )


def add_plant_role(df: pd.DataFrame) -> pd.DataFrame:
    role_mapper = pd.DataFrame(
        [
            ("IC", "petroleum", "peaker"),
            ("ST", "natural_gas", "peaker"),
            ("ST", "coal", "base"),
            ("CC", "natural_gas", "mid"),
            ("GT", "natural_gas", "peaker"),
            ("GT", "petroleum", "peaker"),
            ("ST", "", "peaker"),
            ("GT", "", "peaker"),
            ("IC", "natural_gas", "mid"),
            ("ST", "petroleum", "peaker"),
            ("CC", "coal", "base"),
            ("IC", "", "peaker"),
            ("ST", "other_gas", "peaker"),
            ("GT", "other_gas", "peaker"),
            ("ST", "petroleum_coke", "peaker"),
            ("CC", "", "mid"),
            ("CC", "petroleum", "peaker"),
            ("CC", "other_gas", "peaker"),
            ("IC", "other_gas", "peaker"),
        ],
        columns=["prime_mover", "fuel_group", "plant_role"],
    )
    return df.merge(
        role_mapper,
        on=["prime_mover", "fuel_group"],
        how="left",
        validate="m:1",
    )


def master_unit_list(xl=None):
    try:
        return pd.read_parquet(ROOT_PATH / "patio_data/static_unit_list.parquet")
    except FileNotFoundError:
        sul_map = {
            "Utility_ID": "utility_id",
            "Utility_Name": "utility_name",
            "Plant_Code": "plant_id_eia",
            "Generator_ID": "generator_id",
            "Percent_Owned": "percent_owned",
            "Plant_Name": "plant_name_eia",
            "State": "state",
            "Technology": "technology_description",
            "Prime_Mover": "prime_mover_code",
            "Balancing_Authority_Code": "balancing_authority_code",
            "RTO": "rto",
            "RTO_FERC": "rto_ferc",
            "Nameplate_Capacity": "capacity_mw",
            "Status": "status",
            "Energy_Source_1": "fuel_code",
        }
        sul_dtypes = {
            "utility_id": "Int64",
            "utility_name": str,
            "plant_id_eia": "Int64",
            "generator_id": str,
            "percent_owned": float,
            "plant_name_eia": str,
            "state": str,
            "technology_description": str,
            "prime_mover_code": str,
            "balancing_authority_code": str,
            "rto": str,
            "rto_ferc": str,
            "capacity_mw": float,
            "status": str,
            "fuel_code": str,
        }
        return (
            xl.parse(sheet_name="Master Unit List", header=3)[list(sul_map)]
            .rename(columns=sul_map)
            .astype(sul_dtypes)
            .query(
                "prime_mover_code in @FOSSIL_PRIME_MOVER_MAP & fuel_code in @UDAY_FOSSIL_FUEL_MAP"
            )
            .replace(
                {
                    "prime_mover_code": FOSSIL_PRIME_MOVER_MAP,
                    "fuel_code": UDAY_FOSSIL_FUEL_MAP,
                }
            )
        )


def read_ferc860():
    cache_file = user_cache_path("patio", "rmi") / "860_FERC_matching_cost_regressions.parquet"
    try:
        cost = pd.read_parquet(cache_file)
    except FileNotFoundError:
        cost = pd.read_excel(
            ROOT_PATH / "r_data/860_FERC_matching_cost_regressions - values.xlsx",
            header=0,
        )
        if not cache_file.parent.exists():
            cache_file.parent.mkdir()
        cost.to_parquet(cache_file)
    return cost


def irp_data(asset_data):
    # from hub.irp_data import IrpData
    # irp_data = IrpData()
    # df = irp_data.create_irp_resource_data()
    # ids = (814, 24211, 11208, 11241, 13478, 12685, 5701, 18642, 14354, 19876, 30151, 17543)
    # ids_ = tuple(map(str, ids))
    # s = df.query("utility_id_irp in @ids_")
    # s.to_parquet("/Users/aengel/PycharmProjects/patio_explore/irp_data.parquet")

    mapper_id = {
        "1307": "SWPP",
        "1692": "MISO",
        "5580": "PJM",
        "7570": "MISO",
        "9267": "MISO",
        "12710": "SWPP",
        "21554": "SEC",
        "40211": "MISO",
        "814": "ETR",
        "24211": "TEPC",
        "11241": "ETR",
        "13478": "ETR",
        "12685": "ETR",
        "5701": "EPE",
        "18642": "TVA",
        "14354": "PAC",
        "19876": "186",
        "30151": "WACM",  # TSGT
        "17543": "SC",
    }
    mapper_tech = {
        "Battery Storage": "Batteries",
        "Geothermal": "Geothermal",
        "Hydro": "Conventional Hydroelectric",
        "NGCC": "Natural Gas Fired Combined Cycle",
        "NGCT": "Natural Gas Fired Combustion Turbine",
        "Nuclear": "Nuclear",
        "Offshore Wind": "Offshore Wind Turbine",
        "Pumped Storage": "Hydroelectric Pumped Storage",
        "RICE": "Natural Gas Internal Combustion Engine",
        "Solar": "Solar Photovoltaic",
        "Unspecified": "All Other",
        "Wind": "Onshore Wind Turbine",
    }
    add = (
        asset_data.gens.query(
            "operational_status == 'proposed' & final_ba_code in @mapper_id.values()"
        )
        .copy()
        .groupby(
            [
                "final_ba_code",
                "technology_description",
                pd.Grouper(key="operating_date", freq="YS"),
            ]
        )
        .capacity_mw.sum()
        .reset_index()
        .rename(columns={"capacity_mw": "capacity_860"})
    )
    irp = (
        pd.read_parquet(Path.home() / "patio_data/irp_raw.parquet")
        .query("utility_id_irp in @mapper_id")
        .assign(
            final_ba_code=lambda x: x.utility_id_irp.map(mapper_id),
            operating_date=lambda x: pd.to_datetime(x.year, format="%Y"),
            technology_description=lambda x: x.technology_irp.map(mapper_tech),
        )
        .query("capacity_change > 0 & operating_date.notna()")
        .groupby(["final_ba_code", "technology_description", "operating_date"])
        .capacity_change.sum()
        .reset_index()
    )
    adds = (
        add.merge(
            irp.query("operating_date > '2022'"),
            on=["final_ba_code", "technology_description", "operating_date"],
            validate="1:1",
            how="outer",
        )
        .query("technology_description != 'All Other'")
        .sort_values(["final_ba_code", "technology_description", "operating_date"])
        .assign(
            cum_cap_irp=lambda x: x.groupby(
                ["final_ba_code", "technology_description"]
            ).capacity_change.transform("cumsum"),
            cum_cap_860=lambda x: x.fillna({"capacity_860": 0.0})
            .groupby(["final_ba_code", "technology_description"])
            .capacity_860.transform("cumsum"),
            cum_combined=lambda x: np.maximum(x.cum_cap_irp.fillna(0.0) - x.cum_cap_860, 0.0),
            capacity_mw=lambda x: np.where(
                (x.final_ba_code != np.roll(x.final_ba_code, shift=1))
                | (x.technology_description != np.roll(x.technology_description, shift=1)),
                x.cum_combined,
                x.cum_combined - np.roll(x.cum_combined, shift=1),
            ),
            plant_id_eia=lambda x: -(4 + np.arange(x.shape[0])),
        )
        .query("cum_combined > 0 & operating_date < '2040'")[
            [
                "final_ba_code",
                "plant_id_eia",
                "technology_description",
                "operating_date",
                "capacity_mw",
            ]
        ]
    )

    out = (
        adds.query("technology_description != 'All Other'")
        .merge(
            adds.query("technology_description == 'Batteries'"),
            on=["final_ba_code", "operating_date"],
            suffixes=(None, "_es"),
            how="left",
        )
        .assign(
            plant_id_eia=lambda x: x.plant_id_eia.mask(
                (x.technology_description == "Solar Photovoltaic")
                & (x.plant_id_eia_es.notna()),
                x.plant_id_eia_es,
            ),
            generator_id=lambda x: x.technology_description.map({"Batteries": "es"}).fillna(
                "1"
            ),
            balancing_authority_code_eia=lambda x: x.final_ba_code.replace(
                {"186": "PJM", "ETR": "MISO", "PAC": "PACE"}
            ),
            operational_status="proposed",
            source="irp",
            prime_mover=lambda x: x.technology_description.map(
                {
                    "Batteries": "BA",
                    "Offshore Wind Turbine": "WS",
                    "Solar Photovoltaic": "PV",
                    "Onshore Wind Turbine": "WT",
                    "Natural Gas Fired Combustion Turbine": "GT",
                    "Natural Gas Fired Combined Cycle": "CC",
                    "Nuclear": "ST",
                }
            ),
            fuel_group=lambda x: x.technology_description.map(
                {
                    "Batteries": "other",
                    "Offshore Wind Turbine": "renew",
                    "Solar Photovoltaic": "renew",
                    "Onshore Wind Turbine": "renew",
                    "Natural Gas Fired Combustion Turbine": "natural_gas",
                    "Natural Gas Fired Combined Cycle": "natural_gas",
                    "Nuclear": "nuclear",
                }
            ),
            category=lambda x: x.operational_status.str.cat(
                np.where(
                    x.prime_mover.isin(("CC", "GT", "ST", "IC"))
                    & x.fuel_group.isin(["natural_gas"]),
                    "fossil",
                    np.where(
                        x.technology_description.isin(
                            (
                                "Solar Photovoltaic",
                                "Batteries",
                                "Geothermal",
                                "Onshore Wind Turbine",
                                "Nuclear",
                                "Hydroelectric Pumped Storage",
                                "Offshore Wind Turbine",
                            )
                        ),
                        "clean",
                        "other",
                    ),
                ),
                sep="_",
            ),
            retirement_date=pd.NaT,
            ramp_rate=lambda x: x.capacity_mw,
            duration_hrs=lambda x: x.technology_description.map(
                {"Batteries": 4, "Hydroelectric Pumped Storage": 12}
            ),
        )
    )
    out = out[
        [
            "final_ba_code",
            "plant_id_eia",
            "generator_id",
            "technology_description",
            "operating_date",
            "capacity_mw",
            "balancing_authority_code_eia",
            "operational_status",
            "source",
            "prime_mover",
            "fuel_group",
            "category",
            "retirement_date",
            "ramp_rate",
            "duration_hrs",
        ]
    ]
    out.to_parquet(Path.home() / "patio_data/irp2.parquet")
    return out
