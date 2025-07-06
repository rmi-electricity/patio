from __future__ import annotations

import logging
from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cached_property, lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_STORED
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import pyarrow as pa
from etoolbox.datazip import DataZip
from etoolbox.utils.cloud import get, put
from pandas.util import hash_pandas_object
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from patio.constants import (
    MTDF,
    PATIO_DATA_AZURE_URLS,
    PATIO_DATA_RELEASE,
    RE_TECH,
    RE_TECH_R,
    ROOT_PATH,
)
from patio.data.asset_data import (  # noqa: TC001
    CACHE_PATH,
    USER_DATA_PATH,
    AssetData,
    add_ec_flag,
)
from patio.exceptions import (
    NoCEMS,
    NoEligibleCleanRepowering,
    NoMatchingPlantsAndCEMS,
    NoMaxRE,
    NoNonZeroCEMS,
    NoREData,
    PatioData,
)
from patio.helpers import (
    bbb_path,
    df_query,
    generate_projection_from_historical,
    generate_projection_from_historical_pl,
    get_year_map,
    read_named_range,
    round_coordinates,
    seconds_since_touch,
)

__all__ = ["ProfileData"]

from patio.package_data import PACKAGE_DATA_PATH

LOGGER = logging.getLogger("patio")

# if not (PATIO_DOC_PATH / "data").exists():
#     (PATIO_DOC_PATH / "data").mkdir(parents=True)

pl.enable_string_cache()
BAs_WITH_MISSING_DATA = ()
PLANT_RE_MAX = {
    3: {"solar": 1672, "onshore_wind": 0},
    26: {"solar": 1485, "onshore_wind": 1347},
    59: {"solar": 184, "onshore_wind": 99},
    60: {"solar": 566, "onshore_wind": 309},
    108: {"solar": 643, "onshore_wind": 405},
    113: {"solar": 742, "onshore_wind": 1035},
    130: {"solar": 3131, "onshore_wind": 0},
    136: {"solar": 2851, "onshore_wind": 0},
    160: {"solar": 471, "onshore_wind": 770},
    165: {"solar": 519, "onshore_wind": 303},
    298: {"solar": 2365, "onshore_wind": 1618},
    470: {"solar": 3203, "onshore_wind": 3651},
    525: {"solar": 1027, "onshore_wind": 1461},
    564: {"solar": 1899, "onshore_wind": 0},
    594: {"solar": 137, "onshore_wind": 0},
    602: {"solar": 1187, "onshore_wind": 0},
    628: {"solar": 2094, "onshore_wind": 0},
    645: {"solar": 678, "onshore_wind": 0},
    667: {"solar": 707, "onshore_wind": 0},
    703: {"solar": 4090, "onshore_wind": 0},
    856: {"solar": 1254, "onshore_wind": 759},
    876: {"solar": 1795, "onshore_wind": 1069},
    879: {"solar": 1248, "onshore_wind": 767},
    883: {"solar": 1005, "onshore_wind": 0},
    884: {"solar": 335, "onshore_wind": 200},
    887: {"solar": 2268, "onshore_wind": 1515},
    889: {"solar": 3123, "onshore_wind": 2042},
    963: {"solar": 566, "onshore_wind": 336},
    976: {"solar": 284, "onshore_wind": 180},
    983: {"solar": 2269, "onshore_wind": 0},
    994: {"solar": 3521, "onshore_wind": 0},
    997: {"solar": 692, "onshore_wind": 0},
    1001: {"solar": 2126, "onshore_wind": 1278},
    1012: {"solar": 956, "onshore_wind": 636},
    1040: {"solar": 13, "onshore_wind": 8},
    1047: {"solar": 283, "onshore_wind": 181},
    1073: {"solar": 23, "onshore_wind": 13},
    1082: {"solar": 2874, "onshore_wind": 1599},
    1091: {"solar": 742, "onshore_wind": 436},
    1104: {"solar": 484, "onshore_wind": 291},
    1167: {"solar": 448, "onshore_wind": 266},
    1241: {"solar": 2829, "onshore_wind": 1590},
    1250: {"solar": 827, "onshore_wind": 466},
    1355: {"solar": 417, "onshore_wind": 296},
    1356: {"solar": 4760, "onshore_wind": 0},
    1364: {"solar": 3335, "onshore_wind": 0},
    1379: {"solar": 2684, "onshore_wind": 0},
    1384: {"solar": 210, "onshore_wind": 180},
    1393: {"solar": 652, "onshore_wind": 503},
    1554: {"solar": 55, "onshore_wind": 0},
    1573: {"solar": 659, "onshore_wind": 0},
    1702: {"solar": 1408, "onshore_wind": 0},
    1710: {"solar": 3943, "onshore_wind": 0},
    1733: {"solar": 7018, "onshore_wind": 0},
    1743: {"solar": 1534, "onshore_wind": 0},
    1745: {"solar": 525, "onshore_wind": 0},
    1832: {"solar": 333, "onshore_wind": 0},
    1893: {"solar": 2533, "onshore_wind": 0},
    1915: {"solar": 651, "onshore_wind": 0},
    2079: {"solar": 1185, "onshore_wind": 0},
    2103: {"solar": 7676, "onshore_wind": 0},
    2104: {"solar": 232, "onshore_wind": 158},
    2107: {"solar": 1779, "onshore_wind": 1126},
    2167: {"solar": 2484, "onshore_wind": 1554},
    2168: {"solar": 3931, "onshore_wind": 2298},
    2240: {"solar": 214, "onshore_wind": 120},
    2277: {"solar": 374, "onshore_wind": 197},
    2291: {"solar": 737, "onshore_wind": 415},
    2364: {"solar": 145, "onshore_wind": 89},
    2442: {"solar": 3070, "onshore_wind": 3875},
    2451: {"solar": 1847, "onshore_wind": 2161},
    2712: {"solar": 2464, "onshore_wind": 0},
    2718: {"solar": 39, "onshore_wind": 36},
    2721: {"solar": 2019, "onshore_wind": 0},
    2727: {"solar": 2941, "onshore_wind": 0},
    2790: {"solar": 241, "onshore_wind": 136},
    2817: {"solar": 1249, "onshore_wind": 680},
    2823: {"solar": 2154, "onshore_wind": 1123},
    2828: {"solar": 4897, "onshore_wind": 0},
    2832: {"solar": 3052, "onshore_wind": 0},
    2836: {"solar": 412, "onshore_wind": 0},
    2866: {"solar": 2296, "onshore_wind": 0},
    2876: {"solar": 2386, "onshore_wind": 0},
    2914: {"solar": 28, "onshore_wind": 18},
    2935: {"solar": 7, "onshore_wind": 4},
    2936: {"solar": 0, "onshore_wind": 0},
    2952: {"solar": 1188, "onshore_wind": 729},
    2963: {"solar": 995, "onshore_wind": 576},
    3118: {"solar": 3704, "onshore_wind": 0},
    3122: {"solar": 2267, "onshore_wind": 0},
    3130: {"solar": 1457, "onshore_wind": 764},
    3136: {"solar": 3755, "onshore_wind": 0},
    3140: {"solar": 1267, "onshore_wind": 0},
    3149: {"solar": 644, "onshore_wind": 427},
    3297: {"solar": 642, "onshore_wind": 0},
    3298: {"solar": 1035, "onshore_wind": 0},
    3396: {"solar": 361, "onshore_wind": 291},
    3399: {"solar": 4568, "onshore_wind": 0},
    3403: {"solar": 1752, "onshore_wind": 0},
    3407: {"solar": 1224, "onshore_wind": 1032},
    3470: {"solar": 5436, "onshore_wind": 0},
    3797: {"solar": 512, "onshore_wind": 362},
    3845: {"solar": 1871, "onshore_wind": 1127},
    3935: {"solar": 5884, "onshore_wind": 0},
    3943: {"solar": 2602, "onshore_wind": 0},
    3944: {"solar": 5622, "onshore_wind": 0},
    3948: {"solar": 2271, "onshore_wind": 0},
    3954: {"solar": 2686, "onshore_wind": 0},
    4041: {"solar": 2142, "onshore_wind": 0},
    4050: {"solar": 1104, "onshore_wind": 0},
    4078: {"solar": 2254, "onshore_wind": 0},
    4158: {"solar": 1587, "onshore_wind": 931},
    4162: {"solar": 972, "onshore_wind": 712},
    4271: {"solar": 859, "onshore_wind": 510},
    6002: {"solar": 8939, "onshore_wind": 0},
    6004: {"solar": 3655, "onshore_wind": 0},
    6009: {"solar": 3057, "onshore_wind": 0},
    6017: {"solar": 1485, "onshore_wind": 916},
    6018: {"solar": 1148, "onshore_wind": 0},
    6019: {"solar": 2091, "onshore_wind": 0},
    6021: {"solar": 3022, "onshore_wind": 3479},
    6030: {"solar": 4326, "onshore_wind": 0},
    6034: {"solar": 3282, "onshore_wind": 0},
    6041: {"solar": 3858, "onshore_wind": 0},
    6052: {"solar": 483, "onshore_wind": 460},
    6055: {"solar": 1091, "onshore_wind": 884},
    6064: {"solar": 561, "onshore_wind": 331},
    6065: {"solar": 3498, "onshore_wind": 0},
    6068: {"solar": 3650, "onshore_wind": 2005},
    6071: {"solar": 3705, "onshore_wind": 0},
    6073: {"solar": 1212, "onshore_wind": 0},
    6077: {"solar": 3007, "onshore_wind": 1865},
    6085: {"solar": 1105, "onshore_wind": 674},
    6090: {"solar": 4618, "onshore_wind": 0},
    6095: {"solar": 1355, "onshore_wind": 775},
    6096: {"solar": 3161, "onshore_wind": 1769},
    6098: {"solar": 775, "onshore_wind": 439},
    6101: {"solar": 747, "onshore_wind": 460},
    6113: {"solar": 4174, "onshore_wind": 0},
    6137: {"solar": 1210, "onshore_wind": 794},
    6138: {"solar": 1199, "onshore_wind": 696},
    6139: {"solar": 1868, "onshore_wind": 0},
    6146: {"solar": 5612, "onshore_wind": 0},
    6155: {"solar": 3767, "onshore_wind": 0},
    6165: {"solar": 3799, "onshore_wind": 0},
    6166: {"solar": 2168, "onshore_wind": 0},
    6177: {"solar": 1296, "onshore_wind": 1829},
    6178: {"solar": 1418, "onshore_wind": 1057},
    6179: {"solar": 4278, "onshore_wind": 0},
    6180: {"solar": 5116, "onshore_wind": 0},
    6183: {"solar": 776, "onshore_wind": 570},
    6190: {"solar": 957, "onshore_wind": 755},
    6193: {"solar": 1909, "onshore_wind": 1172},
    6194: {"solar": 948, "onshore_wind": 632},
    6195: {"solar": 1057, "onshore_wind": 646},
    6204: {"solar": 3607, "onshore_wind": 2591},
    6213: {"solar": 2380, "onshore_wind": 1499},
    6248: {"solar": 1191, "onshore_wind": 1114},
    6249: {"solar": 1313, "onshore_wind": 0},
    6250: {"solar": 582, "onshore_wind": 439},
    6254: {"solar": 1916, "onshore_wind": 1098},
    6257: {"solar": 3609, "onshore_wind": 0},
    6264: {"solar": 3394, "onshore_wind": 0},
    6469: {"solar": 2527, "onshore_wind": 1378},
    6481: {"solar": 3114, "onshore_wind": 3939},
    6639: {"solar": 1046, "onshore_wind": 716},
    6641: {"solar": 1781, "onshore_wind": 1247},
    6664: {"solar": 1986, "onshore_wind": 1182},
    6761: {"solar": 694, "onshore_wind": 582},
    6768: {"solar": 761, "onshore_wind": 487},
    6772: {"solar": 354, "onshore_wind": 258},
    6823: {"solar": 1338, "onshore_wind": 955},
    7030: {"solar": 970, "onshore_wind": 681},
    7097: {"solar": 2988, "onshore_wind": 0},
    7213: {"solar": 417, "onshore_wind": 319},
    7343: {"solar": 631, "onshore_wind": 371},
    7504: {"solar": 268, "onshore_wind": 165},
    7790: {"solar": 1229, "onshore_wind": 1933},
    7902: {"solar": 1161, "onshore_wind": 0},
    8023: {"solar": 2782, "onshore_wind": 0},
    8042: {"solar": 3842, "onshore_wind": 0},
    8066: {"solar": 4423, "onshore_wind": 2929},
    8069: {"solar": 2615, "onshore_wind": 4206},
    8102: {"solar": 6551, "onshore_wind": 0},
    8219: {"solar": 454, "onshore_wind": 613},
    8222: {"solar": 1186, "onshore_wind": 634},
    8223: {"solar": 3095, "onshore_wind": 4253},
    8224: {"solar": 690, "onshore_wind": 1013},
    8226: {"solar": 694, "onshore_wind": 0},
    10143: {"solar": 329, "onshore_wind": 169},
    10151: {"solar": 310, "onshore_wind": 230},
    10603: {"solar": 118, "onshore_wind": 59},
    10671: {"solar": 437, "onshore_wind": 334},
    50611: {"solar": 75, "onshore_wind": 45},
    50776: {"solar": 65, "onshore_wind": 35},
    50974: {"solar": 156, "onshore_wind": 100},
    55076: {"solar": 1179, "onshore_wind": 895},
    55749: {"solar": 237, "onshore_wind": 209},
    55856: {"solar": 5755, "onshore_wind": 0},
    56068: {"solar": 3687, "onshore_wind": 0},
    56224: {"solar": 525, "onshore_wind": 744},
    56456: {"solar": 1754, "onshore_wind": 1079},
    56564: {"solar": 1481, "onshore_wind": 1088},
    56609: {"solar": 1283, "onshore_wind": 822},
    56611: {"solar": 2234, "onshore_wind": 1463},
    56671: {"solar": 2774, "onshore_wind": 0},
    56786: {"solar": 83, "onshore_wind": 45},
    56808: {"solar": 435, "onshore_wind": 408},
}
RE_MAP = {
    "Offshore Wind Turbine": "offshore_wind",
    "Onshore Wind Turbine": "onshore_wind",
    "Solar Photovoltaic": "solar",
    "offshore_wind": "offshore_wind",
    "onshore_wind": "onshore_wind",
    "solar": "solar",
}


def choose_years_for_map(h_range=(2008, 2021), f_range0=(2021, 2036), f_range1=(2036, 2040)):
    """Determine historical year to use for each future year, preserving year length.

    Args:
        h_range: years of historical data to sample from
        f_range0: first block of future years
        f_range1: second block of future years (to be repeated)

    Returns:

    """  # noqa: D414
    from collections import Counter
    from random import choice

    h_range_ = range(*h_range)

    def choose(yr, leap_, no_leap_):
        return choice(leap_) if yr % 4 == 0 else choice(no_leap_)  # noqa: S311

    def choose_range(h_yrs, p_yrs):
        leap = [x for x in h_yrs if x % 4 == 0]
        no_leap = [x for x in h_yrs if x % 4 != 0]
        return {yr: choose(yr, leap, no_leap) for yr in p_yrs}

    def make_maps():
        first = choose_range(h_range_, range(*f_range0))
        low_use = [y for y in h_range_ if y not in first.values()]
        try:
            second = choose_range(low_use, range(*f_range1))
        except IndexError:
            return {}
        else:
            return first | second

    return sorted([make_maps() for _ in range(1000)], key=lambda x: len(Counter(x.values())))[
        -1
    ]


@lru_cache
def patio_clean_op_date():
    return (
        read_named_range(bbb_path(), "Tax_Equity_Params")
        .assign(
            technology_description=lambda x: x.Technology.map(
                {
                    "LandbasedWind": "Onshore Wind Turbine",
                    "UtilityPV": "Solar Photovoltaic",
                    "OffShoreWind": "Offshore Wind Turbine",
                    "Utility-Scale Battery Storage": "Batteries",
                    "Nuclear": "Nuclear",
                }
            ),
            operating_date=lambda x: pd.to_datetime(x["Earliest_Build_Year"], format="%Y"),
        )
        .dropna(subset=["technology_description"])
        .set_index("technology_description")
        .operating_date.to_dict()
    )


def select_re_profiles(specs, profiles, id_cols=("plant_id_prof_site", "generator_id")):
    id_cols = list(id_cols)
    ids = specs[id_cols].drop_duplicates().sort_values(id_cols)
    return profiles.loc[:, list(ids.itertuples(index=False, name=None))]


@dataclass
class ProfileData:
    ad: AssetData
    solar_ilr: float = 1.34
    _cems: pd.DataFrame | None = None
    _dfs: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    year_mapper: dict = field(default_factory=dict, repr=False)
    regime: Literal["reference", "limited"] = "reference"

    def __post_init__(self):
        self.norm_cems = {}
        self.year_mapper: dict = get_year_map(max_year=2039)

    def get_ba_data(
        self,
        ba_code: str,
        cost_type="counterfactual",
        kind="adj",
        extend_cems=True,
        re_by_plant=True,
        max_wind_distance=45.0,
        max_solar_distance=10.0,
        min_re_site_mw=25.0,
        cr_eligible_techs: Sequence[str] | None = None,
        colo_techs=(),
        **kwargs,
    ):
        if not re_by_plant:
            raise DeprecationWarning("re_by_plant must be True")
        ba_code = str(ba_code)

        plant_data, cost_data_, bat = self.ad.get_generator_data(
            ba_code, cost_type=cost_type, solar_ilr=self.solar_ilr
        )
        if (
            colo_techs
            and df_query(
                plant_data,
                colo_techs | {"reg_rank": {"comp": "in", "item": colo_techs["reg_rank"]}},
            ).empty
        ):
            raise PatioData(f"No colo opportunities in {ba_code}")
        # TODO would be good to try and put all the profiles together, and all the
        #  specs together, save for patio addded re with the different index
        profiles, re_profiles, re_specs = self.get_ba_profiles(
            ba_code, plant_data, extend_cems
        )
        # in some cases the set of generators from 860 and CEMS differ we
        # select the intersection + proposed
        profiles, plant_data, common = self.align_common_cems_plant_data(
            ba_code, profiles, plant_data, cost_data_
        )
        profiles = (
            generate_projection_from_historical_pl(
                pl.from_pandas(profiles.reset_index()), year_mapper=self.year_mapper
            )
            .to_pandas(use_pyarrow_extension_array=True)
            .pivot(
                index="datetime",
                columns=["plant_id_eia", "generator_id"],
                values="gross_gen",
            )
        )
        # it is possible for profiles to be missing generators because that generator
        # didn't show up in any historical year from year_mapper
        common = sorted(set(common).intersection(set(profiles.columns)))
        plant_data = plant_data.loc[common, :]
        profiles = profiles[common]
        cost_data = generate_projection_from_historical(
            cost_data_.loc[[ind for ind in cost_data_.index if ind[:2] in common], :],
            year_mapper=self.year_mapper,
        ).sort_index()

        plant_data = pd.concat([plant_data, bat], axis=0)

        # need to remove RE associated with plants not in plant_data and
        # re-weight the remaining sites
        icx_rn = {
            "plant_id_eia": "icx_id",
            "generator_id": "icx_gen",
            "technology_description": "icx_tech",
            "operational_status": "icx_status",
        }
        existing_pd = (
            plant_data.reset_index()[
                [
                    "plant_id_eia",
                    "generator_id",
                    "cr_eligible",
                    "technology_description",
                ]
            ]
            .drop_duplicates()
            .rename(columns=icx_rn)
        )
        if cr_eligible_techs is not None:
            existing_pd = existing_pd.query("icx_tech in @cr_eligible_techs")
        LOGGER.warning(
            "solar distance == %s, wind distance == %s", max_solar_distance, max_wind_distance
        )
        re_specs = (
            re_specs.query(
                "capacity_mw_nrel_site >= @min_re_site_mw "
                " & ((technology_description.str.contains('Solar') & distance <= @max_solar_distance)"
                "| (technology_description.str.contains('Wind') & distance <= @max_wind_distance))"
            )
            .merge(
                self.ad.gens.rename(columns=icx_rn)[
                    ["icx_id", "icx_gen", "icx_status", "reg_rank"]
                ],
                on=["icx_id", "icx_gen"],
                how="left",
                validate="m:1",
            )
            .merge(
                existing_pd,
                on=["icx_id", "icx_gen"],
                how="left",
                validate="m:1",
                suffixes=(None, "_rdrop"),
                indicator=True,
            )
            .merge(
                # trying to avoid nans that break avoided cost calculations
                cost_data.reset_index()
                .rename(columns=icx_rn)
                .groupby(["icx_id", "icx_gen"], as_index=False)
                .total_var_mwh.mean(),
                on=["icx_id", "icx_gen"],
                suffixes=(None, "_rdrop"),
                how="left",
                validate="m:1",
            )
            .assign(
                cr_eligible=lambda x: x.cr_eligible.fillna(False),
                ilr=lambda x: np.where(x.re_type == "solar", self.solar_ilr, 1),
                operating_date=lambda x: x.technology_description.map(patio_clean_op_date()),
                retirement_date=pd.NaT,
                total_var_mwh=lambda x: x.total_var_mwh.fillna(x.total_var_mwh_rdrop),
            )
            .filter(regex="^(?!.*rdrop$).*$")
            .query("icx_id > 0")
        )
        if re_specs.query("cr_eligible & _merge == 'both'").empty:
            raise NoEligibleCleanRepowering
        fos_no_re = set(
            existing_pd[existing_pd.cr_eligible.fillna(False)][
                ["icx_id", "icx_gen"]
            ].itertuples(index=False, name=None)
        ) - set(
            re_specs.query("_merge == 'both'")[["icx_id", "icx_gen"]].itertuples(
                index=False, name=None
            )
        )
        if fos_no_re:
            LOGGER.info(
                "%s re missing for these existing generators \n%s",
                ba_code,
                plant_data.loc[
                    sorted(fos_no_re),
                    ["technology_description", "capacity_mw", "operating_date"],
                ]
                .to_string()
                .replace("\n", "\n\t"),
            )
        missing_ = self.ad.get_missing_generators(ba_code, plant_data)

        # create consistent profiles and annual data for `existing_xpatio` renewables
        # for historical operations and pre-operating date counterfactual operations
        existing_re_prof, missing = self.existing_re_helper(missing_, re_profiles)

        missing = (
            generate_projection_from_historical(
                missing.merge(
                    cost_data_[
                        [
                            x
                            for x in cost_data_
                            if x
                            not in (
                                "final_ba_code",
                                "technology_description",
                                "fuel_group",
                            )
                        ]
                    ].reset_index(),
                    on=["plant_id_eia", "generator_id", "datetime"],
                    how="left",
                    validate="1:1",
                ),
                year_mapper=self.year_mapper,
            )
            .set_index(["plant_id_eia", "generator_id", "datetime"])
            .sort_index()
        )

        re_profiles = generate_projection_from_historical_pl(
            re_profiles.select("datetime", *sorted(re_specs.combined.unique())),
            year_mapper=self.year_mapper,
        )

        replaceable = plant_data.query("category == 'existing_fossil'")
        return {
            "ba_code": ba_code,
            "profiles": profiles,
            "plant_data": plant_data,
            "cost_data": cost_data,
            "re_profiles": re_profiles,
            "re_plant_specs": re_specs,
            "year_mapper": self.year_mapper.copy(),
            # "cems_change": cems_change,
            "baseload_replace": {
                k: sorted(
                    set(replaceable.index.get_level_values("plant_id_eia")).intersection(
                        set(v)
                    )
                )
                for k, v in self.ad.temp_nuke_replacements()[ba_code].items()
            },
            "ccs_convert": {
                k: sorted(set(replaceable.index).intersection(set(v)))
                for k, v in self.ad.temp_ccs_conversion()[ba_code].items()
            },
            "missing": missing,
            "fuel_curve": generate_projection_from_historical_pl(
                self.ad.curve.filter(pl.col("ba_code") == ba_code),
                year_mapper=self.year_mapper,
            ).to_pandas(),
            "existing_re_prof": generate_projection_from_historical(
                existing_re_prof, year_mapper=self.year_mapper
            ),
            "exclude_or_mothball": {
                # "57": [703],  # Bowen in Georgia Power 2103
                # "177": [6155],  # Rush Island in Ameren
                # "177": [2103],  # Labadie in Ameren
            }.get(ba_code, []),
            # because we added Vogtle, we remove Scherer and Wansley as in IRA
            "exclude": {
                # "57": [6257, 6052],
                # "177": [
                #     2104,
                #     6155,
                # ],
            }.get(ba_code, []),
            "max_solar_distance": max_solar_distance,
            "max_wind_distance": max_wind_distance,
            "min_re_site_mw": min_re_site_mw,
            "cr_eligible_techs": cr_eligible_techs,
            "colo_techs": colo_techs,
            "regime": self.regime,
            **kwargs,
        }

    def one_plant(self, pid: int, d: dict):
        """Args:
            pid: plant_id_eia
            d: from :meth:`.ProfileData.get_ba_data`

        Returns:

        """  # noqa: D414
        if pid not in PLANT_RE_MAX:
            raise NoMaxRE(f"No Max RE data for {pid=} from ba_code={d['ba_code']}")
        else:
            LOGGER.warning("USING CRUDE CCC3 RE CAP MAX")
        try:
            plant_data = (
                d["plant_data"]
                .query("plant_id_eia == @pid")
                .copy()
                .assign(
                    # remove retirement date so we get all years
                    retirement_date=pd.NaT,
                )
            )
            # re_plant_specs = self._adjust_re_for_fossil_data(
            #     d["re_plant_specs"], plant_data
            # )
            tz_code = self.ad.ba_offsets.loc[d["ba_code"], "tz_code"]
            dt = datetime(self.ad.years[0], 1, 1, 0, tzinfo=ZoneInfo(tz_code))
            re = generate_projection_from_historical(
                pd.concat(
                    [
                        pl.scan_parquet(Path.home() / "patio_data/hyperlocal_re.parquet")
                        .filter(
                            (pl.col("plant_id_eia") == pid)
                            & (pl.col("datetime") >= dt.astimezone(UTC))
                        )
                        .select(["datetime", "solar", "onshore_wind"])
                        .with_columns(
                            pl.col("datetime")
                            .dt.convert_time_zone(tz_code)
                            .dt.replace_time_zone(None)
                        )
                        .collect()
                        .to_pandas()
                        .set_index("datetime")
                    ],
                    keys=[-pid],
                    names=["plant_id_eia", "generator_id"],
                    axis=1,
                ).sort_index(axis=1),
                year_mapper=d["year_mapper"],
            )

            re_cols = ["fos_id", "fos_gen", "latitude", "longitude", "state"]
            pdata = plant_data.reset_index().rename(
                columns={"plant_id_eia": "fos_id", "generator_id": "fos_gen"}
            )[re_cols]
            re_plant_specs = (
                pd.concat(
                    [
                        pdata.assign(
                            technology_description=RE_TECH_R[rtype],
                            ilr=ilr,
                            plant_id_eia=-pid,
                            generator_id=rtype,
                            operating_date=re.index.min() - timedelta(days=30),
                        )
                        for rtype, ilr in (("solar", 1.3), ("onshore_wind", 1.0))
                    ]
                )
                .assign(
                    distance=15.0,
                    ba_code=d["ba_code"],
                    capacity_mw=100.0,
                    ba_weight=len(plant_data) ** -1,
                )
                .sort_values(["plant_id_eia", "generator_id"])
            )
            out = {
                "ba_code": d["ba_code"] + f"_{pid}",
                "capacity": plant_data.capacity_mw.sum(),
                "plant_data": plant_data,
                "profiles": d["profiles"].loc[:, [pid]],
                "cost_data": d["cost_data"].loc[[pid], :],
                "re_plant_specs": re_plant_specs,
                "re_profiles": re,
                "cems_change": MTDF.copy(),
                "baseload_replace": {1: []},
                "ccs_convert": {1: []},
                "old_re_specs": d["old_re_specs"].assign(capacity_mw=0),
                "old_re_profs": d["old_re_profs"],
                "exclude_or_mothball": [],
                "exclude": [],
                "max_re": PLANT_RE_MAX[pid],
                "missing": MTDF.reindex(columns=d["missing"].columns, copy=True),
                "year_mapper": d["year_mapper"],
            }
        except Exception as exc:
            raise RuntimeError(
                f"Unable to select {pid=} from ba_code={d['ba_code']}, {exc!r}"  # noqa: S608
            ) from exc
        else:
            return out

    ###########################################################################
    # BA DATA HELPERS
    ###########################################################################

    def get_ba_profiles(self, ba_code, plant_data, extend_cems):
        cems = self.load_cems(ba_code, extend_cems)[["gross_gen"]]
        cems = (
            cems.reset_index()
            .assign(datetime=lambda x: x.datetime.dt.tz_localize(None))
            .set_index(cems.index.names)
        )
        re_specs, re_profiles = self.hourly_re_by_plant(ba_code)
        re_profiles = re_profiles.with_columns(pl.col("datetime").dt.replace_time_zone(None))
        profiles_wo_nulls = [
            k for k, v in re_profiles.null_count().to_dicts()[0].items() if v == 0
        ]
        if len(profiles_wo_nulls) < re_profiles.shape[0]:
            LOGGER.warning(
                "The following profiles had nulls so will be removed: %s",
                set(re_profiles.columns) - set(profiles_wo_nulls),
            )
            re_profiles = re_profiles.select(*profiles_wo_nulls)

        missing_re_profiles = set(re_specs["combined"].unique()) - set(re_profiles.columns)
        # re_profiles.index = re_profiles.index.tz_localize(None)
        # missing_re_profiles = set(
        #     re_specs[["plant_id_prof_site", "re_type"]]
        #     .drop_duplicates()
        #     .sort_values(["plant_id_prof_site", "re_type"])
        #     .itertuples(index=False, name=None)
        # ) - set(re_profiles.columns)
        if missing_re_profiles:
            LOGGER.warning(
                "%s, filling missing RE profiles with BA averages %s",
                ba_code,
                missing_re_profiles,
            )
            to_create = {
                k: [x for x in missing_re_profiles if k in x]
                for k in {p.split("__")[1] for p in missing_re_profiles}
            }
            for tech, to_make in to_create.items():
                if tech in ("offshore_wind", "onshore_wind", "solar"):
                    re_profiles = re_profiles.with_columns(
                        **{
                            p: pl.mean_horizontal(
                                re_profiles.select(
                                    [x for x in re_profiles.columns if tech in x]
                                )
                            ).cast(pl.Float32)
                            for p in to_make
                        }
                    )
                elif tech in ("Geothermal", "Nuclear"):
                    re_profiles = re_profiles.with_columns(
                        **{p: pl.lit(1.0, pl.Float32) for p in to_make}
                    )
                else:
                    raise RuntimeError(f"Cannot create profile for {tech}.")
        assert not [k for k, v in re_profiles.null_count().to_dicts()[0].items() if v != 0], (
            "filling missing RE profiles introduced nulls"
        )
        #
        # for pid, tech in missing_re_profiles:
        #     re_profiles.loc[:, (pid, tech)] = self.make_proposed_profile(
        #         tech, re_profiles
        #     )
        cems_ids = set(cems.index.droplevel(2).unique())  # noqa: F841

        to_add = []
        if (
            len(
                new := plant_data.query("operational_status == 'proposed'")[
                    ["technology_description", "ilr", "cf_atb"]
                ]
            )
            > 0
        ):
            fos_tech = {
                "Wood/Wood Waste Biomass",
                "Conventional Steam Coal",
                "Natural Gas Fired Combined Cycle",
                "Natural Gas Fired Combustion Turbine",
                "Petroleum Liquids",
                "Natural Gas Internal Combustion Engine",
            }
            techs = {p.partition("__")[-1] for p in re_profiles.columns if "__" in p}
            generic_profs = (
                re_profiles.select("datetime")
                .lazy()
                .with_columns(
                    Geothermal=pl.lit(1.0, pl.Float32),
                    Nuclear=pl.lit(1.0, pl.Float32),
                    **{
                        RE_TECH_R[k]: pl.mean_horizontal(
                            re_profiles.select([x for x in re_profiles.columns if k in x])
                        )
                        for k in techs
                    },
                    **{
                        c: pl.when(pl.col("datetime") == pl.col("datetime").max())
                        .then(pl.lit(1e-10, pl.Float32))
                        .otherwise(pl.lit(0.0, pl.Float32))
                        for c in fos_tech.intersection(set(new.technology_description))
                    },
                )
            )
            prop_profs = (
                pl.from_pandas(new.reset_index())
                .lazy()
                .join(
                    generic_profs.melt(
                        id_vars="datetime",
                        value_vars=generic_profs.columns[1:],
                        value_name="mwh",
                        variable_name="technology_description",
                    ),
                    how="inner",
                    on=["technology_description"],
                )
            )
            pro_df = (
                prop_profs.join(
                    prop_profs.join(
                        pl.from_pandas(plant_data[["cf_atb", "ilr"]].reset_index()).lazy(),
                        on=["plant_id_eia", "generator_id"],
                        how="inner",
                    )
                    .with_columns(
                        cf_rninja_ilr=pl.min_horizontal(pl.col("mwh") * pl.col("ilr"), 1.0)
                    )
                    .group_by("plant_id_eia", "generator_id")
                    .agg(pl.mean("cf_rninja_ilr"), pl.first("cf_atb"))
                    .with_columns(
                        cf_mult=pl.max_horizontal(
                            pl.col("cf_atb") / pl.col("cf_rninja_ilr"), 1.0
                        )
                    )
                    .select("plant_id_eia", "generator_id", "cf_mult"),
                    on=["plant_id_eia", "generator_id"],
                )
                .select(
                    "plant_id_eia",
                    "generator_id",
                    "datetime",
                    pl.when(pl.col("technology_description").is_in(list(RE_TECH)))
                    .then(pl.min_horizontal(pl.col("mwh") * pl.col("cf_mult"), 1.0))
                    .otherwise(pl.col("mwh"))
                    .alias("gross_gen"),
                )
                .sort("plant_id_eia", "generator_id", "datetime")
                .collect()
                .to_pandas(use_pyarrow_extension_array=True)
            )

            missing = (
                pro_df[["plant_id_eia", "generator_id"]]
                .drop_duplicates()
                .merge(
                    new,
                    on=["plant_id_eia", "generator_id"],
                    how="outer",
                    indicator=True,
                )
                .query("_merge != 'both'")
            )
            if not missing.empty:
                LOGGER.error(
                    "Unable to get proposed profiles for %s,\n %r",
                    ba_code,
                    missing[["plant_id_eia", "generator_id", "technology_description"]]
                    .to_string()
                    .replace("\n", "\n\t"),
                )

            to_add.append(pro_df.set_index(cems.index.names))

        return (
            pd.concat([cems, *to_add], axis=0),
            re_profiles,
            re_specs.with_columns(
                technology_description=pl.col("re_type").replace_strict(RE_TECH_R),
                generator_id=pl.col("re_type"),
                energy_community=pl.col("energy_community").fill_null(False).cast(pl.Boolean),
            ).to_pandas(),
        )

    @staticmethod
    def make_proposed_profile(tech, re_profiles):
        if tech in (
            "Wood/Wood Waste Biomass",
            "Conventional Steam Coal",
            "Natural Gas Fired Combined Cycle",
            "Natural Gas Fired Combustion Turbine",
            "Petroleum Liquids",
            "Natural Gas Internal Combustion Engine",
        ):
            # we drop profiles that sum to zero, so we need one value that
            # isn't quite zero
            return np.append(np.zeros(len(re_profiles) - 1), 1e-10)
        if tech in ("Geothermal", "Nuclear"):
            return np.ones_like(re_profiles.iloc[:, 0])
        return (
            re_profiles.loc[:, (slice(None), RE_MAP.get(tech, tech))].mean(axis=1).to_numpy()
        )

    @staticmethod
    def align_common_cems_plant_data(ba_code, cems, plant_data, cost_data=None):
        # in some cases the set of generators from 860 and CEMS differ we
        # select the intersection
        _temp = cems.groupby(level=[0, 1]).sum()["gross_gen"]
        c_set = set(_temp[_temp > 0].index)
        if cems.empty or not c_set:
            raise NoNonZeroCEMS
        p_set = set(plant_data.index)
        common = sorted(c_set.intersection(p_set))
        if cost_data is not None:
            co_set = set(
                cost_data.dropna(
                    axis=0,
                    how="any",
                    subset=[
                        "co2_factor",
                        "final_ba_code",
                        "fuel_per_mwh",
                        "heat_rate",
                        "start_per_kw",
                        "technology_description",
                        "total_var_mwh",
                        "vom_per_mwh",
                    ],
                ).index.droplevel(2)
            )

            common = sorted(set(common).intersection(co_set))
        if not common:
            raise NoMatchingPlantsAndCEMS
        if uncommon := set(plant_data.index) - set(common):
            LOGGER.info(
                "%s these plants will use historical rather than redispatch:\n%s",
                ba_code,
                plant_data.loc[list(uncommon), :]
                .query("operational_status == 'existing'")
                .reset_index()
                .groupby("technology_description", observed=True)
                .agg({"capacity_mw": "sum", "generator_id": "count"})
                .sort_index()
                .to_string()
                .replace("\n", "\n\t"),
            )
        cems = (
            cems[cems.index.droplevel(2).isin(common)]
            .astype(pd.ArrowDtype(pa.float32()))
            .sort_index()
        )
        plant_data = plant_data.loc[common, :]
        return cems, plant_data, common

    @staticmethod
    def _adjust_re_for_fossil_data(re_specs, plant_data):
        """Need to remove RE associated with plants not in plant_data and re-weight
        the remaining sites.
        """
        re_specs_ = re_specs.merge(
            plant_data.query("operational_status == 'existing'")
            .reset_index()[["plant_id_eia", "generator_id"]]
            .rename(columns={"plant_id_eia": "fos_id", "generator_id": "fos_gen"}),
            on=["fos_id", "fos_gen"],
            how="inner",
            validate="m:1",
        )  # .assign(
        #     capacity_for_weight=lambda x: x.groupby(
        #         ["technology_description", "fos_id", "fos_gen"]
        #     ).capacity_mw.transform("sum")
        #     / x.groupby(["technology_description"]).capacity_mw.transform("sum"),
        #     # weight is
        #     ba_weight=lambda x: x.capacity_for_weight * x.weight,
        #     ba_distance=lambda x: x.ba_weight * x.distance,
        # )
        # compare
        id_cols = ["fos_id", "fos_gen", "plant_id_eia", "re_type"]
        c = (
            re_specs.merge(re_specs_[id_cols], on=id_cols, how="right", validate="1:1")
            .reset_index(drop=True)
            .compare(re_specs_.reset_index(drop=True))
        )
        if (
            # if c is empty the second part raises an error
            not c.empty and not c.groupby(level=1, axis=1).sum().query("self > other").empty
        ):
            raise AssertionError("adjusted RE weights are lower, this shouldn't happen")
        if not np.isclose(
            re_specs_.groupby(["re_type", "fos_id", "fos_gen"]).weight.sum(),
            1,
        ).all():
            raise AssertionError("weights that should sum to one do not")
        return re_specs_

    ###########################################################################
    # DATA LOADERS
    ###########################################################################

    def load_cems(self, ba_code, extend_cems):
        ex = "_extended" if extend_cems else ""
        if not (file := USER_DATA_PATH / f"ba_cems{ex}.zip").exists():
            get(PATIO_DATA_AZURE_URLS[f"ba_cems{ex}"], file.parent)
        try:
            with DataZip(file, "r") as z:
                cems = z[ba_code].convert_dtypes(dtype_backend="pyarrow")
        except KeyError as exc:
            raise NoCEMS(f"{ba_code} missing") from exc
        index = ["plant_id_eia", "generator_id", "datetime"]
        if any(x in cems for x in index):
            return cems.set_index(["plant_id_eia", "generator_id", "datetime"])
        return cems

    def hourly_re_by_plant(self, ba_code):
        # if not (file := USER_DATA_PATH / f"re_data_{self.regime}.zip").exists():
        #     download(PATIO_DATA_AZURE_URLS[f"re_data_{self.regime}"], file)
        if not (file := USER_DATA_PATH / "re_data.zip").exists():
            get(PATIO_DATA_AZURE_URLS["re_data"], file.parent)
        try:
            with DataZip(file, "r") as z:
                meta = z[ba_code + "_meta"]
                prof = z[ba_code + "_prof"]
            return meta, prof
        except KeyError as exc:
            raise NoREData(f"{ba_code}") from exc

    def all_re_for_cost_calcs(self):
        to_df = []
        with logging_redirect_tqdm():
            for ba in tqdm(self.ad.all_modelable_generators().final_ba_code.unique()):
                try:
                    meta, _ = self.hourly_re_by_plant(ba)
                    to_df.append(
                        meta.assign(
                            plant_id_eia=lambda x: x.fos_id,
                            re_plant_id=lambda x: x.re_site_id,
                            capacity_mw=1,
                            redispatch_mwh=lambda x: x.cf_ilr_prof_site * 8766,
                            category="patio_clean",
                            technology_description=lambda x: x.re_type.map(RE_TECH_R),
                            year=2028,
                        )
                        .groupby(
                            [
                                "ba_code",
                                "plant_id_eia",
                                "re_plant_id",
                                "year",
                                "re_type",
                            ],
                            as_index=False,
                        )[
                            [
                                "capacity_mw",
                                "redispatch_mwh",
                                "category",
                                "technology_description",
                                "class_atb",
                                "plant_id_prof_site",
                                "cf_prof_site",
                                "cf_ilr_prof_site",
                                "lcoe",
                                "sc_gid",
                                "latitude_nrel_site",
                                "longitude_nrel_site",
                                "distance",
                                "fos_id",
                                "re_site_id",
                            ]
                        ]
                        .first()
                    )
                except Exception as exc:
                    LOGGER.error("%s %r", ba, exc)
        out = pd.concat(to_df)
        (
            out.pipe(
                add_ec_flag,
                key_col="re_site_id",
                lat_lon_cols=("latitude_nrel_site", "longitude_nrel_site"),
            ).to_parquet(Path.home() / "patio_data/all_re_for_cost_calcs.parquet")
        )

    ###########################################################################
    # PROFILE SETUP
    ###########################################################################

    def setup_all(self, keep=True):
        # files = (
        #     Path.home() / f"patio_data/ba_cems.zip",
        #     Path.home() / f"patio_data/ba_cems_extended.zip",
        #     Path.home() / f"patio_data/re_data_{self.max_re_sites}.zip",
        #     Path.home() / f"patio_data/norm_cems_roles_by_ba.parquet",
        # )
        # if keep:
        #     for file in files:
        #         if file.exists():
        #             file.rename(
        #                 str(file)
        #                 .replace(".zip", "_old.zip")
        #                 .replace(".parquet", "_old.parquet")
        #             )
        # else:
        #     for file in files:
        #         file.unlink(missing_ok=True)
        #
        # (CACHE_PATH / "modelable_plants.parquet").unlink(missing_ok=True)

        # self.ba_re_maker()
        # self.ba_cems_maker(extend_cems=False)
        self.ba_cems_maker(extend_cems=True)

    ###########################################################################
    # CEMS PROFILE SETUP
    ###########################################################################

    def ba_cems_maker(self, compression=ZIP_STORED, extend_cems=False, test=False):
        ex = "_extended" if extend_cems else ""

        norm_subplant_cems = self.subplant_cems()

        cw_gen = (
            pl.from_pandas(self.ad.cw)
            .select(["plant_id_eia", "generator_id", "subplant_id", "capacity_mw"])
            .with_columns(pl.col("plant_id_eia").cast(pl.Int32))
            .unique(subset=["plant_id_eia", "generator_id"])
            .lazy()
        )

        def make_cems(df_, ba_):
            tz_code = self.ad.ba_offsets.loc[ba_, "tz_code"]
            dt = datetime(self.ad.years[0], 1, 1, 0, tzinfo=ZoneInfo(tz_code))
            pids = list(df_.index.get_level_values("plant_id_eia").unique())
            out = (
                norm_subplant_cems.filter(
                    pl.col("plant_id_eia").is_in(pids)
                    & (pl.col("operating_datetime_utc") >= dt.astimezone(UTC))
                )
                .with_columns(
                    pl.col("operating_datetime_utc")
                    .dt.convert_time_zone(tz_code)
                    .alias("datetime")
                )
                .join(cw_gen, on=["plant_id_eia", "subplant_id"], how="inner")
                .with_columns(
                    (pl.col("normed_gen") * pl.col("capacity_mw")).alias("gross_gen")
                )
                .select(
                    "plant_id_eia",
                    "generator_id",
                    "datetime",
                    "gross_gen",
                    "capacity_mw",
                )
                .sort(["plant_id_eia", "generator_id", "datetime"])
                .collect()
            )
            if (n := len(out.filter(pl.col("gross_gen") > pl.col("capacity_mw")))) > 0:
                raise AssertionError(
                    f"calculated hourly gross gen exceeded capacity in {n} rows"
                )
            if len(out.select("plant_id_eia", "generator_id", "datetime").unique()) != len(
                out
            ):
                raise AssertionError(
                    "(plant_id_eia, generator_id, datetime) not jointly unique"
                )
            return out.select(
                "plant_id_eia", "generator_id", "datetime", "gross_gen"
            ).to_pandas()

        file = Path.home() / f"patio_data/ba_cems{ex}" if not test else BytesIO()

        with DataZip(file, "w", compression=compression) as z, logging_redirect_tqdm():
            for_loop = tqdm(
                # {
                #     "LGEE": self.ad.all_modelable_bas()["LGEE"],
                #     "NBSO": self.ad.all_modelable_bas()["NBSO"],
                #     **self.ad.all_modelable_bas()
                # }.items(),
                [
                    (k, self.ad.all_modelable_generators().query("final_ba_code == @k"))
                    for k in sorted(self.ad.all_modelable_generators().final_ba_code.unique())
                ],
                desc=f"ba_cems_maker{ex}",
                total=len(self.ad.all_modelable_generators().final_ba_code.unique()),
            )
            for ba_code, plant_data in for_loop:
                for_loop.set_description(desc=f"ba_cems_maker{ex} " + ba_code)
                try:
                    if extend_cems:
                        cems = self.cems_extender(ba_code, plant_data).astype(np.float32)
                    else:
                        cems = make_cems(plant_data, ba_code)
                except Exception as exc:
                    LOGGER.exception("%s %r", ba_code, exc, exc_info=exc)
                else:
                    z[ba_code] = cems
                    z.reset_ids()

    def subplant_cems(self):
        norm_subplant_cems_file = Path.home() / "patio_data/epacems_subplant_norm.parquet"
        if not norm_subplant_cems_file.exists():
            pudl_cems = Path.home() / "pudl-work/output/hourly_emissions_epacems.parquet"
            if not pudl_cems.exists():
                raise FileNotFoundError(
                    "Download the EPA CEMS Hourly Emissions Parquet (1995-2021) file "
                    "from PUDL and place it in `~/pudl-work/output "
                    "(https://github.com/catalyst-cooperative/pudl#nightly-data-builds)"
                )
            LOGGER.warning(
                "WE NEED TO REPROCESS CEMS DATA TO CREATE SUBPLANT PROFILES, THIS "
                "WILL TAKE A LONG TIME (>30 min) AND PROVIDE NO PROGRESS INFO"
            )
            cw_sub = (
                pl.from_pandas(self.ad.cw)
                .select(["plant_id_eia", "emissions_unit_id_epa", "subplant_id"])
                .with_columns(pl.col("plant_id_eia").cast(pl.Int32))
                .unique(subset=["plant_id_eia", "emissions_unit_id_epa"])
                .lazy()
            )
            norm_subplant_cems = (
                pl.scan_parquet(pudl_cems)
                .filter(pl.col("year") >= 2006)
                .join(cw_sub, on=["plant_id_eia", "emissions_unit_id_epa"], how="inner")
                .group_by("plant_id_eia", "subplant_id", "operating_datetime_utc")
                .agg(pl.col("gross_load_mw").sum(), pl.col("heat_content_mmbtu").sum())
                .with_columns(
                    (
                        pl.col("gross_load_mw")
                        / pl.col("gross_load_mw")
                        .max()
                        .over(
                            [
                                "plant_id_eia",
                                "subplant_id",
                                pl.col("operating_datetime_utc").dt.year(),
                            ]
                        )
                    ).alias("normed_gen")
                )
            )
            norm_subplant_cems = norm_subplant_cems.collect()
            norm_subplant_cems.write_parquet(norm_subplant_cems_file)
            return norm_subplant_cems.lazy()
        return pl.scan_parquet(norm_subplant_cems_file)

    def cems_extender(self, ba_code, plant_data, *args):
        """Extend CEMS data backwards using normalized aggregations of plants in the BA with a similar role"""
        # backup = {
        #     "IPCO": "BPAT",
        #     "LGEE": "PJM",
        #     "NWMT": "SWPP",
        #     "SEPA": "DUK",
        # }

        cems, plant_data, common = self.align_common_cems_plant_data(
            ba_code,
            cems=self.load_cems(ba_code, extend_cems=False),
            plant_data=plant_data,
        )

        # assign plants in cems to roles and aggregate to role/hourly
        # calculate normalized hourly profiles for each role
        cemsr_norm = self.norm_cems_roles_by_ba().loc[ba_code, :]
        # because we can have partial years in CEMS data, we want the date for the
        # most recent full year

        lev_year = (  # noqa: F841
            cems.index.get_level_values("datetime")
            .unique()
            .to_frame()
            .groupby(pd.Grouper(freq="YS"))
            .count()
            .query("datetime >= 8760")
            .iloc[-1, :]
            .name.year
        )

        cems = cems.query("datetime.dt.year <= @lev_year")
        cemsr_norm = cemsr_norm.query("datetime.dt.year <= @lev_year")
        full_index = cems.index.get_level_values("datetime").unique()

        new = []
        for (pid, genid), row in plant_data.query(
            f"operating_date > '{self.ad.years[0]}-01-01'"
        ).iterrows():
            try:
                # find the set of datetimes that we need to fill in
                to_fill = full_index.difference(
                    cems.loc[(pid, genid), :].index
                ).drop_duplicates()
                try:
                    # grab the normalized profile for the correct role
                    shape = cemsr_norm.loc[row.plant_role, :].loc[to_fill, :]
                except KeyError:
                    # if not available for

                    try:
                        whole_shape = (
                            self.norm_cems_roles_by_ba()
                            .loc[row.balancing_authority_code_eia, :]
                            .loc[row.plant_role, :]
                            # .loc[to_fill, :]
                        )
                        pd.testing.assert_index_equal(
                            full_index, whole_shape.index.tz_convert(full_index.tz)
                        )
                    except AssertionError:
                        backup = (
                            self.ad.gens.groupby(["state", "final_ba_code"], as_index=False)[
                                ["capacity_mw"]
                            ]
                            .sum()
                            .sort_values(["capacity_mw"])
                            .query("state in @plant_data.state & final_ba_code != @ba_code")
                            .final_ba_code.iloc[-1]
                        )
                        LOGGER.warning(
                            "No full shape for %s in %s, using %s",
                            row.plant_role,
                            ba_code,
                            backup,
                        )
                        whole_shape = (
                            self.norm_cems_roles_by_ba().loc[backup, :].loc[row.plant_role, :]
                            # .loc[to_fill, :]
                        )

                    tz_convert = False
                    if whole_shape.index.tz != to_fill.tz:
                        LOGGER.warning("%s adjusting timezone", pid)
                        whole_shape.index = whole_shape.index.tz_convert("UTC")
                        to_fill = to_fill.tz_convert("UTC")
                        tz_convert = True
                    shape = whole_shape.loc[to_fill, :]
                    if tz_convert:
                        shape.index = shape.index.tz_convert(str(full_index.tz))

            except Exception as exc:
                LOGGER.error("%s %s %s %r", ba_code, pid, genid, exc)
            else:
                # determine the total level of production for the generator in the last year
                # level = cems.loc[
                #     pd.IndexSlice[pid, genid, f"{lev_year}-01-01":], :
                # ].sum()
                new.append(
                    pd.concat(
                        # [shape * level],
                        [shape * row.capacity_mw],
                        axis=0,
                        keys=[(pid, genid)],
                        names=["plant_id_eia", "generator_id"],
                    )
                )
        out = pd.concat([cems, *new], axis=0).sort_index()
        test = (
            out.groupby(["plant_id_eia", "generator_id"])[["gross_gen"]]
            .max()
            .join(plant_data.capacity_mw, how="left")
        )
        # need a little play because of dtype difference
        assert len(test.query("gross_gen > capacity_mw * 1.001")) == 0, (
            "gross_gen exceeds capacity_mw"
        )
        assert out[out.index.duplicated(keep=False)].empty, (
            "duplicates created in `extend_cems`"
        )
        return out

    def norm_cems_roles_by_ba(self):
        """Normalized profiles for each plant role in each BA, profiles are
        normalized annually
        """
        if "norm_cems_roles_by_ba" not in self._dfs:
            try:
                self._dfs["norm_cems_roles_by_ba"] = pd.read_parquet(
                    Path.home() / "patio_data/norm_cems_roles_by_ba.parquet"
                )
            except FileNotFoundError:
                norm_dict = {}
                with logging_redirect_tqdm():
                    for_loop = tqdm(
                        [
                            (
                                k,
                                self.ad.all_modelable_generators.query("final_ba_code == @k"),
                            )
                            for k in sorted(
                                self.ad.all_modelable_generators.final_ba_code.unique()
                            )
                        ],
                        desc="Norm roles",
                        total=len(self.ad.all_modelable_generators.final_ba_code.unique()),
                    )
                    for ba_code, plant_data in for_loop:
                        for_loop.set_description("Norm roles " + ba_code)
                        try:
                            (
                                cems,
                                plant_data,
                                common,
                            ) = self.align_common_cems_plant_data(
                                ba_code,
                                self.load_cems(ba_code, extend_cems=False),
                                plant_data,
                            )
                            cemsr_agg = (
                                cems.reset_index()
                                .merge(
                                    plant_data["plant_role"].reset_index(),
                                    on=["plant_id_eia", "generator_id"],
                                )
                                .groupby(["plant_role", "datetime"])[["gross_gen"]]
                                .sum()
                            )
                            # calculate normalized hourly profiles for each role
                            norm_dict[ba_code] = (
                                cemsr_agg
                                / cemsr_agg.reset_index()
                                .groupby(
                                    [
                                        "plant_role",
                                        pd.Grouper(freq="YS", key="datetime"),
                                    ]
                                )
                                .transform("max")
                                .to_numpy()
                            )
                        except Exception as exc:
                            LOGGER.error("%s %r", ba_code, exc)
                self._dfs["norm_cems_roles_by_ba"] = pd.concat(norm_dict, axis=0).astype(
                    np.float32
                )
                self._dfs["norm_cems_roles_by_ba"].to_parquet(
                    Path.home() / "patio_data/norm_cems_roles_by_ba.parquet"
                )
        return self._dfs["norm_cems_roles_by_ba"]

    def cems_summary(self):
        pudl_cems = (
            pl.scan_parquet(Path.home() / "pudl-work/output/hourly_emissions_epacems.parquet")
            .with_columns(
                (
                    (pl.col("gross_load_mw").fill_null(0.0) != 0.0)
                    & (pl.col("gross_load_mw").fill_null(0.0).shift(1) == 0.0)
                    & (pl.col("plant_id_eia") == pl.col("plant_id_eia").shift(1))
                    & (
                        pl.col("emissions_unit_id_epa")
                        == pl.col("emissions_unit_id_epa").shift(1)
                    )
                )
                .cast(pl.Int32)
                .alias("gen_starts"),
                (
                    (pl.col("heat_content_mmbtu").fill_null(0.0) != 0.0)
                    & (pl.col("heat_content_mmbtu").fill_null(0.0).shift(1) == 0.0)
                    & (pl.col("plant_id_eia") == pl.col("plant_id_eia").shift(1))
                    & (
                        pl.col("emissions_unit_id_epa")
                        == pl.col("emissions_unit_id_epa").shift(1)
                    )
                )
                .cast(pl.Int32)
                .alias("fuel_starts"),
            )
            .sort(["plant_id_eia", "emissions_unit_id_epa", "operating_datetime_utc"])
            .group_by_dynamic(
                index_column="operating_datetime_utc",
                every="1mo",
                period="1mo",
                by=["plant_id_eia", "emissions_unit_id_epa"],
            )
            .agg(
                pl.col("gross_load_mw").sum().alias("gross_generation_mwh"),
                pl.col("gross_load_mw").max().alias("camd_capacity_mw"),
                pl.col("heat_content_mmbtu").sum(),
                pl.col("gen_starts").sum(),
                pl.col("fuel_starts").sum(),
                pl.col("co2_mass_tons").sum(),
            )
            .collect()
        )
        pudl_cems.write_parquet(Path.home() / "patio_data/camd_starts_ms.parquet")

    ###########################################################################
    # RE PROFILE SETUP
    ###########################################################################

    def ba_re_maker(self, compression=ZIP_STORED, test=False, max_distance=50):
        """Create re_data.zip for all BAs."""
        meta_path = CACHE_PATH / "re_curve_to_fossil.parquet"
        if (age := seconds_since_touch(meta_path)) > 3600:
            LOGGER.warning("making nrel curve, this could take a few minutes")
            re_meta = self.ad.curve_for_site_selection(2025, self.solar_ilr)
            LOGGER.warning("done making nrel curve")
        else:
            LOGGER.warning("loading nrel curve: from %s seconds ago", age)
            re_meta = pl.read_parquet(meta_path)

        re_meta = re_meta.filter(
            (
                pl.col("re_type").str.contains("onshore_wind")
                & (pl.col("distance_prof_site") <= 50)
                & (pl.col("distance") <= max_distance)
            )
            | (
                pl.col("re_type").str.contains("offshore_wind")
                & (pl.col("distance_prof_site") <= 50)
            )
            | (
                pl.col("re_type").str.contains("solar")
                & (pl.col("distance_prof_site") <= 101)
                & (pl.col("distance") <= max_distance)
            )
        )

        # re_meta2 = pl.read_parquet(CACHE_PATH / "re_meta_nrel_curve.parquet")
        # re_meta = pd.read_parquet(CACHE_PATH / "re_meta_.parquet")
        self.ad.counterfactual_re().columns.to_frame().reset_index(drop=True)
        if not (
            all_re_path := USER_DATA_PATH.parent / "all_re_new_too_big_tabled.parquet"
        ).exists():
            get("raw-data/all_re_new_too_big_tabled.parquet", all_re_path)
        all_profs = pl.scan_parquet(all_re_path, use_statistics=True).select(
            pl.col("plant_id_eia").alias("plant_id_prof_site"),
            "re_type",
            "datetime",
            "generation",
        )
        file = USER_DATA_PATH / "re_data.zip" if not test else BytesIO()

        # bas = ["MISO", "PJM", *sorted(set(re_meta.ba_code.unique()) - {"MISO", "PJM"})]
        bas = re_meta["ba_code"].unique().sort().to_list()

        with DataZip(file, "w", compression=compression) as z, logging_redirect_tqdm():
            for_loop = tqdm(bas, desc="group profiles by BA")
            for ba_code in for_loop:
                for_loop.set_description(desc="group profiles by BA: " + ba_code)
                try:
                    ba_re_meta, ba_re_prof = self._hourly_re_by_plant(
                        ba_code, re_meta, all_profs
                    )
                    if ba_re_prof.is_empty():
                        raise AssertionError(f"{ba_code} re profile df is empty")
                except Exception as exc:
                    LOGGER.error("%s %r", ba_code, exc)
                else:
                    # for some reason, ba data is not always processed correctly,
                    # adding checks based on errors in past runs
                    assert ba_re_meta["ba_code"].unique().item() == ba_code, (
                        f"{ba_code=} does not match specs"
                    )
                    assert (  # noqa: PT018
                        isinstance(ba_re_prof.columns, list) and "__" in ba_re_prof.columns[1]
                    ), f"{ba_code=} profile column names are in the wrong format"
                    z[ba_code + "_meta"] = ba_re_meta
                    z[ba_code + "_prof"] = ba_re_prof
                    z.reset_ids()
        if not test:
            put(file, f"patio-data/{PATIO_DATA_RELEASE}/")

    def _hourly_re_by_plant(
        self, ba_code: str, re_meta: pl.DataFrame, all_profs: pl.LazyFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create RE data for re_data.zip from raw RE profiles for ``ba_code``."""
        # we need some RE profiles that are not close to fossil plants
        combined = pl.format("{}__{}", "plant_id_prof_site", "re_type").alias("combined")
        re_meta = re_meta.filter(pl.col("ba_code") == ba_code).with_columns(combined)
        try:
            other_re = (
                self.ad.pl_counterfactual_re()
                .filter(pl.col("ba_code") == ba_code)
                .with_columns(combined)
            )

        except KeyError as exc:
            LOGGER.warning("no counterfactual RE for %s, %r", ba_code, exc)
            all_meta = re_meta
        else:
            all_meta = pl.concat([re_meta, other_re], how="diagonal")

        tz_code = self.ad.ba_offsets.loc[ba_code, "tz_code"]
        dt = datetime(self.ad.years[0], 1, 1, 0, tzinfo=ZoneInfo(tz_code))

        profs = (
            all_profs.filter(
                pl.col("plant_id_prof_site").is_in(all_meta["plant_id_prof_site"].unique())
                & (pl.col("datetime") >= dt.astimezone(UTC))
            )
            .select(
                combined,
                pl.col("datetime").dt.convert_time_zone(tz_code),
                pl.col("generation").cast(pl.Float32),
            )
            .collect()
            .pivot(values="generation", index="datetime", on="combined")
        )

        final_list = [x for x in all_meta["combined"].unique() if x in profs.columns]
        return re_meta, profs.select("datetime", *final_list)

    def existing_re_helper(self, missing, re_profiles):
        fyr, lyr = self.ad.years
        missing = (
            pl.from_pandas(missing)
            .sort("plant_id_eia", "generator_id", "datetime")
            .with_columns(pl.col("datetime").cast(pl.Datetime("ns")))
            .lazy()
        )

        existing_re = (
            missing.filter(pl.col("technology_description").is_in(list(RE_TECH)))
            .select(
                "plant_id_eia",
                "generator_id",
                "technology_description",
                pl.col("operating_date").cast(pl.Date),
            )
            .unique()
        )
        if existing_re.collect().is_empty():
            return (
                re_profiles.select("datetime", eia923=pl.lit(0.0), synthetic=pl.lit(0.0))
                .to_pandas()
                .set_index("datetime"),
                missing.collect().to_pandas(),
            )

        techs = {p.partition("__")[-1] for p in re_profiles.columns if "__" in p}
        mean_profiles = (
            re_profiles.select("datetime")
            .lazy()
            .with_columns(
                **{
                    k: pl.mean_horizontal(
                        re_profiles.select([x for x in re_profiles.columns if k in x])
                    ).cast(pl.Float32)
                    for k in techs
                },
            )
            .lazy()
            .melt(
                id_vars="datetime",
                value_vars=list(techs),
                value_name="mwh",
                variable_name="technology_description",
            )
            .with_columns(
                pl.col("technology_description").replace_strict(RE_TECH_R, default=None)
            )
        )
        # combine eia923m with synthesized monthly generation for RE generators before
        # they began operating
        combined_mly = self._fill_pre_op_mly_gen(existing_re, mean_profiles)

        missing_with_filled_re = self._fill_missing_w_pre_op_re(combined_mly, missing)

        prof_out = (
            (
                mean_profiles.with_columns(
                    date=pl.col("datetime")
                    .dt.month_start()
                    .cast(pl.Date)
                    .cast(pl.Datetime("ns"))
                )
                .with_columns(
                    mwh2=pl.col("mwh")
                    / pl.col("mwh").sum().over("technology_description", "date")
                )
                .join(
                    combined_mly.select(
                        "plant_id_eia",
                        "generator_id",
                        pl.col("datetime").alias("date"),
                        "technology_description",
                        "net_generation_mwh",
                        "source",
                    ),
                    on=["technology_description", "date"],
                    how="inner",
                )
                .select(
                    "datetime",
                    "source",
                    (pl.col("mwh2") * pl.col("net_generation_mwh")).alias("mwh"),
                )
                .group_by("datetime", "source")
                .agg(pl.sum("mwh"))
            )
            .sort("datetime", "source")
            .collect()
            .pivot(values="mwh", index="datetime", on="source")
        )
        if "synthetic" not in prof_out.columns:
            prof_out = prof_out.with_columns(synthetic=pl.lit(0.0))

        assert np.isclose(
            missing_with_filled_re.filter(
                pl.col("technology_description").is_in(list(RE_TECH))
            )
            .select("net_generation_mwh")
            .sum(),
            np.nansum(prof_out.sum().to_numpy()),
        ).all()

        return (
            prof_out.to_pandas().set_index("datetime"),
            missing_with_filled_re.to_pandas(),
        )

    @staticmethod
    def _fill_missing_w_pre_op_re(combined_mly, missing):
        annual_re = (
            combined_mly.group_by_dynamic(
                index_column="datetime",
                every="1y",
                period="1y",
                by=["plant_id_eia", "generator_id"],
            )
            .agg(pl.sum("net_generation_mwh"), pl.first("gen_source"))
            .with_columns(
                category=pl.col("gen_source").replace_strict(
                    {
                        "eia923": "existing_xpatio",
                        "synthetic": "old_clean",
                        "combined": "old_clean",
                    },
                    default=None,
                )
            )
            .join(
                missing.group_by("plant_id_eia", "generator_id")
                .agg(pl.all().last())
                .drop(["datetime", "net_generation_mwh", "category"]),
                how="left",
                on=["plant_id_eia", "generator_id"],
            )
            .select(*missing.columns)
        )
        missing_with_filled_re = (
            pl.concat(
                [
                    missing.filter(
                        pl.col("technology_description").is_in(list(RE_TECH)).not_()
                    ),
                    annual_re,
                ],
                how="vertical_relaxed",
            )
            .sort("plant_id_eia", "generator_id", "datetime")
            .collect()
        )
        missing_set = set(
            missing.select("plant_id_eia", "generator_id")
            .unique()
            .collect()
            .to_pandas()
            .itertuples(index=False, name=None)
        )
        filled_set = set(
            missing_with_filled_re.select("plant_id_eia", "generator_id")
            .unique()
            .to_pandas()
            .itertuples(index=False, name=None)
        )
        if missing_set - filled_set == {(64544, "WT2"), (64551, "WT2")}:
            LOGGER.warning(
                "ProfileData._fill_missing_w_pre_op_re is DISCARDING (64544, 'WT2'), (64551, 'WT2')"
            )
            missing = missing.filter(
                ((pl.col("plant_id_eia") == 64544) & (pl.col("generator_id") == "WT2")).not_()
                & (
                    (pl.col("plant_id_eia") == 64551) & (pl.col("generator_id") == "WT2")
                ).not_()
            )
        if missing_set - filled_set == {(64864, "GEN01"), (64864, "CHAPS"), (319, "2")}:
            LOGGER.warning(
                "ProfileData._fill_missing_w_pre_op_re is DISCARDING (64864, 'GEN01'), (64864, 'CHAPS'), (319, '2')"
            )
            missing = missing.filter(
                (
                    (pl.col("plant_id_eia") == 64864) & (pl.col("generator_id") == "GEN01")
                ).not_()
                & (
                    (pl.col("plant_id_eia") == 64864) & (pl.col("generator_id") == "CHAPS")
                ).not_()
                & ((pl.col("plant_id_eia") == 319) & (pl.col("generator_id") == "2")).not_()
            )
        if missing_set - filled_set == {(66626, "WADLE")}:
            LOGGER.warning(
                "ProfileData._fill_missing_w_pre_op_re is DISCARDING (66626, 'WADLE')"
            )
            missing = missing.filter(
                (
                    (pl.col("plant_id_eia") == 66626) & (pl.col("generator_id") == "WADLE")
                ).not_()
            )
        if missing_set - filled_set == {(56458, "CHW-T")}:
            LOGGER.warning(
                "ProfileData._fill_missing_w_pre_op_re is DISCARDING (56458, 'CHW-T')"
            )
            missing = missing.filter(
                (
                    (pl.col("plant_id_eia") == 56458) & (pl.col("generator_id") == "CHW-T")
                ).not_()
            )

        if len(missing_with_filled_re.select("plant_id_eia", "generator_id").unique()) != len(
            missing.select("plant_id_eia", "generator_id").unique().collect()
        ):
            raise AssertionError(
                "the number of generators in existing_xpatio/old_clean changed when we "
                f"added synthetic annual data for RE generators before they began operating, namely:\n{missing_set - filled_set}"
            )
        return missing_with_filled_re

    def _fill_pre_op_mly_gen(self, existing_re, mean_profiles):
        mo_vs_mo_avg = (
            mean_profiles.group_by_dynamic(
                index_column="datetime",
                every="1mo",
                period="1mo",
                by="technology_description",
            )
            .agg(pl.sum("mwh"))
            .with_columns(
                adj=pl.col("mwh")
                / pl.col("mwh")
                .mean()
                .over("technology_description", pl.col("datetime").dt.month()),
                month=pl.col("datetime").dt.month(),
            )
        )
        eia923_re_mly = self.ad.df923m_clean.select(
            pl.col("report_date").alias("date"),
            "plant_id_eia",
            "generator_id",
            "net_generation_mwh",
        ).join(
            existing_re.select("plant_id_eia", "generator_id", "technology_description"),
            on=["plant_id_eia", "generator_id"],
            how="inner",
        )
        synth_re_mly = (
            eia923_re_mly.group_by(
                "plant_id_eia",
                "generator_id",
                pl.col("date").dt.month().alias("month"),
            )
            .agg(
                pl.first("technology_description"),
                pl.mean("net_generation_mwh"),
            )
            .join(mo_vs_mo_avg, on=["technology_description", "month"])
            .select(
                "plant_id_eia",
                "generator_id",
                "technology_description",
                pl.col("datetime").cast(pl.Date).alias("date"),
                (pl.col("net_generation_mwh") * pl.col("adj")).alias(
                    "net_generation_mwh_synth"
                ),
            )
        )
        return (
            eia923_re_mly.rename({"net_generation_mwh": "net_generation_mwh_real"})
            .join(
                synth_re_mly,
                on=["plant_id_eia", "generator_id", "technology_description", "date"],
                how="outer_coalesce",
            )
            .join(
                existing_re,
                on=["plant_id_eia", "generator_id", "technology_description"],
                how="inner",
            )
            .with_columns(
                net_generation_mwh=pl.when(pl.col("date") > pl.col("operating_date"))
                .then(pl.col("net_generation_mwh_real"))
                .otherwise(pl.col("net_generation_mwh_synth"))
                .fill_null(0.0),
                source=pl.when(pl.col("date") > pl.col("operating_date"))
                .then(pl.lit("eia923"))
                .otherwise(pl.lit("synthetic")),
            )
            .select(
                "plant_id_eia",
                "generator_id",
                pl.col("date").cast(pl.Datetime("ns")).alias("datetime"),
                "technology_description",
                "net_generation_mwh",
                "source",
                pl.when(
                    pl.col("source")
                    .n_unique()
                    .over("plant_id_eia", "generator_id", pl.col("date").dt.year())
                    > 1
                )
                .then(pl.lit("combined"))
                .otherwise(pl.col("source"))
                .alias("gen_source"),
            )
            .filter(
                pl.col("datetime").dt.year()
                <= mean_profiles.select(pl.col("datetime").dt.year()).max().collect().item()
            )
            .sort("plant_id_eia", "generator_id", "datetime")
        )

    @staticmethod
    def re_to_parquet():
        import pyarrow.parquet as pq

        full = pl.scan_parquet(Path.home() / "patio_data/all_re_new_too_big.parquet")
        unique = full.select("plant_id_eia", "re_type").unique().collect()
        schema = full.head().collect().to_arrow().schema
        out_path = Path.home() / "patio_data/all_re_new_too_big_tabled.parquet"

        if (ROOT_PATH / "temp/re").exists():
            sol = pl.scan_parquet(ROOT_PATH / "temp/re/*solar.parquet")
            win = pl.scan_parquet(ROOT_PATH / "temp/re/*wind.parquet")
            with_cols = {
                "plant_id_eia": pl.col("plant_id_eia").cast(pl.Int64),
                "datetime": pl.col("__index_level_0__").dt.replace_time_zone("UTC"),
                "plant_id_in_data": pl.col("plant_id_eia"),
            }
            process_new = True
        else:
            sol, win = (
                pl.LazyFrame(schema=full.schema),
                pl.LazyFrame(schema=full.schema),
            )
            with_cols = {}
            process_new = False

        with (
            pq.ParquetWriter(out_path, schema, compression="snappy", version="2.6") as writer,
            logging_redirect_tqdm(),
        ):
            for pid, re_type in tqdm(unique.iter_rows(), total=unique.shape[0]):
                selector = (pl.col("plant_id_eia") == pid) & (pl.col("re_type") == re_type)
                if process_new and "wind" in re_type:
                    new = (
                        win.filter(selector)
                        .with_columns(
                            **with_cols,
                            irradiance_direct=pl.lit(None).cast(pl.Float64),
                            irradiance_diffuse=pl.lit(None).cast(pl.Float64),
                            temperature=pl.lit(None).cast(pl.Float64),
                        )
                        .select(full.columns)
                    )
                elif process_new and "solar" in re_type:
                    new = (
                        sol.filter(selector)
                        .with_columns(**with_cols, wind_speed=pl.lit(None).cast(pl.Float64))
                        .select(full.columns)
                    )
                else:
                    new = pl.LazyFrame(schema=full.schema)
                writer.write_table(
                    # Concat a slice of each state's data from all quarters in a year
                    # and write to parquet to create year-state row groups
                    pl.concat([full.filter(selector), new])
                    # .unique(["plant_id_eia", "re_type", "datetime"])
                    # .sort("datetime")
                    .collect()
                    .to_arrow()
                )

    @staticmethod
    def re_to_parquet_old():
        # inspiration?
        # https://github.com/catalyst-cooperative/pudl/blob/a19bd57dea11fb63eee2ae4d7a03bb8e8e6ce83d/src/pudl/etl/epacems_assets.py#L103
        LOGGER.warning("reading new profiles")
        cols = [
            "plant_id_eia",
            "re_type",
            "datetime",
            "generation",
            "plant_id_in_data",
            "irradiance_direct",
            "irradiance_diffuse",
            "temperature",
            "wind_speed",
        ]
        glob = list(ROOT_PATH.glob("temp/re/*.parquet"))
        LOGGER.warning("this can take multiple minutes without providing any feedback")
        sol = (
            pl.scan_parquet(ROOT_PATH / "temp/re/*solar.parquet")
            .select(
                [
                    pl.col("plant_id_eia").cast(pl.Int64),
                    "re_type",
                    pl.col("__index_level_0__").dt.replace_time_zone("UTC").alias("datetime"),
                    "generation",
                    "irradiance_direct",
                    "irradiance_diffuse",
                    "temperature",
                ]
            )
            .with_columns(
                pl.col("plant_id_eia").alias("plant_id_in_data"),
                wind_speed=pl.lit(None).cast(pl.Float64),
            )
            .select(cols)
        )
        LOGGER.warning("solar scanned")
        win = (
            pl.scan_parquet(ROOT_PATH / "temp/re/*wind.parquet")
            .select(
                [
                    pl.col("plant_id_eia").cast(pl.Int64),
                    "re_type",
                    pl.col("__index_level_0__").dt.replace_time_zone("UTC").alias("datetime"),
                    "generation",
                    "wind_speed",
                ]
            )
            .with_columns(
                pl.col("plant_id_eia").alias("plant_id_in_data"),
                irradiance_direct=pl.lit(None).cast(pl.Float64),
                irradiance_diffuse=pl.lit(None).cast(pl.Float64),
                temperature=pl.lit(None).cast(pl.Float64),
            )
            .select(cols)
        )
        LOGGER.warning("wind scanned")

        LOGGER.warning("this can take multiple minutes without providing any feedback")

        if (Path.home() / "patio_data/all_re.parquet").exists():
            (Path.home() / "patio_data/all_re.parquet").rename(
                Path.home() / "patio_data/all_re_old.parquet"
            )

        old = pl.scan_parquet(Path.home() / "patio_data/all_re_old.parquet").select(cols)
        LOGGER.warning("old scanned")
        full = pl.concat([old, win, sol], parallel=True).sort(
            ["plant_id_eia", "re_type", "datetime"]
        )
        LOGGER.warning("concat added to plan")
        LOGGER.warning("sinking result")
        full.sink_parquet(Path.home() / "patio_data/all_re.parquet", statistics=True)
        LOGGER.warning(
            "combination successful, if things work you can delete the temp "
            "files and `all_re_old.parquet`"
        )
        old_re_temp = ROOT_PATH / "temp/re_o"
        if not old_re_temp.exists():
            (ROOT_PATH / "temp/re").rename(old_re_temp)
        else:
            for file in glob:
                file.rename(old_re_temp / file.name)
            (ROOT_PATH / "temp/re").rmdir()

    @cached_property
    def dt_range_m(self) -> pd.DatetimeIndex:
        return pd.date_range(
            f"01/01/{self.ad.years[0]}",
            f"12/31/{self.ad.years[1] + 1}",
            freq="MS",
        )

    def really_modelable_plants(self):
        try:
            return pd.read_parquet(CACHE_PATH / "modelable_plants.parquet")
        except FileNotFoundError:
            cw_gen = (
                pl.from_pandas(self.ad.cw)
                .select(["plant_id_eia", "generator_id", "subplant_id", "capacity_mw"])
                .with_columns(pl.col("plant_id_eia").cast(pl.Int32))
                .unique(subset=["plant_id_eia", "generator_id"])
                .lazy()
            )
            out = (
                self.subplant_cems()
                .join(cw_gen, on=["plant_id_eia", "subplant_id"], how="inner")
                .with_columns(pl.col("normed_gen").fill_null(0.0).fill_nan(0.0))
                .group_by("plant_id_eia")
                .agg(pl.sum("normed_gen"))
                .filter(pl.col("normed_gen") > 0.0)
                .collect()
            )
            in_cems = set(out.to_pandas().plant_id_eia)
            in_cfl_cost = set(
                self.ad.counterfactual_cost()
                .query("final_ba_code not in ('Alaska', 'HECO', 'NBSO', 'WAUW')")
                .index.get_level_values("plant_id_eia")
            )
            plants = in_cems.intersection(in_cfl_cost)  # noqa: F841

            result = self.ad.modelable_generators().query("plant_id_eia in @plants")
            result.to_parquet(CACHE_PATH / "modelable_plants.parquet")
            return result

    def get_re_for_dl2(self, final_year=2022):
        df = pl.scan_parquet(
            Path.home() / "patio_data/re_curve_to_fossil.parquet"
        ).with_columns(pl.col("fos_id").cast(pl.Int32))
        norm_subplant_cems = (
            self.subplant_cems()
            .select("plant_id_eia", "subplant_id")
            .unique()
            .collect()
            .lazy()
        )
        cw_gen = (
            pl.from_pandas(self.ad.cw)
            .select(["plant_id_eia", "generator_id", "subplant_id"])
            .with_columns(pl.col("plant_id_eia").cast(pl.Int32))
            .unique(subset=["plant_id_eia", "generator_id"])
            .lazy()
        )
        in_cems = (
            norm_subplant_cems.join(cw_gen, on=["plant_id_eia", "subplant_id"], how="inner")
            .select(["plant_id_eia", "generator_id"])
            .unique()
            .sort(["plant_id_eia", "generator_id"])
            .collect()
        )
        too_far = (
            df.filter(
                (
                    pl.col("re_type").str.contains("onshore_wind")
                    & (pl.col("distance_prof_site") > 50)
                    & (pl.col("class_atb") < 10)
                )
                | (
                    pl.col("re_type").str.contains("offshore_wind")
                    & (pl.col("distance_prof_site") > 50)
                    & (pl.col("class_atb") < 12)
                )
                | (
                    pl.col("re_type").str.contains("solar")
                    & (pl.col("distance_prof_site") > 100)
                )
            )
            .rename({"fos_id": "plant_id_eia", "fos_gen": "generator_id"})
            .join(in_cems.lazy(), on=["plant_id_eia", "generator_id"], how="inner")
            .rename({"plant_id_eia": "fos_id", "generator_id": "fos_gen"})
            .collect()
            .to_pandas()
            .pipe(
                round_coordinates,
                tol=0.5,
                in_cols=("latitude_nrel_site", "longitude_nrel_site"),
                out_cols=("latitude", "longitude"),
            )
            .drop_duplicates(subset=["latitude", "longitude", "re_type"])
            .assign(
                plant_id_eia=lambda x: hash_pandas_object(x[["latitude", "longitude"]]).astype(
                    "int64"
                )
            )[
                [
                    "plant_id_eia",
                    "re_type",
                    "latitude",
                    "longitude",
                    "cf_atb",
                    "class_atb",
                ]
            ]
        )
        dl_re = (
            pl.scan_parquet(Path.home() / "patio_data/all_re.parquet", use_statistics=True)
            .group_by(["plant_id_eia", "re_type"])
            .agg(
                pl.col("datetime").dt.year().min().alias("first_year"),
                pl.col("datetime").dt.year().max().alias("last_year"),
            )
            .sort(["plant_id_eia", "re_type"])
            .collect()
            .to_pandas()
        )

        final = too_far.merge(
            dl_re, on=["plant_id_eia", "re_type"], how="left", validate="1:1"
        ).assign(
            technology_description=lambda x: x.re_type,
            dl_first=lambda x: x.last_year.fillna(2005).astype(int) + 1,
            dl_last=final_year,
            years=lambda x: x[["dl_first", "dl_last"]]
            .astype(str)
            .agg("-".join, axis=1)
            .mask(x.dl_first > x.dl_last, "drop"),
        )

        old_ids = pl.read_csv(
            PACKAGE_DATA_PATH / "re_site_ids.csv",
            dtypes={
                "plant_id_eia": pl.Int64,
                "re_type": pl.Utf8,
                "latitude": pl.Float64,
                "longitude": pl.Float64,
                "class_atb": pl.Int64,
            },
        )

        px.scatter_geo(
            pd.concat(
                [too_far.assign(series="new"), old_ids.to_pandas().assign(series="old")]
            ).assign(
                size=1,
                latitude=lambda x: x.latitude.mask(x.re_type == "solar", x.latitude + 0.1),
            ),
            lat="latitude",
            lon="longitude",
            symbol="series",
            color="re_type",
            size="size",
            size_max=2,
            opacity=0.5,
        ).update_traces(marker=dict(line=dict(width=0))).update_layout(  # noqa: C408
            geo_scope="usa"
        ).write_image(ROOT_PATH / "temp/re_sites.pdf")

        pl.concat([old_ids, pl.from_pandas(final).select(old_ids.columns)]).write_csv(
            PACKAGE_DATA_PATH / "re_site_ids.csv"
        )

        final.to_parquet(ROOT_PATH / "temp/to_dl.parquet")

        return final
