"""Interface to download wind and solar profiles from renewables.ninja"""

import json
import logging
import time
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
import requests
from etoolbox.datazip import DataZip
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from patio.constants import ROOT_PATH

LOGGER = logging.getLogger("patio")
__all__ = ["RenewablesNinja", "rename_re", "zip_re"]


def rename_re(re):
    errors = []
    for _i, row in re.iterrows():
        pid, name, tech = (
            row["plant_id_eia"],
            row["name"],
            RenewablesNinja.ptype_fixer(row["technology_description"]),
        )
        old = ROOT_PATH / f"temp/{name}_{tech}.parquet"
        if pid != name:
            new = ROOT_PATH / f"temp/{pid}_{tech}.parquet"
            try:
                pd.read_parquet(old).rename(columns={str(name): str(pid)}).to_parquet(new)
            except Exception as exc:
                LOGGER.error("%r", exc, exc_info=True)  # noqa: G201
                errors.append((pid, name, tech))
            else:
                old.unlink()
            print(f"{old} -> {new}")
        else:
            print(f"{old} no change")
    return errors


def zip_re():
    """Create new DataZip that includes old DataZip and new profiles."""
    files = list(ROOT_PATH.glob("temp/re/*.parquet"))
    with logging_redirect_tqdm():
        if (ROOT_PATH / "all_re.zip").exists():
            (ROOT_PATH / "all_re.zip").rename(ROOT_PATH / "all_re_old.zip")
            add_old = True
        else:
            add_old = False

        with DataZip(ROOT_PATH / "all_re", "w") as znew:
            for f in tqdm(files, total=len(files), desc="from temp dir"):
                znew[f.stem] = pd.read_parquet(f)

            if add_old:
                with DataZip(ROOT_PATH / "all_re_old", "r", ignore_pd_dtypes=True) as zold:
                    for name, df in tqdm(zold.items(), total=len(zold), desc="from old zip"):
                        if name not in znew:  # noqa: SIM102
                            if isinstance(df, pd.DataFrame | pd.Series):
                                znew[name] = df


class RenewablesNinja:
    tokena = "d16413d91e073711a10808cd2bc883e7257e9ddd"
    token0 = "92ed13382e2844a5b8f8bf24cc9bf46a582e48b9"
    token1 = "75a451c1bbf90638af436f8fbad63bf18636ebf3"
    token2 = "5393ce87b475ce4702f04d2ad5aa39fcbf0b5381"
    token3 = "e22dec42a92ba14dd1cd31c025ad5fc40f16b972"
    token4 = "58ca33f994e0e8346144cf65da8307ed1205c505"
    api_base = "https://www.renewables.ninja/api/"
    token = tokena
    _re_type_fixer = {
        "solar": "pv",
        "pv": "pv",
        "wind": "wind",
        "onshore_wind": "wind",
        "offshore_wind": "off_wind",
        "Onshore Wind Turbine": "wind",
        "Offshore Wind Turbine": "off_wind",
        "Solar Photovoltaic": "pv",
    }
    _re_file_fixer = {
        "pv": "solar",
        "wind": "onshore_wind",
        "Wind": "onshore_wind",
        "Onshore Wind Turbine": "onshore_wind",
        "Offshore Wind Turbine": "offshore_wind",
        "Solar Photovoltaic": "solar",
    }
    _token_names = {
        tokena: "a",
        token0: "0",
        token1: "1",
        token2: "2",
        token3: "3",
        token4: "4",
    }
    # TODO deal with offshore wind

    def __init__(self):
        self.tokens = {
            self.tokena: time.time(),
            self.token0: time.time(),
            self.token1: time.time(),
            self.token2: time.time(),
            self.token3: time.time(),
            self.token4: time.time(),
        }

    @property
    def best_token(self):
        return min(self.tokens, key=self.tokens.get)

    @property
    def wait_times(self):
        now = time.time()
        return {k: int(np.ceil(v - now)) for k, v in self.tokens.items()}

    @property
    def str_wait_times(self):
        now = time.time()
        return ", ".join(
            f"{self._token_names[k]}={int(np.ceil(v - now))}" for k, v in self.tokens.items()
        )

    def get_re_profile(
        self,
        re_type: str,
        name: Hashable,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Download hourly renewables data from renewables.ninja

        :param re_type: type of renewable data to download, pv or wind
        :type re_type: str
        :param name: name for the solar data series
        :type name: Hashable
        :param lat: latitude to pull solar data for
        :type lat: float
        :param lon: longitude to pull solar data for
        :type lon: float
        :param date_from: start date in the form YYYY-MM-DD
        :type date_from: str
        :param date_to: end date in the form YYYY-MM-DD
        :type date_to: str
        :param kwargs:

        :return: timeseries of one year's renewable capacity factors for (lat, lon)
        :rtype: DataFrame
        """
        base_args = {
            "lat": lat,
            "lon": lon,
            "date_from": date_from,
            "date_to": date_to,
            "capacity": 1.0,
            "format": "json",
            "local_time": True,
            "raw": True,
        }
        if self._re_type_fixer[re_type.casefold()] == "pv":
            api_suffix = "data/pv"
            args = {
                **base_args,
                "dataset": "merra2",
                "system_loss": 0.1,
                "tracking": 1,
                "tilt": lat,
                "azim": 180,
            }
        elif self._re_type_fixer[re_type.casefold()] == "wind":
            api_suffix = "data/wind"
            args = {
                **base_args,
                "height": 100,
                "turbine": "Vestas V80 2000",
            }
        elif self._re_type_fixer[re_type.casefold()] == "off_wind":
            api_suffix = "data/wind"
            args = {
                **base_args,
                "height": 100,
                "turbine": "Vestas V164 7000",
            }
        else:
            raise NotImplementedError(f"{re_type} not supported, use 'pv' or 'wind'")
        args = {k: kwargs.get(k, v) for k, v in args.items()}
        # print(f" {args['date_from'][:4]}", end="", flush=True)
        r = None
        while r is None:
            r = self._get_prof(api_suffix, args)
        parsed_response = json.loads(r.text)
        return pd.read_json(json.dumps(parsed_response["data"]), orient="index").rename(
            columns={"electricity": "generation"}
        )

    def _get_prof(self, api_suffix, args):
        token = self.best_token
        wait_time = int(self.wait_times[token])
        if wait_time > 0:
            if wait_time > 15:
                LOGGER.warning("%s -> waiting %s seconds", self.str_wait_times, wait_time)
            time.sleep(wait_time)
        else:
            time.sleep(0.5)
        with requests.session() as s:
            s.headers = {"Authorization": "Token " + token}
            r = s.get(self.api_base + api_suffix, params=args)
            if "Error" in r.text:
                self.tokens.update({token: time.time() + self._wait_time(r) + 1})
                # print(".", end="", flush=True)
                return None
            return r

    @staticmethod
    def _wait_time(r):
        return int(f"{r.text}".partition("available in ")[2].partition(" ")[0])

    def get_re_set(
        self,
        data: pd.DataFrame,
        lat: str = "latitude",
        lon: str = "longitude",
        name: str = "plant_id_eia",
        years: str = "years",
        re_type_col: str = "technology_description",
        redownload: Sequence = (),
    ):
        # with DataZip(ROOT_PATH / "all_re") as z:
        #     in_zip = list(z.keys())
        #
        in_zip = []
        re_dir = ROOT_PATH / "temp/re"
        if not re_dir.exists():
            re_dir.mkdir(parents=True)
            in_dir = []
        else:
            in_dir = [x.stem for x in re_dir.glob("*.parquet")]

        existing = sorted(set(in_dir + in_zip) - set(redownload))
        _ = existing  # so ruff doesn't remove it
        data = data.assign(
            stem=lambda x: x.plant_id_eia.astype(str).str.cat(
                x.technology_description.str.casefold(), sep="_"
            )
        ).query("stem not in @existing")

        with logging_redirect_tqdm():
            for _, row in tqdm(data.iterrows(), total=len(data), position=0):
                close = True
                try:
                    loc_list = []
                    re_name = str(row.at[name])
                    re_type = row.at[re_type_col]
                    fi, test, la = row.at[years].partition("-")
                    years_list = range(int(fi), int(la) + 1) if test else [int(fi)]
                    for year in tqdm(years_list, position=1, leave=None):
                        try:
                            loc_list.append(
                                self.get_re_profile(
                                    re_type=re_type,
                                    name=re_name,
                                    lat=row.at[lat],
                                    lon=row.at[lon],
                                    date_from=f"{int(year)}-01-01",
                                    date_to=f"{int(year)}-12-31",
                                )
                            )
                        except NotImplementedError as exc:
                            LOGGER.error("%r", exc)
                    pd.concat(loc_list, axis=0).assign(
                        plant_id_eia=re_name, re_type=re_type
                    ).to_parquet(
                        re_dir
                        / f"{re_name}_{self._re_file_fixer.get(re_type, re_type.casefold())}.parquet"
                    )
                except Exception as exc:
                    LOGGER.error("%r", exc, exc_info=exc)
                finally:
                    if close:
                        print("")

    @classmethod
    def ptype_fixer(cls, item):
        return cls._re_file_fixer.get(item, item)

    @staticmethod
    def optimal_tilt(lat: float) -> float:
        """Method lifted from gsee [1]

        Returns an optimal tilt angle for the given ``lat``, assuming that
        the panel is facing towards the equator, using a simple method from [2].
        This method only works for latitudes between 0 and 50. For higher
        latitudes, a static 40 degree angle is returned.
        These results should be used with caution, but there is some
        evidence that tilt angle may not be that important [3].

        :param lat: Latitude in degrees.
        :type lat: float
        :return: Optimal tilt angle in degrees.

        [1] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py
        [2] http://www.solarpaneltilt.com/#fixed
        [3] http://dx.doi.org/10.1016/j.solener.2010.12.014


        """
        lat = abs(lat)
        if lat <= 25:
            return lat * 0.87
        if lat <= 50:
            return (lat * 0.76) + 3.1
        return 40  # Simply use 40 degrees above lat 50

    @staticmethod
    def agg_8760s(df, groupby_level=0):
        if isinstance(df.columns[0], tuple) and not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df.groupby(axis=0, level=groupby_level).mean()
