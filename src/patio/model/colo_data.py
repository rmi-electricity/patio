import shutil
from _operator import or_
from functools import reduce

import geopandas as gpd
import pandas as pd
import polars as pl
from etoolbox.utils.cloud import get
from etoolbox.utils.pudl import pl_read_pudl, pl_scan_pudl

from patio.constants import PATIO_DATA_AZURE_URLS
from patio.data.asset_data import USER_DATA_PATH, AssetData, add_id_ferc714
from patio.helpers import pl_distance
from patio.model.colo_common import pl_filter
from patio.package_data import PACKAGE_DATA_PATH


def get_lse(lat, lon, pudl_release):
    if not (file := USER_DATA_PATH / "Electric_Retail_Service_Territories.kml").exists():
        if not file.with_suffix(".kml.zip").exists():
            get(PATIO_DATA_AZURE_URLS["lse"], file.with_suffix(".kml.zip"))
        shutil.unpack_archive(file.with_suffix(".kml.zip"), USER_DATA_PATH)
        file.with_suffix(".kml.zip").unlink()
    utils = (
        pl_scan_pudl("core_eia861__yearly_operational_data_misc", release=pudl_release)
        .filter(
            pl.col("retail_sales_mwh").is_not_null()
            & (pl.col("retail_sales_mwh") > 0)
            & pl.col("data_observed")
        )
        .sort("report_date")
        .group_by(id="utility_id_eia")
        .agg(
            pl.last("nerc_region"),
            entity_type_lse=pl.last("entity_type"),
            utility_name_eia_lse=pl.last("utility_name_eia"),
        )
        .collect()
    )
    ba_codes = pl_read_pudl(
        "core_eia861__yearly_balancing_authority", release=pudl_release
    ).filter(pl.col("balancing_authority_code_eia").is_not_null()).group_by(
        pl.col("balancing_authority_name_eia").str.to_lowercase()
    ).agg(pl.last("balancing_authority_code_eia")).to_pandas().set_index(
        "balancing_authority_name_eia"
    ).squeeze().to_dict() | {
        "louisville gas and electric company and kentucky utilities": "LGEE"
    }
    lse = (
        gpd.read_file(file)
        .query(
            "~id.str.contains('NA') & TYPE in ('MUNICIPAL', 'INVESTOR OWNED', 'COOPERATIVE','POLITICAL SUBDIVISION', 'STATE',)"
        )
        .astype({"id": int})
        .assign(
            balancing_authority_code_eia=lambda x: x.CNTRL_AREA.str.casefold().replace(
                ba_codes
            ),
        )
        .pipe(add_id_ferc714, util_id_col="id", pudl_release=pudl_release)
        .merge(utils.to_pandas(), on="id", how="inner", validate="m:1")
    )
    pgeo = pd.DataFrame({"re_site_id": [0], "lat": [lat], "lon": [lon]})
    return pl.from_pandas(
        pd.DataFrame(
            gpd.sjoin(
                gpd.GeoDataFrame(
                    pgeo[["re_site_id"]],
                    geometry=gpd.points_from_xy(x=pgeo["lon"], y=pgeo["lat"], crs="EPSG:4326"),
                ),
                lse,
                how="left",
                predicate="within",
            )
        ).drop(columns=["geometry"])
    )


def get_re_data(lat, lon, sqkm):
    sites = (
        pl.read_csv(PACKAGE_DATA_PATH / "re_site_ids.csv")
        .filter(pl.col("re_type").is_in(("solar", "onshore_wind")))
        .with_columns(lat=lat, lon=lon)
        .pipe(pl_distance, "latitude", "lat", "longitude", "lon")
        .sort("distance")
        .group_by("re_type")
        .agg(pl.first("plant_id_eia"))
    )
    if not (
        all_re_path := USER_DATA_PATH.parent / "all_re_new_too_big_tabled.parquet"
    ).exists():
        get("patio-data/all_re_new_too_big_tabled.parquet", all_re_path)
    profs = (
        pl.scan_parquet(all_re_path)
        .filter(reduce(or_, [pl_filter(**v) for v in sites.iter_rows(named=True)]))
        .collect()
    )
    solar, wind, _ = AssetData.raw_curves(2025, regime="reference", pudl_release="v2025.7.0")
    nrel_site_data = (
        sites.join(
            pl.from_pandas(
                pd.concat([solar.assign(re_type="solar"), wind.assign(re_type="onshore_wind")])
            ),
            on="re_type",
        )
        .with_columns(lat=lat, lon=lon)
        .pipe(pl_distance, "latitude", "lat", "longitude", "lon")
        .sort("distance")
    )
    combi_id = pl.concat_str(pl.lit("0"), pl.lit("0"), "re_type", separator="_")
    # these specs represent the site itself, addl sites might be intereting to include?
    specs = (
        nrel_site_data.group_by("re_type")
        .agg(pl.all().first())
        .rename({"capacity_mw_ac": "capacity_mw_nrel_site"})
        .with_columns(
            icx_genid=pl.lit("0"),
            re_site_id=0,
            combi_id=combi_id,
            area_per_mw=pl.col("area_sq_km") / pl.col("capacity_mw_nrel_site"),
            capacity_mw_nrel_site=(pl.col("capacity_mw_nrel_site") / pl.col("area_sq_km"))
            * sqkm,
            area_sq_km=sqkm,
        )
        .sort("combi_id")
    )

    return specs, profs.with_columns(combi_id=combi_id).pivot(
        on="combi_id", index="datetime", values="generation"
    ).select(
        "datetime",
        *specs["combi_id"],
    )
