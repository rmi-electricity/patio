import io
import json
import logging
import tomllib
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from dispatch.constants import COLOR_MAP
from etoolbox.utils.cloud import rmi_cloud_fs
from etoolbox.utils.misc import all_logging_disabled
from etoolbox.utils.pudl import pl_scan_pudl
from plotly.subplots import make_subplots
from pypdf import PdfReader, PdfWriter
from tqdm.auto import tqdm

from patio.constants import REGION_MAP, TECH_CODES
from patio.data.asset_data import AssetData, clean_atb
from patio.model.colo_common import (
    AEO_MAP,
    COSTS,
    FANCY_COLS,
    LAND,
    f_pmt,
    order_columns,
    pl_exc_fmt,
    pl_filter,
)
from patio.model.colo_lp import Info

LOGGER = logging.getLogger("patio")


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


def sankey(h3):
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


class Results:
    id_cols = ("ba_code", "icx_id", "icx_gen", "regime", "name")
    renamer = pl.col("name").replace(
        {"form_fos": "form_new_fossil", "pure_surplus": "new_fossil"}
    )
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

    def __init__(
        self,
        *results,
        min_served=1,
        min_pct_load_clean=0.6,
        max_ppa=200,
        max_reg_rank=9,
        max_violation=0.50,
        max_attr_rev_export_clean=100,
        font="Gill Sans",
    ):
        self.fs = rmi_cloud_fs()
        self.min_served = min_served
        self.min_pct_load_clean = min_pct_load_clean
        self.max_ppa = max_ppa
        self.max_reg_rank = max_reg_rank
        self.max_violation = max_violation
        self.max_attr_rev_export_clean = max_attr_rev_export_clean
        self.font = font
        self.names = sorted(r.removeprefix("colo_") for r in results)
        self.name_indices = dict(enumerate(self.names)) | {int(r): r for r in self.names}
        with (
            all_logging_disabled(),
            self.fs.open(f"patio-results/colo_{self.last}/patio.toml") as f,
        ):
            self.config = tomllib.load(f)
        self.pudl_release = self.config["project"]["pudl_release"]
        with all_logging_disabled():
            self.ad = AssetData(pudl_release=self.pudl_release)
        self.summaries: dict[str, pl.DataFrame] = {}
        self.flows: dict[str, pl.DataFrame] = {}
        self.load()
        self.fig_path = Path.home() / "patio_data/figures"
        if not self.fig_path.exists():
            if not self.fig_path.parent.exists():
                self.fig_path.parent.mkdir()
            self.fig_path.mkdir()

    def __getitem__(self, item=None):
        return self.summaries[self.norm_name(item)]

    def update_params(self, **kwargs):
        update = False
        for k, v in kwargs.items():
            if k in (
                "min_served",
                "min_pct_load_clean",
                "max_ppa",
                "max_reg_rank",
                "max_violation",
                "max_attr_rev_export_clean",
            ):
                setattr(self, k, v)
                update = True
        if update:
            self.load()

    def load(self):
        self.flows = {}
        for run in self.names:
            try:
                self.summaries[run] = self.get_summary(run)
            except Exception as exc:
                print(f"no summary for {run} {pl_exc_fmt(exc)}")

    def get_aligned(self, run=None, *, land_screen=False):
        run = self.norm_name(run)
        df = self.all_summaries() if run == "all" else self.summaries[run]
        if land_screen:
            df = df.filter(pl.col("land_available"))
        return self.best_at_site(df)

    @property
    def last(self):
        return max(self.names)

    def good_screen[T: pl.DataFrame | pl.LazyFrame](self, df: T) -> T:
        return df.filter(
            (pl.col("run_status") == "SUCCESS")
            & (pl.col("served_pct") >= self.min_served)
            & (pl.col("pct_load_clean") >= self.min_pct_load_clean)
            & (pl.col("ppa_ex_fossil_export_profit") <= self.max_ppa)
            & (pl.col("reg_rank") <= self.max_reg_rank)
            & (pl.col("attr_rev_export_clean") < self.max_attr_rev_export_clean)
        ).filter(
            (pl.col("new_fossil_mw").fill_null(0.0) == 0.0)
            | (
                (pl.col("new_fossil_mw") > 0.0)
                & (pl.col("max_rolling_violation_pct") <= self.max_violation)
            )
        )

    def all_summaries(self):
        return pl.concat(
            [v for _, v in self.summaries.items()],
            how="diagonal_relaxed",
        )

    def best_at_site[T: pl.DataFrame | pl.LazyFrame](self, df: T) -> T:
        colo_cols = df.lazy().collect_schema().names()
        r_col = ("run",) if "run" in colo_cols else ()
        return (
            df.pipe(self.good_screen)
            .sort("load_mw", descending=True)
            .group_by("icx_id", "icx_gen", *r_col)
            .agg(pl.all().first())
            .select(*colo_cols)
            .sort("state", "utility_name_eia", "plant_name_eia")
        )

    def source_for_load(self, run):
        run = self.norm_name(run)
        flows = self.get_flows(run)
        es_for_load = flows.filter(
            (pl.col("target_") == "Load") & (pl.col("source_") == "Storage")
        ).select(*self.id_cols, "value")
        es_allocated = (
            flows.filter(pl.col("target_") == "Storage")
            .with_columns(pl.col("value") / pl.sum("value").over(*self.id_cols))
            .join(es_for_load, on=self.id_cols, how="left", validate="m:1")
            .select(
                *self.id_cols,
                "source_",
                target_=pl.lit("Load"),
                value=pl.col("value") * pl.col("value_right"),
            )
        )
        assert (
            es_allocated.group_by(*self.id_cols)
            .agg(pl.sum("value"))
            .join(es_for_load, on=self.id_cols)
            .fill_nan(0.0)
            .filter((pl.col("value") - pl.col("value_right")).abs() > 0.1)
            .is_empty()
        ), "storage allocation failed"

        return (
            pl.concat(
                [
                    flows.filter(
                        (pl.col("target_") == "Load") & (pl.col("source_") != "Storage")
                    ),
                    es_allocated,
                ]
            )
            .group_by(
                *self.id_cols,
                source_="pct_load_"
                + pl.col("source_")
                .replace({"New Fossil": "Fossil", "Backup": "Fossil"})
                .str.to_lowercase()
                .str.replace(" ", "_"),
            )
            .agg(pl.sum("value"))
            .with_columns(pl.col("value") / pl.sum("value").over(*self.id_cols))
            .sort(*self.id_cols)
            .filter(pl.col("value") > 0)
            .pivot(on="source_", index=self.id_cols, values="value")
        )

    def get_flows(self, run) -> pl.DataFrame:
        run = self.norm_name(run)
        if run in self.flows:
            return self.flows[run]
        with (
            all_logging_disabled(),
            self.fs.open(f"az://patio-results/colo_{run}/colo_flows.parquet") as f,
        ):
            flows_ = pl.read_parquet(f)
        repl = {"export_req": "Required Export", "export_addl": "Surplus Export"}

        out = (
            flows_.group_by(*self.id_cols)
            .agg(cs.contains("__").sum())
            .unpivot(on=cs.contains("__"), index=self.id_cols)
            .select(
                *self.id_cols,
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
        return out

    def get_summary(self, run) -> pl.DataFrame:
        run = self.norm_name(run)
        with (
            all_logging_disabled(),
            self.fs.open(f"az://patio-results/colo_{run}/colo_summary.parquet") as f,
        ):
            colo_ = pl.read_parquet(f)

        if "fuel" not in colo_.columns:
            if run < "202507070053":
                colo_ = colo_.with_columns(fuel=pl.lit("natural_gas"))
            else:
                with (
                    all_logging_disabled(),
                    self.fs.open(f"az://patio-results/colo_{run}/colo.json") as f,
                ):
                    colo_ = colo_.join(
                        pl.from_dicts(json.load(f)["plants"])
                        .select("tech", "status", "fuel", icx_id="pid")
                        .unique(),
                        on=["tech", "status", "icx_id"],
                        how="left",
                        validate="m:1",
                    )

        try:
            with (
                all_logging_disabled(),
                self.fs.open(f"az://patio-results/colo_{run}/colo_re_selected.parquet") as f,
            ):
                re_select = (
                    pl.read_parquet(f)
                    .select(
                        *self.id_cols,
                        "re_site_id",
                        "re_type",
                        "area_per_mw",
                        "capacity_mw",
                        used_acres=pl.col("capacity_mw") * pl.col("area_per_mw") * 247,
                    )
                    .group_by(*self.id_cols, "re_type")
                    .agg(pl.col("used_acres", "capacity_mw").sum())
                    .with_columns(
                        mw_per_acre=pl.col("capacity_mw") / pl.col("used_acres"),
                        acre_per_mw=pl.col("used_acres") / pl.col("capacity_mw"),
                    )
                    .pivot(on="re_type", index=self.id_cols, values="used_acres")
                    .select(
                        *self.id_cols,
                        pl.col("solar").alias("solar_acres"),
                        pl.col("onshore_wind").alias("onshore_wind_acres"),
                    )
                )
            colo_ = colo_.join(re_select, on=self.id_cols, how="left", validate="1:1")
        except:  # noqa: E722
            pass

        utils = (
            colo_.select(
                "icx_id", "icx_gen", pl.col("icx_gen").str.split(",").alias("generator_id")
            )
            .explode("generator_id")
            .unique()
            .join(
                pl.from_pandas(self.ad.own).rename({"plant_id_eia": "icx_id"}),
                on=["icx_id", "generator_id"],
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
                single_owner=pl.col("owner_utility_id_eia")
                .n_unique()
                .over("icx_id", "icx_gen")
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
        cols = [
            "latitude",
            "longitude",
            "plant_name_eia",
            "state",
            "county",
            "parent_name_lse",
        ]
        with_utils = (
            colo_.join(
                pl.from_pandas(self.ad.gens)
                .select(
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
                utils,
                on=["icx_id", "icx_gen"],
                validate="m:1",
                how="left",
                suffix="_utils",
            )
            .select(~cs.contains("_utils"))
        )
        rates = (
            pl_scan_pudl("core_eia861__yearly_sales", self.pudl_release)
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
        )
        r_mw = [f"{t}_mw" for t in ("solar", "onshore_wind", "li", "fe", "new_fossil")]
        out = (
            with_utils.with_columns(
                cf_hist_ratio=pl.col("fossil_cf") / pl.col("historical_fossil_cf"),
                region=pl.col("ba_code").replace(REGION_MAP),
                pct_land_used=1 - pl.col("unused_sqkm_pct"),
                pct_chg_system_co2=pl.col("redispatch_sys_co2") / pl.col("baseline_sys_co2")
                - 1,
                pct_chg_system_cost=pl.col("redispatch_sys_cost_disc")
                / pl.col("baseline_sys_cost_disc")
                - 1,
                pct_overbuild=pl.sum_horizontal(cs.by_name(*r_mw, require_all=False))
                / pl.col("load_mw"),
                regimes=pl.col("regime").n_unique().over("icx_id", "icx_gen", "name"),
                own_short=pl.col("owner_utility_name_eia")
                .str.split(" ")
                .list.gather([0, 1], null_on_oob=True)
                .list.join(" "),
                plant_name_short=pl.col("plant_name_eia")
                .str.split(" ")
                .list.gather([0, 1], null_on_oob=True)
                .list.join(" "),
                utility_name_eia_lse_=pl.col("utility_name_eia_lse")
                .str.split("|")
                .list.first(),
                balancing_authority_code_eia=pl.col("balancing_authority_code_eia").replace(
                    {"CPLE": "DUK"}
                ),
            )
            .join(rates, on="utility_name_eia_lse_", how="left", validate="m:1")
            .select(pl.exclude("utility_name_eia_lse_"))
            .join(
                self._cc_cost(),
                on=["balancing_authority_code_eia", "state"],
                how="left",
                validate="m:1",
            )
        )
        try:
            out = (
                out.join(LAND, on="icx_id", how="left", validate="m:1")
                .with_columns(
                    required_acres=pl.sum_horizontal(
                        "solar_acres", "onshore_wind_acres", "load_mw"
                    )
                )
                .with_columns(
                    land_available=pl.col("buildable_acres").fill_null(0.0).fill_nan(0.0) * 1.5
                    >= pl.col("required_acres")
                )
            )
        except KeyError:
            pass
        out = out.join(
            self.best_at_site(out).select(*self.id_cols, best_at_site=pl.lit(True)),
            on=self.id_cols,
            how="left",
            validate="m:1",
        ).with_columns(best_at_site=pl.col("best_at_site").fill_null(False))
        assert (
            out.group_by("icx_id", "icx_gen").agg(pl.sum("best_at_site"))["best_at_site"].max()
            == 1
        )
        try:
            with (
                all_logging_disabled(),
                self.fs.open(f"az://patio-results/colo_{run}/colo_annual.parquet") as f,
            ):
                annual = (
                    pl.scan_parquet(f)
                    .group_by(*self.id_cols)
                    .agg(cs.numeric().sum())
                    .select(
                        *self.id_cols,
                        capex=pl.sum_horizontal(cs.contains("capex")),
                        itc=pl.sum_horizontal(cs.contains("itc")),
                        load_fossil_mcoe=pl.col("cost_load_fossil") / pl.col("load_fossil"),
                    )
                    .collect()
                )
                out = out.join(
                    annual,
                    on=self.id_cols,
                    how="left",
                    validate="1:1",
                ).with_columns(
                    capex_per_mw=pl.col("capex") / pl.col("load_mw"),
                    net_capex=pl.sum_horizontal("capex", "itc"),
                )
        except Exception as exc:
            print(f"no annual data for {run} {pl_exc_fmt(exc)}")

        try:
            self.flows[run] = self.get_flows(run)
        except Exception as exc:
            print(f"no flows for {run} {pl_exc_fmt(exc)}")
        else:
            out = (
                out.join(
                    self.flows[run]
                    .select(*self.id_cols, has_flows=pl.sum("value").over(*self.id_cols) > 0)
                    .unique(),
                    on=self.id_cols,
                    how="left",
                )
                .with_columns(has_flows=pl.col("has_flows").fill_null(False))
                .join(
                    self.source_for_load(run),
                    on=self.id_cols,
                    how="left",
                )
            )

        return (
            out.join(
                self.good_screen(out).select(*self.id_cols, good=pl.lit(True)),
                on=self.id_cols,
                how="left",
            )
            .with_columns(good=pl.col("good").fill_null(False), run=pl.lit(run))
            .pipe(order_columns)
            .sort("state", "utility_name_eia_lse", "icx_id")
        )

    def _cc_cost(self) -> pl.DataFrame:
        atb = (
            clean_atb(
                case="Market",
                scenario="Moderate",
                report_year=2024,
                pudl_release=self.pudl_release,
            )
            .filter(
                (pl.col("technology_description") == "Natural Gas Fired Combined Cycle")
                & (pl.col("projection_year") == 2026)
            )
            .select("heat_rate_mmbtu_per_mwh", "fom_per_kw", "vom_per_mwh")
            .to_dict(as_series=False)
        )
        r_disc = ((1 + COSTS["discount"]) / (1 + COSTS["inflation"])) - 1
        return (
            pl_scan_pudl(
                "core_eiaaeo__yearly_projected_fuel_cost_in_electric_sector_by_type",
                self.pudl_release,
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
            .collect()
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
        )

    def parse_log(self, run: str, *, local: bool = False) -> pl.DataFrame:
        run = self.norm_name(run)
        if local:
            return pl.read_ndjson(Path.home() / f"patio_data/{run}/log.jsonl")
        else:
            with self.fs.open(f"patio-results/colo_{run}/log.jsonl") as f:
                return pl.read_ndjson(f)

    def norm_name(self, run: str | int) -> str:
        if run is None:
            return self.last
        if isinstance(run, int):
            return self.name_indices[run]
        return run.removeprefix("colo_")

    def summary_stats(self, cols=()):
        weights = pl.col("load_mw") / pl.col("load_mw").sum()
        return pl.concat(
            self.get_aligned(r).select(
                run=pl.lit(r),
                num=pl.count("name"),
                load_mw=pl.col("load_mw").sum(),
                weighted_ppa=(pl.col("ppa_ex_fossil_export_profit") * weights).sum(),
                weighted_pct_clean=(pl.col("pct_load_clean") * weights).sum(),
                weighted_pct_served=(pl.col("served_pct") * weights).sum(),
                **{f"weighted_{c}": (pl.col(c) * weights).sum() for c in cols},
            )
            for r in self.names
        )

    def for_dataroom(self, run=None, *, clip=False) -> pl.DataFrame:
        run = self.norm_name(run)
        out = (
            self.summaries[run]
            .filter(pl.col("good"))
            .with_columns(pl.col("tech").replace(TECH_CODES), name=self.renamer)
        )
        if missing_cols := set(FANCY_COLS) - set(out.columns):
            LOGGER.warning("%s missing columns %s", run, missing_cols)
            out = out.with_columns(**dict.fromkeys(missing_cols))
        out = out.sort(*self.id_cols).select(*list(FANCY_COLS)).rename(FANCY_COLS)
        if clip:
            out.write_clipboard()
        return out

    def for_xl(self, run=None, *, clip=False) -> pl.DataFrame:
        out = self.all_summaries() if run == "all" else self.summaries[self.norm_name(run)]
        out = out.filter(pl.col("run_status") == "SUCCESS")
        if clip:
            out.write_clipboard()
        return out

    def case_sheets(self, run=None, *, land_screen=False) -> list[go.Figure]:
        run = self.norm_name(run)
        subplots = self.fig_case_subplots(
            run,
            test=False,
            land_screen=land_screen,
        )

        file = self.fig_path / f"{run} screened.pdf"
        if file.exists():
            file.unlink()
        with PdfWriter(file) as pdf:
            for plot in tqdm(subplots):
                # if "**" not in plot.layout.title.text:
                #     continue
                o = PdfReader(
                    io.BytesIO(
                        plot.to_image(
                            format="pdf", height=plot.layout.height, width=plot.layout.width
                        )
                    )
                )
                for page in o.pages:
                    if page.get_contents() is not None:
                        pdf.add_page(page)
                    else:
                        print(page)
        return subplots

    def fig_case_subplots(self, run=None, *, land_screen=False, test=False) -> list[go.Figure]:
        colo = self.summaries[run].with_columns(name=self.renamer)
        if land_screen:
            colo = colo.filter(pl.col("land_available"))
        good = self.good_screen(colo)
        pl_good = (
            colo.filter(
                (pl.col("run_status") == "SUCCESS")
                & pl.col("good").max().over("icx_id", "icx_gen")
            )
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
                self.colors,
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
            zip(cs.expand_selector(pl_good, cs.contains("attr_")), self.colors, strict=False)
        )
        good = good.select(
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
            self.flows[run]
            .with_columns(
                name=self.renamer,
                full_name=pl.concat_str(self.renamer, "regime", separator="<br>"),
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
        pl_good_parts = (
            pl_good.with_columns(pl.col("net_capex").fill_null(0.0))
            .sort("icx_id")
            .partition_by("icx_id", "icx_gen", as_dict=True)
        )
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
        axes = {
            "xaxis": {"title": "% clean", "tickformat": ".0%", "range": [0.5, 1.05]},
            "yaxis": {"title": "ppa", "tickformat": "$.0f", "range": [0, 350]},
            "yaxis2": {"title": "ppa attribution", "tickprefix": "$"},
            "yaxis3": {"title": "% of load capacity", "tickformat": ".0%"},
            "yaxis4": {"title": "% of load energy", "tickformat": ".0%"},
            "yaxis5": {"title": "% of generation", "tickformat": ".0%", "range": [0, 1.19]},
            "yaxis6": {"title": "fossil capacity factor", "tickformat": ".0%"},
            "yaxis7": {"title": "capex/MW load", "tickprefix": "$"},
            "yaxis8": {"title": "acres", "title_standoff": 1},
        }
        axes = {k: v | {"title_standoff": 1} for k, v in axes.items()}
        subplots = []

        for j, (keys, pid) in enumerate(tqdm(pl_good_parts.items())):
            try:
                pid = pid.sort("name", "regime")
                names = pid.select("full_name").to_series()
                rows = max(8, len(pid))
                title = list(
                    pid.select(
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
                    .to_series()
                )
                if len(title) != 1:
                    print(title)
                plt = (
                    make_subplots(
                        rows=rows,
                        cols=2,
                        specs=[[{"type": "scatter"}, {"type": "sankey"}]] * rows,
                        subplot_titles=[k for n in names for k in ["", n]],
                        vertical_spacing=0.025,
                        horizontal_spacing=0.02,
                    )
                    .update_layout(
                        title=("** " if keys[0] in good["icx_id"].to_numpy() else "")
                        + title[0],
                        template="ggplot2",
                        font_family=self.font,
                        height=50 + 380 * rows,
                        width=1400,
                        barmode="stack",
                        **axes,
                        legend={
                            "title": "Scenario name",
                            "xref": "paper",
                            "yref": "paper",
                            "x": 1.025,
                            "y": 0.97,
                        },
                        margin_r=200,
                    )
                    .add_annotation(
                        text=f"<b>Bolded</b> Scenario name means <br> Load Served ≥ {self.min_served:.2%}"
                        f"<br> & Hourly Matched Clean ≥ {self.min_pct_load_clean:.0%}<br> & PPA ≤ "
                        f"${self.max_ppa:,.0f}/MWh{'<br>Land confirmed' if land_screen else ''}",
                        xref="paper",
                        yref="paper",
                        align="left",
                        showarrow=False,
                        font_size=12,
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
                    pl.col("source_").is_in(("Solar", "Wind"))
                    & pl.col("target_").is_in(("Load",))
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
                costs = {
                    "attr_cost_clean": "Clean Cost",
                    "attr_cost_load_fossil": "Fossil Cost",
                }
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
                                    "solar_mw",
                                    "li_mw",
                                    "onshore_wind_mw",
                                    "fe_mw",
                                    "new_fossil_mw",
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
                for (target,), part in source_flows.partition_by(
                    "source_", as_dict=True
                ).items():
                    if target == "Solar":
                        plt.add_scatter(
                            x=part["full_name"],
                            y=[
                                source_flows.group_by("full_name")
                                .agg(pl.sum("share"))["share"]
                                .max()
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
                for target in (
                    "Required Export",
                    "Load",
                    "Storage",
                    "Surplus Export",
                    "Curtailment",
                ):
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
                        plt.add_scatter(
                            y=[com_rate] * 3, text=["", "commercial<br>rate", ""], **rkwgs
                        )
                    if (ind_rate := pid["ind_rate"][0]) is not None:
                        plt.add_scatter(
                            y=[ind_rate] * 3, text=["", "industrial<br>rate", ""], **rkwgs
                        )
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
            except Exception as exc:
                LOGGER.error("%s", keys, exc_info=exc)
        return subplots

    def package_econ_data(self, run=None, addl_filter=None, *, local=False):
        run = self.norm_name(run)
        colo_dir = Path.home() / "patio_data" / f"colo_{run}"
        if not colo_dir.exists():
            colo_dir.mkdir()
        econ_dir = colo_dir / "econ"
        if not econ_dir.exists():
            econ_dir.mkdir()

        good = self.good_screen(self.summaries[run])
        if addl_filter:
            good = good.filter(addl_filter)
        selector = ~cs.by_dtype(pl.List(pl.Int64)) & ~cs.by_dtype(pl.Duration)
        all_itr = [
            (info, o_ids)
            for info in set(Info.from_df(good))
            for o_ids in (
                good.filter(info.filter()).select("regime", "ix").iter_rows(named=True)
            )
        ]
        for file in tqdm(
            ("summary", "annual", "cost_detail", "flows", "re_selected", "full"), position=0
        ):
            with (
                all_logging_disabled(),
                self.fs.open(f"patio-results/colo_{run}/colo_{file}.parquet") as f,
            ):
                data = pl.read_parquet(f)
            for info, o_ids in tqdm(all_itr, position=1):
                try:
                    ftsv_scr = info.filter(**o_ids)
                    pdir = econ_dir / info.file(**o_ids, suffix="")
                    if not pdir.exists():
                        pdir.mkdir()
                    data.filter(ftsv_scr).select(selector).write_csv(pdir / f"{file}.csv")
                except Exception as e:
                    print(f"{info} {o_ids} {pl_exc_fmt(e)}")

    def compare(
        self,
        a,
        b,
        *,
        on=(
            "balancing_authority_code_eia",
            "plant_name_eia",
            "utility_name_eia_lse",
            "owner_utility_name_eia",
            "state",
            "icx_id",
            "tech",
            "status",
            "name",
            "regime",
        ),
        values=(
            "run_status",
            "load_mw",
            "served_pct",
            "pct_load_clean",
            "ppa_ex_fossil_export_profit",
        ),
        clip=False,
    ) -> pl.DataFrame:
        c = (
            self.summaries[self.norm_name(a)]
            .select(*on, *values)
            .join(
                self.summaries[self.norm_name(b)].select(*on, *values),
                on=on,
                suffix="_b",
                how="full",
                coalesce=True,
            )
        ).filter((pl.col("run_status") == "SUCCESS") | (pl.col("run_status_b") == "SUCCESS"))
        c = c.select(list(dict.fromkeys(on) | dict.fromkeys(sorted(c.columns))))
        if clip:
            c.write_clipboard()
        return c

    def fig_scatter_geo(self, run=None, sixe_max=25, *, land_screen=False) -> go.Figure:
        run = self.norm_name(run)
        data = self.get_aligned(run, land_screen=land_screen)
        kwargs = (
            {"facet_row": "run", "category_orders": sorted(data["run"].unique())}
            if run == "all"
            else {}
        )
        cmax = data["ppa_ex_fossil_export_profit"].max()
        cmax = int(cmax + (100 - cmax) % 100)
        return (
            px.scatter_geo(
                data.sort("run"),
                lat="latitude",
                lon="longitude",
                hover_name="plant_name_eia",
                hover_data=[
                    "pct_load_clean",
                    "ba_code",
                    "tech",
                    "balancing_authority_code_eia",
                    "state",
                    "utility_name_eia",
                ],
                color="ppa_ex_fossil_export_profit",
                size="load_mw",
                height=800,
                width=900,
                locationmode="USA-states",
                size_max=sixe_max,
                **kwargs,
            )
            .update_geos(fitbounds="locations", scope="usa")
            .update_coloraxes(
                colorbar_title="PPA<br>Price",
                cmax=cmax,
                cmid=int(cmax / 2),
                cmin=0,
                colorscale="portland",
                colorbar_tickformat="$.0f",
                colorbar_len=0.565,
            )
            .update_layout(
                legend_title="Percent<br>Clean Energy",
                showlegend=False,
                font_family=self.font,
                font_size=8,
                template="ggplot2",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
            )
            .update_traces(textposition="top center")
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        )

    def fig_selection_map(
        self,
        run=None,
        *,
        land_screen=False,
        px_size_max=8,
        fossil_size=0.15,
        selected_size=1,
        potential_size=0.8,
    ) -> go.Figure:
        run = self.norm_name(run)
        runs = list(self.summaries) if run == "all" else (run,)

        all_re = []
        for r in runs:
            with (
                all_logging_disabled(),
                self.fs.open(f"patio-results/colo_{r}/all_re_sites.parquet") as f,
            ):
                all_re.append(
                    pl.read_parquet(f)
                    .cast({"re_site_id": pl.Int32})
                    .filter(pl.col("re_type") != "offshore_wind")
                    .with_columns(run=pl.lit(r))
                )
        all_re = pl.concat(all_re).join(
            pl_scan_pudl("core_eia860m__changelog_generators", self.pudl_release)
            .sort("valid_until_date", descending=True)
            .group_by(pl.col("plant_id_eia").alias("icx_id"))
            .agg(pl.col("latitude", "longitude").first())
            .collect(),
            on="icx_id",
            how="left",
            validate="m:1",
        )

        # with (
        #     all_logging_disabled(),
        #     self.fs.open(f"patio-results/colo_{run}/all_re_sites.parquet") as f,
        # ):
        #     all_re = (
        #         pl.scan_parquet(f)
        #         .cast({"re_site_id": pl.Int32})
        #         .filter(pl.col("re_type") != "offshore_wind")
        #         .join(
        #             pl_scan_pudl("core_eia860m__changelog_generators", self.pudl_release)
        #             .sort("valid_until_date", descending=True)
        #             .group_by(pl.col("plant_id_eia").alias("icx_id"))
        #             .agg(pl.col("latitude", "longitude").first()),
        #             on="icx_id",
        #             how="left",
        #             validate="m:1",
        #         )
        #         .with_columns(run=pl.lit(run))
        #         .collect()
        #     )
        re_selected = []
        for r in runs:
            with (
                all_logging_disabled(),
                self.fs.open(f"patio-results/colo_{r}/colo_re_selected.parquet") as f,
            ):
                re_selected.append(
                    pl.read_parquet(f)
                    .cast({"re_site_id": pl.Int32})
                    .with_columns(run=pl.lit(run))
                )
        re_selected = pl.concat(re_selected)

        # with (
        #     all_logging_disabled(),
        #     self.fs.open(f"patio-results/colo_{run}/colo_re_selected.parquet") as f,
        # ):
        #     re_selected = pl.read_parquet(f).cast({"re_site_id": pl.Int32}) .with_columns(run=pl.lit(run))

        potential = (
            all_re.filter(pl.col("distance") < 25)
            .select(
                "run",
                plant_id="re_site_id",
                tech="re_type",
                icx_id="icx_id",
                color=pl.col("re_type").replace(
                    {"solar": "Solar", "onshore_wind": "Onshore Wind"}
                ),
                cat=pl.lit("potential"),
                size=potential_size
                * pl.col("capacity_mw_nrel_site").log()
                / pl.col("capacity_mw_nrel_site").log().max().over("icx_id"),
                latitude="latitude_nrel_site",
                longitude="longitude_nrel_site",
            )
            .unique()
            .sort("tech", "size", descending=[False, True])
        )
        selected = (
            self.get_aligned(run, land_screen=land_screen)
            .select("run", "icx_id", "icx_gen", "regime", "name", "latitude", "longitude")
            .join(
                re_selected.with_columns(
                    re_site_id=pl.when(pl.col("re_type") == "solar")
                    .then(pl.lit(1))
                    .otherwise(pl.col("re_site_id"))
                )
                .group_by(
                    "run", "icx_id", "icx_gen", "re_type", "re_site_id", "regime", "name"
                )
                .agg(
                    pl.col("latitude_nrel_site", "longitude_nrel_site").first(),
                    pl.sum("capacity_mw"),
                ),
                on=["run", "icx_id", "icx_gen", "regime", "name"],
                how="left",
            )
            .select(
                "run",
                plant_id="re_site_id",
                tech="re_type",
                icx_id="icx_id",
                color=pl.col("re_type").replace(
                    {"solar": "Solar selected", "onshore_wind": "Onshore Wind selected"}
                ),
                cat=pl.lit("selected"),
                size=selected_size,
                # size=1.5*pl.col("capacity_mw").log(2)
                # / pl.col("capacity_mw").log(2).max().over("icx_id"),
                capacity_mw="capacity_mw",
                capacity_mw_lim=pl.lit(0.0),
                latitude=pl.when(pl.col("re_type") == "onshore_wind")
                .then(pl.col("latitude_nrel_site"))
                .otherwise(pl.col("latitude"))
                .cast(pl.Float64),
                longitude=pl.when(pl.col("re_type") == "onshore_wind")
                .then(pl.col("longitude_nrel_site"))
                .otherwise(pl.col("longitude"))
                .cast(pl.Float64),
            )
            .sort("tech", descending=False)
        )
        fossil = all_re.select(
            "run",
            plant_id="icx_id",
            tech="icx_tech",
            icx_id="icx_id",
            color=pl.lit("Fossil"),
            cat=pl.lit("fossil"),
            size=pl.lit(fossil_size),
            latitude="latitude",
            longitude="longitude",
        ).unique()

        map_df = pl.concat([potential, selected, fossil], how="diagonal_relaxed").filter(
            pl.col("size") > 0.0
        )
        return (
            px.scatter_geo(
                map_df,
                lat="latitude",
                lon="longitude",
                size="size",
                color="color",
                template="ggplot2",
                hover_data="capacity_mw",
                size_max=px_size_max,
                color_discrete_map={k + " selected": v for k, v in COLOR_MAP.items()}
                | {
                    "Onshore Wind selected": "#529cba",
                    "aOnshore Wind selected": "#529cba",
                    "Fossil selected": "#000000",
                    "Load": "#000000",
                    "Fossil": "#000000",
                    # "Solar": "#fff8d5",
                    # "Onshore Wind": "#ebf5f9",
                    "Solar": "#ffeda9",
                    "Onshore Wind": "#c9e3f0",
                },
                facet_row="run",
            )
            .update_geos(fitbounds="locations", scope="usa")
            .update_traces(
                marker={"opacity": 1, "line": {"width": 0}}, selector={"mode": "markers"}
            )
        )

    def fig_indicator_distributions(
        self,
        metrics: tuple = ("load_mw", "ppa_ex_fossil_export_profit", "pct_overbuild"),
        color="good",
    ):
        data = (
            self.all_summaries()
            .unpivot(on=metrics, index=list({"run", "good", *self.id_cols, "tech", color}))
            .with_columns(pl.col("run").str.replace("2025", ""))
        )

        def fix(axis):
            try:
                num = int(axis.anchor.replace("y", ""))
            except ValueError:
                num = 1
            if num > len(metrics):
                match = num % len(metrics)
                match = {1: "", 0: len(metrics)}.get(match, match)
                axis.update(matches=f"x{match}")
            else:
                dmax = data.filter(pl.col("variable") == metrics[num - 1])["value"].max()
                dmin = data.filter(pl.col("variable") == metrics[num - 1])["value"].min()
                axis.update(range=[dmin, dmax], matches=None)

        return (
            px.histogram(
                data,
                x="value",
                facet_row="run",
                facet_col="variable",
                height=600,
                template="ggplot2",
                color=color,
                barmode="overlay",
                opacity=0.75,
                category_orders={"run": sorted(data["run"].unique())},
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .update_yaxes(matches=None, showticklabels=True)
            .update_xaxes(matches=None)
            .for_each_xaxis(fix)
            .update_layout(font_family=self.font)
        )

    def fig_supply_curve(self, run=None, *, land_screen=False) -> go.Figure:
        run = self.norm_name(run)

        if run == "all":
            plt = make_subplots(
                len(self.names),
                subplot_titles=self.names,
                vertical_spacing=0.05,
                shared_xaxes=True,
            )
            for i, r in enumerate(self.names, 1):
                plt.add_traces(self.fig_supply_curve(r, land_screen=land_screen).data, i, 1)
            return (
                plt.update_yaxes(tickformat="$.0f", title="PPA Price ($/MWh)")
                .update_layout(
                    template="ggplot2",
                    font_family=self.font,
                    height=25 + 275 * i,
                    width=700,
                    margin={"t": 25, "b": 5, "l": 5, "r": 0},
                    **{f"xaxis{i}_title": "GW", f"xaxis{i}_tickformat": ",.0f"},
                )
                .add_hline(y=100, line_color="white", line_width=1, opacity=1)
            )

        own_rn = {  # noqa: F841
            "Virginia Electric & Power Co": "Dominion",
            "Duke Energy Progress - (NC)": "Duke",
            "Duke Energy Carolinas, LLC": "Duke",
            "Southern Power Co": "Southern",
            "Old Dominion Electric Coop": "Old Dominion Coop",
            "North Carolina El Member Corp": "Member Corp",
            "SEPG Operating Services, LLC WCP": "SEPG",
            "Middle River Power II, LLC": "Middle River",
            "Doswell Ltd Partnership": "Doswell",
        }
        d_ = (
            self.get_aligned(run, land_screen=land_screen)
            .to_pandas()
            .sort_values(["ppa_ex_fossil_export_profit"])
            .assign(
                load_gw=lambda x: x.load_mw / 1000,
                x_=lambda x: ((x.load_gw.shift(1).fillna(0) / 2) + x.load_gw / 2).astype(
                    float
                ),
                x=lambda x: x.x_.cumsum(),
            )
        )
        cmax = d_["ppa_ex_fossil_export_profit"].max()
        cmax = int(cmax + (100 - cmax) % 100)

        return (
            go.Figure(
                go.Bar(
                    x=d_["x"],
                    width=d_["load_gw"],
                    y=d_["ppa_ex_fossil_export_profit"],
                    showlegend=False,
                    # text=d_["plant_name_short"]
                    # + "<br>"
                    # # + d_["state"]+ " "
                    #      + d_["balancing_authority_code_eia"].replace(
                    #     own_rn
                    # ),
                    customdata=d_[
                        [
                            "plant_name_eia",
                            "state",
                            "utility_name_eia",
                            "tech",
                            "solar_mw",
                            "onshore_wind_mw",
                            "li_mw",
                            "pct_load_clean",
                            "load_mw",
                        ]
                    ].fillna(0.0),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    "PPA: %{y:$,.0f}<br>"
                    "Load MW: %{customdata[8]:.0f}<br>"
                    "State: %{customdata[1]}<br>"
                    "Utility: %{customdata[2]}<br>"
                    "Tech: %{customdata[3]}<br>"
                    "ES, Solar, Wind: %{customdata[6]:.0f}, %{customdata[4]:.0f}, %{customdata[5]:.0f}<br>"
                    "Pct Clean: %{customdata[7]:.1%}<br>"
                    "<extra></extra>",
                    marker=dict(  # noqa: C408
                        color=d_["pct_load_clean"],
                        cmin=0.5,
                        cmax=1,
                        colorbar=dict(  # noqa: C408
                            title="Percent<br>Clean<br>Energy",
                            tickformat=".0%",
                            outlinewidth=0,
                            tickcolor="rgb(237,237,237)",
                            ticklen=6,
                            ticks="inside",
                            len=0.65,
                        ),
                        colorscale="portland_r",
                        line_width=0.01,
                    ),
                )
            )
            .update_yaxes(tickformat="$.0f", range=[0, cmax], title="PPA Price ($/MWh)")
            .update_xaxes(tickformat=",.0f", title="GW")
            .update_layout(
                template="ggplot2",
                font_family=self.font,
                height=500,
                width=700,
                margin={"t": 5, "b": 5, "l": 5, "r": 0},
            )
            .add_hline(y=100, line_color="white", line_width=1, opacity=1)
        )

    def hourly_analysis(
        self,
        run,
    ):
        pl.scan_parquet(Path.home() / "patio_data")


# def flows_w_allocated_storage(run, fs=None):
#     idc = ["ba_code", "icx_id", "icx_gen", "regime", "name", "ix"]
#     repl = {"Surplus Export": "Export", "Required Export": "Export"}
#     for_util = (
#         get_flows(run, fs)
#         .group_by(*idc, "source_", pl.col("target_").replace(repl))
#         .agg(pl.sum("value"))
#     )
#     via_storage = (
#         for_util.filter(pl.col("source_") == "Storage")
#         .select(*idc, "target_", "value")
#         .join(
#             for_util.filter(pl.col("target_") == "Storage").select(
#                 *idc,
#                 "source_",
#                 es=pl.sum("value").over(*idc, "source_") / pl.sum("value").over(*idc),
#             ),
#             on=idc,
#             how="left",
#         )
#         .select(*idc, "source_", "target_", es=pl.col("value") * pl.col("es"))
#     )
#     out = (
#         for_util.join(via_storage, on=[*idc, "source_", "target_"], how="left")
#         .with_columns(value=pl.sum_horizontal("value", "es"))
#         .filter((pl.col("target_") != "Storage") & (pl.col("source_") != "Storage"))
#         .pivot(on="target_", index=[*idc, "source_"], values="value")
#         .sort("icx_id", "regime", "ix", "source_")
#     )
#     assert (
#         out.group_by(*idc)
#         .agg(pl.col("Load", "Export").sum())
#         .join(
#             for_util.filter(pl.col("target_").is_in(("Export", "Load"))).pivot(
#                 on="target_", index=idc, values="value", aggregate_function="sum"
#             ),
#             on=idc,
#             validate="1:1",
#             how="left",
#         )
#         .filter(
#             ((pl.col("Load").fill_nan(None) - pl.col("Load_right")).abs() > 1)
#             | ((pl.col("Export").fill_nan(None) - pl.col("Export_right")).abs() > 1)
#         )
#         .is_empty()
#     )
#     return out
