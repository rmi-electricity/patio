"""Functional entry point for PATIO."""

import logging
import logging.config
import logging.handlers
import multiprocessing
import shutil
import sys
import tomllib
import warnings as warnings_
from datetime import datetime
from pathlib import Path

import click

from patio import __version__
from patio.constants import PATIO_DOC_PATH, ROOT_PATH
from patio.model.ba_level import BAs
from patio.model.colo_core import main as colo_main

RELATIVE_ROOT = ROOT_PATH.relative_to(Path.home())
CONF_KWARGS = {
    "type": str,
    "default": str(RELATIVE_ROOT / "patio.toml"),
    "show_default": True,
    "help": "path to patio.toml config file relative to home",
}


def logging_process(queue, configurer, now):
    configurer(now)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            if "numba" not in record.name or record.levelno > 20:
                logger = logging.getLogger(record.name)
                logger.handle(record)
        except Exception:
            import sys
            import traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def setup_logger():
    now = datetime.now().strftime("%Y%m%d%H%M")
    if not (PATIO_DOC_PATH / "logs").exists():
        (PATIO_DOC_PATH / "logs").mkdir()
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(levelname)s|%(module)s|L%(lineno)d|%(process)d]: %(message)s"
            },
            "detailed": {
                "format": "[%(levelname)s|%(module)s|L%(lineno)d|%(process)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "json": {
                "()": "etoolbox.utils.logging_utils.JSONFormatter",
                "fmt_keys": {
                    "level": "levelname",
                    "message": "message",
                    "timestamp": "timestamp",
                    "logger": "name",
                    "module": "module",
                    "function": "funcName",
                    "line": "lineno",
                    "thread_name": "threadName",
                    "process": "process",
                },
            },
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "simple",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": PATIO_DOC_PATH / f"logs/{now}.log",
            },
            "file_json": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": PATIO_DOC_PATH / f"logs/{now}.jsonl",
            },
        },
        "loggers": {"root": {"level": "INFO", "handlers": ["stderr", "file", "file_json"]}},
    }
    logging.config.dictConfig(config)
    return now


@click.group(help="patio CLI")
@click.version_option(__version__)
def main():
    pass


@main.command(name="repower")
@click.option("--bas", "-b", type=str, default="", help="BAs to run separated by commas.")
@click.option("--config", "-C", **CONF_KWARGS)
@click.option("--warnings", "-w", is_flag=True, default=False, help="Show warnings")
@click.option("--local", "-l", is_flag=True, default=False, help="do not upload to Azure")
def clean_repower(bas, config, warnings, local):
    """Run clean repowering model."""
    multiprocessing.set_start_method("spawn")
    now = setup_logger()

    with open(Path.home() / config, "rb") as f:
        config = tomllib.load(f)

    if bas:
        config["project"]["balancing_authorities"] = [s.strip() for s in bas.split(",")]
    elif not config["project"]["balancing_authorities"]:
        config["project"]["balancing_authorities"] = None

    kwargs = (
        config["data"]
        | config["cr"]["project"]
        | {
            "colo_techs": config["colo"]["data"],
        }
    )

    if not warnings:
        warnings_.simplefilter("ignore")

    try:
        patio = BAs(
            name="BAs_" + now,
            solar_ilr=config["project"]["solar_ilr"],
            data_kwargs=kwargs,
            bas=config["project"]["balancing_authorities"],
            by_plant=False,
            regime=config["project"]["regime"],
            pudl_release=config["project"]["pudl_release"],
        )
        patio.run_all()
        patio.write_output()
        print(patio.name)
        if not local:
            print("uploading to patio-results/...")
            patio.upload_output()
    finally:
        pass


@main.command(name="profiles")
@click.option("--config", "-C", **CONF_KWARGS)
@click.option("--warnings", "-w", is_flag=True, default=False, help="Show warnings")
def profiles(config, warnings):
    """Setup profiles."""
    now = setup_logger()

    with open(Path.home() / config, "rb") as f:
        config = tomllib.load(f)

    if not warnings:
        warnings_.simplefilter("ignore")

    patio = BAs(
        name="BAs_" + now,
        solar_ilr=config["project"]["solar_ilr"],
        bas=config["project"]["balancing_authorities"],
        by_plant=False,
        regime=config["project"]["regime"],
        pudl_release=config["project"]["pudl_release"],
        scenario_configs=[],
    )
    patio.pd.setup_all()


@main.command(name="specs")
@click.option("--config", "-C", **CONF_KWARGS)
@click.option("--warnings", "-w", is_flag=True, default=False, help="Show warnings")
def specs(config, warnings):
    """Just export fossil specs."""
    now = setup_logger()

    with open(Path.home() / config, "rb") as f:
        config = tomllib.load(f)

    kwargs = (
        config["data"]
        | config["cr"]["project"]
        | {
            "colo_techs": config["colo"]["data"],
        }
    )
    if not warnings:
        warnings_.simplefilter("ignore")
    patio = BAs(
        name="BAs_" + now,
        solar_ilr=config["project"]["solar_ilr"],
        data_kwargs=kwargs,
        bas=config["project"]["balancing_authorities"],
        by_plant=False,
        regime=config["project"]["regime"],
        pudl_release=config["project"]["pudl_release"],
        scenario_configs=[],
    )
    patio.get_fossils_specs()


@main.group()
def colo():
    pass


@colo.command(name="data")
@click.option("--bas", "-b", type=str, default="", help="BAs to run separated by commas.")
@click.option("--config", "-C", **CONF_KWARGS)
@click.option("--warnings", "-w", is_flag=True, default=False, help="Show warnings")
def colo_data(bas, config, warnings):
    now = setup_logger()

    with open(Path.home() / config, "rb") as f:
        config = tomllib.load(f)

    if bas:
        config["project"]["balancing_authorities"] = [s.strip() for s in bas.split(",")]
    elif not config["project"]["balancing_authorities"]:
        config["project"]["balancing_authorities"] = None

    kwargs = (
        config["data"]
        | config["cr"]["project"]
        | {
            "colo_techs": config["colo"]["data"],
        }
    )
    if not warnings:
        warnings_.simplefilter("ignore")
    patio = BAs(
        name="BAs_" + now,
        solar_ilr=config["project"]["solar_ilr"],
        data_kwargs=kwargs,
        bas=config["project"]["balancing_authorities"],
        by_plant=False,
        regime=config["project"]["regime"],
        pudl_release=config["project"]["pudl_release"],
        scenario_configs=[],
    )
    patio.run_all()
    print(f"patio-colo -d '{patio.colo_dir}'", file=sys.stderr)
    print()
    print(patio.colo_dir)
    to_delete = patio.dir_path
    colo_dir = Path.home() / patio.colo_dir
    del patio
    shutil.rmtree(to_delete)
    shutil.copyfile(PATIO_DOC_PATH / f"logs/{now}.log", colo_dir / "data.log")
    shutil.copyfile(PATIO_DOC_PATH / f"logs/{now}.jsonl", colo_dir / "data_log.jsonl")


@colo.command(name="run")
@click.option(
    "--run-dir",
    "-D",
    type=str,
    default="",
    help="path from home, overrides `run_dir_path` in patio.toml",
)
@click.option("--config", "-C", **CONF_KWARGS)
@click.option(
    "--source-data",
    "-L",
    type=str,
    default="",
    help="path from home, overrides `data_path` and `colo_json_path` in patio.toml",
)
@click.option("--resume", "-r", is_flag=True, default=False, help="resume most recent run")
@click.option("--plants", "-p", type=str, default="", help="override of plants to run")
@click.option(
    "--workers",
    "-w",
    type=int,
    default=-1,
    help="number of workers in colo analysis, default is core_count / 2 for Apple Silicon, otherwise core_count - 1",
)
@click.option("--keep", "-k", is_flag=True, default=False, help="keep intermediate files")
@click.option("--local", "-l", is_flag=True, default=False, help="do not upload to Azure")
@click.option("--saved", "-S", type=str, default="", help="re-run all plants/configs")
@click.option(
    "--assemble",
    "-A",
    is_flag=True,
    default=False,
    help="assemble and upload finished results",
)
def colo_run(
    run_dir, config, source_data, resume, plants, workers, keep, local, saved, assemble
):
    """Run colo."""
    colo_main(
        run_dir=run_dir,
        config_dir=config,
        source_data=source_data,
        resume=resume,
        plants=plants,
        workers=workers,
        keep=keep,
        local=local,
        saved=saved,
        assemble=assemble,
    )
