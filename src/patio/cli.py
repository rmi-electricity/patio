"""Functional entry point for PATIO."""

import logging
import logging.config
import logging.handlers
import multiprocessing
import shutil
import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

from patio.constants import PATIO_DOC_PATH, ROOT_PATH


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


def setup_logger(now):
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
            # "queue_handler": {
            #     "class": "logging.handlers.QueueHandler",
            #     "handlers": ["stderr", "file_json", "file"],
            #     "respect_handler_level": True
            # }
        },
        "loggers": {"root": {"level": "INFO", "handlers": ["stderr", "file", "file_json"]}},
        # "root": {"level": "INFO", "handlers": ["stderr", "file", "file_json"]},
    }
    # root = logging.getLogger()
    # # root.setLevel(logging.INFO)
    # log_path = PATIO_DOC_PATH / f"logs/{now}.log"
    # fh = logging.handlers.WatchedFileHandler(log_path, "a")
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    # fh.setLevel(logging.INFO)
    # format_file = logging.Formatter(
    #     "[%(levelname)s|%(module)s|L%(lineno)d|%(process)d] %(asctime)s: %(message)s",
    #     datefmt="%Y-%m-%dT%H:%M:%S%z",
    # )
    # format_stream = logging.Formatter(
    #     "[%(levelname)s|%(module)s|L%(lineno)d|%(process)d]: %(message)s",
    # )
    # fh.setFormatter(format_file)
    # ch.setFormatter(format_stream)
    # root.addHandler(fh)
    # root.addHandler(ch)

    logging.config.dictConfig(config)

    # for lo in (
    #     "PIL.PngImagePlugin",
    #     "numba.core.ssa",
    #     "numba.core.byteflow",
    #     "numba.core.interpreter",
    #     "numba.core.typeinfer",
    # ):
    #     logger_ = logging.getLogger(lo)
    #     logger_.setLevel(logging.WARNING)


def main():
    multiprocessing.set_start_method("spawn")

    import argparse

    now = datetime.now().strftime("%Y%m%d%H%M")
    setup_logger(now)
    # queue = multiprocessing.Manager().Queue(-1)
    #
    # lp = threading.thread(
    #     target=logging_process, args=(queue, setup_logger, now)
    # )
    # lp.start()
    # h = logging.handlers.QueueHandler(queue)
    # ROOT_LOGGER = logging.getLogger()
    # ROOT_LOGGER.addHandler(h)
    # ROOT_LOGGER.propagate = False
    # logger = logging.getLogger("patio")

    from patio.model.ba_level import BAs

    parser = argparse.ArgumentParser(description="Run PATIO.")
    parser.add_argument(
        "-C, --config",
        type=str,
        default=str(ROOT_PATH / "patio.toml"),
        required=False,
        help="path to patio.toml config file",
        dest="config",
    )
    parser.add_argument(
        "-b, --bas",
        type=str,
        help="BAs to run separated by commas.",
        default=None,
        required=False,
        dest="bas",
    )
    parser.add_argument(
        "-w, --warnings",
        action="store_true",
        help="Show warnings",
        default=False,
        required=False,
        dest="warnings",
    )
    parser.add_argument(
        "-S, --setup",
        action="store_true",
        help="Setup profiles",
        default=False,
        required=False,
        dest="setup",
    )
    parser.add_argument(
        "-s, --specs",
        action="store_true",
        help="Just export fossil specs",
        default=False,
        required=False,
        dest="specs",
    )
    parser.add_argument(
        "-l, --local",
        action="store_true",
        help="do not upload to Azure",
        default=False,
        required=False,
        dest="local",
    )
    args = parser.parse_args()
    print(args)

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    if (bas := args.bas) is not None:
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
    cargs = {"scenario_configs": []} if config["project"]["run_type"] == "colo" else {}
    print(f"{kwargs=}\n{cargs=}")

    if not args.warnings:
        warnings.simplefilter("ignore")

    try:
        patio = BAs(
            name="BAs_" + now,
            solar_ilr=config["project"]["solar_ilr"],
            data_kwargs=kwargs,
            bas=config["project"]["balancing_authorities"],
            by_plant=False,
            regime=config["project"]["regime"],
            pudl_release=config["project"]["pudl_release"],
            **cargs,
        )
        if args.specs:
            patio.get_fossils_specs()
        elif args.setup:
            patio.pd.setup_all()
        else:
            patio.run_all()
            if config["project"]["run_type"] == "colo":
                print(f"patio-colo -d '{patio.colo_dir}'", file=sys.stderr)
                print()
                print(patio.colo_dir)
                to_delete = patio.dir_path
                colo_dir = Path.home() / patio.colo_dir
                del patio
                shutil.rmtree(to_delete)
                shutil.copyfile(PATIO_DOC_PATH / f"logs/{now}.log", colo_dir / "data.log")
                shutil.copyfile(
                    PATIO_DOC_PATH / f"logs/{now}.jsonl", colo_dir / "data_log.jsonl"
                )
            else:
                patio.write_output()

                print(patio.name)
                if not args.local:
                    print("uploading to patio-results/...")
                    patio.upload_output()
    finally:
        pass
        # queue.put_nowait(None)
        # lp.join()
        # if args.shutdown:
        #     from azure.identity import DefaultAzureCredential
        #
        #     from azure.mgmt.compute import ComputeManagementClient
        #     """
        #     # PREREQUISITES
        #         pip install azure-identity
        #         pip install azure-mgmt-compute
        #     # USAGE
        #         python virtual_machine_power_off_maximum_set_gen.py
        #
        #         Before run the sample, please set the values of the client ID, tenant ID and client secret
        #         of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
        #         AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
        #         https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
        #     """
        #
        #     client = ComputeManagementClient(
        #         credential=DefaultAzureCredential(),
        #         subscription_id="{subscription-id}",
        #     )
        #
        #     client.virtual_machines.begin_power_off(
        #         resource_group_name="RMI-SP-CFE-CLEANREPO-RG",
        #         vm_name="RMI-SP-CFE-CLEANREPO-VM",
        #     ).result()
        #     print("Shutting down...")
        # os.system("")


if __name__ == "__main__":
    main()
