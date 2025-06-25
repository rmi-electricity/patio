"""Functional entry point for PATIO."""

import logging
import logging.config
import logging.handlers
import multiprocessing
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

from patio.constants import PATIO_DOC_PATH, PATIO_PUDL_RELEASE, TECH_CODES


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
    # parser.add_argument(
    #     "-t, --test",
    #     action="store_true",
    #     help="run in test mode",
    #     default=False,
    #     required=False,
    #     dest="test",
    # )
    parser.add_argument(
        "-P, --pudl_release",
        type=str,
        help="pudl release, ",
        default=PATIO_PUDL_RELEASE,
        required=False,
        dest="pudl_release",
    )
    parser.add_argument(
        "-n, --name-prefix",
        type=str,
        help="the name of the run, ",
        default="BAs",
        required=False,
        dest="name_prefix",
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
        "-p, --plant",
        action="store_true",
        help="Run each coal plant individually.",
        default=False,
        required=False,
        dest="by_plant",
    )
    parser.add_argument(
        "-f, --figs",
        action="store_true",
        help="Include figs",
        default=False,
        required=False,
        dest="include_figs",
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
        "-s, --specs",
        action="store_true",
        help="Just export fossil specs",
        default=False,
        required=False,
        dest="specs",
    )
    parser.add_argument(
        "-d, --data-setup",
        action="store_true",
        help="Process BA profile data",
        default=False,
        required=False,
        dest="setup",
    )
    parser.add_argument(
        "-r, --re-limits-dispatch",
        type=str,
        help="`-r True` for re_limits_dispatch == True only (default). `-r False` for re_limits_dispatch == False only. `-r both` for both.",
        default="True",
        required=False,
        dest="re_limits_dispatch",
    )
    parser.add_argument(
        "-L, --limited",
        action="store_true",
        help="use limited renewable regime instead of reference https://www.nrel.gov/gis/wind-supply-curves.html",
        default=False,
        required=False,
        dest="limited",
    )
    parser.add_argument(
        "-l, --local",
        action="store_true",
        help="do not upload to Azure",
        default=False,
        required=False,
        dest="local",
    )
    parser.add_argument(
        "-k, --km-max-re-distance",
        type=str,
        help="maximum distance between fossil and clean repowering sites in km",
        default="45",
        required=False,
        dest="max_re_distance",
    )
    parser.add_argument(
        "-t, --techs",
        type=str,
        help="type codes of fossil plants that should be considered for clean repowering separated by commas (e.g. `coal` or `coal,NGST`)",
        default=None,
        required=False,
        dest="cr_eligible_techs",
    )
    parser.add_argument(
        "-c, --colo-techs",
        type=str,
        help="type codes of fossil plants that should be considered for co-located load (e.g. `coal` or `coal,NGST`)",
        default=None,
        required=False,
        dest="colo_techs",
    )
    # parser.add_argument(
    #     "--colo-only",
    #     action="store_true",
    #     help="only run colo analysis on counterfactual, without running and CR portfolios",
    #     default=False,
    #     required=False,
    #     dest="colo_only",
    # )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="shutdown system after run completes",
        default=False,
        required=False,
        dest="shutdown",
    )
    args = parser.parse_args()
    print(args)

    if (bas := args.bas) is not None:
        bas = [s.strip() for s in bas.split(",")]

    kwargs = {"include_figs": args.include_figs}

    if (cr_eligible_techs := args.cr_eligible_techs) is not None:
        kwargs = kwargs | {
            "cr_eligible_techs": [
                TECH_CODES[t.strip().casefold()] for t in cr_eligible_techs.split(",")
            ]
        }
    if (colo_techs := args.colo_techs) is not None:
        kwargs = kwargs | {
            "colo_techs": [TECH_CODES[t.strip().casefold()] for t in colo_techs.split(",")]
        }
    cargs = {"scenario_configs": []} if colo_techs else {}

    kwargs = (
        kwargs
        | {
            "re_limits_dispatch": {
                "all": "both",
                "both": "both",
                "true": True,
                "false": False,
            }[args.re_limits_dispatch.casefold()]
        }
        | {"max_re_distance": float(args.max_re_distance)}
    )
    # ROOT_LOGGER.info("data_kwargs %s", kwargs)

    if not args.warnings:
        warnings.simplefilter("ignore")

    try:
        patio = BAs(
            name=args.name_prefix + "_" + now,
            solar_ilr=1.34,
            data_kwargs={"re_by_plant": True} | kwargs,
            bas=bas,
            by_plant=args.by_plant,
            regime="limited" if args.limited else "reference",
            pudl_release=args.pudl_release,
            # queue=queue,
            **cargs,
        )
        if args.specs:
            patio.get_fossils_specs()
        elif args.setup:
            patio.pd.setup_all()
        else:
            patio.run_all()
            if colo_techs is not None:
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
