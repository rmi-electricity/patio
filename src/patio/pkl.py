from __future__ import annotations

import logging
import pickle
from pathlib import Path  # noqa: TC003
from typing import Any

LOGGER = logging.getLogger("patio")
__all__ = ["load_pickle", "save_pickle"]
MOD_MAP = {
    "CEMS": "patio.data.cems",
    "clean_all_cems": "patio.data.cems",
    "all_cems_daily": "patio.data.cems",
    "all_cems_normalized_daily": "patio.data.cems",
    "all_cems_monthly_fuel": "patio.data.cems",
    "AssetData": "patio.data.asset_data",
    "ProfileData": "patio.data.profile_data",
    "master_unit_list": "patio.data.asset_data",
    "show": "patio.helpers",
    "isclose": "patio.helpers",
    "find_na": "patio.helpers",
    "check_lat_lons": "patio.helpers",
    "distance_arrays": "patio.helpers",
    "lstrip0": "patio.helpers",
    "agg_profile": "patio.helpers",
    "fix_cems_datetime": "patio.helpers",
    "combine_cols": "patio.helpers",
    "add_states": "patio.helpers",
    "fix_na_neg_col": "patio.helpers",
    "adjust_col_for_pct_owned": "patio.helpers",
    "cat_multiindex_as_col": "patio.helpers",
    "round_coordinates": "patio.helpers",
    "distance": "patio.helpers",
    "ninja_profile_fix": "patio.helpers",
    "mo_to_days": "patio.helpers",
    "df_product": "patio.helpers",
    "prep_for_re_ninja": "patio.helpers",
    "combine_profiles": "patio.helpers",
    "Plant": "patio.model",
    "Utility": "patio.model",
    "Plants": "patio.model",
}


def load_pickle(name: Path) -> Any:
    """Function to load an object from a pickle

    :param name: the path of the pickle to load
    :type name: Path
    :return: contents from the pickle
    :rtype: object
    """
    try:
        with open(name.with_suffix(".pkl"), "rb") as f:
            temp = pickle.load(f)  # noqa: S301
        return temp
    except (ModuleNotFoundError, TypeError):
        with open(name.with_suffix(".pkl"), "rb") as f:
            temp = renamed_load(f)
        return temp


def save_pickle(contents: Any, name: Path) -> None:
    """Function to save to an object as a pickle

    :param contents: the contents of the pickle
    :type contents: object
    :param name: the path to use for the pickle
    :type name: Path
    :return: None
    """
    with open(name.with_suffix(".pkl"), "wb") as output:
        pickle.dump(contents, output, pickle.HIGHEST_PROTOCOL)


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if "patio" not in module:
            pass
        elif name in MOD_MAP:
            renamed_module = MOD_MAP[name]
        try:
            return super().find_class(renamed_module, name)
        except ModuleNotFoundError as exc:
            LOGGER.error("unable to load %s from %s", name, renamed_module, exc_info=exc)
            return None


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
