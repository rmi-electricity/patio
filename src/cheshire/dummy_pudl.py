"""Example of a module that contains functions for PUDL."""

from etoolbox.utils.pudl import pd_read_pudl


def use_a_table_from_pudl():
    """Use a table from the PUDL GCS bucket.

    This is an example of a function that uses a table from PUDL.
    """
    return (
        pd_read_pudl("out_eia__yearly_plants").groupby("state").plant_name_eia.count()
    )
