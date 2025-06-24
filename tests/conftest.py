import shutil
from pathlib import Path

import pytest

from patio.data.asset_data import AssetData
from patio.data.profile_data import ProfileData


@pytest.fixture(scope="session")
def copt_license_etc():
    import os

    for var, val in {
        "COPT_HOME": "/Users/aengel/Applications/copt72",
        "COPT_LICENSE_DIR": "/Users/aengel/Applications/copt72",
        # "DYLD_LIBRARY_PATH": ""
    }.items():
        # if var not in os.environ:
        #     val = subprocess.getoutput(["echo", f"${var}"])
        os.environ[var] = val


@pytest.fixture(scope="session")
def asset_data():
    return AssetData()


@pytest.fixture(scope="session")
def profile_data(asset_data):
    return ProfileData(asset_data, solar_ilr=1.34, regime="reference")


@pytest.fixture(scope="session")
def profile_data_limited(asset_data):
    return ProfileData(asset_data, solar_ilr=1.34, regime="limited")


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def temp_dir(test_dir) -> Path:
    """Return the path to a temp directory that gets deleted on teardown."""
    out = test_dir / "temp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(exist_ok=True)
    return out
    # shutil.rmtree(out)
