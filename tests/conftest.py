import os
import shutil
from pathlib import Path

import pytest

from patio.data.asset_data import AssetData
from patio.data.profile_data import ProfileData


@pytest.fixture(scope="session")
def copt_license_etc():
    for var in ("COPT_HOME", "COPT_LICENSE_DIR"):
        os.environ[var] = str(Path.home() / "Applications/copt72")


@pytest.fixture
def os_solver():
    o_solver = os.environ.get("SOLVER", None)
    os.environ["SOLVER"] = "HIGHS"
    yield None
    if o_solver is not None:
        os.environ["SOLVER"] = o_solver
    else:
        del os.environ["SOLVER"]


@pytest.fixture(scope="session")
def asset_data():
    return AssetData()


@pytest.fixture(scope="session")
def profile_data(asset_data):
    return ProfileData(asset_data, solar_ilr=1.34, regime="reference")


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def temp_dir(test_dir):
    """Return the path to a temp directory that gets deleted on teardown."""
    out = test_dir / "temp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(exist_ok=True)
    out.joinpath("results").mkdir()
    yield out
    shutil.rmtree(out)
