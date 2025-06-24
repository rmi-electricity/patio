"""A dummy unit test so pytest has something to do."""

import logging
from pathlib import Path

import pytest

from cheshire.dummy import do_something

logger = logging.getLogger(__name__)


def test_nothing(test_dir: Path) -> None:
    """A dummy test that relies on our dummy fixture."""
    assert isinstance(test_dir, Path)  # nosec: B101
    assert test_dir.exists()  # nosec: B101
    assert test_dir.is_dir()  # nosec: B101


@pytest.mark.parametrize(
    ("a", "b", "expected_c"),
    [
        (1, 1, 2),
        (3, 5, 8),
        (13, 22, 35),
    ],
)
def test_something(a: int, b: int, expected_c: int) -> None:
    """Test the dummy function from our dummy module to generate coverage.

    This function also demonstrates how to parametrize a test.

    """
    c = do_something(a, b)
    assert c == expected_c  # nosec: B101


def test_use_a_table_from_pudl() -> None:
    """Test that we can run a function that uses a table from the PUDL GCS bucket."""
    from cheshire.dummy_pudl import use_a_table_from_pudl

    df = use_a_table_from_pudl()
    assert df.loc["CA"] > df.loc["AK"]
