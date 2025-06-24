"""A template repository for a Python package.

Created by Catalyst Cooperative, modified by RMI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("patio")
except PackageNotFoundError:
    print("Version unknown because package is not installed.")
    __version__ = "unknown"
