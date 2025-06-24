=======================================================================================
PACKAGE_NAME Release Notes
=======================================================================================

.. _release-v0-3-0:

---------------------------------------------------------------------------------------
0.3.0 (2024-XX-XX)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
* Switching coverage to `Coveralls <coveralls.io>`_.
* Moving ``doc8``, ``pytest``, and ``rstcheck`` configs to ``pyproject.toml``.
* Updates to ``docs-to_gh-pages.yml`` so it works with the version from
  ``setuptools-scm``.
* Version bumps, including testing against Python 3.11.
* Drop ``flake8``, ``isort``, ``bandit`` and others in favor of ``ruff``.
* Reorganized and simplified ``README.rst`` to more clearly provide instructions for
  using the ``cheshire`` template, and to describe what it contains.
* Change GHA CI to only use pip and tox rather than mamba and tox. This means that
  users are more likely to be able to completely ignore those configuration files.
* Change ``sphinx`` command in ``tox.ini`` so that warnings are not treated as errors.
  In my experience lots of things that seem fine raise warnings but don't cause problems
  in the docs so treating warnings as errors just makes tox fail. Also many of the
  warnings don't seem fixable without changing package code in undesirable ways.
* Bump Python version to 3.12.
* Switch to using ``ruff format`` from ``black`` in pre-commit hooks.
* Use :mod:`importlib` instead of :mod:`pkg_resources` in ``console_scripts_test.py``.
* Include configurations that enable and examples that demonstrate accessing PUDL data
  from GCS locally and in GitHub Actions.
* Update to support new way to setup PUDL GCP using ``rmi.etoolbox``.
* Update to support new way to access PUDL on AWS using ``rmi.etoolbox`` requiring
  much less setup than the GCP version.
* Pre-commit hook version bumps.



.. _release-v0-2-0:

---------------------------------------------------------------------------------------
0.2.0 (2022-08-24)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
* Adapting to RMI internal project needs.
* Removing coverage tracking functionality.
* Docs are now built and deployed to `GitHub Pages <https://pages.github.com>`__.
* Moving ``bandit`` and ``mypy`` configurations to ``pyproject.toml``.
* Replacing ``setup.py`` with ``setup.cfg``.
* Removing ``docker`` and PyPI distribution functionality.


Known Issues
^^^^^^^^^^^^
* It's also good to list any remaining known problems, and link to their issues too.

.. _release-v0-1-0:

---------------------------------------------------------------------------------------
0.1.0 (2022-04-29)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
* This is the first fully functional and documented version of our template repository.

Known Issues
^^^^^^^^^^^^
* Need to get some user feedback!
* Still need to look at updating our Code of Conduct. See :issue:`12`
