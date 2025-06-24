***************************************************************************************
Cheshire: a Python Template Repository for RMI created by Catalyst Cooperative
***************************************************************************************

.. image:: https://github.com/rmi-electricity/cheshire/workflows/tox-pytest/badge.svg
   :target: https://github.com/rmi-electricity/cheshire/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://github.com/rmi-electricity/cheshire/workflows/docs/badge.svg
   :target: https://rmi-electricity.github.io/cheshire/
   :alt: GitHub Pages Status

.. image:: https://coveralls.io/repos/github/rmi-electricity/cheshire/badge.svg
   :target: https://coveralls.io/github/rmi-electricity/cheshire

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/uv
    :alt: uv

.. contents::
   :depth: 2

.. readme-intro

This template repository helps make new Python projects easier to set up and more
uniform. It contains a lot of infrastructure surrounding a minimal Python package named
``cheshire`` (the cat who isn't entirely there...). This template is mostly a lightly
modified copy of Catalyst Cooperative's
`cheshire <https://github.com/catalyst-cooperative/cheshire>`_ but with alterations
for private work and alternative tools.

The goal of this template is to provide a uniform starting point for Python projects,
with reasonable configurations for a suite of common tools. It is by no means
comprehensive but generally errs on including a kind of tool rather excluding it. In
other words, it includes a lot of things that are not necessary and likely not worth
getting to work for a basic Python project.

Getting Started
=======================================================================================
Please read this whole getting started section before beginning.

Setup your development machine using uv
---------------------------------------------------------------------------------------
We recommend using `uv <https://github.com/astral-sh/uv>`__ to manage python
installations, virtual environments, and packages instead of conda or mamba. If you
don't have `uv <https://github.com/astral-sh/uv>`__ installed you can install it using
homebrew with the instructions below. More information and alternative installation
instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.

.. code-block:: zsh

   brew install uv
   uv python install 3.13
   uv tool update-shell

There are a number of development tools that we recommend using and installing globally
using uv. That way they can be installed once and used by multiple projects. The
instructions below assume these tools are installed as follows.

.. code-block:: zsh

   uv tool install tox --with tox-uv
   uv tool install pre-commit --with pre-commit-uv
   uv tool install ruff

In the future, these tools can be upgraded by running ``uv tool upgrade --all``.

.. Note::

  In this repository we effectively disable uv's locking functionality by including
  ``uv.lock`` in ``.gitignore``. This simplifies library development and dependabot.
  See `the lockfile <https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile>`_
  for information on this functionality. To use this feature as it was intended,
  remove ``uv.lock`` from ``.gitignore``.

Create a new repository from this template
---------------------------------------------------------------------------------------
* Choose a name for the new package that you are creating.
* The name of the repository should be the same as the name of the new Python package
  you are going to create. e.g. a repository at ``rmi-electricity/cheshire`` should
  be used to define a package named ``cheshire``.
* Click the green ``Use this template`` to create a new Python project repo.
  See `these instructions for using a template <https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template>`__.
* Create a release with a version tag if there isn't one already. This is required
  because various tools use it to set the version dynamically. See
  `managing releases <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>`__
  for more information.
* Clone the new repository to your development machine.
* Create the virtual environment by running ``uv sync --extra dev --extra doc`` in the top level of the repository.
* Run ``pre-commit install`` in the newly cloned repository to install the `pre-commit hooks <https://pre-commit.com/>`__ defined in ``.pre-commit-config.yaml``.
* Run ``tox`` from the top level of the repository to verify that everything is working correctly.


Rename the package and distribution
---------------------------------------------------------------------------------------
Once your forked version of the ``cheshire`` package is working, you can change the
package and distribution names in your new repo to reflect the name of your package.
The **package name** is determined by the name of the directory under ``src/`` which
contains the source code, and is the name you'll use to import the package for use in
a program, script, or notebook. E.g.:

.. code:: python

  import cheshire

The **distribution name** is the name that is used to install the software using a
program like  ``pip`` or uv. We are using the ``rmi`` namespace for the
packages that we publish, so the ``dispatch`` package would have the distribution
name ``rmi.dispatch``. The distribution name is determined by the ``name`` argument
under ``[project]`` in ``pyproject.toml``. See :pep:`423` for more on Python package
naming conventions. You will want to search the ``pyproject.toml`` file and replace
**all** references to ``cheshire`` with your package's name.

The package and distribution names are used throughout the files in the template
repository, and they all need to be replaced with the name of your new package.

* Rename the ``src/cheshire`` directory to reflect the new package name.
* Search for ``cheshire`` and replace it as appropriate everywhere.
  Sometimes this will be with a distribution name like ``rmi.cheshire`` or ``rmi-cheshire`` and  sometimes this will be the importable package name ``cheshire``.
  You can use ``grep -r`` to search recursively through all of the files for the word ``cheshire`` at the command line, or use the search-and-replace functionality of your IDE / text editor.
  (Global search in PyCharm is command+shift+f)

Now that everything is renamed, make sure all the renaming worked properly by running
``tox`` from the top level of the repository to verify that everything is working
correctly. If it passes, you can commit your new skeleton package and get to work!

.. Warning::

  Unless you have relatively complete tests of your package, you will want to disable
  ``.github/workflows/bot-auto-merge.yml`` by either commenting out its contents or
  deleting the file. If you do this, do the same with ``.github/dependabot.yml``.

  If you leave these GitHub Actions in place with insufficient tests, GitHub might break
  your package by upgrading dependencies to version that are not compatible with your
  package.

What this template provides
=======================================================================================

Python Package Skeleton
---------------------------------------------------------------------------------------
* Dummy code for a skeleton python package with the following structure:

  * The ``src`` directory contains the code that will be packaged and deployed on the
    user system. That code is in a directory with the same name as the package.
  * A simple python module (``dummy.py``), and a separate module providing a command
    line interface to that module (``cli.py``) are included as examples.
  * A module (``dummy_pudl.py``) that includes an example of how to access PUDL data.
  * Any files in the ``src/package_data/`` directory will also be packaged and deployed.

* Instructions for ``pip`` on how to install the package and configurations for a
  number of tools in ``pyproject.toml`` including the following:

  * Package dependencies, including "extras" -- additional optional
    package dependencies that can be installed in special circumstances: ``dev``,
    ``doc```, and ``tests``.
    Note: if you follow the instructions above and install ``ruff``, ``tox``, and ``pre-commit`` globally, you do not need to install the ``tests`` extras yourself.
  * The CLI deployed using a ``console_script`` entrypoint.
  * ``setuptools_scm`` to obtain the package's version directly from ``git`` tags.
  * What files (beyond the code in ``src/`` are included in or excluded from the package
    on the user's system.
  * Configurations for ``ruff``, ``doc8``, and ``rstcheck`` described in the
    `Code Formatting and Linters`_ section below.

Testing Tools
---------------------------------------------------------------------------------------

Pytest Testing Framework
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* A skeleton `pytest <https://docs.pytest.org/>`_ testing setup is included in the
  ``tests/`` directory.
* Session-wide test fixtures, additional command line options, and other pytest
  configuration can be added to ``tests/conftest.py``
* Exactly what pytest commands are run during continuous integration is controlled by
  Tox.

Test Coordination with Tox
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* We define several different test environments for use with Tox in ``pyproject.toml`` in the sections starting with ``[tool.tox``.
* `Tox <https://tox.wiki/en/latest/>`__ is used to run pytest in an isolated Python virtual environment.
* We also use Tox to coordinate running the code linters and building the documentation.

Test Coverage
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* We use Tox, pytest, and `coverage <https://coverage.readthedocs.io>`__
  to measure and record what percentage of our codebase is being tested, and to
  identify which modules, functions, and individual lines of code are not being
  exercised by the tests.
* When you run ``tox`` a summary of the test coverage will be printed at the end of
  the tests (assuming they succeed).

See `GitHub Actions`_ for additional tools that track coverage statistics.

Code Quality Tools
---------------------------------------------------------------------------------------

Git Pre-commit Hooks
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* A variety of sanity checks are defined as git pre-commit hooks -- they run any time
  you try to make a commit, to catch common issues before they are saved. Many of these
  hooks are taken from the excellent `pre-commit project <https://pre-commit.com/>`__.
* The hooks are configured in ``.pre-commit-config.yaml``, see
  `Code Formatting and Linters`_ for details.
* For them to run automatically when you try to make a commit, you **must** install the
  pre-commit hooks in your cloned repository first. This only has to be done once by
  running ``pre-commit install`` in your local repo.
* These checks are run as part of our GitHub automations, which will fail if the
  pre-commit hooks fail.

Additional comments on using Pre-commit
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Most git GUI tools work with pre-commit but don't work that well. The terminal based
``git`` is usually the safer choice. See
`notes on git for <https://github.com/rmi-electricity/.github-private/blob/main/profile/notes_on_git.md>`__
for recommendations and instructions.

Code Formatting and Linters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
To avoid the tedium of meticulously formatting all the code ourselves, and to ensure a
standard style of formatting and syntactical idioms across the codebase, we use several
automatic code formatters, which run as pre-commit hooks. The following formatters are
included in the template ``.pre-commit-config.yaml``:

* `Deterministic formatting with ruff (similar to black) <https://docs.astral.sh/ruff/formatter/>`__
* Fix some of the issues found by `ruff <https://github.com/charliermarsh/ruff>`__,
  including to:

  * Use only absolute import paths
  * Standardize the sorting of imports
  * Remove unnecessary f-strings
  * Upgrade type hints for built-in types
  * Upgrade Python syntax

* Clear outputs in Jupyter notebooks using `nbstripout <https://github.com/kynan/nbstripout>`_.

To catch additional errors before commits are made, and to ensure uniform formatting
across the codebase, we also use `ruff <https://github.com/charliermarsh/ruff>`__  as
a linter, as well as other tools, to identify issues in code and documentation files.
They don't change the files, but they will raise an error or warning when something
doesn't look right so you can fix it.

* `ruff <https://github.com/charliermarsh/ruff>`__ is an extremely fast Python linter,
  written in Rust that replaces a number of other tools including:

  * `flake8 <https://github.com/PyCQA/flake8>`__ is an extensible Python linting
    framework, with a bunch of plugins.
  * `bandit <https://bandit.readthedocs.io/en/latest/>`__ identifies code patterns known
    to cause security issues.

* `doc8 <https://github.com/pycqa/doc8>`__ and `rstcheck
  <https://github.com/myint/rstcheck>`__ look for formatting issues in our docstrings
  and the standalone ReStructuredText (RST) files under the ``docs/`` directory.

See for
`tests and linters <https://github.com/rmi-electricity/.github-private/blob/main/profile/notes_on_tests_and_linters.md>`__
some advice on how to avoid getting bogged down making the linter happy.


Documentation Tools
---------------------------------------------------------------------------------------
* We build our documentation using `Sphinx <https://www.sphinx-doc.org/en/master/>`__.
* Standalone docs files are stored under the ``docs/`` directory, and the Sphinx
  configuration is there in ``conf.py`` as well.
* We use `Sphinx AutoAPI <https://sphinx-autoapi.readthedocs.io/en/latest/>`__ to
  convert the docstrings embedded in the python modules under ``src/`` into additional
  documentation automatically.
* The top level documentation index simply includes this ``README.rst``, the
  ``LICENSE.txt`` and ``code_of_conduct.rst`` files are similarly referenced. The only
  standalone documentation file under ``docs/`` right now is the ``release_notes.rst``.
* Unless you're debugging something specific, the docs should always be built using
  ``tox -e docs`` as that will lint the source files using ``doc8`` and ``rstcheck``,
  and wipe previously generated documentation to build everything from scratch. The docs
  are also rebuilt as part of the normal Tox run.

Documentation Publishing
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* We use the `GitHub Pages <https://pages.github.com>`__ service to host our
  documentation.
* When you open a PR or push to ``dev`` or ``main``, the associated
  documentation is automatically built and stored in a ``gh-pages`` branch.
* To make the documentation available, go to the repositories settings. Select
  'Pages' under 'Code and automation', select 'Deploy from a branch' and then
  select the ``gh-pages`` branch and then ``/(root)``, and click save.
* The documentation should then be available at https://rmi-electricity.github.io/<repo-name>/.

GitHub Automations
---------------------------------------------------------------------------------------

Dependabot
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
We use GitHub's `Dependabot <https://docs.github.com/en/code-security/dependabot/dependabot-version-updates>`__
to automatically update the allowable versions of packages we depend on. This applies
to both the Python dependencies specified in ``pyproject.toml`` and to the versions of
the `GitHub Actions <https://docs.github.com/en/actions>`__ that we employ. The
dependabot behavior is configured in ``.github/dependabot.yml``.

For Dependabot's PRs to automatically get merged, your repository must have access to
the correct organization secrets and the ``rmi-electricity auto-merge Bot`` GitHub App.
Contact Alex Engel for help setting this up.

GitHub Actions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Under ``.github/workflows`` are YAML files that configure the `GitHub Actions
<https://docs.github.com/en/actions>`__ associated with the repository. We use GitHub
Actions to:

* Run continuous integration using `tox <https://tox.wiki>`__ on several different
  versions of Python.
* Build and publish docs to GitHub Pages.
* Merge passing dependabot PRs.

* When the tests are run via the ``tox-pytest`` workflow in GitHub Actions, the test
  coverage data from the ``coverage.info`` output is uploaded to a service called
  `Coveralls <https://coveralls.io>`__ that saves historical data about our test
  coverage, and provides a nice visual representation of the data -- identifying which
  subpackages, modules, and individual lines of are being tested. For example, here are
  the results
  `for the cheshire repo <https://coveralls.io/github/rmi-electricity/cheshire>`__.
