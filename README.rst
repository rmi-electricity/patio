***************************************************************************************
Patio: an electricity model
***************************************************************************************

.. image:: https://github.com/rmi-electricity/patio/workflows/pytest/badge.svg
   :target: https://github.com/rmi-electricity/patio/actions?query=workflow%3Apytest
   :alt: Tox-PyTest Status

.. image:: https://github.com/rmi-electricity/patio/workflows/docs/badge.svg
   :target: https://rmi-electricity.github.io/patio/
   :alt: GitHub Pages Status

.. image:: https://coveralls.io/repos/github/rmi-electricity/patio/badge.svg
   :target: https://coveralls.io/github/rmi-electricity/patio

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh
    :alt: pixi

.. contents::
   :depth: 2

.. readme-intro

Getting Started
=======================================================================================
See
`setting up your development environment <https://github.com/rmi-electricity/.github-private/blob/main/profile/notes_on_dev_env.md>`_
for general tool installation and configuration.

Setup patio environment
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Unfortunately, geopandas doesn't work properly when installed using uv (or pip) so we
use patio with pixi.

.. code-block:: zsh

   brew install pixi

Navigate to the cloned ``patio`` repository and run the following to setup patio
with pixi.

.. code-block:: zsh

   pre-commit install
   pixi install

If the pixi command is not recognized after restarting the terminal you may need to
add it to your path.

.. code-block:: zsh

   echo 'export PATH=~/.pixi/bin:$PATH'  >> ~/.zshenv

If you have not yet initialized `etoolbox <https://github.com/RMI/etoolbox>`__ on your
computer, run this command and follow the instructions:

.. code-block:: zsh

   pixi run etb cloud init

The FRED_API_KEY and BLS_KEY must be set as environment variables.
To set the environment variable ``FRED_API_KEY`` with the value ``abcd1234``, use the following command:

.. code-block:: zsh

   echo 'export FRED_API_KEY=abcd1234' >> ~/.zshenv

The value can be retrieved in R using its key:

.. code-block:: R

   FRED_API_KEY <- Sys.getenv("FRED_API_KEY")

Using RStudio
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
For RStudio to use the patio environment created in the previous step,
you must open it from the terminal with the following command:

.. code-block:: zsh

   pixi run rstudio


To setup your IDE to use the pixi environment see these guides
`PyCharm <https://pixi.sh/v0.20.1/ide_integration/pycharm/>`_,
`RStudio <https://pixi.sh/v0.20.1/ide_integration/r_studio/>`_.

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
