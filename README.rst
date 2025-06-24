***************************************************************************************
Patio: an electricity model
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
   pre-commit install

In the future, these tools can be upgraded by running ``uv tool upgrade --all``.

.. Note::

  In this repository we effectively disable uv's locking functionality by including
  ``uv.lock`` in ``.gitignore``. This simplifies library development and dependabot.
  See `the lockfile <https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile>`_
  for information on this functionality. To use this feature as it was intended,
  remove ``uv.lock`` from ``.gitignore``.

Setup patio environment
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: zsh

   uv sync --extra dev

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
