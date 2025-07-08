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
you must open it from the terminal with the following command. The first time
running this may take a long time as additional R packages are downloaded and compiled.

.. code-block:: zsh

   pixi run rstudio

Reticulate and RStudio can behave strangely with pixi environments, additional
environment variables may help. But mean that reticulate will always use this environment
no matter the project.

.. code-block:: zsh

   echo "export RETICULATE_PYTHON=$(pixi run -e econ 'which python')" >> ~/.zshenv

Additional information on setting up your IDE to use the pixi environment see these guides
`PyCharm <https://pixi.sh/v0.20.1/ide_integration/pycharm/>`_,
`RStudio <https://pixi.sh/v0.20.1/ide_integration/r_studio/>`_.

Additional comments on using Pre-commit
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Most git GUI tools work with pre-commit but don't work that well. The terminal based
``git`` is usually the safer choice. See
`notes on git for <https://github.com/rmi-electricity/.github-private/blob/main/profile/notes_on_git.md>`__
for recommendations and instructions.

Running the model
=======================================================================================
To run the resource model:

.. code-block:: zsh

   pixi run patio

To run the economic model:

.. code-block:: zsh

   pixi run patio-econ < model-run-datestr >
