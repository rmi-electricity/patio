#!/bin/bash

# update conda-meta/history if it has blank pixi default
# see https://github.com/rstudio/reticulate/issues/1184#issuecomment-2655718703
if ! grep -q "# cmd:" "$CONDA_PREFIX/conda-meta/history"; then
    echo -e "# cmd: $CONDA_PREFIX/bin/conda" > $CONDA_PREFIX/conda-meta/history
fi
#echo -e "# cmd: $CONDA_PREFIX/bin/conda" > $CONDA_PREFIX/conda-meta/history

# if not there, get BBB file
if [ ! -f "BBB Fossil Transition Analysis Inputs.xlsm" ]; then
    {
      etb cloud get "patio-restricted/BBB Fossil Transition Analysis Inputs.xlsm"
    } || {
         echo "etb cloud init must run before BBB config can be downloaded"
    }
fi
