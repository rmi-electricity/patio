#!/bin/bash

# update conda-meta/history if it has blank pixi default
# see https://github.com/rstudio/reticulate/issues/1184#issuecomment-2655718703
if ! grep -q "# cmd:" "$CONDA_PREFIX/conda-meta/history"; then
    echo -e "# cmd: $CONDA_PREFIX/bin/conda" > $CONDA_PREFIX/conda-meta/history
fi

# install blsR and tidyxl as they are not available from conda-forge
R -e 'tryCatch(library(blsR), error = function(e) install.packages("blsR", repos = "http://cran.us.r-project.org")); tryCatch(library(tidyxl), error = function(e) install.packages("tidyxl", repos = "http://cran.us.r-project.org"))'
