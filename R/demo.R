library(reticulate)

# use patio's python environment
use_python(gsub("lib/R", "bin/python", R.home()))

# create and get path to cache directory for persistent intermediate data
platformdirs <- import("platformdirs")
cache_dir <- platformdirs$user_cache_dir("patio", ensure_exists=TRUE)

# load cloud tools
cloud <- import("etoolbox.utils.cloud")
