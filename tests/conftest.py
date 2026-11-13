# On macOS, torch (pulled in via sentence-transformers) and LightGBM each ship
# their own OpenMP runtime. If torch loads first, LightGBM training segfaults
# once pyarrow is also in the process. Importing lightgbm before any test
# module loads torch keeps a single OpenMP runtime in charge.
import lightgbm  # noqa: F401
