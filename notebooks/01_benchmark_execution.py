# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Notebook 1: Benchmark Execution Pipeline
#
# This notebook walks through the **execution phase** of the AI weather
# benchmarking workflow:
#
# 1. Configure the project environment (paths, catalogs)
# 2. Detect available benchmark models and their date ranges
# 3. Run the deterministic benchmark (AIFS, FuXi, GraphCast)
# 4. Run the probabilistic benchmark (AIFS Ensemble, GenCast)
# 5. Discover and verify the generated NetCDF outputs
#
# All heavy-lifting functions (model detection, output discovery, etc.)
# are imported from `ui/pages/short_medium_benchmarking.py`, which acts
# as the shared `utils` module for both training notebooks.
#
# **Prerequisites:**
# - `pip install fpdf2` (for Notebook 2 PDF reports)
# - MOMP package discoverable via `PYTHONPATH`
# - Observation data in `data/external/CHIRPS_IMERG/` and `data/external/ENACTS/`
# - Model forecasts in `data/external/{AIFS,fuxi,graphcast,AIFS_ENS,gencast}/0p25/`

# %% [markdown]
# ## 1. Environment Setup
#
# We bootstrap the project paths using the MOMP loader, then import the
# shared utilities from the Streamlit dashboard file.

# %%
import os
import sys
import warnings
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

warnings.filterwarnings("ignore")
get_ipython().run_line_magic("matplotlib", "inline")

# ------------------------------------------------------------------
# 1a. Load MOMP configuration to discover the project root
# ------------------------------------------------------------------
import momp.lib.loader as loader

get_cfg = loader.get_cfg
cfg = get_cfg()
cfg.probabilistic = True          # NOTE: fixed trailing-comma bug
base_dir = Path(cfg.base_dir)

# Walk up from momp/ to find the folder containing both data/ and apps/
def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "apps").exists():
            return p
        if p == p.parent:
            break
    return start

PROJECT_ROOT = _find_project_root(base_dir)

# Inject project root into sys.path so `import config` resolves correctly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# 1b. Import shared utilities from the Streamlit dashboard
#     (short_medium_benchmarking.py acts as utils.py)
# ------------------------------------------------------------------
# Add the ui/pages directory so we can import the dashboard module
_ui_pages = PROJECT_ROOT / "ui" / "pages"
if str(_ui_pages) not in sys.path:
    sys.path.insert(0, str(_ui_pages))

import short_medium_benchmarking as utils

# Pull out the constants and helpers we need
PROJECT_ROOT          = utils.PROJECT_ROOT
EXTERNAL_DATA_DIR     = utils.EXTERNAL_DATA_DIR
ROMP_DEMO_OUT_DIR     = utils.ROMP_DEMO_OUT_DIR
ROMP_DEMO_FIG_DIR     = utils.ROMP_DEMO_FIG_DIR
CONFIG_BENCHMARK_MODEL_CATALOG = utils.CONFIG_BENCHMARK_MODEL_CATALOG
detect_benchmark_models        = utils.detect_benchmark_models
discover_map_outputs           = utils.discover_map_outputs

print("PROJECT_ROOT      :", PROJECT_ROOT)
print("EXTERNAL_DATA_DIR :", EXTERNAL_DATA_DIR)
print("ROMP_DEMO_OUT_DIR :", ROMP_DEMO_OUT_DIR)
print("ROMP_DEMO_FIG_DIR :", ROMP_DEMO_FIG_DIR)

# %% [markdown]
# ## 2. Detect Available Benchmark Models
#
# The utility `detect_benchmark_models()` scans every directory listed in
# `BENCHMARK_MODEL_CATALOG`, reads the NetCDF files inside, and infers the
# date range each model covers. This is what the dashboard uses to populate
# the **Models** table.

# %%
model_registry = detect_benchmark_models()

if not model_registry:
    print("No benchmark models were detected.")
    print("Check BENCHMARK_MODEL_CATALOG in config.py and verify that")
    print("the model directories exist under:", EXTERNAL_DATA_DIR)
else:
    display(
        pd.DataFrame(model_registry)[
            ["Model", "Category", "Type", "Coverage", "Date Range"]
        ]
    )

# %% [markdown]
# ## 3. Run the Deterministic Benchmark
#
# The deterministic pipeline evaluates **AIFS, FuXi, and GraphCast** using
# the metrics:
#
# | Metric | Description |
# |---|---|
# | **FAR** (False Alarm Rate) | Fraction of forecast onset cells that were not observed |
# | **MR** (Miss Rate) | Fraction of observed onset cells that were missed |
# | **MAE** (Mean Absolute Error) | Average timing error in days |
#
# We invoke `run_momp.py` via `subprocess` so the output is captured
# cleanly inside the notebook.

# %%
def run_benchmark(mode: str, model: str = None, capture: bool = True):
    """
    Run the MOMP benchmark pipeline.

    Parameters
    ----------
    mode : "det" or "prob"
    model : optional model key (e.g. "AIFS", "FuXi")
    capture : if True, suppress stdout/stderr and return the CompletedProcess
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "run_momp.py"), "--mode", mode]
    if model:
        cmd.extend(["--model", model])

    print(f"Running: {' '.join(cmd)}")

    if capture:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  -> completed successfully.")
        else:
            print(f"  -> FAILED (exit code {result.returncode})")
            print("STDERR:", result.stderr[-500:])
        return result
    else:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return None

# Run ALL deterministic models
print("=" * 60)
print("DETERMINISTIC BENCHMARK")
print("=" * 60)
run_benchmark("det")

# %% [markdown]
# ### 3.1 Run a Single Deterministic Model
#
# If you only want to re-evaluate one model (e.g. after changing a config
# parameter), use the `--model` flag.

# %%
run_benchmark("det", model="AIFS")

# %% [markdown]
# ## 4. Run the Probabilistic Benchmark
#
# The probabilistic pipeline evaluates **AIFS Ensemble** and **GenCast**
# using ensemble-based metrics:
#
# | Metric | Description |
# |---|---|
# | **BS** (Brier Score) | Mean squared error of probability forecasts |
# | **RPS** (Ranked Probability Score) | Generalization of BS to ordered categories |
# | **AUC** | Area under the ROC curve |
# | **Reliability** | Calibration of probability forecasts |

# %%
print("=" * 60)
print("PROBABILISTIC BENCHMARK")
print("=" * 60)
run_benchmark("prob")

# %% [markdown]
# ### 4.1 Run a Single Probabilistic Model

# %%
run_benchmark("prob", model="AIFS_ENS")

# %% [markdown]
# ## 5. Discover Generated Benchmark Outputs
#
# After running the pipeline, we scan `ROMP_DEMO_OUT_DIR` for NetCDF files
# containing the computed metrics. The utility `discover_map_outputs()`
# returns a list of records describing each output file:
#
# - `path`         : absolute path to the NetCDF file
# - `model_label`  : human-readable model name
# - `windows`      : lead-time windows present in the file
# - `metrics`      : which metrics were computed

# %%
map_outputs = discover_map_outputs()

if not map_outputs:
    print("No benchmark outputs were found.")
    print("Run the benchmark cells above first, then re-run this cell.")
else:
    outputs_df = pd.DataFrame(map_outputs)
    outputs_df["windows"] = outputs_df["windows"].map(lambda x: ", ".join(x))
    outputs_df["metrics"] = outputs_df["metrics"].map(lambda x: ", ".join(x))
    display(outputs_df)
    print(f"\nTotal outputs discovered: {len(map_outputs)}")

# %% [markdown]
# ## 6. Quick Verification
#
# Sanity-check that the expected files exist and contain the expected
# variables.

# %%
import xarray as xr

def verify_output(path: str):
    """Open a NetCDF and print its structure."""
    try:
        ds = xr.open_dataset(path)
        print(f"\n{Path(path).name}")
        print(f"  dimensions : {dict(ds.dims)}")
        print(f"  variables  : {list(ds.data_vars)}")
        ds.close()
    except Exception as e:
        print(f"  ERROR opening {path}: {e}")

# Verify the first 3 outputs
for record in map_outputs[:3]:
    verify_output(record["path"])

# %% [markdown]
# ## 7. Summary
#
# At this point you have:
#
# - A populated `data/ROMP_OUT/et/output/` directory with one NetCDF per
#   model per lead-time window, containing FAR / MR / MAE (deterministic)
#   or BS / RPS / AUC / Reliability (probabilistic).
# - A populated `data/ROMP_OUT/et/figure/` directory with PNG plots.
# - A `map_outputs` list ready for analysis in **Notebook 2**.
#
# **Next:** open `02_benchmark_analysis.ipynb` to visualize the results.

# %%
# Save the discovered outputs for Notebook 2
import pickle
_cache_path = PROJECT_ROOT / "notebooks" / ".benchmark_cache.pkl"
_cache_path.parent.mkdir(parents=True, exist_ok=True)
with open(_cache_path, "wb") as f:
    pickle.dump({"map_outputs": map_outputs, "model_registry": model_registry}, f)
print(f"Cache saved to {_cache_path}")