---
title: Demo 3
teaching: 1
exercises: 0
questions:
- "How do we set up and run AI weather models locally in a container?"
- "What are the core commands needed to go from environment setup to generating a forecast figure?"
- "How do we tailor the model execution to specific use cases like onset, cessation, or temperature exceedance?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Execute a streamlined 4-command workflow to build a container, run a model, and generate use-case-specific outputs."
keypoints:
- "A streamlined 4-command workflow (build container, run model, get output, generate figure) simplifies local AI forecasting."
- "Use-case flags in the code allow groups to tailor outputs for onset/cessation, temperature exceedance, or precipitation exceedance."
- "Jupyter notebooks provide an interactive environment for executing and visualizing the AI weather models."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Running Your First AI Weather Forecast

## End-to-End Local Model Execution and Use-Case Application


### 1. Environment and Workflow Setup
- Utilize the Jupyter notebook environment for interactive model execution.
- Introduce the distilled 4-command workflow designed to simplify the end-to-end process:
  1. Build the container.
  2. Run the model.
  3. Get the output for discussion.
  4. Generate the use-case-specific figure.

### 2. Use-Case Group Execution
- Split participants into their designated use-case groups. 
- Ensure each group uses the specific flag in the code that distinguishes their use case:
  - **Onset/Cessation:** Led by Mesfin, Aryan, supported by Panchali.
  - **Temperature Exceedance:** Led by Docko, supported by Narayana.
  - **Precipitation Exceedance (Short-run rainfall):** Led by Koomi, supported by Shruti.

### 3. Output Generation and Discussion Prep
- Groups execute their specific model runs within the container.
- Participants generate the final figure (Command 4) and prepare their outputs for discussion and evaluation in subsequent sessions.



# Complete Rainy Season Onset Benchmarking Lesson

## Using the ROMP/MOMP Framework with AI Weather Models

---

## 📋 Lesson Overview

This lesson teaches you how to benchmark AI weather forecast models (AIFS, FuXi, GraphCast, GenCast, AIFS-ENS) against observational rainfall data (ENACTS) for Ethiopian rainy season onset prediction. You will learn both **deterministic** and **probabilistic** evaluation workflows using the ROMP (Rainy season Onset Metrics Package) framework.

---

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the theoretical foundations of rainy season onset detection
2. Configure and run deterministic benchmarks (MAE, FAR, Miss Rate)
3. Configure and run probabilistic benchmarks (BSS, RPS, AUC, Reliability)
4. Interpret spatial skill maps and model ranking scoreboards
5. Customize onset criteria, regions, and verification windows
6. Build interactive dashboards for forecast verification

---

## Theoretical Background

### What is Rainy Season Onset?

In Ethiopian agriculture, the **onset of the rainy season** (Kiremt: June–September) determines planting dates for millions of smallholder farmers. A late or false onset signal can lead to:

- Crop failure due to planting too early
- Lost growing days due to planting too late
- Food insecurity at regional scale

**Onset Definition (used in ROMP):**
The onset is declared when:
- At least `wet_init` mm of rain falls (initial trigger)
- Followed by `wet_spell` consecutive days with ≥ `wet_threshold` mm/day
- Without a subsequent dry spell of `dry_spell` days with < `dry_threshold` mm/day
- Within a defined search window (e.g., May 1 – September 30)

### Benchmarking Metrics

#### Deterministic Metrics (Single Forecast)
| Metric | Formula Concept | Interpretation |
|--------|----------------|----------------|
| **MAE** (Mean Absolute Error) | \|forecast_onset − obs_onset\| | Average error in days |
| **FAR** (False Alarm Rate) | false_alarms / (hits + false_alarms) | % of predicted onsets that didn't occur |
| **MR** (Miss Rate) | misses / (hits + misses) | % of actual onsets that were missed |

#### Probabilistic Metrics (Ensemble Forecasts)
| Metric | Interpretation |
|--------|----------------|
| **BSS** (Brier Skill Score) | Improvement over climatology (higher = better) |
| **RPS** (Ranked Probability Score) | Distance between forecast & observed CDF (lower = better) |
| **AUC** (Area Under ROC Curve) | Discrimination ability (0.5 = no skill, 1.0 = perfect) |
| **Reliability** | Calibration — does 70% probability ≈ 70% observed frequency? |

### Models in the Benchmark

| Model | Type | Origin | Resolution |
|-------|------|--------|-----------|
| **AIFS** | Deterministic | ECMWF | ~25 km (0.25°) |
| **FuXi** | Deterministic | Fudan University | ~25 km |
| **GraphCast** | Deterministic | Google DeepMind | ~25 km |
| **AIFS-ENS** | Probabilistic (50 members) | ECMWF | ~50 km |
| **GenCast** | Probabilistic (diffusion) | Google DeepMind | ~50 km |

---

##  Project Structure & Setup

### Directory Layout

```
ai-weather/
├── ai_weather/
│   ├── config.py                    # Central configuration
│   └── benchmarking/
│       ├── ROMP/
│       │   └── momp/
│       │       ├── driver.py        # Main execution entry point
│       │       ├── app/             # High-level workflow
│       │       ├── stats/           # Onset detection algorithms
│       │       ├── metrics/         # Error & skill calculations
│       │       ├── params/          # config.in + region_def.py
│       │       ├── lib/             # Core utilities
│       │       ├── io/              # NetCDF I/O
│       │       ├── graphics/        # Plotting
│       │       └── utils/           # Shared helpers
│       └── benchmarking.py          # Benchmark utilities
├── data/
│   ├── external/
│   │   ├── AIFS/                    # AIFS forecast NetCDFs
│   │   ├── fuxi/                    # FuXi forecast NetCDFs
│   │   ├── graphcast/               # GraphCast forecast NetCDFs
│   │   ├── AIFS_ENS/               # AIFS Ensemble NetCDFs
│   │   ├── gencast/                 # GenCast NetCDFs
│   │   ├── ENACTS/                  # Reference observations
│   │   ├── ENACTS_regridded_025/    # Regridded obs (0.25°)
│   │   ├── CHIRPS_IMERG/           # Satellite rainfall
│   │   ├── shapefile/               # Ethiopia boundaries
│   │   └── jjas_100mm_rainfall_mask_0p25.nc
│   └── ROMP_OUT/
│       └── et/
│           ├── output/              # NetCDF metric results
│           └── figure/              # PNG/PDF plots
├── notebooks/                       # Jupyter notebooks
├── ui/
│   ├── app.py                       # Streamlit main app
│   └── pages/
│       └── short_medium_benchmarking.py
└── pyproject.toml
```

### Environment Setup

```bash
# Option A: Conda (recommended for scientific Python)
conda create -n momp "python>=3.10" -y
conda activate momp

# Option B: venv
python -m venv .venv-momp
source .venv-momp/bin/activate  # Linux/Mac
# .venv-momp\Scripts\activate.bat  # Windows

# Install ROMP from source
cd ai_weather/benchmarking/ROMP
pip install -U pip
pip install -e .  # Editable install for development

# Verify
python -c "import momp; print(momp.__file__)"
```

### Python Dependencies

```
numpy, pandas, xarray, netCDF4, matplotlib, scipy,
geopandas, seaborn, regionmask, gcsfs, zarr, cartopy,
loguru, python-dotenv, ipywidgets, tqdm
```

---

## Configuration System

### Central Configuration (`config.py`)

The `config.py` file defines all paths, catalogs, and default parameters:

```python
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Project root detection
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Key directories
DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ROMP/MOMP paths
BMARK_ROOT = PROJ_ROOT / "ai_weather" / "benchmarking"
ROMP_ROOT = BMARK_ROOT / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"

# Output directories
ROMP_DEMO_ET_DIR = DATA_DIR / "ROMP_OUT" / "et"
ROMP_DEMO_FIG_DIR = ROMP_DEMO_ET_DIR / "figure"
ROMP_DEMO_OUT_DIR = ROMP_DEMO_ET_DIR / "output"
```

### 3.2 Model Catalog

The catalog maps model names to their data directories and metadata:

```python
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": EXTERNAL_DATA_DIR / "AIFS",
        "model_name": "AIFS",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": EXTERNAL_DATA_DIR / "AIFS_ENS",
        "model_name": "AIFS_ENS",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": EXTERNAL_DATA_DIR / "graphcast",
        "model_name": "GraphCast",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
     "GenCast": {
        "label": "Probabilistic · GenCast",
        "dir": EXTERNAL_DATA_DIR / "gencast",
        "model_name": "GenCast",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
}
```

### 3.3 Onset Criteria Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `wet_init` | 1 mm | Minimum initial rainfall trigger |
| `wet_threshold` | 20 mm | Daily rainfall threshold for "wet day" |
| `wet_spell` | 3 days | Consecutive wet days required |
| `dry_threshold` | 1 mm | Below this = "dry day" |
| `dry_spell` | 7 days | Max dry spell allowed after onset |
| `dry_extent` | 0 | Spatial dry extent check |
| `start_date` | (2019, 5, 1) | Search window start |
| `end_date` | (2022, 7, 31) | Search window end |
| `verification_window_list` | ((1, 15),) | Forecast lead days |
| `tolerance_days_list` | (3,) | ± days tolerance for onset match |

---

## 📚 Module 4: Running the Benchmark (Jupyter Notebook)

### Cell 1: Documentation (Markdown)

```markdown
# ROMP Benchmarking Notebook

This notebook executes the MOMP driver pipeline for deterministic and
probabilistic evaluation of AI weather models against ENACTS observations.

## Workflow
1. Load configuration → 2. Detect onset → 3. Compute metrics →
4. Save NetCDF → 5. Generate figures
```

### Cell 2: Environment & Path Setup

```python
import os
import sys
import runpy
from pathlib import Path

# --- Determine Project Root ---
if (Path.cwd() / "pyproject.toml").exists():
    PROJ_ROOT = Path.cwd()
elif (Path.cwd().parent / "pyproject.toml").exists():
    PROJ_ROOT = Path.cwd().parent
else:
    PROJ_ROOT = Path().resolve().parent

# --- Add config to path ---
CONFIG_DIR = PROJ_ROOT / "ai_weather"
if not CONFIG_DIR.exists():
    CONFIG_DIR = PROJ_ROOT / "al_weather"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

# --- Patch missing typing imports ---
import builtins
from typing import Dict, Any
builtins.Dict = Dict
builtins.Any = Any

# --- Import config ---
import config

ROMP_ROOT = config.ROMP_ROOT.resolve()
MOMP_PKG_DIR = config.MOMP_PKG_DIR.resolve()

for p in [str(ROMP_ROOT), str(PROJ_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import momp
import momp.params as params
from momp.lib.loader import get_cfg

print(f"✓ Project Root : {PROJ_ROOT}")
print(f"✓ ROMP Root    : {ROMP_ROOT}")
print(f"✓ MOMP Pkg Dir : {MOMP_PKG_DIR}")
```

### Cell 3: Helper Functions

```python
def resolve_path(p):
    """Resolve relative paths against PROJ_ROOT."""
    if p and not str(p).startswith("/"):
        return str(PROJ_ROOT / p)
    return str(p)


def configure_benchmark_mode(cfg, mode="det"):
    """Configure MOMP for deterministic or probabilistic evaluation."""
    mode = mode.lower().strip()
    is_prob = mode in ["prob", "probabilistic"]
    target_category = "Probabilistic" if is_prob else "Deterministic"

    selected_models = [
        meta for meta in config.BENCHMARK_MODEL_CATALOG.values()
        if meta.get("category") == target_category
    ]

    if not selected_models:
        raise ValueError(f"No models for category '{target_category}'.")

    model_names = tuple(meta["model_name"] for meta in selected_models)
    model_dirs = tuple(resolve_path(meta["dir"]) for meta in selected_models)
    num_models = len(selected_models)

    cfg.model_list = model_names
    cfg.model_dir_list = model_dirs
    cfg.model_var_list = ("tp",) * num_models
    cfg.unit_cvt_list = (None,) * num_models
    cfg.file_pattern_list = ("{}.nc",) * num_models

    if is_prob:
        cfg.probabilistic = True
        cfg.members = "All"
        cfg.onset_percentage_threshold = 0.5
        cfg.BS = True; cfg.RPS = True; cfg.AUC = True
        cfg.Reliability = True; cfg.skill_score = True
        cfg.FAR = False; cfg.MAE = False; cfg.MR = False
        cfg.plot_heatmap_bss_auc = True
        cfg.plot_reliability = True
        cfg.plot_panel_heatmap_skill = True
        cfg.plot_bar_bss_rpss_auc = True
        cfg.plot_spatial_far_mr_mae = False
        cfg.plot_panel_heatmap_error = False
    else:
        cfg.probabilistic = False
        cfg.FAR = True; cfg.MAE = True; cfg.MR = True
        cfg.BS = False; cfg.RPS = False; cfg.AUC = False
        cfg.Reliability = False; cfg.skill_score = False
        cfg.plot_spatial_far_mr_mae = True
        cfg.plot_panel_heatmap_error = True
        cfg.plot_heatmap_bss_auc = False
        cfg.plot_reliability = False
        cfg.plot_panel_heatmap_skill = False
        cfg.plot_bar_bss_rpss_auc = False

    print(f"Configured {target_category} mode: {list(model_names)}")
```

### Cell 4: Main Execution Function

```python
def execute_momp_benchmark(mode="det"):
    """Run the full MOMP benchmarking pipeline."""
    print("=" * 60)
    print(f"MOMP Benchmark [{mode.upper()} MODE]")
    print("=" * 60)

    # Auth
    os.environ.setdefault('EARTHDATA_AUTH', 'netrc')
    try:
        config.setup_earthdata_auth()
    except Exception as e:
        print(f"⚠ Auth warning: {e}")

    # Metadata patch for uninstalled package
    import importlib.metadata
    from importlib.metadata import PackageNotFoundError
    original_version = importlib.metadata.version
    original_distribution = importlib.metadata.distribution

    class DummyDist:
        version = "0.0.1"

    importlib.metadata.version = lambda p: "0.0.1" if p.lower() == "momp" else original_version(p)
    importlib.metadata.distribution = lambda p: DummyDist() if p.lower() == "momp" else original_distribution(p)

    # Load config
    cfg = get_cfg()
    cfg.pkg_dir = str(MOMP_PKG_DIR)
    cfg.work_dir = resolve_path(getattr(cfg, "work_dir", str(ROMP_ROOT)))
    cfg.obs_dir = resolve_path(getattr(cfg, "obs_dir", "data/external/ENACTS_regridded_025"))
    cfg.ref_model_dir = resolve_path(getattr(cfg, "ref_model_dir", "data/external/ENACTS"))
    cfg.nc_mask = resolve_path(getattr(cfg, "nc_mask", "data/external/jjas_100mm_rainfall_mask_0p25.nc"))

    if hasattr(cfg, "shpfile_dir") and cfg.shpfile_dir:
        cfg.shpfile_dir = resolve_path(cfg.shpfile_dir)

    cfg.dir_out = str(config.ROMP_DEMO_OUT_DIR)
    cfg.dir_fig = str(config.ROMP_DEMO_FIG_DIR)

    # Configure mode
    configure_benchmark_mode(cfg, mode=mode)
    cfg.model_dir_list = tuple(resolve_path(d) for d in cfg.model_dir_list)

    # CRITICAL: Disable parallel for Jupyter
    cfg.parallel = False

    # Sync to params
    for attr in dir(cfg):
        if not attr.startswith("_"):
            setattr(params, attr, getattr(cfg, attr))
    params.parallel = False

    os.makedirs(cfg.dir_out, exist_ok=True)
    os.makedirs(cfg.dir_fig, exist_ok=True)

    print(f"  work_dir : {cfg.work_dir}")
    print(f"  pkg_dir  : {cfg.pkg_dir}")
    print(f"  obs_dir  : {cfg.obs_dir}")
    print(f"  models   : {cfg.model_list}")
    print(f"  parallel : {cfg.parallel}")
    print("-" * 60)

    # Run driver
    driver_path = MOMP_PKG_DIR / "driver.py"
    assert driver_path.exists(), f"Driver not found: {driver_path}"

    sys.stdout.flush()
    try:
        runpy.run_path(str(driver_path), run_name="__main__")
        print(f"\n[{mode.upper()}] benchmark completed.")
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback; traceback.print_exc()
    finally:
        importlib.metadata.version = original_version
        importlib.metadata.distribution = original_distribution
```

### Cell 5: Execute

```python
# Run deterministic benchmark
execute_momp_benchmark("det")

# Run probabilistic benchmark (uncomment when ready)
# execute_momp_benchmark("prob")
```

---

## Understanding the Pipeline

### What `driver.py` Does (Step by Step)

```
┌─────────────────────────────────────────────────────────────┐
│                    MOMP Driver Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Load config.in parameters                                │
│ 2. Read observation NetCDF (ENACTS 0.25°)                   │
│ 3. Read forecast model NetCDFs (AIFS, FuXi, GraphCast)     │
│ 4. Apply spatial mask (Ethiopia highland JJAS mask)         │
│ 5. Detect onset in observations → onset_DOY_obs             │
│ 6. Detect onset in each model → onset_DOY_model            │
│ 7. Compute metrics:                                         │
│    • Deterministic: MAE, FAR, Miss Rate (per grid cell)     │
│    • Probabilistic: BSS, RPS, AUC, Reliability             │
│ 8. Save spatial metric NetCDFs to output/                   │
│ 9. Generate figures (heatmaps, maps, reliability plots)     │
│ 10. Save summary CSVs                                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Diagram

```
ENACTS (obs)          Model Forecasts
     │                     │
     ▼                     ▼
┌──────────┐        ┌──────────────┐
│ Onset    │        │ Onset        │
│ Detection│        │ Detection    │
│ (obs)    │        │ (per model)  │
└────┬─────┘        └──────┬───────┘
     │                     │
     ▼                     ▼
┌─────────────────────────────────┐
│     Comparison & Metrics        │
│  • Per-grid-cell MAE/FAR/MR    │
│  • Binned by lead time         │
│  • Spatial aggregation         │
└────────────┬────────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐   ┌──────────┐
│ NetCDF  │   │ Figures  │
│ Output  │   │ (PNG/PDF)│
└─────────┘   └──────────┘
```

---

## Interpreting Results

### Output Files

After running, check:

```python
from pathlib import Path

out_dir = Path(config.ROMP_DEMO_OUT_DIR)
fig_dir = Path(config.ROMP_DEMO_FIG_DIR)

print("=== NetCDF Outputs ===")
for f in sorted(out_dir.glob("*.nc")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")

print("\n=== Figures ===")
for f in sorted(fig_dir.glob("*.png")):
    print(f"  {f.name}")
```

### 6.2 Loading and Inspecting Results

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load a spatial metrics file
nc_files = sorted(out_dir.glob("spatial_metrics_*.nc"))
if nc_files:
    ds = xr.open_dataset(nc_files[0])
    print(ds)
    print(f"\nVariables: {list(ds.data_vars)}")
    print(f"Coords: {list(ds.coords)}")
```

### 6.3 Model Ranking Scoreboard

```python
import pandas as pd

def build_scoreboard():
    """Aggregate results from all spatial metric NetCDFs."""
    rows = []
    for nc_path in sorted(out_dir.glob("spatial_metrics_*.nc")):
        if "climatology" in nc_path.name.lower():
            continue
        try:
            ds = xr.open_dataset(nc_path)
            # Find MAE, FAR, MR variables
            mae_var = next((v for v in ds.data_vars if "mae" in v.lower()), None)
            far_var = next((v for v in ds.data_vars if "far" in v.lower() or "false_alarm" in v.lower()), None)
            mr_var = next((v for v in ds.data_vars if "miss" in v.lower()), None)

            if mae_var and far_var and mr_var:
                model = nc_path.stem.replace("spatial_metrics_", "")
                rows.append({
                    "Model": model,
                    "MAE (days)": float(ds[mae_var].mean(skipna=True)),
                    "FAR (%)": float(ds[far_var].mean(skipna=True)) * 100,
                    "Miss Rate (%)": float(ds[mr_var].mean(skipna=True)) * 100,
                })
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)
        df["Composite Score"] = (
            df["MAE (days)"] +
            0.08 * df["FAR (%)"] +
            0.06 * df["Miss Rate (%)"]
        )
        return df.sort_values("Composite Score").reset_index(drop=True)
    return pd.DataFrame()

scoreboard = build_scoreboard()
print(scoreboard.to_string(index=False))
```

### 6.4 Visualizing Spatial Skill

```python
def plot_spatial_metric(nc_path, var_keyword="mae", title=""):
    """Plot a 2D spatial metric map."""
    ds = xr.open_dataset(nc_path)
    var_name = next((v for v in ds.data_vars if var_keyword in v.lower()), None)
    if not var_name:
        print(f"No variable matching '{var_keyword}' found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ds[var_name].plot(ax=ax, cmap="RdYlGn_r", add_colorbar=True)
    ax.set_title(title or f"{var_name} — {Path(nc_path).stem}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(fig_dir / f"lesson_{var_keyword}_map.png", dpi=150)
    plt.show()

# Example usage
# plot_spatial_metric(nc_files[0], "mae", "AIFS Mean Absolute Error (days)")
```

### 6.5 Skill Location Identification

```python
def find_top_skill_locations(nc_path, top_n=10):
    """Find grid cells with best forecast skill."""
    ds = xr.open_dataset(nc_path)
    mae_var = next((v for v in ds.data_vars if "mae" in v.lower()), None)
    far_var = next((v for v in ds.data_vars if "far" in v.lower()), None)
    mr_var = next((v for v in ds.data_vars if "miss" in v.lower()), None)

    if not all([mae_var, far_var, mr_var]):
        return pd.DataFrame()

    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    df = pd.DataFrame({
        "lat": ds[mae_var][lat_name].values.repeat(ds[mae_var][lon_name].size),
        "lon": np.tile(ds[mae_var][lon_name].values, ds[mae_var][lat_name].size),
        "MAE": ds[mae_var].values.reshape(-1),
        "FAR": ds[far_var].values.reshape(-1),
        "MR": ds[mr_var].values.reshape(-1),
    }).dropna()

    # Filter: MAE ≤ 7 days, FAR ≤ 35%, MR ≤ 60%
    df = df[(df["MAE"] <= 7) & (df["FAR"] <= 0.35) & (df["MR"] <= 0.60)]
    df["Skill"] = df["MAE"] + 0.08 * df["FAR"] * 100 + 0.06 * df["MR"] * 100

    return df.sort_values("Skill").head(top_n).reset_index(drop=True)
```

---

## 📚 Module 7: Probabilistic Evaluation

### 7.1 Ensemble Onset Probability

For probabilistic models (AIFS-ENS, GenCast), each grid cell has multiple ensemble members. The onset probability is:

```
P(onset) = (number of members detecting onset) / (total members)
```

The `onset_percentage_threshold = 0.5` means onset is declared if ≥50% of members agree.

### 7.2 Running Probabilistic Mode

```python
execute_momp_benchmark("prob")
```

### 7.3 Interpreting Probabilistic Outputs

```python
# After probabilistic run, check for:
prob_files = sorted(out_dir.glob("*prob*.nc")) + sorted(out_dir.glob("*ens*.nc"))
print(f"Probabilistic outputs: {[f.name for f in prob_files]}")

# Reliability diagram interpretation:
# - Perfect reliability: points on the diagonal
# - Overconfident: curve below diagonal (forecast prob > observed freq)
# - Underconfident: curve above diagonal
```

### 7.4 Brier Skill Score Interpretation

| BSS Value | Interpretation |
|-----------|----------------|
| > 0 | Better than climatology |
| = 0 | No improvement over climatology |
| < 0 | Worse than climatology |
| = 1 | Perfect forecast |

---

## 📚 Module 8: Building the Streamlit Dashboard

### 8.1 Architecture

The Streamlit app (`app.py`) uses a modular page system:

```python
PAGE_MODULE_MAPPING = {
    "SM_Benchmarking": "ui.pages.short_medium_benchmarking",
    # ... other pages ...
}
```

### 8.2 Benchmarking Page Structure

The `short_medium_benchmarking.py` page has four sections:

1. **Model Selection** — Radio buttons for Deterministic/Probabilistic, cards for each model
2. **Run Section** — Button triggers `subprocess.run()` calling `run_momp.py --mode det|prob`
3. **Spatial Analysis** — Interactive NetCDF variable plotting, figure galleries
4. **Model Ranking** — Composite score table from all spatial metric outputs

### 8.3 Running the Dashboard

```bash
cd ai-weather
streamlit run ui/app.py
```

### 8.4 Key Dashboard Functions

```python
# From benchmarking.py — dynamically discovers available presets
def get_benchmark_presets() -> dict[str, dict]:
    """Scans ROMP_DEMO_OUT_DIR for real NetCDF outputs."""
    nc_files = sorted(ROMP_DEMO_OUT_DIR.glob("*.nc"))
    presets = {}
    for f in nc_files:
        key = f.stem
        is_ens = "ENS" in f.name or "probabilistic" in f.name.upper()
        presets[key] = {
            "label": f.stem.replace("_", " ").title(),
            "category": "Probabilistic AI" if is_ens else "Deterministic AI",
            "file_path": str(f.resolve()),
        }
    return presets
```

### 8.5 Subprocess Execution Pattern

```python
def run_momp_benchmark(mode: str) -> Dict[str, Any]:
    """Execute run_momp.py as a subprocess (avoids Jupyter deadlock)."""
    cmd = [sys.executable, str(run_momp_path), "--mode", mode]
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
        timeout=1800,  # 30 min
    )
    return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}
```

---

## 📚 Module 9: Advanced Customization

### 9.1 Custom Region Definitions

Edit `params/region_def.py` or provide a shapefile:

```python
# In config.in or cfg object:
cfg.region = "Ethiopia"
cfg.shpfile_dir = str(SHAPEFILE_DIR)
cfg.polygon = False  # Set True for custom polygon
cfg.nc_mask = str(DEFAULT_MASK_NC)  # Spatial mask NetCDF
```

### 9.2 Changing Onset Criteria

```python
# More lenient onset (earlier detection):
cfg.wet_threshold = 15  # Lower threshold
cfg.wet_spell = 2       # Fewer consecutive days needed
cfg.dry_spell = 10      # Allow longer dry spells

# Stricter onset (fewer false alarms):
cfg.wet_threshold = 25
cfg.wet_spell = 5
cfg.dry_spell = 5
```

### 9.3 Custom Verification Windows

```python
# Evaluate at multiple lead times:
cfg.verification_window_list = ((1, 5), (6, 10), (11, 15), (16, 30))
cfg.tolerance_days_list = (1, 3, 5, 7)
cfg.day_bins = ((1, 5), (6, 10), (11, 15), (16, 30))
```

### 9.4 Adding a New Model

Add to `BENCHMARK_MODEL_CATALOG` in `config.py`:

```python
"PanguWeather": {
    "label": "Deterministic · Pangu-Weather",
    "dir": EXTERNAL_DATA_DIR / "pangu",
    "model_name": "PanguWeather",
    "probabilistic": False,
    "category": "Deterministic",
    "slow": False,
},
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Button does nothing in Jupyter | `parallel=True` deadlock | Set `cfg.parallel = False` |
| `NameError: Dict` | Missing typing import in config.py | Add `from typing import Dict, Any` |
| `pkg_dir = .../ROMPmomp` | Missing slash in config.in | Override: `cfg.pkg_dir = str(MOMP_PKG_DIR)` |
| Auth hang | Interactive password prompt | Use `~/.netrc` file |
| `FileNotFoundError` for NetCDF | Wrong data paths | Check `data/external/` contents |
| Empty figures | No data in mask region | Verify mask aligns with model grid |
| Cartopy crashes | Missing system libraries | `conda install -c conda-forge cartopy` |

---


---

*© 2026 Ethiopian Meteorological Institute | Human-Centered Weather Forecasts Initiative | University of Chicago*