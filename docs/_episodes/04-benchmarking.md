---
title: Benchmarking 
teaching: 30
exercises: 15
questions:
- "How do we systematically compare AI weather models against climatological baselines?"
- "What are the best practices for safely loading and standardizing NetCDF datasets?"
- "How can we dynamically discover and categorize model outputs for evaluation?"
objectives:
- "Understand the core configuration parameters for onset and spell benchmarking."
- "Implement robust `xarray` loading functions to handle inconsistent NetCDF dimensions."
- "Build dynamic preset discovery to categorize deterministic vs. probabilistic AI models."
- "Generate automated summary scoreboards from geospatial output files."
keypoints:
- "Standardizing time dimensions ('day' vs 'time') is critical for robust xarray operations."
- "Dynamic file discovery using `pathlib` prevents hardcoded pipeline failures."
- "Benchmarking requires clear separation between reference data (climatology) and target models (e.g., AIFS)."
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




# AI Forecast Benchmarking Pipeline

When deploying Artificial Intelligence forecasting systems (like ECMWF's AIFS) in operational weather workflows, we must rigorously benchmark them against trusted baselines, such as historical climatology. 

This lesson walks through the Python architecture required to load, standardize, categorize, and evaluate geospatial forecast data using `xarray`, `pandas`, and `pathlib`.

---

## 1. Defining the Benchmark Configuration

Before processing data, we must define the rules of our evaluation. This includes the time periods, spatial resolution, and meteorological thresholds (e.g., what constitutes a "wet" or "dry" spell).

```python
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Centralized configuration imports (example)
# from config import APP_DIR, ROMP_DEMO_OUT_DIR, etc.

BENCHMARK_METHOD = {
    "climatology_years": tuple(range(2015, 2023)),
    "test_years": tuple(range(2019, 2023)),
    "resolution_label": "0.25° (~25 km)",
    "mask_path": Path("data/external/jjas_seasonal_mask_0p25.nc").resolve(),
    "mask_label": "JJAS seasonal 0.25° mask used as available highland-season proxy",
    "wet_init": 1,
    "wet_threshold": 20,      # mm/day
    "wet_spell": 3,           # consecutive days
    "dry_threshold": 1,       # mm/day
    "dry_spell": 7,           # consecutive days
    "start_month_day": (5, 1),
    "end_month_day": (9, 30),
}
```

> **Tip: Meteorological Thresholds**
> 
> The `wet_threshold` and `dry_spell` parameters are highly region-dependent. Always validate these thresholds against local meteorological agency standards before running large-scale benchmarks.
{: .tip}

---

## 2. Safely Loading Geospatial Data

NetCDF files from different modeling systems often have inconsistent dimension naming (e.g., some use `time`, others use `day`). A robust pipeline must standardize this before any analysis occurs.

```python
def _safe_open_dataset(nc_path: str | Path) -> xr.Dataset:
    """Safely loads a NetCDF file and handles day/time dimension renaming."""
    path = Path(nc_path)
    ds = xr.open_dataset(path)
    
    # Standardize time dimension
    if "day" in ds.dims:
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.swap_dims({"day": "time"})
        elif "time" not in ds.coords and "time" not in ds.dims:
            ds = ds.rename({"day": "time"})
            
    return ds
```

> **Warning: Memory Management**
> 
> While `xr.open_dataset` is lazy, chaining operations on large, unchunked NetCDF files can cause memory spikes. For production pipelines, consider adding `chunks={"time": "auto"}` to the `open_dataset` call.
{: .warning}

---

## 3. Dynamic Preset Discovery

Hardcoding file paths leads to brittle pipelines. Instead, we can scan an output directory and dynamically build a registry of available benchmark datasets, automatically categorizing them as *Deterministic* or *Probabilistic*.

```python
def get_benchmark_presets(output_dir: Path) -> dict[str, dict]:
    """Builds available benchmark presets dynamically from real NetCDF files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(list(output_dir.glob("*.nc")))
    
    presets = {}
    for f in nc_files:
        fname = f.name
        key = fname.replace(".nc", "")
        
        # Heuristic to determine if the model is probabilistic/ensemble
        is_ens = "ENS" in fname or "probabilistic" in fname.upper()
        
        presets[key] = {
            "label": fname.replace("_", " ").replace(".nc", "").title(),
            "category": "Probabilistic AI" if is_ens else "Deterministic AI",
            "description": f"Real NetCDF output dataset: {fname}",
            "model_list": ["AIFS_ENS" if is_ens else "AIFS"],
            "probabilistic": is_ens,
            "file_path": str(f.resolve()),
        }
        
    return presets
```

---

## 4. Evaluating and Summarizing Results

Once the benchmark runs, we need to extract meaningful summaries. We can combine `xarray` for spatial aggregation and `pandas` for tabular reporting to create a "scoreboard" of model performance.

```python
def runtime_benchmark_scoreboard(output_dir: Path) -> pd.DataFrame:
    """Scans all netCDF files in output directory for scoreboard ranking."""
    nc_files = sorted(list(output_dir.glob("*.nc")))
    rows = []
    
    for f in nc_files:
        try:
            with _safe_open_dataset(f) as ds:
                vars_list = list(ds.data_vars)
                # Calculate mean for the first 3 numeric variables as a quick health check
                means = {
                    v: float(ds[v].mean().values) 
                    for v in vars_list[:3] 
                    if np.issubdtype(ds[v].dtype, np.number)
                }
                
                rows.append({
                    "Dataset": f.name,
                    "Variables": ", ".join(vars_list),
                    "Primary Mean Value": list(means.values())[0] if means else 0.0
                })
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")
            
    return pd.DataFrame(rows)
```

---

## 5. Identifying Skill Locations

To understand *where* an AI model adds value, we can extract the top-performing grid cells. This function converts an `xarray` Dataset into a `pandas` DataFrame, filters out missing values, and returns the top 15 locations.

```python
def identify_skill_locations(nc_path: str | Path) -> pd.DataFrame:
    """Extracts top grid cells for detailed spatial analysis."""
    with _safe_open_dataset(nc_path) as ds:
        if "lat" in ds.dims and "lon" in ds.dims:
            # Convert to dataframe, drop NaNs (e.g., ocean masks), and get top 15
            df = ds.to_dataframe().reset_index().dropna().head(15)
            return df
            
    return pd.DataFrame()
```

---

## Exercises

> **Exercise 1: Modify Benchmark Thresholds**
> 
> 1. Copy the `BENCHMARK_METHOD` dictionary into your local environment.
> 2. Modify the `wet_threshold` to `15` and the `dry_spell` to `10`.
> 3. Write a short print statement that outputs the `mask_label` to verify the configuration loaded correctly.
{: .exercise}

> **Exercise 2: Extend the Scoreboard**
> 
> The `runtime_benchmark_scoreboard` function currently calculates the `mean()`. 
> 1. Modify the dictionary comprehension inside the function to also calculate the `std()` (standard deviation) for each variable.
> 2. Add a new column to the `rows.append()` dictionary called `"Std Dev"` that captures this value.
{: .exercise}

> **Exercise 3: Handle Missing Files Gracefully**
> 
> The `get_benchmark_presets` function returns an empty dictionary if the folder is empty. 
> 1. Add an `if not presets:` block to the end of the function.
> 2. Inside this block, create a "fallback" preset dictionary with a `label` of "No Data Found" and `probabilistic` set to `False`, ensuring the pipeline doesn't crash downstream.
{: .exercise}

---

## Summary

Building a benchmarking pipeline requires more than just running a model; it requires robust data engineering. By using `pathlib` for dynamic file discovery, `xarray` for safe dimension standardization, and `pandas` for clear reporting, we create a reproducible system that can reliably evaluate AI weather models against operational baselines.
```
