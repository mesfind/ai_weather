# %% [markdown]
# # Forecast Benchmarking — Step-by-Step Reproduction Notebook
#
# This notebook reproduces all outputs from the Streamlit benchmarking dashboard:
#
# 1. Detect available benchmark models and their data ranges
# 2. Discover generated benchmark NetCDF outputs
# 3. Plot metric maps (Values, Skill, Difference, Swipe comparison)
# 4. Build the All Metrics comparison table
# 5. Compute detailed metric distribution statistics
# 6. Display benchmark output figures
# 7. Interactive Onset Time-Series Analysis (optional)
# 8. Generate complete PDF report
#
# **Prerequisites:**
# - MOMP package installed: `pip install -e apps/benchmarking/ROMP`
# - fpdf2 for PDF generation: `pip install fpdf2`
# - Data files in `data/external/` directory

# %% [markdown]
# ## 1. Environment Setup

# %%
import os
import sys
import warnings
from pathlib import Path
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from functools import lru_cache
from contextlib import contextmanager
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw
from IPython.display import display, HTML, Image as IPImage, FileLink

warnings.filterwarnings("ignore")
%matplotlib inline

# %% [markdown]
# ## 2. Project Configuration
#
# Import paths and catalogs from the project's `config.py`.

# %%
try:
    from config import (
        PROJECT_ROOT,
        ROMP_DEMO_FIG_DIR,
        ROMP_DEMO_OUT_DIR,
        BENCHMARK_OBS_CATALOG,
        EXTERNAL_DATA_DIR,
        ONSET_REFERENCE_DIR,
        BENCHMARK_MODEL_CATALOG as CONFIG_BENCHMARK_MODEL_CATALOG,
    )
    import config
except Exception as exc:
    raise RuntimeError(
        "Could not import project configuration. "
        "Make sure you are running from the project root and config.py exists."
    ) from exc

print("PROJECT_ROOT:", PROJECT_ROOT)
print("ROMP_DEMO_FIG_DIR:", ROMP_DEMO_FIG_DIR)
print("ROMP_DEMO_OUT_DIR:", ROMP_DEMO_OUT_DIR)

# %% [markdown]
# ## 3. Benchmark Constants
#
# Define human-readable labels, metric keywords, and thresholds.

# %%
MODEL_DISPLAY_LABELS = {
    "AIFS": "AIFS Single v2",
    "AIFS_ENS": "AIFS Ensemble v2",
    "FuXi": "FuXi",
    "GraphCast": "GraphCast",
    "GenCast": "GenCast",
}

LIVE_FORECAST_RECENT_YEARS = 2

REFERENCE_CATEGORY_ORDER = [
    "Deterministic AI",
    "Probabilistic AI",
    "Climatology / Baseline",
    "Other",
]

MAP_METRICS = {
    "False Alarm Rate": {
        "keywords": ["false_alarm_rate", "false_alarm", "far"],
        "fraction": True,
        "unit": "%",
        "threshold": 35.0,
        "cmap": "Reds",
        "description": "Fraction of forecast onset grid cells that were not observed onset cells. Lower is better.",
    },
    "Miss Rate": {
        "keywords": ["miss_rate", "miss", "mr"],
        "fraction": True,
        "unit": "%",
        "threshold": 60.0,
        "cmap": "Reds",
        "description": "Fraction of observed onset grid cells that the forecast missed. Lower is better.",
    },
    "Mean Absolute Error": {
        "keywords": ["mean_absolute_error", "mean_mae", "mae"],
        "fraction": False,
        "unit": "days",
        "threshold": 7.0,
        "cmap": "YlOrRd",
        "description": "Average absolute error in onset timing, measured in days. Lower is better.",
    },
}

FIGURE_TYPE_TOKENS = {
    "Spatial Metrics (MAE / FAR / Miss Rate)": ["spatial_metrics"],
    "Portrait Panel": ["panel_portrait", "portrait"],
    "Reliability Diagram": ["reliability"],
    "Skill Scores Heatmap": ["skill_scores", "heatmap"],
    "Climatology Onset": ["climatology_onset"],
}

FIGURE_HUMAN_LABELS = {
    "panel_portrait_mae_far_mr_AIFS_30day.png": "Panel Portrait: MAE / FAR / MR - AIFS (30-day)",
    "panel_portrait_mae_far_mr_prob_30day.png": "Panel Portrait: MAE / FAR / MR - Probabilistic (30-day)",
    "spatial_metrics_climatology_1-15.png": "Spatial Metrics: Climatology (Days 1-15)",
    "spatial_metrics_climatology_16-30.png": "Spatial Metrics: Climatology (Days 16-30)",
    "spatial_metrics_prob_1-15.png": "Spatial Metrics: Probabilistic (Days 1-15)",
    "spatial_metrics_prob_16-30.png": "Spatial Metrics: Probabilistic (Days 16-30)",
}

WINDOW_TOLERANCE_DAYS = {"Days 1–15": 3, "Days 16–30": 5}

# %% [markdown]
# ## 4. Generic Helper Functions

# %%
def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""

def _meta_get(meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(meta, dict):
        return default
    if key in meta:
        return meta[key]
    target = key.strip().lower()
    for k, v in meta.items():
        if str(k).strip().lower() == target:
            return v
    return default

def _find_var(ds: xr.Dataset, keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        for v in ds.data_vars:
            if kw in str(v).lower():
                return str(v)
    return None

def _safe_open_dataset(nc_path: str | Path) -> xr.Dataset:
    ds = xr.open_dataset(Path(nc_path))
    if "day" in ds.dims:
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.swap_dims({"day": "time"})
        elif "time" not in ds.coords and "time" not in ds.dims:
            ds = ds.rename({"day": "time"})
    return ds

def classify_figure_type(filename: str) -> str:
    fname = str(filename).lower()
    for label, tokens in FIGURE_TYPE_TOKENS.items():
        for token in tokens:
            if token in fname:
                return label
    return "Other"

def format_figure_label(path_str: str) -> str:
    name = Path(path_str).name
    if name in FIGURE_HUMAN_LABELS:
        return FIGURE_HUMAN_LABELS[name]
    stem = Path(path_str).stem.lower()
    
    if stem.startswith("panel_portrait"):
        label = "Panel Portrait"
    elif stem.startswith("spatial_metrics"):
        label = "Spatial Metrics"
    elif "reliability" in stem:
        label = "Reliability Diagram"
    elif "skill_scores_heatmap" in stem or "heatmap" in stem:
        label = "Skill Scores Heatmap"
    elif "climatology_onset" in stem:
        label = "Climatology Onset"
    else:
        label = Path(path_str).stem.replace("_", " ").title()
    
    metric = None
    if "mae_far_mr" in stem:
        metric = "MAE / FAR / MR"
    elif "mae" in stem:
        metric = "MAE"
    elif "far" in stem:
        metric = "FAR"
    elif "mr" in stem:
        metric = "Miss Rate"
    
    subject = None
    if "climatology" in stem:
        subject = "Climatology"
    elif "aifs_ens" in stem:
        subject = "AIFS Ensemble"
    elif "prob" in stem:
        subject = "Probabilistic"
    elif "aifs" in stem:
        subject = "AIFS"
    elif "fuxi" in stem:
        subject = "FuXi"
    elif "graphcast" in stem:
        subject = "GraphCast"
    elif "gencast" in stem:
        subject = "GenCast"
    
    window = None
    if "1-15" in stem:
        window = "Days 1-15"
    elif "16-30" in stem:
        window = "Days 16-30"
    elif "30day" in stem:
        window = "30-day"
    
    parts = [label]
    if metric:
        parts.append(metric)
    details = []
    if subject:
        details.append(subject)
    if window:
        details.append(window)
    if details:
        parts.append("- " + " - ".join(details))
    return " ".join(parts)

# %% [markdown]
# ## 5. Detect Available Benchmark Models
#
# Scan model directories to detect valid date ranges from NetCDF files.

# %%
def _extract_time_range_from_nc(nc_path: Path):
    try:
        ds = xr.open_dataset(nc_path, decode_cf=True, decode_times=True)
    except Exception:
        try:
            ds = xr.open_dataset(nc_path, decode_cf=False, decode_times=False)
        except Exception:
            return None, None
    try:
        time_name = None
        for name in ds.coords:
            lname = str(name).lower()
            if lname in {"time", "t", "date", "datetime", "valid_time"} or "time" in lname:
                time_name = str(name)
                break
        if time_name is None:
            for name in ds.coords:
                try:
                    if np.issubdtype(ds[name].dtype, np.datetime64):
                        time_name = str(name)
                        break
                except Exception:
                    pass
        if time_name is None:
            return None, None
        da = ds[time_name]
        if np.issubdtype(da.dtype, np.number):
            return None, None
        values = np.asarray(da.values).ravel()
        if values.size == 0:
            return None, None
        dates = []
        for v in values[:5000]:
            try:
                ts = pd.Timestamp(v)
                if not pd.isna(ts):
                    dates.append(ts.date())
            except Exception:
                try:
                    dates.append(date(int(v.year), int(v.month), int(v.day)))
                except Exception:
                    pass
        if not dates:
            return None, None
        return min(dates), max(dates)
    except Exception:
        return None, None
    finally:
        try:
            ds.close()
        except Exception:
            pass

def _scan_filename_date_range(nc_files: List[Path]):
    pairs, years = [], set()
    for path in nc_files:
        for match in re.finditer(r"((?:19|20)\d{2})(?:[-_]?(\d{2}))?", path.stem):
            try:
                year = int(match.group(1))
            except Exception:
                continue
            years.add(year)
            month_raw = match.group(2)
            if month_raw:
                try:
                    month = int(month_raw)
                    if 1 <= month <= 12:
                        pairs.append((year, month))
                except Exception:
                    continue
    if pairs:
        min_y, min_m = min(pairs)
        max_y, max_m = max(pairs)
        return date(min_y, min_m, 1), date(max_y, max_m, monthrange(max_y, max_m)[1])
    if years:
        return date(min(years), 1, 1), date(max(years), 12, 31)
    return None, None

def _detect_model_data_date_range(model_dir: Path):
    if not model_dir or str(model_dir).strip() in {"", "."}:
        return None, None
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None, None
    try:
        nc_files = list(model_dir.rglob("*.nc"))
    except Exception:
        return None, None
    if not nc_files:
        return None, None
    def file_year(p: Path) -> int:
        m = re.search(r"(?:19|20)\d{2}", p.stem)
        return int(m.group(0)) if m else 0
    files_sorted = sorted(nc_files, key=lambda p: (file_year(p), p.name))
    sample = files_sorted if len(files_sorted) <= 10 else files_sorted[:5] + files_sorted[-5:]
    starts, ends = [], []
    for nc_path in sample:
        s, e = _extract_time_range_from_nc(nc_path)
        if s and e:
            starts.append(s)
            ends.append(e)
    if starts and ends:
        return min(starts), max(ends)
    return _scan_filename_date_range(files_sorted)

def detect_benchmark_models() -> List[Dict[str, Any]]:
    detected = []
    current_year = datetime.now().year
    catalog = CONFIG_BENCHMARK_MODEL_CATALOG or {}
    for raw_key, meta in catalog.items():
        key = _clean(raw_key)
        if not key:
            continue
        model_name = _clean(_meta_get(meta, "model_name", key))
        label = MODEL_DISPLAY_LABELS.get(key, model_name or key)
        dir_raw = _clean(_meta_get(meta, "dir", ""))
        if not dir_raw:
            continue
        start_date, end_date = _detect_model_data_date_range(Path(dir_raw))
        if start_date is None or end_date is None:
            continue
        probabilistic = bool(_meta_get(meta, "probabilistic", False))
        category = _clean(_meta_get(meta, "category", "Deterministic")) or "Deterministic"
        is_live = end_date.year >= current_year - LIVE_FORECAST_RECENT_YEARS
        type_label = "LIVE FORECAST" if is_live else "HISTORICAL"
        if probabilistic:
            type_label += " - Ensemble"
        if (start_date.month == 1 and start_date.day == 1 and end_date.month == 12 and end_date.day == 31):
            date_range_label = f"{start_date.year} to {end_date.year}"
        else:
            date_range_label = f"{start_date:%Y-%m} to {end_date:%Y-%m}"
        detected.append({
            "Model": label,
            "Category": "Probabilistic" if probabilistic else "Deterministic",
            "Type": type_label,
            "Coverage": "AIWP",
            "Date Range": date_range_label,
            "_row_key": key,
            "_exec_key": key,
            "_model_name": model_name,
            "_start_date": start_date,
            "_end_date": end_date,
            "_probabilistic": probabilistic,
        })
    preferred = ["AIFS", "AIFS_ENS", "FuXi", "GraphCast", "GenCast"]
    detected.sort(key=lambda r: preferred.index(r["_row_key"]) if r["_row_key"] in preferred else 999)
    return detected

model_registry = detect_benchmark_models()
if not model_registry:
    print("No benchmark models were detected. Check config.BENCHMARK_MODEL_CATALOG.")
else:
    display(pd.DataFrame(model_registry)[["Model", "Category", "Type", "Coverage", "Date Range"]])

# %% [markdown]
# ## 6. Discover Generated Benchmark Outputs

# %%
def _map_find_window_dim(da: xr.DataArray) -> Optional[str]:
    for dim in da.dims:
        if "window" in str(dim).lower():
            return str(dim)
    return None

def _map_infer_window_label(nc_path: Path) -> str:
    s = str(nc_path).lower()
    if any(t in s for t in ["1-15", "1_15", "days_1_15", "d1_15", "lead_1_15"]):
        return "Days 1–15"
    if any(t in s for t in ["16-30", "16_30", "days_16_30", "d16_30", "lead_16_30"]):
        return "Days 16–30"
    return "Days 1–15"

def _map_infer_model_label(nc_path: Path) -> Optional[str]:
    s = str(nc_path).lower()
    if "climatology" in s or "traditional" in s:
        return "Traditional Climatology"
    parent = nc_path.parent
    if parent and parent != ROMP_DEMO_OUT_DIR and parent.name != ROMP_DEMO_OUT_DIR.name:
        return MODEL_DISPLAY_LABELS.get(parent.name, parent.name.replace("_", " ").title())
    if "aifs_ens" in s:
        return "AIFS Ensemble v2"
    if "gencast" in s:
        return "GenCast"
    if "graphcast" in s:
        return "GraphCast"
    if "fuxi" in s:
        return "FuXi"
    if "aifs" in s:
        return "AIFS Single v2"
    return None

@lru_cache(maxsize=1)
def discover_map_outputs() -> List[Dict[str, Any]]:
    outputs = []
    if not ROMP_DEMO_OUT_DIR.exists():
        return outputs
    for nc_path in sorted(ROMP_DEMO_OUT_DIR.rglob("*.nc")):
        metrics_found = []
        first_da = None
        try:
            with xr.open_dataset(nc_path) as ds:
                for metric_key, metric_meta in MAP_METRICS.items():
                    var = _find_var(ds, metric_meta["keywords"])
                    if var:
                        metrics_found.append(metric_key)
                        if first_da is None:
                            first_da = ds[var]
                if not metrics_found:
                    continue
                wdim = _map_find_window_dim(first_da)
                if wdim:
                    nw = int(first_da.sizes[wdim])
                    if nw == 1:
                        windows = [_map_infer_window_label(nc_path)]
                    elif nw == 2:
                        windows = ["Days 1–15", "Days 16–30"]
                    else:
                        windows = [f"Window {i + 1}" for i in range(nw)]
                else:
                    windows = [_map_infer_window_label(nc_path)]
        except Exception:
            continue
        model_label = _map_infer_model_label(nc_path)
        if model_label is None:
            continue
        outputs.append({
            "path": str(nc_path.resolve()),
            "model_label": model_label,
            "windows": windows,
            "metrics": metrics_found,
        })
    return outputs

map_outputs = discover_map_outputs()
if not map_outputs:
    print("No benchmark map outputs were found.")
else:
    outputs_df = pd.DataFrame(map_outputs)
    outputs_df["windows"] = outputs_df["windows"].map(lambda x: ", ".join(x))
    outputs_df["metrics"] = outputs_df["metrics"].map(lambda x: ", ".join(x))
    display(outputs_df)

# %% [markdown]
# ## 7. Select Analysis Parameters
#
# Edit these variables to reproduce specific dashboard views.

# %%
METRIC = "Mean Absolute Error"
LEAD_WINDOW = "Days 1–15"
MODEL_LABEL = "AIFS Single v2"

selected_output = next((o for o in map_outputs if o["model_label"] == MODEL_LABEL and LEAD_WINDOW in o["windows"] and METRIC in o["metrics"]), None)

if selected_output is None:
    print("Selected model/metric/window was not found.")
    display(pd.DataFrame(map_outputs))
else:
    print("Selected output file:", selected_output["path"])

# %% [markdown]
# ## 8. Metric Map Plotting Functions

# %%
def _map_select_window(da: xr.DataArray, wl: str) -> xr.DataArray:
    wdim = _map_find_window_dim(da)
    if wdim:
        return da.isel({wdim: 1 if wl == "Days 16–30" and da.sizes[wdim] > 1 else 0})
    return da

def _map_lon_lat_names(da: xr.DataArray):
    lon = next((c for c in ["lon", "longitude"] if c in da.coords), None)
    lat = next((c for c in ["lat", "latitude"] if c in da.coords), None)
    return lon, lat

@lru_cache(maxsize=128)
def load_map_da_cached(nc_path_str: str, metric_key: str, lead_window: str) -> Optional[xr.DataArray]:
    try:
        ds = xr.open_dataset(Path(nc_path_str))
    except Exception:
        return None
    metric_meta = MAP_METRICS.get(metric_key)
    if not metric_meta:
        ds.close()
        return None
    var = _find_var(ds, metric_meta["keywords"])
    if not var:
        ds.close()
        return None
    da = _map_select_window(ds[var], lead_window)
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    if metric_meta.get("fraction", False):
        arr = np.asarray(da.values, dtype=float)
        if np.isfinite(arr).any() and float(np.nanmax(arr)) <= 1.5:
            da = da * 100.0
    da = da.load()
    ds.close()
    return da

def plot_map(da, title, cmap="viridis", unit="", vmin=None, vmax=None, diverging=False):
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if diverging:
        lim = float(np.nanmax(np.abs(arr))) or 1.0
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    lon, lat = _map_lon_lat_names(da)
    try:
        if lon and lat:
            da.plot.imshow(ax=ax, x=lon, y=lat, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs={"label": unit or title})
        else:
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, label=unit or title)
    except Exception:
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=unit or title)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

def plot_map_image(da, cmap="viridis", vmin=None, vmax=None):
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    fig = plt.figure(figsize=(6, 4), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig

def fig_to_bytes(fig, tight: bool = True) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight" if tight else None, pad_inches=0 if not tight else 0.1)
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def subtract_maps(da_a, da_b):
    try:
        da_a, da_b = xr.align(da_a, da_b, join="inner")
        return da_a - da_b
    except Exception:
        return None

def create_swipe_comparison_image(left_bytes: bytes, right_bytes: bytes, split_pct: float):
    try:
        img_l = Image.open(io.BytesIO(left_bytes)).convert("RGB")
        img_r = Image.open(io.BytesIO(right_bytes)).convert("RGB")
    except Exception:
        return None
    if img_l.size != img_r.size:
        img_r = img_r.resize(img_l.size, Image.Resampling.LANCZOS)
    w, h = img_l.size
    sx = max(0, min(w, int(w * split_pct / 100.0)))
    comp = Image.new("RGB", (w, h), color="white")
    if sx > 0:
        comp.paste(img_l.crop((0, 0, sx, h)), (0, 0))
    if sx < w:
        comp.paste(img_r.crop((sx, 0, w, h)), (sx, 0))
    draw = ImageDraw.Draw(comp)
    if 0 < sx < w:
        draw.line([(sx, 0), (sx, h)], fill=(37, 99, 235), width=4)
    return comp

# %% [markdown]
# ## 9. Render Metric Maps

# %%
if selected_output is not None:
    base_da = load_map_da_cached(selected_output["path"], METRIC, LEAD_WINDOW)
    if base_da is not None:
        metric_meta = MAP_METRICS[METRIC]
        
        # Values Map
        print("--- Values Map ---")
        fig = plot_map(base_da, f"{METRIC} - {MODEL_LABEL} - {LEAD_WINDOW}", cmap=metric_meta["cmap"], unit=metric_meta["unit"])
        if fig:
            display(fig)
            plt.close(fig)
        
        # Skill Map
        print("\n--- Skill Map ---")
        skill_da = (base_da <= metric_meta["threshold"]).astype(float).where(base_da.notnull())
        fig = plot_map(skill_da, f"Skill - {METRIC}", cmap="RdYlGn", vmin=0, vmax=1)
        if fig:
            display(fig)
            plt.close(fig)

# %%
# Difference and Swipe Comparison
COMPARE_MODEL = "FuXi"
COMPARE_WINDOW = LEAD_WINDOW

compare_output = next((o for o in map_outputs if o["model_label"] == COMPARE_MODEL and COMPARE_WINDOW in o["windows"] and METRIC in o["metrics"]), None)

if compare_output is not None and base_da is not None:
    compare_da = load_map_da_cached(compare_output["path"], METRIC, COMPARE_WINDOW)
    if compare_da is not None:
        # Difference Map
        print("--- Difference Map ---")
        diff_da = subtract_maps(base_da, compare_da)
        if diff_da is not None:
            fig = plot_map(diff_da, f"Difference - {MODEL_LABEL} minus {COMPARE_MODEL}", diverging=True)
            if fig:
                display(fig)
                plt.close(fig)
            
        # Swipe Comparison
        print("\n--- Swipe Comparison ---")
        vmin = float(np.nanmin(np.concatenate([base_da.values[np.isfinite(base_da.values)], compare_da.values[np.isfinite(compare_da.values)]])))
        vmax = float(np.nanmax(np.concatenate([base_da.values[np.isfinite(base_da.values)], compare_da.values[np.isfinite(compare_da.values)]])))
        left_fig = plot_map_image(base_da, cmap=metric_meta["cmap"], vmin=vmin, vmax=vmax)
        right_fig = plot_map_image(compare_da, cmap=metric_meta["cmap"], vmin=vmin, vmax=vmax)
        if left_fig and right_fig:
            swipe_image = create_swipe_comparison_image(fig_to_bytes(left_fig, False), fig_to_bytes(right_fig, False), 50)
            if swipe_image:
                print(f"Left: {MODEL_LABEL} - {LEAD_WINDOW}")
                print(f"Right: {COMPARE_MODEL} - {COMPARE_WINDOW}")
                display(swipe_image)

# %% [markdown]
# ## 10. All Metrics Table

# %%
def _extract_spatial_mean(da, convert_fraction=False):
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    val = float(np.nanmean(arr))
    return val / 100.0 if convert_fraction and val > 1.5 else val

@lru_cache(maxsize=128)
def load_metric_values_by_window_cached(nc_path_str: str):
    nc_path = Path(nc_path_str)
    dw = _map_infer_window_label(nc_path)
    res = {}
    try:
        with xr.open_dataset(nc_path) as ds:
            for mn, kws, is_frac in [
                ("Mean Absolute Error days", ["mean_absolute_error", "mean_mae", "mae"], False),
                ("Miss Rate fraction", ["miss_rate", "miss", "mr"], True),
                ("False Alarm Rate fraction", ["false_alarm_rate", "false_alarm", "far"], True),
            ]:
                var = _find_var(ds, kws)
                if not var:
                    continue
                wdim = _map_find_window_dim(ds[var])
                if wdim:
                    nw = int(ds[var].sizes[wdim])
                    if nw == 1:
                        wls = [dw]
                    elif nw == 2:
                        wls = ["Days 1–15", "Days 16–30"]
                    else:
                        wls = [f"Window {i + 1}" for i in range(nw)]
                    for i, wl in enumerate(wls[:nw]):
                        val = _extract_spatial_mean(ds[var].isel({wdim: i}), is_frac)
                        if val is not None:
                            res.setdefault(wl, {})[mn] = val
                else:
                    val = _extract_spatial_mean(ds[var], is_frac)
                    if val is not None:
                        res.setdefault(dw, {})[mn] = val
    except Exception:
        pass
    return res

def build_metrics_table_html(table, models, windows):
    metric_order = ["Mean Absolute Error days", "Miss Rate fraction", "False Alarm Rate fraction"]
    present = {k[0] for k in table.keys()}
    ordered = [m for m in metric_order if m in present] + sorted(present - set(metric_order))
    clim = "Traditional Climatology"
    css = "<style>.all-metrics-table{width:100%;border-collapse:collapse;font-size:0.9rem;min-width:700px}.all-metrics-table th,.all-metrics-table td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right}.all-metrics-table th{background:#f8fafc;color:#0f1b2a;font-weight:600}.all-metrics-table td:first-child,.all-metrics-table th:first-child{text-align:left}</style>"
    parts = [css, '<div style="overflow-x:auto;"><table class="all-metrics-table"><thead><tr><th rowspan="2">METRIC</th>']
    for w in windows:
        parts.append(f'<th colspan="{len(models)}">{w}</th>')
    parts.append("</tr><tr>")
    for _ in windows:
        for m in models:
            parts.append(f"<th>{m}</th>")
    parts.append("</tr></thead><tbody>")
    for metric in ordered:
        parts.append(f"<tr><td>{metric}</td>")
        for w in windows:
            cv = table.get((metric, w, clim))
            diffs = [abs(table.get((metric, w, m)) - cv) for m in models if m != clim and table.get((metric, w, m)) is not None and cv is not None]
            scale = max(diffs) if diffs else None
            for m in models:
                val = table.get((metric, w, m))
                if val is None:
                    parts.append("<td>-</td>")
                    continue
                style = ""
                if m == clim:
                    style = "background-color: rgba(148,163,184,0.12); font-weight:600;"
                elif cv is not None and scale:
                    diff = val - cv
                    if abs(diff) > 1e-6:
                        better = diff < 0
                        intensity = min(abs(diff) / scale, 1.0)
                        alpha = 0.08 + (0.35 * intensity)
                        color = "34,197,94" if better else "239,68,68"
                        style = f"background-color: rgba({color},{alpha:.2f});"
                fmt = f"{val:.1f}" if metric.startswith("Mean Absolute Error") else f"{val:.3f}"
                parts.append(f'<td style="{style}">{fmt}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)

tbl = {}
mods = set()
wins = set()
for o in map_outputs:
    vals = load_metric_values_by_window_cached(o["path"])
    for w, metrics in vals.items():
        for mn, val in metrics.items():
            tbl[(mn, w, o["model_label"])] = val
            mods.add(o["model_label"])
            wins.add(w)

if tbl:
    pref_m = ["GraphCast", "FuXi", "AIFS Single v2", "AIFS Ensemble v2", "GenCast", "Traditional Climatology"]
    om = [m for m in pref_m if m in mods] + sorted(mods - set(pref_m))
    ow = [w for w in ["Days 1–15", "Days 16–30"] if w in wins] + sorted(wins - {"Days 1–15", "Days 16–30"})
    display(HTML(build_metrics_table_html(tbl, om, ow)))
else:
    print("No All Metrics table could be built from the available outputs.")

# %% [markdown]
# ## 11. Detailed Metric Distributions

# %%
def _format_distribution_value(value: float, is_fraction: bool) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.1f}%" if is_fraction else f"{value:.1f} d"

def compute_metric_distribution_stats() -> pd.DataFrame:
    outs = discover_map_outputs()
    unique_outputs = {}
    for item in outs:
        model_label = item.get("model_label")
        if not model_label:
            continue
        path = item.get("path")
        if not path:
            continue
        for wl in item.get("windows", []):
            unique_outputs.setdefault((model_label, wl), path)
    rows = []
    for (model_label, wl), path in unique_outputs.items():
        for metric_key, metric_meta in MAP_METRICS.items():
            da = load_map_da_cached(path, metric_key, wl)
            if da is None:
                continue
            arr = np.asarray(da.values, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            rows.append({
                "Model": model_label,
                "Lead Window": wl,
                "Metric": metric_key,
                "Mean": float(np.mean(arr)),
                "Min": float(np.min(arr)),
                "Median": float(np.median(arr)),
                "P75": float(np.percentile(arr, 75)),
                "P90": float(np.percentile(arr, 90)),
                "Max": float(np.max(arr)),
                "fraction": bool(metric_meta.get("fraction", False)),
            })
    return pd.DataFrame(rows)

distribution_stats = compute_metric_distribution_stats()
if distribution_stats.empty:
    print("No distribution statistics could be computed.")
else:
    display(distribution_stats)
    numeric_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]
    styled = distribution_stats.style.background_gradient(subset=numeric_cols, cmap="RdYlGn_r")
    display(styled)

# %% [markdown]
# ## 12. Benchmark Output Figures

# %%
def classify_reference_asset(path: Path) -> str:
    s = str(path).lower()
    prob_keys = ["aifs_ens", "ens", "ensemble", "probabilistic", "gencast", "brier", "rps", "reliability", "spread", "rank"]
    clim_keys = ["climatology", "baseline", "traditional"]
    det_keys = ["aifs", "fuxi", "graphcast", "deterministic", "mae", "far", "mr"]
    if any(k in s for k in prob_keys):
        return "Probabilistic AI"
    if any(k in s for k in clim_keys):
        return "Climatology / Baseline"
    if any(k in s for k in det_keys):
        return "Deterministic AI"
    return "Other"

def collect_reference_figures() -> Dict[str, List[str]]:
    grouped = {cat: [] for cat in REFERENCE_CATEGORY_ORDER}
    if not ROMP_DEMO_FIG_DIR.exists():
        return grouped
    for path in sorted(ROMP_DEMO_FIG_DIR.rglob("*.png")):
        cat = classify_reference_asset(path)
        grouped.setdefault(cat, []).append(str(path.resolve()))
    return grouped

reference_figures = collect_reference_figures()
if not any(reference_figures.values()):
    print("No benchmark output figures were found.")
else:
    for category in REFERENCE_CATEGORY_ORDER:
        paths = reference_figures.get(category, [])
        if not paths:
            continue
        print(f"\n{category} ({len(paths)} figures)")
        print("=" * 60)
        type_groups = {}
        for path_str in paths:
            ftype = classify_figure_type(Path(path_str).name)
            type_groups.setdefault(ftype, []).append(path_str)
        for ftype, fpaths in type_groups.items():
            print(f"\n{ftype} ({len(fpaths)})")
            print("-" * 40)
            for path_str in fpaths[:4]:
                print(format_figure_label(path_str))
                display(IPImage(filename=path_str, width=500))

# %% [markdown]
# ## 13. Generate PDF Report
#
# Requires `fpdf2`: `pip install fpdf2`

# %%
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False
    FPDF = object
    print("fpdf2 is not installed. PDF generation is disabled.")

if PDF_AVAILABLE:
    print("PDF Generator initialized. Run the full generate_benchmark_pdf_report() function to export the report.")
    print("The report will be saved to: data/ROMP_OUT/et/reports/")