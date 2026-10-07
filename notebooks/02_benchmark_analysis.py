# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Notebook 2: Benchmark Analysis & Visualization
#
# This notebook reproduces every visualization from the Streamlit dashboard:
#
# 1. Metric maps (Values, Skill, Difference, Swipe comparison)
# 2. The **All Metrics** comparison table (HTML with color-coded cells)
# 3. Detailed metric distribution statistics (Mean, Min, Median, P75, P90, Max)
# 4. Benchmark output figures (grouped by category)
# 5. Complete PDF report generation
#
# All visualization functions are imported from
# `ui/pages/short_medium_benchmarking.py` (the shared `utils` module).
#
# **Prerequisites:**
# - Run Notebook 1 first to generate the benchmark outputs.
# - `pip install fpdf2` for PDF report generation.

# %% [markdown]
# ## 1. Environment Setup

# %%
import os
import sys
import io
import warnings
from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from IPython.display import display, HTML, Image as IPImage, FileLink

warnings.filterwarnings("ignore")
get_ipython().run_line_magic("matplotlib", "inline")

# ------------------------------------------------------------------
# 1a. Bootstrap project root (same logic as Notebook 1)
# ------------------------------------------------------------------
import momp.lib.loader as loader
cfg = loader.get_cfg()
base_dir = Path(cfg.base_dir)

def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "apps").exists():
            return p
        if p == p.parent:
            break
    return start

PROJECT_ROOT = _find_project_root(base_dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# 1b. Import shared utilities
# ------------------------------------------------------------------
_ui_pages = PROJECT_ROOT / "ui" / "pages"
if str(_ui_pages) not in sys.path:
    sys.path.insert(0, str(_ui_pages))

import short_medium_benchmarking as utils

# Constants
MAP_METRICS              = utils.MAP_METRICS
MODEL_DISPLAY_LABELS     = utils.MODEL_DISPLAY_LABELS
REFERENCE_CATEGORY_ORDER = utils.REFERENCE_CATEGORY_ORDER
WINDOW_TOLERANCE_DAYS    = utils.WINDOW_TOLERANCE_DAYS
ROMP_DEMO_OUT_DIR        = utils.ROMP_DEMO_OUT_DIR
ROMP_DEMO_FIG_DIR        = utils.ROMP_DEMO_FIG_DIR

# Helpers — model & output discovery
discover_map_outputs     = utils.discover_map_outputs

# Helpers — map loading & plotting
load_map_da_cached       = utils.load_map_da_cached
plot_map                 = utils.plot_map
plot_map_image           = utils.plot_map_image
fig_to_bytes             = utils.fig_to_bytes
subtract_maps            = utils.subtract_maps
create_swipe_comparison_image = utils.create_swipe_comparison_image

# Helpers — metrics & tables
load_metric_values_by_window_cached = utils.load_metric_values_by_window_cached
build_metrics_table_html            = utils.build_metrics_table_html
compute_metric_distribution_stats   = utils.compute_metric_distribution_stats
_format_distribution_value          = utils._format_distribution_value

# Helpers — figures & PDF
classify_figure_type      = utils.classify_figure_type
format_figure_label       = utils.format_figure_label
classify_reference_asset  = utils.classify_reference_asset
collect_reference_figures = utils.collect_reference_figures
generate_benchmark_pdf_report = utils.generate_benchmark_pdf_report

print("Utilities imported from:", utils.__file__)

# %% [markdown]
# ## 2. Load Discovered Outputs
#
# If you ran Notebook 1 in the same session, the outputs are cached.
# Otherwise, re-discover them from disk.

# %%
import pickle

_cache_path = PROJECT_ROOT / "notebooks" / ".benchmark_cache.pkl"
if _cache_path.exists():
    with open(_cache_path, "rb") as f:
        _cache = pickle.load(f)
    map_outputs   = _cache["map_outputs"]
    model_registry = _cache["model_registry"]
    print(f"Loaded {len(map_outputs)} outputs from cache.")
else:
    map_outputs    = discover_map_outputs()
    model_registry = utils.detect_benchmark_models()
    print(f"Discovered {len(map_outputs)} outputs from disk.")

if not map_outputs:
    print("No outputs found. Please run Notebook 1 first.")

# %% [markdown]
# ## 3. Select Analysis Parameters
#
# Edit these three variables to reproduce any dashboard view.
# Available metrics: `False Alarm Rate`, `Miss Rate`, `Mean Absolute Error`
# Available windows: `Days 1–15`, `Days 16–30`

# %%
METRIC      = "Mean Absolute Error"
LEAD_WINDOW = "Days 1–15"
MODEL_LABEL = "AIFS Single v2"

selected_output = next(
    (
        o for o in map_outputs
        if o["model_label"] == MODEL_LABEL
        and LEAD_WINDOW in o["windows"]
        and METRIC in o["metrics"]
    ),
    None,
)

if selected_output is None:
    print("Selected combination not found. Available outputs:")
    display(pd.DataFrame(map_outputs))
else:
    print("Selected output file:", selected_output["path"])

# %% [markdown]
# ## 4. Metric Maps
#
# ### 4.1 Values Map
#
# The raw metric value at each grid cell. For FAR and MR, values are
# already in percent (0–100). For MAE, values are in days.

# %%
base_da = None

if selected_output is not None:
    base_da = load_map_da_cached(selected_output["path"], METRIC, LEAD_WINDOW)
    if base_da is not None:
        metric_meta = MAP_METRICS[METRIC]
        fig = plot_map(
            base_da,
            f"{METRIC} — {MODEL_LABEL} — {LEAD_WINDOW}",
            cmap=metric_meta["cmap"],
            unit=metric_meta["unit"],
        )
        if fig:
            display(fig)
            plt.close(fig)

# %% [markdown]
# ### 4.2 Skill Map
#
# Each grid cell is classified as **skilful** (1) if the metric value is
# below the threshold, or **unskilful** (0) otherwise. Thresholds:
#
# - FAR: 35%
# - MR: 60%
# - MAE: 7 days

# %%
if base_da is not None:
    metric_meta = MAP_METRICS[METRIC]
    threshold = metric_meta["threshold"]
    skill_da = (base_da <= threshold).astype(float).where(base_da.notnull())

    fig = plot_map(skill_da, f"Skill — {METRIC} (threshold={threshold})",
                   cmap="RdYlGn", vmin=0, vmax=1)
    if fig:
        display(fig)
        plt.close(fig)

# %% [markdown]
# ### 4.3 Difference Map
#
# Shows where the selected model is better (blue) or worse (red) than a
# comparison model. The difference is computed as:
#
#     difference = selected_model - comparison_model
#
# For lower-is-better metrics, **negative values (blue) mean the selected
# model is better**.

# %%
COMPARE_MODEL  = "FuXi"
COMPARE_WINDOW = LEAD_WINDOW

compare_output = next(
    (
        o for o in map_outputs
        if o["model_label"] == COMPARE_MODEL
        and COMPARE_WINDOW in o["windows"]
        and METRIC in o["metrics"]
    ),
    None,
)

if compare_output is not None and base_da is not None:
    compare_da = load_map_da_cached(compare_output["path"], METRIC, COMPARE_WINDOW)
    if compare_da is not None:
        diff_da = subtract_maps(base_da, compare_da)
        if diff_da is not None:
            fig = plot_map(
                diff_da,
                f"Difference — {MODEL_LABEL} minus {COMPARE_MODEL}",
                diverging=True,
            )
            if fig:
                display(fig)
                plt.close(fig)

# %% [markdown]
# ### 4.4 Swipe Comparison
#
# A static composite image showing the selected model on the left and the
# comparison model on the right, split at 50%. In the Streamlit dashboard
# this is interactive (drag the slider).

# %%
if compare_output is not None and base_da is not None:
    compare_da = load_map_da_cached(compare_output["path"], METRIC, COMPARE_WINDOW)
    if compare_da is not None:
        metric_meta = MAP_METRICS[METRIC]

        # Compute shared color scale
        all_vals = np.concatenate([
            base_da.values[np.isfinite(base_da.values)],
            compare_da.values[np.isfinite(compare_da.values)],
        ])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        left_fig  = plot_map_image(base_da,    cmap=metric_meta["cmap"], vmin=vmin, vmax=vmax)
        right_fig = plot_map_image(compare_da, cmap=metric_meta["cmap"], vmin=vmin, vmax=vmax)

        if left_fig and right_fig:
            swipe_image = create_swipe_comparison_image(
                fig_to_bytes(left_fig, False),
                fig_to_bytes(right_fig, False),
                split_pct=50,
            )
            if swipe_image:
                print(f"Left : {MODEL_LABEL} — {LEAD_WINDOW}")
                print(f"Right: {COMPARE_MODEL} — {COMPARE_WINDOW}")
                display(swipe_image)

# %% [markdown]
# ## 5. All Metrics Comparison Table
#
# This reproduces the **All Metrics** dashboard tab. The table compares
# every discovered model across both lead-time windows for all three
# core metrics.
#
# Color coding:
# - **Gray** — climatology baseline
# - **Green** — better than climatology
# - **Red** — worse than climatology

# %%
tbl  = {}
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
    pref_m = [
        "GraphCast", "FuXi", "AIFS Single v2",
        "AIFS Ensemble v2", "GenCast", "Traditional Climatology",
    ]
    om = [m for m in pref_m if m in mods] + sorted(mods - set(pref_m))
    ow = [w for w in ["Days 1–15", "Days 16–30"] if w in wins] + \
         sorted(wins - {"Days 1–15", "Days 16–30"})

    display(HTML(build_metrics_table_html(tbl, om, ow)))
else:
    print("No All Metrics table could be built from the available outputs.")

# %% [markdown]
# ## 6. Detailed Metric Distributions
#
# For each (model, lead-window, metric) combination, we compute
# distribution statistics directly from the gridded output:
#
#     Mean, Min, Median, P75, P90, Max
#
# This reveals whether a model performs well on average but has extreme
# errors in specific regions.

# %%
distribution_stats = compute_metric_distribution_stats()

if distribution_stats.empty:
    print("No distribution statistics could be computed.")
else:
    # Plain table
    display(distribution_stats)

    # Heatmap-styled table
    numeric_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]
    styled = distribution_stats.style.background_gradient(
        subset=numeric_cols, cmap="RdYlGn_r"
    )
    display(styled)

# %% [markdown]
# ### 6.1 Per-Model Breakdown
#
# Group the distributions by model and lead window, matching the layout
# of the dashboard's **Detailed Metric Distributions** section.

# %%
metric_order  = {"False Alarm Rate": 0, "Miss Rate": 1, "Mean Absolute Error": 2}
window_order  = {"Days 1–15": 0, "Days 16–30": 1}

stats_sorted = distribution_stats.copy()
stats_sorted["metric_sort"] = stats_sorted["Metric"].map(lambda m: metric_order.get(m, 999))
stats_sorted["window_sort"] = stats_sorted["Lead Window"].map(lambda w: window_order.get(w, 999))
stats_sorted = stats_sorted.sort_values(["Model", "window_sort", "metric_sort"])

for (model_label, window_label), group_df in stats_sorted.groupby(
    ["Model", "Lead Window"], sort=False
):
    display_model = "Climatology" if model_label == "Traditional Climatology" else model_label
    print(f"\n{display_model.upper().replace(' ', '_')} — {window_label.upper()}")
    tolerance = WINDOW_TOLERANCE_DAYS.get(window_label)
    if tolerance is not None:
        print(f"  +/- {tolerance} day tolerance")

    display_rows = []
    for _, row in group_df.iterrows():
        is_frac = bool(row["fraction"])
        display_rows.append({
            "METRIC": row["Metric"],
            "MEAN":   _format_distribution_value(row["Mean"],   is_frac),
            "MIN":    _format_distribution_value(row["Min"],    is_frac),
            "MEDIAN": _format_distribution_value(row["Median"], is_frac),
            "P75":    _format_distribution_value(row["P75"],    is_frac),
            "P90":    _format_distribution_value(row["P90"],    is_frac),
            "MAX":    _format_distribution_value(row["Max"],    is_frac),
            "UNIT":   "fraction" if is_frac else "days",
        })
    display(pd.DataFrame(display_rows))

# %% [markdown]
# ## 7. Benchmark Output Figures
#
# Figures generated by the MOMP pipeline are grouped into four categories:
#
# - **Deterministic AI** — AIFS, FuXi, GraphCast outputs
# - **Probabilistic AI** — AIFS Ensemble, GenCast outputs
# - **Climatology / Baseline** — traditional climatology references
# - **Other** — anything that doesn't match the above
#
# Within each category, figures are further grouped by type:
#
# - Spatial Metrics (MAE / FAR / Miss Rate)
# - Portrait Panel
# - Reliability Diagram
# - Skill Scores Heatmap
# - Climatology Onset

# %%
reference_figures = collect_reference_figures()

if not any(reference_figures.values()):
    print("No benchmark output figures were found.")
    print("Check:", ROMP_DEMO_FIG_DIR)
else:
    for category in REFERENCE_CATEGORY_ORDER:
        paths = reference_figures.get(category, [])
        if not paths:
            continue

        print(f"\n{'=' * 60}")
        print(f"{category} ({len(paths)} figures)")
        print('=' * 60)

        type_groups = {}
        for path_str in paths:
            ftype = classify_figure_type(Path(path_str).name)
            type_groups.setdefault(ftype, []).append(path_str)

        for ftype, fpaths in type_groups.items():
            print(f"\n  {ftype} ({len(fpaths)})")
            print("  " + "-" * 40)
            for path_str in fpaths[:6]:
                print(f"  {format_figure_label(path_str)}")
                display(IPImage(filename=path_str, width=500))

# %% [markdown]
# ## 8. Generate Complete PDF Report
#
# The PDF report bundles everything above into a single document:
#
# - Executive summary
# - Models evaluated
# - Benchmark inputs & shared settings
# - Fair comparison plan
# - Verification scores
# - Detailed metric distributions
# - All benchmark output figures
#
# The report is saved to `data/ROMP_OUT/et/reports/`.

# %%
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False
    print("fpdf2 is not installed. Install with: pip install fpdf2")

if PDF_AVAILABLE:
    from datetime import datetime

    # Build the report context
    report_context = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ground_truth": "ENACTS",
        "benchmark_coverage": "Ethiopia (Default)",
        "forecast_window": "1-15 days",
        "shared_settings": {
            "wet_day_threshold": 20.0,
            "minimum_wet_day_rainfall": 1.0,
            "wet_spell_length": 3,
            "dry_spell_limit": 7,
            "dry_spell_search_extension": 0,
            "baseline_forecast": "climatology",
        },
        "active_rows": model_registry,
        "fair_plan": None,
        "map_outputs": map_outputs,
        "scores_df": pd.DataFrame(),
        "distribution_stats": distribution_stats,
        "uncertainty_df": pd.DataFrame(),
        "significance_df": pd.DataFrame(),
        "figures": reference_figures,
    }

    print("Generating PDF report...")
    pdf_bytes = generate_benchmark_pdf_report(report_context)

    report_dir = PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = report_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    print(f"Report saved to: {pdf_path}")
    display(FileLink(pdf_path))

# %% [markdown]
# ## 9. Summary
#
# You have now reproduced every visualization from the Streamlit dashboard:
#
# | Section | Dashboard Tab | Output |
# |---|---|---|
# | Metric maps | Metric Map | Values / Skill / Difference / Swipe PNGs |
# | Comparison table | All Metrics | HTML table with color coding |
# | Distribution stats | All Metrics (lower) | Per-model tables |
# | Output figures | Benchmark Outputs | Grouped PNG gallery |
# | PDF report | Benchmark Report & Export | Downloadable PDF |
#
# **Next steps for training participants:**
#
# 1. Change `METRIC`, `LEAD_WINDOW`, or `MODEL_LABEL` in Section 3 and
#    re-run Sections 4–5 to explore different views.
# 2. Compare deterministic vs probabilistic models by switching
#    `MODEL_LABEL` between `"AIFS Single v2"` and `"AIFS Ensemble v2"`.
# 3. Inspect the generated PDF report for a printable summary.