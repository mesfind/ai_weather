from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ==============================================================================
# Spatial Data & Shapefile Directories
# ==============================================================================
STATION_DATA_PATH = DATA_DIR / "stations.csv"
STATION_CSV_PATH = EXTERNAL_DATA_DIR / "RF_Station_Grid_Format.csv"
DEFAULT_MASK_NC = EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
SHAPEFILE_DIR = EXTERNAL_DATA_DIR / "shapefile"
SHAPEFILE_PATH = SHAPEFILE_DIR / "eth_admbnda_admin3_csa_bofedb_2021.geojson"
DEFAULT_MASK_NC = DATA_DIR / "chirps_jjas_seasonal_mask_ethiopia_0p25.nc"
# ETH_GEOJSON_PATH = SHAPEFILE_DIR / "ETH_Zones.geojson"
ETH_GEOJSON_PATH = DATA_DIR / "ethiopia.geojson"
ZONE_WOREDA_SHP = SHAPEFILE_DIR / "ETH_Zones_Woreda.geojson"
WOREDA_SHAPEFILE_PATH = SHAPEFILE_DIR
SHAPEFILE = SHAPEFILE_PATH
STUDYAREA_SHAPEFILE = ZONE_WOREDA_SHP

# ==============================================================================
# Onset Benchmarking directory structure
# ==============================================================================
APP_DIR = PROJ_ROOT
BMARK_ROOT = APP_DIR / "ai_weather" / "benchmarking"
ROMP_ROOT = BMARK_ROOT / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"  
ROMP_DEMO_ET_DIR = DATA_DIR / "ROMP_OUT" / "et"
ROMP_DEMO_FIG_DIR = ROMP_DEMO_ET_DIR / "figure"
ROMP_DEMO_OUT_DIR = ROMP_DEMO_ET_DIR / "output"
ROMP_RUNTIME_ROOT = APP_DIR / ".romp_runtime"



# ==============================================================================
# Benchmarking Catalogs (Mapped to data/external)
# ==============================================================================
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": EXTERNAL_DATA_DIR / "AIFS",
        "model_name": "AIFS",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": EXTERNAL_DATA_DIR / "fuxi",
        "model_name": "FuXi",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": EXTERNAL_DATA_DIR / "graphcast",
        "model_name": "GraphCast",
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
    "GenCast": {
        "label": "Probabilistic · GenCast",
        "dir": EXTERNAL_DATA_DIR / "gencast",
        "model_name": "GenCast",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
}
BENCHMARK_OBS_CATALOG: Dict[str, Dict[str, Any]] = {
    "ENACTS": {
        "label": "ENACTS gauge-gridded rainfall",
        "dir": EXTERNAL_DATA_DIR / "ENACTS",
        "file_pattern": "{}_0p25.nc",
        "var": "precip",
        "category": "dashboard",
    },
    "CHIRPS": {
        "label": "CHIRPS rainfall",
        "dir": EXTERNAL_DATA_DIR / "CHIRPS_IMERG",
        "file_pattern": "{}.nc",
        "var": "precip",
        "category": "dashboard",
    },
    "IMERG": {
        "label": "IMERG rainfall",
        "dir": EXTERNAL_DATA_DIR / "CHIRPS_IMERG",
        "file_pattern": "{}.nc",
        "var": "precip",
        "category": "dashboard",
    },
}

# ==============================================================================
# ROMP / MOMP WORKFLOW CONFIGURATION
# ==============================================================================
project_name = "deterministic benchmarking with ENACTS data"
# Direct absolute paths for ROMP execution
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
# Internal data layout & model list
layout = ("model", "verification_window")
model_list = ("AIFS", "fuxi", "graphcast")
# Observation settings
obs = "ENACTS"
obs_dir = str((EXTERNAL_DATA_DIR / "ENACTS").resolve())
obs_file_pattern = ("{}_0p25.nc",)
obs_var = "precip"
obs_unit_cvt = None
# Reference Model (Benchmark) Settings
ref_model = "climatology"
ref_model_dir = str((EXTERNAL_DATA_DIR / "ENACTS").resolve())
ref_model_file_pattern = "{}_0p25.nc"
ref_model_var = "precip"
ref_model_unit_cvt = None
# Forecast Model Settings
model_dir_list = (
    str((EXTERNAL_DATA_DIR / "AIFS").resolve()),
    str((EXTERNAL_DATA_DIR / "fuxi").resolve()),
    str((EXTERNAL_DATA_DIR / "graphcast").resolve()),
)
model_var_list = ("tp", "tp", "tp")
unit_cvt_list = (None, None, None)
file_pattern_list = ("{}.nc", "{}.nc", "{}.nc")
# Spatial & Mask Definitions
region = "Ethiopia"
nc_mask = str((EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc").resolve())
shpfile_dir = str(SHAPEFILE_DIR.resolve())
polygon = False
# Physics & Onset Criteria Logic
wet_init = 1
wet_threshold = 20
wet_spell = 3
dry_threshold = 1
dry_spell = 7
dry_extent = 0
thresh_file = None
thresh_var = None
onset_percentage_threshold = 0.5
# Temporal Settings
start_date = (2019, 5, 1)
end_date = (2022, 7, 31)
start_year_clim = 2015
end_year_clim = 2022
init_days = (0, 3)
date_filter_year = 2024
# Evaluation & Verification Parameters
verification_window_list = ((1, 15),)
tolerance_days_list = (3,)
max_forecast_day = 15
day_bins = ((1, 5), (6, 10), (11, 15))
# Metrics Selection
FAR = True
MAE = True
MR = True
probabilistic = False
members = "All"
BS = True
RPS = True
AUC = True
Reliability = True
skill_score = True
# Output & Plotting Settings
dir_out = str(ROMP_DEMO_OUT_DIR.resolve())
dir_fig = str(ROMP_DEMO_FIG_DIR.resolve())
save_fig = True
save_nc_spatial_far_mr_mae = True
save_csv_score = True
save_nc_climatology = True
plot_spatial_far_mr_mae = True
plot_heatmap_bss_auc = True
plot_reliability = True
plot_climatology_onset = True
plot_panel_heatmap_error = True
plot_panel_heatmap_skill = True
plot_bar_bss_rpss_auc = True
show_plot = False
show_panel = False
parallel = True
debug = False
# Reserved / Legacy compatibility
mok = (5, 15)
years = None
years_clim = None
fallback_date = None

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
