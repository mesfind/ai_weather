import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables if present
load_dotenv()

# ==============================================================================
# Dynamic Project Root & Directory Resolution
# ==============================================================================
CURRENT_FILE = Path(__file__).resolve()
MODULE_DIR = CURRENT_FILE.parent.parent

_HOME = Path.home().resolve()


def _find_root(start: Path, markers) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if p == _HOME or p == p.parent:
            break
        if any((p / m).exists() for m in markers):
            return p
    return None


_env_root = os.getenv("AI_WEATHER_REPO")
PROJ_ROOT = (
    Path(_env_root).expanduser().resolve() if _env_root
    else _find_root(MODULE_DIR, ("data",))
    or _find_root(MODULE_DIR, ("pyproject.toml", ".git"))
    or MODULE_DIR
)

_PROJ_ROOT = os.getenv("PROJ_ROOT", os.getcwd())

DATA_DIR = os.getenv("ROMP_DATA_DIR", os.path.join(MODULE_DIR, "data"))
DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENACTS_DIR = EXTERNAL_DATA_DIR / "ENACTS_regridded_025"
CHIRPS_DIR = EXTERNAL_DATA_DIR / "CHIRPS_IMERG"
JJAS_MASK_FILE = DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
JJAS_DIR = JJAS_MASK_FILE  # backward-compatible alias (it is a file, not a directory)
MONSOON_DIR = PROCESSED_DATA_DIR / "Monsoon" / "2026" 

# ROMP & MOMP Internal Package Directories
APPS_DIR = MODULE_DIR / "apps" if (MODULE_DIR / "apps").exists() else PROJ_ROOT / "ai_weather" / "apps"
ROMP_ROOT = APPS_DIR / "benchmarking" / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"



# Output & Graphics Directory Structures
ROMP_OUT_DIR = DATA_DIR / "ROMP_OUT" / "et" / "output"
ROMP_FIG_DIR = DATA_DIR / "ROMP_OUT" / "et" / "figure"

ROMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROMP_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Register core package paths in sys.path
for path_obj in [PROJ_ROOT, MODULE_DIR, ROMP_ROOT, MOMP_PKG_DIR]:
    path_str = str(path_obj.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
del path_obj, path_str


def resolve_path(p: Any, root: Optional[Path] = None) -> str:
    """Resolves relative paths against PROJ_ROOT if not absolute."""
    if not p:
        return ""
    if root is None:
        root = PROJ_ROOT
    path_obj = Path(p).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(root) / path_obj
    return str(path_obj.resolve())


def setup_earthdata_auth() -> None:
    """
    Placeholder for Earthdata authentication setup, referenced by the
    benchmark entry point. Configure a .netrc file or credentials here
    as needed for your environment.
    """
    pass


# ==============================================================================
# BENCHMARKING CATALOG — single source of truth for model → mode mapping
# ==============================================================================
# `category` and `probabilistic` MUST always agree for every entry.
# compute_mode_overrides() validates this at call time and raises loudly
# on any mismatch rather than silently picking the wrong models.
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": DATA_DIR / "external/AIFS",
        "model_name": "AIFS",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": DATA_DIR / "external/fuxi",
        "model_name": "FuXi",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": DATA_DIR / "external/graphcast",
        "model_name": "GraphCast",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": DATA_DIR / "external/AIFS_ENS/0p25",
        "model_name": "AIFS_ENS",
        "probabilistic": True,
        "category": "Probabilistic",
    },
    "gencast": {
        "label": "Probabilistic · GenCast",
        "dir": DATA_DIR / "external/gencast/0p25",
        "model_name": "gencast",
        "probabilistic": True,
        "category": "Probabilistic",
    },
}


def _catalog_models(category: str, catalog: Optional[Dict[str, Any]] = None):
    catalog = catalog or BENCHMARK_MODEL_CATALOG
    return [m for m in catalog.values() if m["category"] == category]


# ==============================================================================
# SANITY CHECKS ON THE CATALOG ITSELF (fail at import time, not mid-benchmark)
# ==============================================================================
for _key, _meta in BENCHMARK_MODEL_CATALOG.items():
    _expected_bool = (_meta.get("category") == "Probabilistic")
    if _meta.get("probabilistic") != _expected_bool:
        raise ValueError(
            f"BENCHMARK_MODEL_CATALOG['{_key}'] is inconsistent: "
            f"category='{_meta.get('category')}' but probabilistic={_meta.get('probabilistic')}. "
            f"These must always agree."
        )
del _key, _meta, _expected_bool

# ==============================================================================
# 1. CORE STRUCTURE & DIMENSIONS (mode-independent project metadata only)
# ==============================================================================
project_name = "probabilistic benchmarking with ENACTS data"
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
layout = ("model", "verification_window")

# ==============================================================================
# 2. OBSERVATION DATA SETTINGS
# ==============================================================================
# Pick the observation dataset with the ROMP_OBS env var ("ENACTS" or "CHIRPS").
_OBS_SETS = {
    "ENACTS": dict(dir=ENACTS_DIR, pattern=("{}_0p25.nc",), var="precip"),
    "CHIRPS": dict(dir=CHIRPS_DIR, pattern=("{}.nc",), var="RAINFALL"),
}
obs = os.getenv("ROMP_OBS", "ENACTS").upper()
if obs not in _OBS_SETS:
    raise ValueError(f"ROMP_OBS must be one of {list(_OBS_SETS)}, got '{obs}'")
_obs = _OBS_SETS[obs]

obs_dir = resolve_path(_obs["dir"])
obs_file_pattern = _obs["pattern"]
obs_var = _obs["var"]
obs_unit_cvt = None

# ==============================================================================
# 3. REFERENCE MODEL (BENCHMARK) SETTINGS
# ==============================================================================
ref_model = "climatology"
ref_model_dir = obs_dir
ref_model_file_pattern = _obs["pattern"][0]
ref_model_var = _obs["var"]
ref_model_unit_cvt = None

# ==============================================================================
# 4. REGION DEFINITIONS
# ==============================================================================
region = "Ethiopia"
nc_mask = resolve_path(JJAS_MASK_FILE)
shpfile_dir = EXTERNAL_DATA_DIR / "shapefile"
polygon = False

# ==============================================================================
# 5. ONSET CRITERIA (Physics Logic)
# ==============================================================================
wet_init = 1
wet_threshold = 20
wet_spell = 3
dry_threshold = 1
dry_spell = 7
dry_extent = 21  # must be >= dry_spell
thresh_file = None
thresh_var = None

# ==============================================================================
# 6. TEMPORAL SETTINGS (Dates & Years)
# ==============================================================================
start_date = (2007, 5, 1)
end_date = (2023, 7, 31)
start_year_clim = 2007
end_year_clim = 2024
init_days = (0, 3)
init_type = "weekly"
date_filter_year = 2024

# ==============================================================================
# 7. EVALUATION & VERIFICATION PARAMETERS
# ==============================================================================
verification_window_list = ((1, 15),)
tolerance_days_list = (3,)
max_forecast_day = 15
day_bins = ((1, 5), (6, 10), (11, 15))

# ==============================================================================
# 8. OUTPUT & GRAPHICS SETTINGS (mode-independent only)
# ==============================================================================
dir_out = os.getenv("ROMP_OUT_DIR", str(ROMP_OUT_DIR.resolve()))
dir_fig = os.getenv("ROMP_FIG_DIR", str(ROMP_FIG_DIR.resolve()))

save_fig = True
save_nc_spatial_far_mr_mae = False
save_csv_score = True
save_nc_climatology = True

plot_climatology_onset = True

show_plot = False
show_panel = False

# ==============================================================================
# 9. PARALLELIZATION & DEBUG
# ==============================================================================
parallel = True
debug = False

# ==============================================================================
# RESERVED / LEGACY PARAMETERS
# ==============================================================================
mok = (5, 15)
years = None
years_clim = None
fallback_date = None

# ===============================
# calibartion with idr 
#================================

apply_idr_calibration = False
# ==============================================================================
# SANITY CHECKS (warn early instead of failing later with an empty map)
# ==============================================================================
for _label, _p in (("data directory", DATA_DIR), ("observation directory", Path(obs_dir)),
                   ("region mask (nc_mask)", Path(nc_mask))):
    if not _p.exists():
        warnings.warn(f"config: {_label} not found: {_p} (PROJ_ROOT={PROJ_ROOT})", stacklevel=2)
del _label, _p

# ==============================================================================
# MODE-DEPENDENT OVERRIDES — the ONE function that defines everything that
# differs between deterministic and probabilistic evaluation.
#
# This replaces the old `configure_benchmark_mode(cfg, mode)`, which mutated
# a cfg object in place and required manually keeping two model lists
# (deter_model_list / prob_model_list) in sync with a separately hand-typed
# model_dir_list. That pattern is gone: model lists are derived from
# BENCHMARK_MODEL_CATALOG, and every metric/plot flag is set atomically
# alongside `probabilistic`, so nothing can drift out of sync.
#
# This function is PURE — it returns a plain dict and touches no global
# state. It is applied by momp.lib.loader.build_cfg() on every get_cfg()
# call, using the `mode` value from config.in / ROMP_MODE / apply_overrides().
# ==============================================================================
def compute_mode_overrides(mode: str = "det", catalog: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a dict of config overrides for the given mode. Does not mutate
    any object — callers are responsible for applying the result to
    wherever their configuration actually lives (see momp.lib.loader).
    """
    mode_str = str(mode).lower().strip()
    if mode_str not in ["det", "deterministic", "prob", "probabilistic"]:
        raise ValueError("Invalid mode. Choose 'det'/'deterministic' or 'prob'/'probabilistic'.")

    is_prob = mode_str in ["prob", "probabilistic"]
    target_category = "Probabilistic" if is_prob else "Deterministic"
    catalog = catalog if catalog is not None else BENCHMARK_MODEL_CATALOG

    selected_models = []
    for key, meta in catalog.items():
        meta_is_prob = meta.get("probabilistic")
        meta_category = meta.get("category")
        if meta_is_prob is None or meta_category is None:
            raise ValueError(f"Catalog entry '{key}' is missing 'probabilistic' or 'category'.")
        expected_bool = (meta_category == "Probabilistic")
        if meta_is_prob != expected_bool:
            raise ValueError(
                f"Catalog entry '{key}' inconsistent: category='{meta_category}' "
                f"but probabilistic={meta_is_prob}. Fix BENCHMARK_MODEL_CATALOG."
            )
        if meta_category == target_category:
            selected_models.append(meta)

    if not selected_models:
        raise ValueError(f"No models found in catalog for category '{target_category}'.")

    n = len(selected_models)
    overrides: Dict[str, Any] = {
        "mode": mode_str,
        "probabilistic": is_prob,
        "model_list": tuple(m["model_name"] for m in selected_models),
        "model_dir_list": tuple(resolve_path(m["dir"]) for m in selected_models),
        "model_var_list": ("tp",) * n,
        "unit_cvt_list": (None,) * n,
        "file_pattern_list": ("{}.nc",) * n,
        "skill_score": True,
    }

    if is_prob:
        overrides.update(
            members="All",
            onset_percentage_threshold=0.6,
            BS=True, RPS=True, AUC=True, Reliability=True,
            FAR=False, MAE=False, MR=False,
            plot_heatmap_bss_auc=True,
            plot_reliability=True,
            plot_panel_heatmap_skill=True,
            plot_bar_bss_rpss_auc=True,
            plot_spatial_far_mr_mae=False,
            plot_panel_heatmap_error=False,
        )
    else:
        overrides.update(
            members=None,
            onset_percentage_threshold=None,
            FAR=True, MAE=True, MR=True,
            BS=False, RPS=False, AUC=False, Reliability=False,
            plot_spatial_far_mr_mae=True,
            plot_panel_heatmap_error=True,
            plot_heatmap_bss_auc=False,
            plot_reliability=False,
            plot_panel_heatmap_skill=False,
            plot_bar_bss_rpss_auc=False,
        )

    print(f"[config] mode overrides computed for {'PROBABILISTIC' if is_prob else 'DETERMINISTIC'}: "
          f"models={overrides['model_list']}")
    return overrides