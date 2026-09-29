import importlib.resources as resources
import os
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

from momp.utils.practical import set_dir
from momp.lib.control import init_dataclass
from momp.lib.convention import Setting
from momp.lib.assertion import ROMPValidator, ROMPConfigError
from momp.lib.parser import ensure_config_exists
from .parser import create_parser

# Detect if running inside Jupyter / IPykernel environment
IS_JUPYTER = "ipykernel" in sys.modules or "IPython" in sys.modules

package = "momp"
with resources.as_file(resources.files(package)) as p:
    base_dir = Path(p)

print(f"package base dir {base_dir}")

_VALID_MODE_CHOICES = ["det", "deterministic", "prob", "probabilistic"]


def get_config_path_and_mode_pre_parse():
    """
    Resolve the config file path AND an optional explicit --mode/-m flag,
    before the rest of loader.py runs.

    Config path priority:
      1. Explicit -p/--param CLI argument
      2. ROMP_CONFIG_PATH environment variable
      3. ./params/config.in in the current working directory
      4. The packaged default (params/config.in)

    Mode priority (see build_cfg() below for how this is applied):
      An explicit -m/--mode CLI flag always wins over everything else —
      it forcibly overrides model_list, model_dir_list, and every
      metric/plot flag, even if the config file sets them explicitly.
      If not given here, mode falls back to the config file's own
      `mode` variable, then ROMP_MODE, then "det".
    """
    pre_parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,  # same reasoning as parser.py: this pre-parser
                             # also runs parse_known_args() against the full
                             # sys.argv, so it's exposed to flags meant for
                             # the *other* parser (create_parser). Disabling
                             # abbreviation matching here too prevents any
                             # future flag added to either parser from
                             # silently prefix-matching and corrupting a
                             # flag meant for the other.
    )
    pre_parser.add_argument("-p", "--param", default=None)
    pre_parser.add_argument(
        "-m", "--mode",
        default=None,
        choices=_VALID_MODE_CHOICES,
        help=(
            "Evaluation mode: 'det'/'deterministic' or 'prob'/'probabilistic'. "
            "Overrides model_list, model_dir_list, and all metric/plot flags — "
            "takes priority over the config file's own `mode` value and over "
            "the ROMP_MODE environment variable."
        ),
    )
    cli_args = [] if IS_JUPYTER else None
    args, _ = pre_parser.parse_known_args(args=cli_args)

    cwd_default = Path.cwd() / "params" / "config.in"
    resolved_path = (
        args.param
        or os.environ.get("ROMP_CONFIG_PATH")
        or (str(cwd_default) if cwd_default.exists() else None)
        or "params/config.in"  # packaged fallback default
    )
    return resolved_path, args.mode


config_path, _cli_mode = get_config_path_and_mode_pre_parse()
requested_path = ensure_config_exists(config_path)

config_item = set_dir(requested_path)

if not config_item.exists():
    config_item = base_dir / requested_path

if not config_item.exists():
    raise FileNotFoundError(f"Could not find configuration file: {requested_path}")

print(f"Using config file: {config_item}")
if _cli_mode:
    print(f"Mode explicitly set via CLI: {_cli_mode.upper()} (overrides config file and ROMP_MODE)")

if isinstance(config_item, Path):
    with open(config_item, "r") as f:
        params_in = f.read()
else:
    with resources.as_file(config_item) as actual_path:
        with open(actual_path, "r") as f:
            params_in = f.read()

params_in = "\n".join(
    line for line in params_in.splitlines() if not line.strip().startswith("#")
)

# Execute configuration string into current context
exec(params_in)

# Dynamically sanitize hardcoded '/Users/mesfind/data' paths
PROJ_DATA_DIR = os.environ.get("ROMP_DATA_DIR") or str(Path.cwd() / "data")
for key, val in list(globals().items()):
    if isinstance(val, str) and "/Users/mesfind/data" in val:
        globals()[key] = val.replace("/Users/mesfind/data", PROJ_DATA_DIR)

# Safely resolve all defined path variables
path_keys = [
    "work_dir", "pkg_dir", "ref_model_dir", "dir_out",
    "dir_fig", "obs_dir", "thresh_file", "shpfile_dir", "nc_mask"
]

for p_key in path_keys:
    if globals().get(p_key) is not None:
        p_val = globals()[p_key]

        if not isinstance(p_val, (str, os.PathLike)):
            hint = ""
            if isinstance(p_val, tuple) and len(p_val) == 1:
                hint = (
                    f" This looks like a trailing comma turned a string into "
                    f"a 1-element tuple. Check for `{p_key} = ...,` "
                    f"(with a trailing comma) in '{requested_path}' and "
                    f"remove the comma."
                )
            raise TypeError(
                f"Config error in '{requested_path}': '{p_key}' must be a "
                f"string or path, but got {type(p_val).__name__} with value "
                f"{p_val!r}.{hint}"
            )

        p_obj = Path(p_val).expanduser()
        if not p_obj.is_absolute():
            globals()[p_key] = set_dir(str(p_obj))
        else:
            globals()[p_key] = p_obj.resolve()

print("work_dir = ", globals().get("work_dir"))
print("pkg_dir = ", globals().get("pkg_dir"))
print("ref_model_dir = ", globals().get("ref_model_dir"))
print("out_dir = ", globals().get("dir_out"))
print("out_fig = ", globals().get("dir_fig"))
print("obs_dir = ", globals().get("obs_dir"))
print("thresh_file = ", globals().get("thresh_file"))
print("shpfile_dir = ", globals().get("shpfile_dir"))
print("nc_mask = ", globals().get("nc_mask"))

if globals().get("dir_fig"):
    os.makedirs(globals()["dir_fig"], exist_ok=True)
if globals().get("dir_out"):
    os.makedirs(globals()["dir_out"], exist_ok=True)

excluded_vars = {"f", "config_file_path", "params_in", "actual_path", "config_item"}

dic = {
    var: globals()[var]
    for var in globals()
    if not var.startswith("__")
    and not callable(globals()[var])
    and var not in globals().get("__builtins__", {})
    and var not in excluded_vars
    and var != "excluded_vars"
}

# ------------------------------------------------------------------------------
# Locate and import config.py (project-level config, defining
# BENCHMARK_MODEL_CATALOG and compute_mode_overrides()).
# ------------------------------------------------------------------------------
def _locate_and_import_project_config():
    try:
        import config as project_config
        return project_config
    except ModuleNotFoundError:
        pass

    candidate_dirs = []

    work_dir_val = globals().get("work_dir")
    if work_dir_val:
        candidate_dirs.append(str(work_dir_val))

    for env_var in ("AI_WEATHER_REPO", "PROJ_ROOT"):
        val = os.environ.get(env_var)
        if val:
            candidate_dirs.append(val)

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.py").exists():
            candidate_dirs.append(str(parent))

    for d in candidate_dirs:
        if d and os.path.isfile(os.path.join(d, "config.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import config as project_config
                return project_config
            except ModuleNotFoundError:
                continue

    raise ImportError(
        "Could not locate or import 'config.py' (project-level config, "
        "defining BENCHMARK_MODEL_CATALOG and compute_mode_overrides()). "
        f"Tried: sys.path, work_dir ({work_dir_val!r}), the AI_WEATHER_REPO "
        "and PROJ_ROOT environment variables, and walking up from "
        "momp/lib/loader.py. Fix by setting AI_WEATHER_REPO (or PROJ_ROOT) "
        "to the directory containing config.py, or ensure it is importable "
        "on sys.path."
    )


project_config = _locate_and_import_project_config()

_cfg = None
_cfg_ns = None
_setting = None
_overrides: dict = {}


def apply_overrides(**kwargs):
    """
    Register configuration overrides programmatically (e.g. from a
    notebook), without touching any config file. A `mode=` override here
    behaves the same as the CLI --mode flag: it forces model_list,
    model_dir_list, and every metric/plot flag for that mode, even if
    they were set explicitly elsewhere.
    """
    global _overrides, _cfg, _cfg_ns
    _overrides.update(kwargs)
    if _cfg is not None:
        _cfg.update(kwargs)
    if _cfg_ns is not None:
        for k, v in kwargs.items():
            setattr(_cfg_ns, k, v)


def reset_cfg():
    """Fully clear cached config/singletons — call when switching modes
    or config files within a single long-lived process (e.g. Jupyter)."""
    global _cfg, _cfg_ns, _setting, _overrides
    _cfg = None
    _cfg_ns = None
    _setting = None
    _overrides = {}


def build_cfg(cli_args=None):
    """
    Build configuration. Mode resolution, in priority order:
      1. Explicit -m/--mode CLI flag, or apply_overrides(mode=...) —
         FORCES every mode-dependent key (model_list, model_dir_list,
         FAR/BS/etc., plot flags), overriding whatever the config file
         set for them.
      2. The config file's own `mode` variable, or the ROMP_MODE
         environment variable, default "det" — only FILLS IN keys the
         config file left unset, so a fully explicit config (e.g. a
         training/demo file with its own model_list and metric flags)
         is never silently overridden.
    """
    if cli_args is None and IS_JUPYTER:
        cli_args = []

    base = dict(dic)

    explicit_mode = _cli_mode or _overrides.get("mode")
    if explicit_mode:
        mode = explicit_mode
        mode_overrides = project_config.compute_mode_overrides(mode)
        base.update(mode_overrides)  # explicit intent always wins outright
    else:
        mode = base.get("mode") or os.environ.get("ROMP_MODE", "det")
        mode_overrides = project_config.compute_mode_overrides(mode)
        for key, value in mode_overrides.items():
            base.setdefault(key, value)  # only fills gaps

    args = create_parser(base, cli_args=cli_args)
    cfg = base.copy()
    args_dict = vars(args)

    cli_overrides = {
        key: value
        for key, value in args_dict.items()
        if key in cfg and value is not None
    }
    cfg.update(cli_overrides)
    cfg.update(_overrides)  # explicit programmatic overrides always win last
    return cfg


def _check_model_year_coverage(cfg_dict):
    """
    Lightweight preflight sanity check: for each configured model, verify
    that a data file exists for every year in [start_date year, end_date
    year]. This does NOT replace real file reads at run time — it's a
    fast (filesystem stat only, no parallel workers, no xarray) warning
    that surfaces "this model has no data for year X" in under a second
    at startup, instead of after several minutes of parallel, multi-year
    processing failing partway through inside a ProcessPoolExecutor
    worker (which is what happens today if a model's available years
    don't cover the configured date range).
    """
    model_list = cfg_dict.get("model_list") or ()
    model_dir_list = cfg_dict.get("model_dir_list") or ()
    file_pattern_list = cfg_dict.get("file_pattern_list") or ()
    start_date = cfg_dict.get("start_date")
    end_date = cfg_dict.get("end_date")

    if not (model_list and model_dir_list and file_pattern_list and start_date and end_date):
        return  # not enough info to check; skip silently

    if not (len(model_list) == len(model_dir_list) == len(file_pattern_list)):
        return  # lengths already mismatched; ROMPValidator handles that error separately

    try:
        start_year = int(start_date[0])
        end_year = int(end_date[0])
    except (TypeError, IndexError, ValueError):
        return

    if start_year > end_year:
        return  # ROMPValidator's _check_temporal already flags this

    warnings_found = []
    for model, mdir, pattern in zip(model_list, model_dir_list, file_pattern_list):
        if not mdir or not os.path.isdir(mdir):
            warnings_found.append(f"  • {model}: directory does not exist: {mdir}")
            continue

        missing_years = [
            y for y in range(start_year, end_year + 1)
            if not os.path.exists(os.path.join(mdir, pattern.format(y)))
        ]
        if missing_years:
            preview = missing_years[:3]
            suffix = "..." if len(missing_years) > 3 else ""
            warnings_found.append(
                f"  • {model}: missing {len(missing_years)} year(s) in "
                f"[{start_year}-{end_year}] — e.g. {preview}{suffix} "
                f"(dir: {mdir}, pattern: {pattern})"
            )

    if warnings_found:
        print(
            "\nWARNING: some configured years have no data file for one or more models:\n"
            + "\n".join(warnings_found)
            + "\nThe run will likely fail partway through with a FileNotFoundError. "
              "Consider narrowing start_date/end_date (and start_year_clim/"
              "end_year_clim, if used) to the years all listed models actually "
              "cover.\n"
        )


def get_cfg(cli_args=None):
    """Returns the SAME SimpleNamespace instance on every call within this
    process (singleton), so mutations are visible everywhere downstream."""
    global _cfg, _cfg_ns

    if _cfg is None:
        _cfg = build_cfg(cli_args)

    if _cfg_ns is None:
        try:
            validator = ROMPValidator(_cfg)
            validator.validate()
            print("Configuration validated!")
        except ROMPConfigError as e:
            print(f"Error: Invalid Config!!! {e}")
            if IS_JUPYTER or _cfg.get("debug"):
                raise ROMPConfigError(f"Invalid configuration: {e}") from e
            else:
                sys.exit(1)

        _check_model_year_coverage(_cfg)

        _cfg_ns = SimpleNamespace(**_cfg)

    return _cfg_ns


def get_setting(cli_args=None):
    global _setting
    if _setting is None:
        cfg = get_cfg(cli_args)
        _setting = init_dataclass(Setting, vars(cfg))
    return _setting