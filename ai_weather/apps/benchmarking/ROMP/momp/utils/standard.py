import pandas as pd


# Standard dimension order: time-like dims first, then spatial
_STANDARD_DIM_ORDER = ["init_time", "time", "member", "step", "lat", "lon"]


def _find_coord(coord_list, keywords, purpose):
    """
    Find the coordinate name matching any of `keywords` (case-insensitive
    substring match). Raises a clear, actionable error instead of a bare
    IndexError when nothing matches — the previous behavior of `[...][0]`
    on an empty list gave no indication of which file/coordinate caused
    the failure, which made every naming mismatch a multi-step debugging
    exercise. This shows the actual available coordinate names directly
    in the error, so a genuinely new naming convention (e.g. a different
    model version using "lead_time" instead of "step") can be identified
    and fixed in one step.
    """
    matches = [v for v in coord_list if any(k in v.lower() for k in keywords)]
    if not matches:
        raise KeyError(
            f"Could not identify the '{purpose}' coordinate. Searched for "
            f"names containing any of {keywords} (case-insensitive) among "
            f"this file's actual coordinates: {list(coord_list)}. "
            f"This file likely uses a naming convention not yet covered — "
            f"add the correct keyword to the relevant list in "
            f"momp/utils/standard.py."
        )
    return matches[0]


def dim_order_fmt(ds):
    """
    Transpose all variables in ds so their dimensions follow the standard order:
        (init_time, time, member, step, lat, lon)

    Only the dimensions actually present in each variable are used — missing ones
    are skipped.  For example, a variable with dims (lat, lon, init_time, step)
    becomes (init_time, step, lat, lon).

    Any dimension not listed in _STANDARD_DIM_ORDER is appended at the end in
    their original relative order, so unknown dims are never dropped.
    """
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        current_dims = list(da.dims)

        # Build target order: known dims in standard order, then any unknown dims
        ordered = [d for d in _STANDARD_DIM_ORDER if d in current_dims]
        remainder = [d for d in current_dims if d not in _STANDARD_DIM_ORDER]
        target_dims = ordered + remainder

        if target_dims != current_dims:
            da = da.transpose(*target_dims)

        new_vars[var] = da

    return ds.assign(new_vars)


def dim_fmt(ds):
    """Standardize dimension names"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        lat_coords = _find_coord(coord_list, ["lat"], "latitude")
        lon_coords = _find_coord(coord_list, ["lon"], "longitude")
        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        time_coords = _find_coord(coord_list, ["time", "date"], "time")
        ds = ds.rename({time_coords: "time"})

    return dim_order_fmt(ds)


def dim_fmt_model(ds):
    """Standardize dimension names for deterministic reforecast model data"""
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        lat_coords = _find_coord(coord_list, ["lat"], "latitude")
        lon_coords = _find_coord(coord_list, ["lon"], "longitude")
        ds = ds.rename({lat_coords: "lat", lon_coords: "lon"})

    if "init_time" not in coord_list:
        time_coords = _find_coord(
            coord_list,
            ["init_time", "time", "forecast_reference_time", "date"],
            "init_time",
        )
        ds = ds.rename({time_coords: "init_time"})

    if "step" not in coord_list:
        # FIX: Added "lead_time", "lead", "step", "forecast_period", "fhour"
        # to cover all common naming conventions across different model outputs.
        #
        # Your AIFS files use "lead_time" for the forecast lead dimension,
        # which was not in the original keyword list ["day", "prediction_timedelta"].
        step_coords = _find_coord(
            coord_list,
            [
                "step",
                "lead_time",
                "lead",
                "day",
                "prediction_timedelta",
                "forecast_period",
                "fhour",
                "forecast_hour",
            ],
            "step",
        )
        ds = ds.rename({step_coords: "step"})

    # convert TimedeltaIndex to integer (days)
    if isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    return dim_order_fmt(ds)


def dim_fmt_model_ensemble(ds):
    """Standardize dimension names for probabilistic reforecast model data"""

    ds = dim_fmt_model(ds)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        ensemble_coords = _find_coord(
            coord_list,
            ["number", "sample", "member", "realization", "ensemble"],
            "ensemble member",
        )
        ds = ds.rename({ensemble_coords: "member"})

    return dim_order_fmt(ds)