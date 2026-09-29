from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from momp.io.input import load_imd_rainfall
from momp.io.input import load_thresh_file
from momp.stats.detect import detect_observed_onset
from momp.graphics.func_map import spatial_metrics_map
from momp.utils.practical import restore_args


# Defaults for the map. Any of them can be overridden by passing the same name
# as a keyword (e.g. ax=my_ax, cmap="viridis"); overlapping keys are removed from
# **kwargs first so they are never passed twice.
_MAP_DEFAULTS = dict(
    fig=None, ax=None, figsize=(8, 6), cmap="YlOrRd", n_colors=10,
    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
    show_ylabel=True, title="onset day", rect_box=True,
)


def spatial_onset(year, *, obs_dir, obs_file_pattern, obs_var,
                  thresh_file, thresh_var, wet_threshold,
                  date_filter_year, start_date, end_date,
                  wet_init, wet_spell, dry_spell, dry_threshold, dry_extent,
                  fallback_date, mok, **kwargs):
    """Map the observed onset day-of-year for one year.

    Forecast-only settings (members, init_days, max_forecast_day, day_bins,
    onset_percentage_threshold, ...) are no longer required: they are still
    accepted and forwarded through **kwargs when present.
    """
    kwargs = restore_args(spatial_onset, kwargs, locals())

    thresh_slice = load_thresh_file(**kwargs)

    print("Loading observational rainfall data...")
    rainfall_ds = load_imd_rainfall(year, **kwargs)

    print("Detecting observed onset...")
    onset_da = detect_observed_onset(rainfall_ds, thresh_slice, year, **kwargs)

    # Day-of-year for datetime results; keep NaT/NaN as NaN
    if np.issubdtype(onset_da.dtype, np.datetime64):
        onset_doy = onset_da.dt.dayofyear.astype(float).where(~onset_da.isnull())
    else:
        onset_doy = onset_da.astype(float)

    # Use the first verification window unless one was given explicitly
    if kwargs.get("verification_window") is None:
        windows = kwargs.get("verification_window_list")
        if windows:
            kwargs["verification_window"] = windows[0]

    plot_kw = dict(_MAP_DEFAULTS)
    for key in _MAP_DEFAULTS:
        if key in kwargs:
            plot_kw[key] = kwargs.pop(key)

    # Guard: an empty / all-NaN grid makes cartopy build a degenerate map extent,
    # which surfaces later as a cryptic GEOSException (unclosed LinearRing).
    n_valid = int(onset_doy.notnull().sum())
    print(f"Onset grid {dict(onset_doy.sizes)}; cells with a detected onset: {n_valid}")
    if onset_doy.size == 0 or n_valid == 0:
        for dim in onset_doy.dims:
            if dim in onset_doy.coords and onset_doy[dim].size:
                print(f"  {dim}: {float(onset_doy[dim].min()):.3f} .. {float(onset_doy[dim].max()):.3f}")
        print("No onset to plot. Check: region selection (empty grid?), start_date/end_date, "
              "wet/dry thresholds, fallback_date, and obs_var / obs_file_pattern.")
        return onset_doy

    # cartopy's gridliner can raise GEOSException inside plt.tight_layout() with some
    # cartopy/shapely combinations, so skip tight_layout for this call
    # (upgrading cartopy is the real fix).
    with patch.object(plt, "tight_layout", lambda *a, **k: None):
        spatial_metrics_map(onset_doy, f"observation year {year}", **plot_kw, **kwargs)

    return onset_doy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from momp.lib.loader import get_cfg

    cfg = get_cfg()
    params = cfg.copy() if isinstance(cfg, dict) else vars(cfg).copy()
    params.pop("year", None)  # passed positionally

    spatial_onset(2020, **params)
    plt.show()