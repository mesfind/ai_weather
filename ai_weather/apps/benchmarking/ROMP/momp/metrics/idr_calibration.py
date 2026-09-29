"""
IDR (Isotonic Distributional Regression) calibration for ROMP onset
forecasts, adapting the gridded IDR engine (momp/stats/idr.py) to
onset-probability calibration.

Covariate X: raw `predicted_prob` (ensemble-derived exceedance fraction).
Target y:    `observed_onset` (0/1).
Calibrated probability = 1 - F(0), the IDR-calibrated P(onset=1).
Cross-validation: Leave-One-Year-Out (idr_loyo_cv's philosophy), since
archives here are short (6-10 years) and fitting/scoring on the same
years would overstate skill.
Grouping: fit separately per lead-time bin (bin_label).

Scope: PROBABILISTIC track only (BS/RPS/AUC). Extending this to make
deterministic-track models (FAR/MAE/MR) comparable on the same footing
is a follow-up requiring explicit choices (covariate, aggregation
level) — not attempted here to avoid guessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from momp.stats.idr import IDRModel, idr_loyo_cv  # adjust import path to match where idr.py lives


def _pivot_bin_to_grid(df_bin: pd.DataFrame) -> tuple[xr.DataArray, xr.DataArray]:
    """Pivot one lead-time bin's long-format rows into (init_time, lat, lon) grids."""
    df_bin = df_bin.sort_values(["init_time", "lat", "lon"])
    forecast_da = df_bin.set_index(["init_time", "lat", "lon"])["predicted_prob"].to_xarray()
    obs_da = df_bin.set_index(["init_time", "lat", "lon"])["observed_onset"].to_xarray()
    return forecast_da, obs_da


def _loyo_calibrated_probability(
    forecast_da: xr.DataArray,
    obs_da: xr.DataArray,
    min_valid: int,
    min_onset_events: int,
) -> xr.DataArray:
    """
    Leave-one-year-out calibrated P(onset=1) per (init_time, lat, lon).

    Duplicates idr_loyo_cv's fold loop directly against IDRModel,
    because the LOYO kernel in idr.py returns {crps, q10, q50, q90,
    pit} and doesn't (yet) expose the exceedance-probability output
    that idr_on_grid's kernel does. If this becomes a hot path,
    consider adding exceedance_thresholds support to idr_loyo_cv /
    _loyo_kernel directly so both paths share one engine — not done
    here to avoid modifying already-optimized code without being able
    to test it against real data.
    """
    years = forecast_da["init_time"].dt.year.values
    unique_years = np.unique(years)
    lat_vals = forecast_da["lat"].values
    lon_vals = forecast_da["lon"].values

    out = np.full(forecast_da.shape, np.nan)
    X_all = forecast_da.values  # (init_time, lat, lon)
    y_all = obs_da.values

    for i in range(len(lat_vals)):
        for j in range(len(lon_vals)):
            X_cell, y_cell = X_all[:, i, j], y_all[:, i, j]

            for yr in unique_years:
                test_mask = years == yr
                train_mask = ~test_mask

                X_tr, y_tr = X_cell[train_mask], y_cell[train_mask]
                valid_tr = np.isfinite(X_tr) & np.isfinite(y_tr)

                if valid_tr.sum() < min_valid:
                    continue
                if np.sum(y_tr[valid_tr] > 0.1) < min_onset_events:
                    continue
                if len(np.unique(y_tr[valid_tr])) < 2:
                    continue

                X_te = X_cell[test_mask]
                valid_te = np.isfinite(X_te)
                if not valid_te.any():
                    continue

                model = IDRModel(increasing=True).fit(X_tr[valid_tr], y_tr[valid_tr])
                preds = model.predict(X_te[valid_te])
                p_onset = 1.0 - np.asarray(preds.cdf_at(0.0)).reshape(-1)  # P(y > 0)

                te_positions = np.where(test_mask)[0][valid_te]
                out[te_positions, i, j] = p_onset

    return xr.DataArray(
        out, dims=["init_time", "lat", "lon"],
        coords={"init_time": forecast_da["init_time"], "lat": forecast_da["lat"], "lon": forecast_da["lon"]},
        name="calibrated_prob",
    )


def calibrate_onset_probabilities(
    forecast_obs_df: pd.DataFrame,
    min_valid: int = 10,
    min_onset_events: int = 5,
    use_parallel: bool = False,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Adds `calibrated_prob`, `idr_crps`, `idr_pit`, `idr_calibrated`
    columns to forecast_obs_df via per-bin, leave-one-year-out IDR.

    Rows where calibration couldn't be fit (too few training samples
    or onset events in a fold — common at the edges of a short
    archive) fall back to raw `predicted_prob`, flagged via
    `idr_calibrated=False` so downstream reporting can tell calibrated
    rows apart from fallback rows rather than silently mixing them.
    """
    required_cols = {"init_time", "lat", "lon", "bin_label", "predicted_prob", "observed_onset"}
    missing = required_cols - set(forecast_obs_df.columns)
    if missing:
        raise ValueError(f"forecast_obs_df is missing required columns: {missing}")

    out_frames = []

    for bin_label, df_bin in forecast_obs_df.groupby("bin_label"):
        forecast_da, obs_da = _pivot_bin_to_grid(df_bin)
        forecast_da = forecast_da.sortby("init_time")
        obs_da = obs_da.sortby("init_time")

        n_years = len(np.unique(forecast_da["init_time"].dt.year.values))
        if n_years < 3:
            # Too few independent years for a meaningful LOYO fold.
            df_bin = df_bin.copy()
            df_bin["calibrated_prob"] = df_bin["predicted_prob"]
            df_bin["idr_crps"] = np.nan
            df_bin["idr_pit"] = np.nan
            df_bin["idr_calibrated"] = False
            out_frames.append(df_bin)
            continue

        cv_ds = idr_loyo_cv(
            forecast=forecast_da, obs=obs_da,
            spatial_dims=("lat", "lon"), time_dim="init_time",
            min_valid=min_valid, min_wet_days=min_onset_events,
            use_parallel=use_parallel, n_jobs=n_jobs,
        )
        calibrated_prob_da = _loyo_calibrated_probability(
            forecast_da, obs_da, min_valid=min_valid, min_onset_events=min_onset_events,
        )

        calibrated_df = calibrated_prob_da.to_dataframe(name="calibrated_prob").reset_index()
        crps_df = cv_ds["crps"].to_dataframe().reset_index()
        pit_df = cv_ds["pit"].to_dataframe().reset_index()

        merged = df_bin.merge(calibrated_df, on=["init_time", "lat", "lon"], how="left")
        merged = merged.merge(crps_df, on=["init_time", "lat", "lon"], how="left")
        merged = merged.merge(pit_df, on=["init_time", "lat", "lon"], how="left")
        merged = merged.rename(columns={"crps": "idr_crps", "pit": "idr_pit"})

        merged["idr_calibrated"] = merged["calibrated_prob"].notna()
        merged["calibrated_prob"] = merged["calibrated_prob"].fillna(merged["predicted_prob"])

        out_frames.append(merged)

    return pd.concat(out_frames, ignore_index=True)