"""Result views — the plug-in point for the visualization code.

Every view has the signature

    def view(ds: xr.Dataset, ctx: VizContext) -> None

and draws into the current Streamlit container (st.pyplot / st.plotly_chart /
st.image …). `ds` is already cropped to the selected region and lead time and
follows demo3/contract.py (tp mm per 6 h, t2m K, z500 m2 s-2,
dims ensemble × lead_time × lat × lon).

To plug in new plots, replace a function below or point VIEWS at your own
module. The functions here are PLACEHOLDERS so the app works end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from .regions import Box


@dataclass
class VizContext:
    model_name: str
    kind: str                 # "deterministic" | "ensemble"
    init: pd.Timestamp
    lead_hours: int
    region_name: str
    box: Box
    use_case: str             # "Heat" | "Precipitation" | "Onset"
    settings: dict = field(default_factory=dict)   # thresholds etc. from the sidebar
    synthetic: bool = False


def crop(ds: xr.Dataset, box: Box, lead_hours: int) -> xr.Dataset:
    return ds.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max),
                  lead_time=slice(0, lead_hours))


def daily_precip(ds: xr.Dataset) -> xr.DataArray:
    """24 h totals [mm/day] per forecast day (day 1 = hours 6–24)."""
    tp = ds.tp.sel(lead_time=ds.lead_time > 0)
    day = ((tp.lead_time - 1) // 24 + 1).rename("day")
    return tp.groupby(day).sum("lead_time", min_count=4).rename("tp_daily")


# ── placeholder drawing helpers ──────────────────────────────────────────────

def _map(da: xr.DataArray, box: Box, title: str, cmap: str, units: str, contours=None):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cf
        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([box.lon_min, box.lon_max, box.lat_min, box.lat_max], crs=ccrs.PlateCarree())
        kw = dict(transform=ccrs.PlateCarree())
        ax.add_feature(cf.COASTLINE, linewidth=0.7)
        ax.add_feature(cf.BORDERS, linewidth=0.5, linestyle=":")
    except Exception:  # cartopy data unavailable offline
        fig, ax = plt.subplots(figsize=(7, 5))
        kw = {}
    m = ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, shading="auto", **kw)
    if contours is not None:
        ax.contour(da.lon, da.lat, contours, colors="k", linewidths=0.6, **kw)
    fig.colorbar(m, ax=ax, shrink=0.8, label=units)
    ax.set_title(title, fontsize=11)
    return fig


def _lead_picker(ds: xr.Dataset, key: str) -> int:
    leads = [int(h) for h in ds.lead_time.values]
    return st.select_slider("Forecast hour", leads, value=leads[min(len(leads) - 1, 4)], key=key,
                            format_func=lambda h: f"+{h} h (day {h / 24:g})")


def _placeholder_note():
    st.caption("Placeholder view — to be replaced by the Demo 3 visualization module.")


# ── views ────────────────────────────────────────────────────────────────────

def temperature_view(ds: xr.Dataset, ctx: VizContext) -> None:
    _placeholder_note()
    t = ds.t2m - 273.15
    h = _lead_picker(ds, "t2m_lead")
    field_ = t.sel(lead_time=h).mean("ensemble")
    thr = ctx.settings.get("heat_threshold_c")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(_map(field_, ctx.box, f"2 m temperature, +{h} h (ensemble mean)", "RdYlBu_r", "°C"))
    with c2:
        area = t.mean(["lat", "lon"])
        fig = go.Figure()
        for m in area.ensemble.values:
            fig.add_scatter(x=area.valid_time.values, y=area.sel(ensemble=m).values,
                            mode="lines", name=f"member {int(m)}")
        if thr is not None:
            fig.add_hline(y=thr, line_dash="dash", annotation_text=f"{thr} °C")
        fig.update_layout(title="Area-mean 2 m temperature", yaxis_title="°C", height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    if ctx.use_case == "Heat" and thr is not None:
        frac = (t.max("lead_time") > thr).mean("ensemble") * 100
        st.pyplot(_map(frac, ctx.box, f"% of members with T2m > {thr} °C at any time", "Reds", "%"))


def precipitation_view(ds: xr.Dataset, ctx: VizContext) -> None:
    _placeholder_note()
    daily = daily_precip(ds)
    days = [int(d) for d in daily.day.values]
    d = st.select_slider("Forecast day", days, value=days[0], key="tp_day")
    thr = ctx.settings.get("precip_threshold_mm")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(_map(daily.sel(day=d).mean("ensemble"), ctx.box,
                       f"Daily precipitation, day {d} (ensemble mean)", "YlGnBu", "mm/day"))
    with c2:
        area = daily.mean(["lat", "lon"])
        fig = go.Figure()
        for m in area.ensemble.values:
            fig.add_bar(x=area.day.values, y=area.sel(ensemble=m).values, name=f"member {int(m)}")
        fig.update_layout(title="Area-mean daily precipitation", xaxis_title="forecast day",
                          yaxis_title="mm/day", barmode="group", height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    if ctx.use_case == "Precipitation" and thr is not None:
        prob = (daily > thr).any("day").mean("ensemble") * 100
        st.pyplot(_map(prob, ctx.box, f"% of members with a day above {thr} mm", "Blues", "%"))


def z500_view(ds: xr.Dataset, ctx: VizContext) -> None:
    _placeholder_note()
    h = _lead_picker(ds, "z500_lead")
    z = (ds.z500.sel(lead_time=h).mean("ensemble") / 9.80665)
    st.pyplot(_map(z, ctx.box, f"500 hPa geopotential height, +{h} h", "viridis", "m", contours=z))


def onset_view(ds: xr.Dataset, ctx: VizContext) -> None:
    _placeholder_note()
    s = ctx.settings
    wet_thr, wet_days = s.get("wet_threshold_mm", 20), s.get("wet_spell_days", 3)
    st.info(f"Placeholder: first forecast day where the {wet_days}-day rainfall total reaches "
            f"{wet_thr} mm. The dry-spell check ({s.get('dry_spell_days', 7)} dry days within "
            f"{s.get('dry_extent_days', 21)} days) is not applied here, since it needs rainfall "
            "beyond the forecast window.")
    daily = daily_precip(ds).mean("ensemble")
    roll = daily.rolling(day=wet_days, min_periods=wet_days).sum()
    hit = roll >= wet_thr
    first = xr.where(hit.any("day"), hit.argmax("day") + 1, np.nan)
    st.pyplot(_map(first, ctx.box, "First day meeting the wet-spell rule", "viridis_r", "forecast day"))


VIEWS = {
    "Temperature": temperature_view,
    "Precipitation": precipitation_view,
    "Geopotential (z500)": z500_view,
    "Onset": onset_view,
}
