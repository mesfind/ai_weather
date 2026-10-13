"""Compare up to 4 models for the same start date, lead time and region.

Saved runs load straight away; models without one run on the Spark one after
another (never two at once, so they don't compete for memory), and each joins
the chart as soon as it finishes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from . import catalog, jobs, obs, onset, store
from .regions import Box
from .results import crop

MAX_MODELS = 4
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706"]


def _plan(keys, init, lead_days, box, region_name, use_case, synthetic):
    """Per model: ('ready', path) | ('run', RunRequest) | ('skip', reason)."""
    out = {}
    for k in keys:
        m = catalog.BY_KEY[k]
        members = m.default_members if m.kind == "ensemble" else 1
        errs = [msg for lvl, msg in catalog.check(m, init, lead_days, members, box, region_name, use_case)
                if lvl == "error"]
        need = catalog.USE_CASE_NEEDS.get(use_case)
        path = store.find_any(k, init, lead_days * 24, synthetic)
        if need in m.missing:
            out[k] = ("skip", f"{m.name} does not forecast {catalog.VAR_LABELS[need]}")
        elif path:
            out[k] = ("ready", str(path))
        elif errs:
            out[k] = ("skip", errs[0].split(". ")[0])
        elif m.live or synthetic:
            out[k] = ("run", store.RunRequest(k, init, lead_days * 24, members, synthetic))
        else:
            out[k] = ("skip", f"no saved run, and {m.name} can't run live on this Spark")
    return out


@st.cache_data(max_entries=32, show_spinner=False)
def _area_series(path: str, mtime: float, box_t: tuple, init: pd.Timestamp, lead_days: int,
                 var: str) -> xr.DataArray:
    """Daily area-mean series (ensemble, time): rain (mm/day) over CHIRPS land points, or daily
    maximum 2 m temperature (°C, from the 6-hourly values)."""
    box = Box(*box_t)
    with xr.open_dataset(path) as ds:
        sub = crop(ds, Box(box.lat_min - 1, box.lat_max + 1, box.lon_min - 1, box.lon_max + 1),
                   lead_days * 24).load()
    if var == "tp":
        fc = onset.forecast_daily(sub, init)
        try:
            last = obs.last_day()
            like = obs.daily(box, last, last).isel(time=0, drop=True)
            fc = onset.to_grid(fc, like)
        except (OSError, FileNotFoundError):
            fc = fc.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max))
        return onset.area_mean(fc)
    t = sub.t2m.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max)) - 273.15
    t = t.sel(lead_time=t.lead_time > 0)
    day = init.normalize() + pd.to_timedelta((t.lead_time.values - 1) // 24, unit="D")
    t = t.assign_coords(day=("lead_time", day)).groupby("day").max("lead_time").rename(day="time")
    return onset.area_mean(t)


@st.cache_data(max_entries=16, show_spinner="Getting observations…")
def _obs_series(box_t: tuple, init: pd.Timestamp, lead_days: int, var: str) -> xr.DataArray | None:
    box = Box(*box_t)
    end = init.normalize() + pd.Timedelta(days=lead_days - 1)
    if var == "tp":
        try:
            return onset.area_mean(obs.daily(box, init.normalize(), end))
        except (OSError, FileNotFoundError):
            return None
    # ERA5 2 m temperature from Google's copy (as in the event movie), daily max of 6-hourly values
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "event_movie"))
    import run_event_movie as rem
    da = rem._era5("2m_temperature", init.normalize(), end + pd.Timedelta(hours=18),
                   (box.lon_min, box.lon_max, box.lat_min, box.lat_max))
    da = (da - 273.15).resample(time="1D").max()
    return onset.area_mean(da)


def _status_rows(plan, state):
    rows = []
    for k, (what, info) in plan.items():
        name = catalog.BY_KEY[k].name
        if k in state["paths"]:
            rows.append((name, "loaded"))
        elif what == "skip":
            rows.append((name, f"not included: {info}"))
        elif state.get("running") == k:
            s = jobs.status(state["job"])
            rows.append((name, f"running on the Spark: {s.get('message', '')} "
                               f"({100 * s.get('progress', 0):.0f}%, {s.get('elapsed_s', 0):.0f} s)"))
        elif k in state["failed"]:
            rows.append((name, f"run failed: {state['failed'][k]}"))
        else:
            rows.append((name, "waiting its turn"))
    return pd.DataFrame(rows, columns=["Model", "Status"])


def section(init, lead_days, box: Box, region_name: str, use_case: str, synthetic: bool,
            defn: onset.Definition | None, main_job_running: bool) -> None:
    ss = st.session_state
    var = "t2m" if use_case == "Heat" else "tp"
    keys = st.multiselect(f"Models to compare (up to {MAX_MODELS})", [m.key for m in catalog.MODELS],
                          default=[], max_selections=MAX_MODELS, key="cmp_keys",
                          format_func=lambda k: catalog.BY_KEY[k].name)
    if not keys:
        st.caption("Pick 2–4 models. Each uses this page's start date, lead time and region; "
                   "ensembles use their default number of members.")
        return
    plan = _plan(keys, init, lead_days, box, region_name, use_case, synthetic)
    sig = (tuple(keys), str(init), lead_days, synthetic, use_case,
           (box.lat_min, box.lat_max, box.lon_min, box.lon_max))
    state = ss.get("cmp")
    if state is None or state["sig"] != sig:
        state = ss["cmp"] = dict(sig=sig, paths={}, queue=[], running=None, job=None, failed={},
                                 started=False)
    to_run = [k for k, (w, _) in plan.items() if w == "run"]
    label = ("Load and compare" if not to_run else
             f"Load saved runs and run {len(to_run)} model{'s' if len(to_run) > 1 else ''} one after another")
    if not state["started"]:
        if st.button(label, type="primary", key="cmp_go"):
            state["started"] = True
            state["paths"] = {k: info for k, (w, info) in plan.items() if w == "ready"}
            state["queue"] = list(to_run)
            st.rerun()
        if to_run:
            est = sum(catalog.load_timings().get(k, {}).get("run_s", 0) +
                      catalog.load_timings().get(k, {}).get("load_s", 0) for k in to_run)
            st.caption(f"ⓘ {', '.join(catalog.BY_KEY[k].name for k in to_run)} "
                       f"{'have' if len(to_run) > 1 else 'has'} no saved run for these settings "
                       f"(about {est / 60:.0f} min on the Spark in total).")
        st.dataframe(_status_rows(plan, state).assign(
            Status=lambda d: d.Status.where(d.Status.str.startswith("not included"),
                                            ["saved run" if plan[k][0] == "ready" else "will run"
                                             for k in plan])), hide_index=True, width="stretch")
        return

    busy = bool(state["running"] or state["queue"])
    # only this block refreshes while runs are going, not the whole page
    st.fragment(_live, run_every=3 if busy else None)(
        plan, state, keys, init, lead_days, box, region_name, use_case, var, defn, main_job_running)


def _live(plan, state, keys, init, lead_days, box, region_name, use_case, var, defn, main_job_running):
    was_busy = bool(state["running"] or state["queue"])
    # advance the queue: one run at a time, and not while the main run is going
    if state["running"]:
        s = jobs.status(state["job"])
        if s["state"] == "done":
            state["paths"][state["running"]] = s["output"]
            state["running"] = state["job"] = None
        elif s["state"] not in ("queued", "running", "unknown"):
            state["failed"][state["running"]] = s.get("message", "see the job log")
            state["running"] = state["job"] = None
    if not state["running"] and state["queue"] and not main_job_running:
        k = state["queue"].pop(0)
        state["running"], state["job"] = k, jobs.launch(plan[k][1])

    st.dataframe(_status_rows(plan, state), hide_index=True, width="stretch")
    if main_job_running and state["queue"] and not state["running"]:
        st.caption("Waiting for the forecast above to finish before starting the next run.")
    if state["paths"]:
        chart(state["paths"], keys, init, lead_days, box, region_name, use_case, var, defn)
    if was_busy and not (state["running"] or state["queue"]):
        st.rerun()  # all done: one full rerun stops the timer


def chart(paths, keys, init, lead_days, box, region_name, use_case, var, defn):
    box_t = (box.lat_min, box.lat_max, box.lon_min, box.lon_max)
    fig = go.Figure()
    unit = "mm/day" if var == "tp" else "°C"
    for i, k in enumerate(k for k in keys if k in paths):
        p = Path(paths[k])
        ser = _area_series(str(p), p.stat().st_mtime, box_t, init, lead_days, var)
        name, color = catalog.BY_KEY[k].name, COLORS[i % len(COLORS)]
        mean = ser.mean("ensemble") if "ensemble" in ser.dims else ser
        if "ensemble" in ser.dims and ser.sizes["ensemble"] > 1:
            lo, hi = ser.min("ensemble"), ser.max("ensemble")
            fig.add_scatter(x=np.concatenate([ser.time.values, ser.time.values[::-1]]),
                            y=np.concatenate([hi.values, lo.values[::-1]]), fill="toself",
                            line=dict(width=0), fillcolor=color, opacity=0.15, showlegend=False,
                            hoverinfo="skip", name=f"{name} range")
            name += f" (mean of {ser.sizes['ensemble']})"
        fig.add_scatter(x=mean.time.values, y=mean.values, mode="lines+markers", name=name,
                        line=dict(color=color, width=2))
    obs_name = "CHIRPS" if var == "tp" else "ERA5"
    want_obs = var == "tp" or st.checkbox("Add ERA5 observations (fetched from the cloud; "
                                          "a minute or two the first time)", key="cmp_obs")
    if want_obs:
        o = _obs_series(box_t, init, lead_days, var)
        if o is not None and o.sizes.get("time", 0):
            fig.add_scatter(x=o.time.values, y=o.values, mode="lines+markers", name=f"Observed ({obs_name})",
                            line=dict(color="black", width=2, dash="dash"))
        elif var == "tp":
            st.caption("No CHIRPS observations for these dates yet.")
    what = {"tp": "area-mean daily rainfall (land)", "t2m": "area-mean daily maximum 2 m temperature"}[var]
    fig.update_layout(title=f"{region_name}: {what}", yaxis_title=unit, height=420,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    if var == "tp":
        st.caption("Each model's rainfall is put on the CHIRPS grid and averaged over land, so it "
                   "compares like for like with the observations. Shaded: range across members.")
    else:
        st.caption("Daily maximum of the 6-hourly values (models and ERA5 alike). Shaded: range "
                   "across members.")
    if use_case == "Onset" and defn is not None:
        onset_table(paths, keys, init, lead_days, box, region_name, defn)


def onset_table(paths, keys, init, lead_days, box, region_name, defn):
    from .onset_view import _day_field, compute
    box_t = (box.lat_min, box.lat_max, box.lon_min, box.lon_max)
    search_from = onset.search_start(defn, region_name, init)
    rows, obs_row = [], None
    for k in keys:
        if k not in paths:
            continue
        p = Path(paths[k])
        r = compute(str(p), p.stat().st_mtime, box_t, init, lead_days, defn, search_from)
        land = r["land"]
        n = max(1, int(land.sum()))
        f = r["f_on"]
        conf = (f.status == onset.CONFIRMED)
        poss = (f.status == onset.POSSIBLE)
        fday, _ = _day_field(f, r["init_day"], land)
        during = fday >= 1
        share = lambda m: 100 * float((m & land).sum()) / n / (f.sizes.get("ensemble", 1))
        rows.append((catalog.BY_KEY[k].name, f"{share(during & conf):.0f}%", f"{share(during & poss):.0f}%"))
        if obs_row is None and r["o_on"] is not None:
            oday, _ = _day_field(r["o_on"], r["init_day"], land)
            n_days = (r["fc_end"] - r["init_day"]).days + 1
            obs_row = ("Observed (CHIRPS)", f"{100 * float(((oday >= 1) & (oday <= n_days) & land).sum()) / n:.0f}%", "")
    if obs_row:
        rows.append(obs_row)
    st.markdown(f"**Onset during the forecast** ({defn.name}, searching from {search_from:%-d %b %Y})")
    st.dataframe(pd.DataFrame(rows, columns=["", "Onset (confirmed)", "Possible onset"]),
                 hide_index=True, width="stretch")
    st.caption("Share of land points (averaged over members for ensembles).")
