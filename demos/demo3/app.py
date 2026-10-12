"""Demo 3 — Running Your First AI Weather Forecast.

    streamlit run app.py

Step 1 (in the terminal) starts the container and this app. In the app:
Step 2 choose model, start date, lead time, region and use case;
Step 3 run the model (or load a saved run); then explore the results.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import xarray as xr

from demo3 import catalog, jobs, movie, results, store, theme
from demo3.regions import COUNTRIES, GROUPS, Box, country_box

st.set_page_config(page_title="Demo 3 · AI Weather Forecast", layout="wide")
theme.apply()


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@st.cache_resource(max_entries=4)
def open_output(path: str, mtime: float) -> xr.Dataset:
    return xr.open_dataset(path).load()


def event_results(path, m, full, run_req, box, region_name, lead_days, use_case):
    """Panchali's obs-vs-forecast event movie and event-track map for the loaded run."""
    kind, var = ("heat", "t2m") if use_case == "Heat" else ("precip", "tp")
    if var in catalog.missing_in_run(full):
        other = "Precipitation" if use_case == "Heat" else "Heat"
        st.info(f"{m.name} does not forecast {catalog.VAR_LABELS[var]}, so there is no "
                f"{use_case.lower()} movie for this run. Switch the use case to {other}, or "
                "choose another model.")
        return
    st.caption("Animated map of observations (ERA5) next to the forecast, with the regional "
               "series building up day by day, then a map of the event's footprint and track.")
    c2, c3 = st.columns(2)
    reduce = c2.selectbox("Regional series", ["mean", "max", "p95", "p99"], key="movie_reduce",
                          index=1 if kind == "heat" else 0,
                          format_func={"mean": "area mean", "max": "area maximum",
                                       "p95": "95th percentile", "p99": "99th percentile"}.get)
    auto = c3.checkbox("Automatic threshold", value=True, key="movie_auto",
                       help="High percentile of the observations (p95 heat, p97.5 rain)")
    threshold = None if auto else c3.number_input(
        "Threshold (°C)" if kind == "heat" else "Threshold (mm/day)",
        value=40.0 if kind == "heat" else 50.0, key="movie_thr")
    run_days = int(full.lead_time.max()) // 24
    days = min(lead_days, run_days)
    obs_end = pd.Timestamp(catalog.era5_dates()[1])  # last day of (preliminary) ERA5
    if run_req.init >= obs_end:
        st.info(f"Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so there is nothing yet "
                f"to compare this {run_req.init:%-d %b %Y} forecast with.")
        return
    if run_req.init + pd.Timedelta(days=days) > obs_end:
        days = max(1, (obs_end - run_req.init).days)
        st.info(f"Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so the movie covers "
                f"the first {days} day{'s' if days > 1 else ''} of the forecast.")
    req = movie.request_for(path, m.name, kind, run_req.init, days, box,
                            region_name, reduce, threshold)
    s = movie.status(req)

    if s["state"] == "none":
        if st.button("Make event movie", type="primary", key="movie_go"):
            movie.launch(req)
            st.rerun()
        st.caption("The first movie for a new date fetches ERA5 observations from the cloud and "
                   "can take a few minutes; later ones reuse the cached data.")
    elif s["state"] == "running":
        st.info("Rendering the movie… (fetching observations and drawing one frame per day)")
        with st.expander("Log"):
            st.code(s["log"] or "…")
        time.sleep(3)
        st.rerun()
    elif s["state"] == "failed":
        st.error(f"Movie failed: {s.get('error')}")
        with st.expander("Log", expanded=True):
            st.code(s["log"])
        if st.button("Try again", key="movie_retry"):
            movie.launch(req)
            st.rerun()
    else:
        if s.get("gif"):
            # st.image shows only a GIF's first frame; embed it so it animates
            gif64 = base64.b64encode(Path(s["gif"]).read_bytes()).decode()
            st.html(f'<img src="data:image/gif;base64,{gif64}" style="width:100%" '
                    'alt="Observations vs forecast, by valid day">')
            st.caption("Observations vs forecast, by valid day")
            st.download_button("Download movie (GIF)", Path(s["gif"]).read_bytes(),
                               file_name=Path(s["gif"]).name, mime="image/gif", key="movie_dl")
        if s.get("png"):
            st.image(s["png"], caption="Event footprint and track")
            st.download_button("Download track map (PNG)", Path(s["png"]).read_bytes(),
                               file_name=Path(s["png"]).name, mime="image/png", key="track_dl")
        st.caption("Observations are ERA5 reanalysis. AIFS and several other models train on ERA5, "
                   "so this is not a fully independent check.")


ss = st.session_state
ss.setdefault("job_id", None)
ss.setdefault("result", None)        # (path, request fields) of the run being viewed

timings = catalog.load_timings()
saved = store.list_saved()
gpu = gpu_name()

# ── sidebar: developer options ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Developer options")
    synthetic = st.toggle("Use synthetic output", value=False,
                          help="Generate fake fields in the output format instead of running a model. "
                               "For building the visualizations before the model runners exist.")
    st.caption(f"Output folder: `{store.OUTPUT_DIR}`")
    if not saved.empty:
        st.markdown("**Saved runs**")
        st.dataframe(saved.drop(columns="path"), hide_index=True, width="stretch")

# ── header ───────────────────────────────────────────────────────────────────
theme.header("AI Forecast Lab", "Demo 3", f"{'GPU: ' + gpu if gpu else 'No GPU detected'}", live=bool(gpu))
theme.page_title("Running Your First AI Weather Forecast",
                 "Pick a model, a start date and a region, run the forecast on your Spark, "
                 "and compare it with what was observed.")

ready = [m for m in catalog.MODELS if m.live]
free_gb = shutil.disk_usage(store.OUTPUT_DIR if store.OUTPUT_DIR.exists() else "/").free / 1e9
theme.stats([
    ("Models", str(len(catalog.MODELS)), theme.BLUE),
    ("Live on this Spark", f"{len(ready)} / {len(catalog.MODELS)}", theme.TEAL),
    ("Saved runs", str(len(saved)), theme.MUTED),
    ("Free disk", f"{free_gb:,.0f} GB", theme.ORANGE),
])

# ── Step 2a: model ───────────────────────────────────────────────────────────
theme.section("Select model", "Step 2 ·")
mode = st.radio("Mode", ["Deterministic", "Probabilistic"], horizontal=True, label_visibility="collapsed")
kind = "deterministic" if mode == "Deterministic" else "ensemble"
choices = [m for m in catalog.MODELS if m.kind == kind]

# Store the model's key, not the Model object: a session that outlives a code update
# would otherwise hold an object of the old class.
model_key = st.selectbox("Model", [m.key for m in choices], label_visibility="collapsed",
                         format_func=lambda k: f"{catalog.BY_KEY[k].name}  "
                                               f"{catalog.timing_label(catalog.BY_KEY[k], timings)}")
model = catalog.BY_KEY[model_key]
def model_tags(m: catalog.Model) -> str:
    kind_tag = theme.tag(m.kind, "ens" if m.kind == "ensemble" else "det")
    return kind_tag + (theme.tag("live", "ok") if m.live else theme.tag("saved runs only", "warn"))


theme.cards([{
    "name": m.name, "org": m.org, "tags": model_tags(m),
    "note": f"{catalog.timing_label(m, timings)} · starts from {m.init_source}"
            + "".join(f" · no {catalog.VAR_LABELS[v]}" for v in m.missing),
    "color": theme.PURPLE if m.kind == "ensemble" else theme.TEAL,
    "selected": m.key == model.key, "dim": not m.live and not synthetic,
} for m in choices])
if model.notes:
    st.caption(f"ⓘ {model.notes}")
if model.slow:
    st.caption("ⓘ This model is slow on the Spark. In class, load a saved run where possible.")
st.markdown("**What this model needs on the Spark**")
results.requirements(model, timings, store.OUTPUT_DIR)

# ── Step 2b: forecast setup ──────────────────────────────────────────────────
theme.section("Forecast setup")
c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
init_lo, init_hi = model.date_range()
default_init = min(max(date(2025, 7, 1), init_lo), init_hi)
init_day = c1.date_input("Start date (UTC)", value=default_init, min_value=init_lo,
                         max_value=init_hi, key=f"init_{model.key}")
init_hour = c2.selectbox("Hour", [0, 12], format_func=lambda h: f"{h:02d}Z")
lead_days = c3.slider("Lead time (days)", 1, model.max_lead_days, min(10, model.max_lead_days))
members = (c4.number_input("Ensemble members", 1, model.max_members, model.default_members)
           if model.kind == "ensemble" else 1)
if model.kind != "ensemble":
    c4.text_input("Ensemble members", "1 (deterministic)", disabled=True)
init = pd.Timestamp(init_day) + pd.Timedelta(hours=init_hour)
st.caption(f"ⓘ Start dates for {model.name}: {init_lo:%-d %b %Y} – {init_hi:%-d %b %Y}. "
           f"{model.date_reason()}")

# ── Step 2c: region ──────────────────────────────────────────────────────────
theme.section("Region")
r1, r2 = st.columns([1, 3])
region_mode = r1.radio("Region", ["Country", "Custom box"], label_visibility="collapsed")
if region_mode == "Country":
    group_of = {c: g for g, cs in GROUPS.items() for c in cs}
    country = r2.selectbox("Country", COUNTRIES, format_func=lambda c: f"{c}  ·  {group_of[c]}")
    box = country_box(country)
    region_name = country
    r2.caption(f"Plot area (country plus a margin): {box.label()}")
else:
    b1, b2, b3, b4 = r2.columns(4)
    box = Box(b1.number_input("Lat min", -90.0, 90.0, 3.0, 0.25),
              b2.number_input("Lat max", -90.0, 90.0, 15.0, 0.25),
              b3.number_input("Lon min", -180.0, 180.0, 33.0, 0.25),
              b4.number_input("Lon max", -180.0, 180.0, 48.0, 0.25))
    region_name = "Custom region"
    if not box.is_valid():
        r2.error("Min must be below max; latitude within ±90°, longitude within ±180°.")

# ── Step 2d: use case ────────────────────────────────────────────────────────
theme.section("What do you want to look at?")
use_case = st.segmented_control("Use case", ["Heat", "Precipitation"], default="Precipitation",
                                label_visibility="collapsed") or "Precipitation"
st.caption("**Heat:** daily maximum 2 m temperature (°C).  **Precipitation:** daily rainfall "
           "total (mm). The result is a movie of the forecast next to the observations, and a "
           "map of where the event went.")

problems = catalog.check(model, init, lead_days, int(members), box, region_name, use_case)
for level, msg in problems:
    (st.error if level == "error" else st.warning)(msg)
blocked = any(level == "error" for level, _ in problems)

# ── Step 3: run ──────────────────────────────────────────────────────────────
theme.section("Run the forecast", "Step 3 ·")
req = store.RunRequest(model.key, init, lead_days * 24, int(members), synthetic)
saved_path = store.find_saved(req)
theme.pills([("Model", model.name), ("Mode", mode), ("Start", f"{init:%Y-%m-%d %HZ}"),
             ("Lead", f"{lead_days} days"), ("Members", str(members)), ("Region", region_name),
             ("Use case", use_case)] + ([("Output", "synthetic")] if synthetic else []))

can_run = (model.live or synthetic) and box.is_valid() and not blocked
b1, b2 = st.columns([2, 3])
if saved_path:
    clicked = b1.button("Load saved run", type="primary", width="stretch",
                        disabled=not box.is_valid() or blocked)
    b2.markdown(theme.tag("saved run found — loads instantly", "ok"), unsafe_allow_html=True)
else:
    clicked = b1.button("Run forecast", type="primary", width="stretch",
                        disabled=not can_run or ss.job_id is not None)
    b2.markdown(theme.tag(f"will run on the Spark {catalog.timing_label(model, timings)}", "det")
                if can_run else theme.tag("fix the settings above first" if blocked else
                                          f"no saved run for these settings, and {model.name} can't run live yet", "warn"),
                unsafe_allow_html=True)

if clicked:
    if saved_path:
        ss.result = (str(saved_path), req)
    else:
        ss.job_id = jobs.launch(req)
        ss.result = None
    st.rerun()

if ss.job_id:
    s = jobs.status(ss.job_id)
    if s["state"] in ("queued", "running", "unknown"):
        st.progress(s.get("progress", 0.0), text=f"{s.get('message', '')} · {s['elapsed_s']:.0f} s elapsed")
        with st.expander("Log"):
            st.code(jobs.log_tail(ss.job_id) or "…")
        time.sleep(2)
        st.rerun()
    elif s["state"] == "done":
        ss.result = (s["output"], req)
        ss.job_id = None
        st.success(f"Forecast finished in {s.get('runtime_s', s['elapsed_s']):.0f} s.")
        st.rerun()
    else:
        st.error(f"Run failed: {s.get('message')}")
        with st.expander("Log", expanded=True):
            st.code(jobs.log_tail(ss.job_id, 40))
        if st.button("Dismiss"):
            ss.job_id = None
            st.rerun()

# ── Results ──────────────────────────────────────────────────────────────────
if ss.result:
    path, run_req = ss.result
    p = Path(path)
    if not p.exists():
        st.warning("The saved output file is gone; run the forecast again.")
    else:
        full = open_output(path, p.stat().st_mtime)
        m = catalog.BY_KEY[run_req.model]
        theme.section("Results")
        if full.attrs.get("synthetic"):
            st.caption("ⓘ Synthetic output — not a real forecast.")
        st.markdown(f"**How this forecast was made** · {m.name}, started "
                    f"{run_req.init:%-d %b %Y %HZ} from {full.attrs.get('init_source', m.init_source)}")
        results.pipeline(path, full, m, timings)
        movie_tab, maps_tab = st.tabs(["Event movie (vs observations)", "Forecast maps"])
        with movie_tab:
            event_results(path, m, full, run_req, box, region_name, lead_days, use_case)
        with maps_tab:
            ds = results.crop(full, box, lead_days * 24)
            if ds.sizes["lat"] < 2 or ds.sizes["lon"] < 2:
                st.error(f"{region_name} holds fewer than 2 grid points of {m.name}'s "
                         f"{m.grid_deg:g}° grid; choose a larger region.")
            else:
                results.field_maps(ds, m, box, region_name)
        data = results.netcdf_bytes(path, p.stat().st_mtime,
                                    (box.lat_min, box.lat_max, box.lon_min, box.lon_max),
                                    lead_days * 24)
        st.download_button(f"Download this forecast for {region_name} (NetCDF, "
                           f"{len(data) / 1e6:.1f} MB)", data, mime="application/x-netcdf",
                           file_name=f"{m.key}_{run_req.init:%Y%m%dT%H}_{lead_days}d_"
                                     f"{region_name.replace(' ', '_').lower()}.nc")
        st.caption(f"Variables tp (mm per 6 h), t2m (K), z500 (m² s⁻²) by member and lead time. "
                   f"The full global file is on the Spark at `{p.relative_to(store.OUTPUT_DIR.parent)}` "
                   f"({p.stat().st_size / 1e6:.0f} MB).")
